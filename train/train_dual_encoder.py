import os
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from model.dual_encoder import DualEncoder
from dataset.dual_encoder_train_dataset import DualEncoderTrainingSet
from utils.similarity_functions import dot_product, l1_distance, l2_distance


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO
)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu_count > 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def save_model(args, model, update_steps):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_to_save = model.module if args.gpu_count > 1 else model
    save_dir = os.path.join(args.checkpoint_dir, "step-{}".format(update_steps))
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, "args.bin"))


def train_epoch(args, model, criterion, optimizer, scheduler, train_dataloader, writer,
    update_steps, train_loss, similarity_func):
    model.train()
    device = model.device
    train_dataloader = tqdm(train_dataloader)

    model.zero_grad()
    for step, batch in enumerate(train_dataloader):
        query_input_ids = batch["query_input_ids"].to(device)
        doc_input_ids = batch["doc_input_ids"].to(device)
        query_attention_mask = batch["query_attention_mask"].to(device)
        doc_attention_mask = batch["doc_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        if args.loss_type == "cross-entropy":
            labels = labels[:,0]

        query_embeddings = model(query_input_ids, query_attention_mask)
        doc_embeddings = model(doc_input_ids, doc_attention_mask)

        rel_score = similarity_func(query_embeddings, doc_embeddings)

        loss = criterion(rel_score, labels)
        if args.gpu_count > 1:
            loss = loss.mean()
        if args.gradient_accumulate_steps > 1:
            loss = loss / args.gradient_accumulate_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        train_loss += loss.item()

        if (step + 1) % args.gradient_accumulate_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            update_steps += 1
            description = "Avg. batch train loss:{:.6f}".format(train_loss / update_steps)
            train_dataloader.set_description(description)

            if args.logging_steps > 0 and update_steps % args.logging_steps == 0:
                writer.add_scalar('lr', scheduler.get_lr()[0], update_steps)
                writer.add_scalar("train/loss", train_loss / update_steps, update_steps)
            
            if args.save_steps > 0 and update_steps % args.save_steps == 0:
                save_model(args, model, update_steps)


def train(args):
    # annotate devices
    if "cpu" in args.device or not torch.cuda.is_available():
        device = torch.device("cpu")
        args.gpu_count = 0
    else:
        _, ids = args.device.split(":")
        device_id_list = [int(idx) for idx in ids.split(",")]
        device = torch.device("cuda:{}".format(device_id_list[0]))
        args.gpu_count = len(device_id_list)
    
    # init tensorboard writer
    writer = SummaryWriter(args.logging_dir)

    # set seed
    set_seed(args)

    # prepare training data
    train_dataset = DualEncoderTrainingSet(
        max_query_length=args.max_query_length,
        max_doc_length=args.max_doc_length,
        tripplets_and_qrels_dir=args.tripplets_and_qrels_dir,
        collection_memmap_dir=args.collection_memmap_dir,
        tokenize_dir=args.tokenize_dir
    )
    train_sampler = SequentialSampler(train_dataset)
    assert int(args.batch_size_per_gpu) % 2 == 0
    args.batch_size = args.batch_size_per_gpu * max(1, args.gpu_count)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=DualEncoderTrainingSet.collate_func,
    )
    total_steps = len(train_dataloader) // args.gradient_accumulate_steps * args.train_epochs

    # init model
    config = BertConfig.from_pretrained(args.load_model_path)
    config.return_dict = False
    config.repr_normalized = args.repr_normalized
    if args.repr_type not in ["cls", "avg"]:
        raise ValueError("Only support `cls` and `avg` for argument `repr_type`!")
    else:
        config.repr_type = args.repr_type
    model = DualEncoder.from_pretrained(args.load_model_path, config=config)
    if args.gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=device_id_list)
    model.to(device)

    # init loss
    if args.loss_type == 'hinge':
        criterion = nn.MultiLabelMarginLoss()
    elif args.loss_type == 'cross-entropy':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise ValueError('Only support `hinge` and `cross-entropy` for argument `loss_type`!')

    # init optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # init similarity function
    if args.similarity_type == "dot-product":
        similarity_func = dot_product
    elif args.similarity_type == "L1":
        similarity_func = l1_distance
    elif args.similarity_type == "L2":
        similarity_func = l2_distance
    else:
        raise ValueError("Only support `dot-product`, `L1` and `L2` for argument `similarity_type`!")
    
    train_loss = 0.0
    update_steps = 0
    for epoch in range(args.train_epochs):
        logger.info("********Train epoch {}********".format(epoch + 1))
        train_epoch(
            args=args,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            writer=writer,
            update_steps=update_steps,
            train_loss=train_loss,
            similarity_func=similarity_func
        )
        save_model(args, model, "epoch-{}".format(epoch + 1))


def get_args():
    parser = argparse.ArgumentParser()
    # data related args
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/msmarco-passage/collection_memmap/complete")
    parser.add_argument("--tokenize_dir", type=str, default="./data/msmarco-passage/tokenize")
    parser.add_argument("--tripplets_and_qrels_dir", type=str, default="./data/msmarco-passage/tripplets_and_qrels")
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=256)
    # train setting args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulate_steps", type=int, default=2)
    parser.add_argument("--batch_size_per_gpu", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # model parameters args
    parser.add_argument("--load_model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--repr_type", type=str, default="avg", help="choose from cls / avg")
    parser.add_argument("--repr_normalized", action="store_true")
    parser.add_argument("--similarity_type", type=str, default="dot-product", help="choose from dot-product / L1 / L2")
    parser.add_argument("--loss_type", type=str, default='hinge',help='choose from cross-entropy / hinge')
    # save and logging args
    parser.add_argument("--checkpoint_dir", type=str, default="./data/msmarco-passage/checkpoints")
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--logging_dir", type=str, default="./data/msmarco-passage/log")
    parser.add_argument("--logging_steps", type=int, default=100)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logger.info(args)
    train(args)


if __name__ == "__main__":
    main()