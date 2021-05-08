import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import BertConfig
from torch.utils.data import DataLoader, SequentialSampler

from dataset.dual_encoder_eval_docset import DualEncoderEvalDocSet
from model.dual_encoder import DualEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO
)


def initilaze_embedding_memmap(embedding_dir, collection_size, embedding_size):
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    pids_memmap_dir = os.path.join(embedding_dir, "pids.memmap")
    pids_memmap = np.memmap(
        pids_memmap_dir,
        mode="w+",
        dtype="int32",
        shape=(collection_size, )
    )
    embedding_memmap_dir = os.path.join(embedding_dir, "doc_embeddings.memmap")
    embedding_memmap = np.memmap(
        embedding_memmap_dir,
        mode="w+",
        dtype="float32",
        shape=(collection_size, embedding_size)
    )
    return pids_memmap, embedding_memmap


def precompute_embeddings(load_model_path, collection_memmap_dir, embedding_dir,
    device, batch_size_per_gpu, max_doc_length):

    # annotate devices
    if "cpu" in device or not torch.cuda.is_available():
        device = torch.device("cpu")
        gpu_count = 0
    else:
        _, ids = device.split(":")
        device_id_list = [int(idx) for idx in ids.split(",")]
        device = torch.device("cuda:{}".format(device_id_list[0]))
        gpu_count = len(device_id_list)
    logger.info("   precompute on device:{}".format(device))

    # init model
    config = BertConfig.from_pretrained(load_model_path)
    model = DualEncoder.from_pretrained(load_model_path, config=config)
    if gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=device_id_list)
    model.to(device)

    # loading eval collection dataset
    eval_doc_dataset = DualEncoderEvalDocSet(
        collection_memmap_dir=collection_memmap_dir,
        max_doc_length=max_doc_length
    )
    eval_doc_sampler = SequentialSampler(eval_doc_dataset)
    batch_size = batch_size_per_gpu * max(1, gpu_count)
    eval_doc_dataloader = DataLoader(
        eval_doc_dataset,
        batch_size=batch_size,
        sampler=eval_doc_sampler,
        collate_fn=DualEncoderEvalDocSet.collate_func
    )
    collection_size = len(eval_doc_dataset)
    total_steps = len(eval_doc_dataloader)
    docid_to_memmapid = eval_doc_dataset.get_doc_id_to_memmap_id()

    # annotate output memmap
    doc_id_memmap, embedding_memmap = initilaze_embedding_memmap(
        embedding_dir=embedding_dir,
        collection_size=collection_size,
        embedding_size=config.hidden_size
    )

    # generate doc embeddings
    model.eval()
    for batch in tqdm(eval_doc_dataloader, desc="precompute doc embedding", total=total_steps):
        doc_input_ids = batch["doc_input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pids = batch["pids"]
        doc_embedding = model(
            input_ids=doc_input_ids,
            attention_mask=attention_mask
        )
        doc_embedding = doc_embedding.detach().cpu().numpy()
        for idx, doc_id in enumerate(pids):
            memmap_id = docid_to_memmapid[doc_id]
            doc_id_memmap[memmap_id] = doc_id
            embedding_memmap[memmap_id] = doc_embedding[idx]