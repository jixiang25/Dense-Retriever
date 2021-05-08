import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from queue import PriorityQueue
from torch.utils.data import SequentialSampler, DataLoader
from transformers import BertConfig

from dataset.dual_encoder_eval_queryset import DualEncoderEvalQuerySet
from model.dual_encoder import DualEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO
)


def load_doc_embeddings(embedding_dir, hidden_size):
    pids_memmap_dir = os.path.join(embedding_dir, "pids.memmap")
    pids_memmap = np.memmap(
        pids_memmap_dir,
        mode="c",
        dtype="int32"
    )
    doc_embedding_memmap_dir = os.path.join(embedding_dir, "doc_embeddings.memmap")
    collection_size = len(pids_memmap)
    doc_embedding_memmap = np.memmap(
        doc_embedding_memmap_dir, 
        mode="c",
        shape=(collection_size, hidden_size),
        dtype="float32"
    )
    return pids_memmap, doc_embedding_memmap


def output_retrieved_ranking_result(retrieved_result_dir, retrieved_result, hit_num):
    if not os.path.exists(retrieved_result_dir):
        os.makedirs(retrieved_result_dir)
    score_file = os.path.join(retrieved_result_dir, "top{}.score.txt".format(hit_num))
    rank_file = os.path.join(retrieved_result_dir, "top{}.rank.txt".format(hit_num))
    with open(score_file, "w") as fout_score, open(rank_file, "w") as fout_rank:
        for query_id, query_pq in retrieved_result.items():
            if query_pq.qsize() != hit_num:
                raise ValueError("Query {} should have {} hits".format(query_id, hit_num))
            for i in range(hit_num):
                doc_score, doc_id = query_pq.get_nowait()
                fout_score.write("{}\t{}\t{}\n".format(query_id, doc_id, doc_score))
                fout_rank.write("{}\t{}\t{}\n".format(query_id, doc_id, hit_num - i))


def retrieve_docs(load_model_path, device, batch_size, hit_num, embedding_dir, tokenize_dir, 
    retrieved_result_dir, docs_per_gpu, max_query_length):

    # annotate devices
    if "cpu" in device or not torch.cuda.is_available():
        device = torch.device("cpu")
        gpu_count = 0
    else:
        _, ids = device.split(":")
        device_id_list = [int(idx) for idx in ids.split(",")]
        device = torch.device("cuda:{}".format(device_id_list[0]))
        gpu_count = len(device_id_list)
    if gpu_count > 1:
        raise ValueError("Only single gpu should be annotated while online retrieving")
    logger.info("   online inference queries on device:{}".format(device))

    # init model
    config = BertConfig.from_pretrained(load_model_path)
    model = DualEncoder.from_pretrained(load_model_path, config=config)
    model.to(device)

    # loading eval queries dataset
    eval_query_dataset = DualEncoderEvalQuerySet(
        tokenize_dir=tokenize_dir,
        max_query_length=max_query_length
    )
    eval_query_sampler = SequentialSampler(eval_query_dataset)
    eval_query_dataloader = DataLoader(
        eval_query_dataset,
        sampler=eval_query_sampler,
        batch_size=batch_size,
        collate_fn=DualEncoderEvalQuerySet.collate_func
    )
    total_steps = len(eval_query_dataloader)

    # load doc embeddings
    pids_memmap, doc_embedding_memmap = load_doc_embeddings(
        embedding_dir=embedding_dir,
        hidden_size=config.hidden_size
    )

    # inference query representation, and retrieve nearest doc embeddings
    model.eval()
    doc_num = len(pids_memmap)
    doc_block_num = doc_num // docs_per_gpu + 1 if doc_num % docs_per_gpu != 0 else doc_num // docs_per_gpu
    hit_num = min(hit_num, doc_num)
    topk_num = min(hit_num, docs_per_gpu)
    query_rank = {}

    for doc_block_id in range(doc_block_num):
        logger.info("   retrieve in doc_block_id:{}".format(doc_block_id))
        start_doc_id = doc_block_id * docs_per_gpu
        end_doc_id = min((doc_block_id + 1) * docs_per_gpu, doc_num)
        logger.info("   start_doc_id: {}\tend_doc_id: {}".format(start_doc_id, end_doc_id))
        block_doc_embedding = torch.from_numpy(doc_embedding_memmap[start_doc_id: end_doc_id]).to(device)
        for batch in tqdm(eval_query_dataloader, desc="online retrieve docs", total=total_steps):
            query_input_ids = batch["query_input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            qids = batch["qids"]
            query_embedding = model(
                input_ids=query_input_ids,
                attention_mask=attention_mask
            )
            for idx, qid in enumerate(qids):
                if qid not in query_rank:
                    query_rank[qid] = PriorityQueue(maxsize=hit_num)
                if config.similarity_type == "dot-product":
                    score = torch.sum(query_embedding[idx] * block_doc_embedding, dim=-1)
                elif config.similarity_type == "L1":
                    score = -torch.sum(torch.abs(query_embedding[idx] - block_doc_embedding), dim=-1)
                elif config.similarity_type == "L2":
                    score = -torch.sum(torch.pow(query_embedding[idx] - block_doc_embedding, exponent=2), dim=-1)
                else:
                    raise ValueError("Similarity type `{}` does not exsits!".format(config.similarity_type))
                top_score, top_indices = torch.topk(score, k=topk_num)
                top_indices = top_indices + start_doc_id
                top_score, top_indices = top_score.cpu(), top_indices.cpu().numpy()
                cur_pq = query_rank[qid]
                for i in range(topk_num):
                    doc_id = int(pids_memmap[top_indices[i]])
                    doc_score = top_score[i].item()
                    if cur_pq.full():
                        lowest_score, lowest_doc_id = cur_pq.get_nowait()
                        if lowest_score >= doc_score:
                            cur_pq.put_nowait((lowest_score, lowest_doc_id))
                        else:
                            cur_pq.put_nowait((doc_score, doc_id))
                    else:
                        cur_pq.put_nowait((doc_score, doc_id))
    
    # output result for metric calculating
    output_retrieved_ranking_result(
        retrieved_result_dir=retrieved_result_dir,
        retrieved_result=query_rank,
        hit_num=hit_num
    )