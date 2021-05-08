import argparse
import os
import json
import shutil
import logging
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO
)


def preprocess_complete_collection(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # tokenize collection and convert it to memmap
    collection_dir = os.path.join(args.official_data_dir, "collection.tsv")
    collection_size = sum(1 for line in open(collection_dir))
    max_doc_length = 512

    complete_collection_dir = os.path.join(args.collection_memmap_dir, "complete")
    if not os.path.exists(complete_collection_dir):
        os.makedirs(complete_collection_dir)
    token_ids_memmap = np.memmap(
        os.path.join(complete_collection_dir, "token_ids.complete.memmap"),
        dtype="int32",
        mode="w+",
        shape=(collection_size, max_doc_length)
    )
    pids_memmap = np.memmap(
        os.path.join(complete_collection_dir, "pids.complete.memmap"),
        dtype="int32",
        mode="w+",
        shape=(collection_size, )
    )
    lengths_memmap = np.memmap(
        os.path.join(complete_collection_dir, "lengths.complete.memmap"),
        dtype="int32",
        mode="w+",
        shape=(collection_size, )
    )

    with open(collection_dir) as fin:
        for idx, line in enumerate(tqdm(fin, total=collection_size, desc="Complete collection preprocessing")):
            doc_id, doc_content = line.split("\t")
            doc_id = int(doc_id)
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc_content))
            doc_length = min(max_doc_length, len(token_ids))
            pids_memmap[idx] = doc_id
            lengths_memmap[idx] = doc_length
            token_ids_memmap[idx, :doc_length] = token_ids[:doc_length]

    # tokenize queries in train set
    if not os.path.exists(args.tokenize_dir):
        os.makedirs(args.tokenize_dir)
    queries_train_dir = os.path.join(args.official_data_dir, "queries.train.tsv")
    tokenized_queries_train_dir = os.path.join(args.tokenize_dir, "tokenized_queries.train.json")
    with open(queries_train_dir) as fin, open(tokenized_queries_train_dir, "w") as fout:
        for line in fin:
            query_id, query_content = line.split("\t")
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query_content))
            fout.write(json.dumps({
                "id": int(query_id),
                "ids": token_ids,
            }) + "\n")


def preprocess_eval_collection(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # tokenize eval collection and convert it to memmap
    eval_toprank_dir = os.path.join(args.official_data_dir, "top1000.dev")
    eval_collection_id = set()
    with open(eval_toprank_dir) as fin:
        for line in fin:
            query_id, doc_id, _, _ = line.split("\t")
            doc_id = int(doc_id)
            eval_collection_id.add(doc_id)

    eval_collection_size = len(eval_collection_id)
    max_doc_length = 512

    eval_collection_dir = os.path.join(args.collection_memmap_dir, "eval")
    if not os.path.exists(eval_collection_dir):
        os.makedirs(eval_collection_dir)
    token_ids_memmap = np.memmap(
        os.path.join(eval_collection_dir, "token_ids.memmap"),
        dtype="int32",
        mode="w+",
        shape=(eval_collection_size, max_doc_length)
    )
    pids_memmap = np.memmap(
        os.path.join(eval_collection_dir, "pids.memmap"),
        dtype="int32",
        mode="w+",
        shape=(eval_collection_size, )
    )
    lengths_memmap = np.memmap(
        os.path.join(eval_collection_dir, "lengths.memmap"),
        dtype="int32",
        mode="w+",
        shape=(eval_collection_size, )
    )

    idx = 0
    collection_dir = os.path.join(args.official_data_dir, "collection.tsv")
    collection_size = sum(1 for line in open(collection_dir))
    with open(collection_dir) as fin:
        for line in tqdm(fin, total=collection_size, desc="Eval collection preprocessing"):
            doc_id, doc_content = line.split("\t")
            doc_id = int(doc_id)
            if doc_id not in eval_collection_id:
                continue
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc_content))
            doc_length = min(max_doc_length, len(token_ids))
            pids_memmap[idx] = doc_id
            lengths_memmap[idx] = doc_length
            token_ids_memmap[idx, :doc_length] = token_ids[:doc_length]
            idx += 1

    # tokenize queries in train set
    if not os.path.exists(args.tokenize_dir):
        os.makedirs(args.tokenize_dir)
    queries_dev_dir = os.path.join(args.official_data_dir, "queries.dev.small.tsv")
    tokenized_queries_dev_dir = os.path.join(args.tokenize_dir, "tokenized_queries.dev.json")
    with open(queries_dev_dir) as fin, open(tokenized_queries_dev_dir, "w") as fout:
        for line in fin:
            query_id, query_content = line.split("\t")
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query_content))
            fout.write(json.dumps({
                "id": int(query_id),
                "ids": token_ids,
            }) + "\n")


def generate_tripplets(args):
    if not os.path.exists(args.tripplets_and_qrels_dir):
        os.makedirs(args.tripplets_and_qrels_dir)
    source_tripplets_file = os.path.join(args.official_data_dir, "qidpidtriples.train.small.tsv")
    target_tripplets_file = os.path.join(args.tripplets_and_qrels_dir, "tripplets.train.tsv")
    shutil.copyfile(source_tripplets_file, target_tripplets_file)


def generate_qrels(args):
    if not os.path.exists(args.tripplets_and_qrels_dir):
        os.makedirs(args.tripplets_and_qrels_dir)
    source_qrels_file = os.path.join(args.official_data_dir, "qrels.train.tsv")
    target_qrels_file = os.path.join(args.tripplets_and_qrels_dir, "qrels.train.tsv")
    shutil.copyfile(source_qrels_file, target_qrels_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_data_dir", type=str, default="./data/msmarco-passage/official_data")
    parser.add_argument("--tripplets_and_qrels_dir", type=str, default="./data/msmarco-passage/tripplets_and_qrels")
    parser.add_argument("--tokenize_dir", type=str, default="./data/msmarco-passage/tokenize")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/msmarco-passage/collection_memmap")
    parser.add_argument("--preprocess_eval_collection", action="store_true")
    parser.add_argument("--preprocess_complete_collection", action="store_true")
    parser.add_argument("--generate_tripplets", action="store_true")
    parser.add_argument("--generate_qrels", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logger.info(args)
    if args.preprocess_complete_collection:
        preprocess_complete_collection(args)
    if args.preprocess_eval_collection:
        preprocess_eval_collection(args)
    if args.generate_tripplets:
        generate_tripplets(args)
    if args.generate_qrels:
        generate_qrels(args)


if __name__ == "__main__":
    main()