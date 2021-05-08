import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer


class CollectionDataset(object):
    def __init__(self, collection_memmap_dir):
        pid_memmap_dir = os.path.join(collection_memmap_dir, "pids.memmap")
        self.pids = np.memmap(pid_memmap_dir, dtype='int32')
        length_memmap_dir = os.path.join(collection_memmap_dir, "lengths.memmap")
        self.lengths = np.memmap(length_memmap_dir, dtype='int32')
        self.collection_size = len(self.pids)
        token_ids_memmap_idr = os.path.join(collection_memmap_dir, "token_ids.memmap")
        self.token_ids = np.memmap(token_ids_memmap_idr, dtype='int32', shape=(self.collection_size, 512))

    def __len__(self):
        return self.collection_size

    def __getitem__(self, idx):
        return self.token_ids[idx, :self.lengths[idx]].tolist()


class QueryDataset(object):
    def __init__(self, tokenize_dir, mode):
        self.queries = {}
        tokenized_queries_dir = os.path.join(tokenize_dir, "tokenized_queries.{}.json".format(mode))
        with open(tokenized_queries_dir) as fin:
            for line in tqdm(fin, desc="loading queries for {}".format(mode)):
                data = json.loads(line)
                self.queries[int(data["id"])] = data["ids"]
    
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]