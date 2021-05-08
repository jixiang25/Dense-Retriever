import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from dataset import CollectionDataset, QueryDataset


class DualEncoderEvalQuerySet(Dataset):
    def __init__(self, tokenize_dir, max_query_length):
        super(DualEncoderEvalQuerySet, self).__init__()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.queries = QueryDataset(tokenize_dir, "dev")
        self.qids = []
        for qid in self.queries.queries.keys():
            self.qids.append(qid)

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        query_input_ids = [self.cls_id] + self.queries[self.qids[idx]][:self.max_query_length - 2] + [self.sep_id]
        data = {
            "query_input_ids": query_input_ids,
            "qid": qid
        }
        return data

    @classmethod
    def _pack_tensor_2D(cls, lst, default, length=None):
        batch_size = len(lst)
        length = max(len(l) for l in lst) if length is None else length
        packed_tensor = default * torch.ones((batch_size, length), dtype=torch.int64)
        for i, l in enumerate(lst):
            packed_tensor[i,:len(l)] = torch.tensor(l, dtype=torch.int64)
        return packed_tensor

    @classmethod
    def collate_func(cls, batch):
        qids = [x["qid"] for x in batch]
        query_input_ids = [x["query_input_ids"] for x in batch]
        attention_mask = [[1 for i in range(len(x))] for x in query_input_ids]
        data = {
            "qids": qids,
            "query_input_ids": cls._pack_tensor_2D(query_input_ids, default=0),
            "attention_mask": cls._pack_tensor_2D(attention_mask, default=0)
        }
        return data