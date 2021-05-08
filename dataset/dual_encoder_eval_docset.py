import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from dataset import CollectionDataset, QueryDataset


class DualEncoderEvalDocSet(Dataset):
    def __init__(self, collection_memmap_dir, max_doc_length):
        super(DualEncoderEvalDocSet, self).__init__()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_doc_length = max_doc_length
        self.collection = CollectionDataset(collection_memmap_dir)
        self.pids = self.collection.pids

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        doc_input_ids = [self.cls_id] + self.collection[idx][:self.max_doc_length - 2] + [self.sep_id]
        data = {
            "doc_input_ids": doc_input_ids,
            "pid": pid
        }
        return data

    def get_doc_id_to_memmap_id(self):
        docid_to_memmapid = dict()
        for memmap_id, doc_id in enumerate(self.pids):
            docid_to_memmapid[doc_id] = memmap_id
        return docid_to_memmapid

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
        pids = [x["pid"] for x in batch]
        doc_input_ids = [x["doc_input_ids"] for x in batch]
        attention_mask = [[1 for i in range(len(x))] for x in doc_input_ids]
        data = {
            "pids": pids,
            "doc_input_ids": cls._pack_tensor_2D(doc_input_ids, default=0),
            "attention_mask": cls._pack_tensor_2D(attention_mask, default=0)
        }
        return data