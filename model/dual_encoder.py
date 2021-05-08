import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel


def _average_sequence_embeddings(sequence_output, mask):
    sequence_length = (mask == 1).sum(dim=-1)
    sequence_length = torch.clamp(sequence_length, 1, None)
    sum_embedding = torch.sum(sequence_output * mask[:, :, None], dim=1)
    average_embedding = sum_embedding / sequence_length[:, None]
    return average_embedding


class DualEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(DualEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
        self.repr_type = config.repr_type
        self.repr_normalized = config.repr_normalized

    def forward(self, input_ids, attention_mask):
        sequence_output, cls_embeddings = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # choose representation via `repr_type`
        if self.repr_type == "avg":
            text_embedding = _average_sequence_embeddings(
                sequence_output=sequence_output,
                mask=attention_mask
            )
        else:
            text_embedding = cls_embeddings
        # execuate normalization if `is_normalize` is true
        if self.repr_normalized:
            text_embedding = F.normalize(text_embedding, dim=1)
        return text_embedding