import torch
import torch.nn.functional as F


def dot_product(query_embeddings, doc_embeddings):
    return query_embeddings.matmul(doc_embeddings.transpose(1, 0))


def l1_distance(query_embeddings, doc_embeddings):
    # batch_size * batch_size * embedding_dim
    difference_matrix = torch.abs(query_embeddings.unsqueeze(dim=1) - doc_embeddings.unsqueeze(dim=0))
    # batch_size * batch_size
    distance_matrix = -torch.sum(difference_matrix, dim=-1)
    return distance_matrix


def l2_distance(query_embeddings, doc_embeddings):
    # batch_size * batch_size * embedding_dim
    difference_matrix = torch.pow(query_embeddings.unsqueeze(dim=1) - doc_embeddings.unsqueeze(dim=0), exponent=2)
    # batch_size * batch_size
    distance_matrix = -torch.sqrt(torch.sum(difference_matrix, dim=-1))
    return distance_matrix