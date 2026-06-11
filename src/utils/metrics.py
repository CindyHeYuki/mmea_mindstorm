import torch
import numpy as np


def compute_hits_mrr(emb1: torch.Tensor, emb2: torch.Tensor, test_pairs, ks=(1, 5, 10)):
    """
    Compute Hits@k and MRR for entity alignment evaluation.

    Args:
        emb1: embeddings for KG1 entities, shape (N1, D)
        emb2: embeddings for KG2 entities, shape (N2, D)
        test_pairs: list of (e1_id, e2_id) ground-truth alignment pairs
        ks: tuple of k values for Hits@k
    """
    emb1 = emb1.float()
    emb2 = emb2.float()

    test_pairs = torch.tensor(test_pairs, dtype=torch.long)
    e1_ids = test_pairs[:, 0]
    e2_ids = test_pairs[:, 1]

    query = emb1[e1_ids]          # (T, D)
    gallery = emb2                 # (N2, D)

    # Cosine similarity matrix
    sim = torch.mm(query, gallery.t())   # (T, N2)
    ranks = (sim.argsort(dim=-1, descending=True) == e2_ids.unsqueeze(1)).nonzero()[:, 1] + 1

    results = {}
    for k in ks:
        results[f"hits@{k}"] = (ranks <= k).float().mean().item()
    results["mrr"] = (1.0 / ranks.float()).mean().item()
    return results
