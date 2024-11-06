import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from csi.base.model.predict import predict
from csi.base.utils import batch_to_device
from csi.training.loop.utils import split_by_batch_size

logger = logging.getLogger(__name__)


def compute_ndcg(cfg, model: nn.Module, val_loader: DataLoader, top_k: int = 100) -> float:
    embeddings, cliques, track_ids, num_other_tracks_in_clique = [], [], [], []

    for batch in val_loader:
        batch = batch_to_device(batch, cfg.environment.device)
        outs = predict(model, batch)
        embs = outs["embedding"]
        embeddings.append(embs)
        cliques.append(batch["clique"].reshape(-1, 1))
        num_other_tracks_in_clique.append(batch["num_other_tracks_in_clique"].reshape(-1, 1))

    embeddings = torch.vstack(embeddings).detach().cpu().numpy()
    cliques = torch.vstack(cliques).detach().cpu().numpy()
    num_other_tracks_in_clique = torch.vstack(num_other_tracks_in_clique).detach().cpu().numpy()

    if cfg.nearest_neighbors_search.normalize_embeddings:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    emb_indices = np.arange(len(embeddings))
    mini_batch_size = 5000
    embeddingsT = embeddings.T
    ndcg_sum = 0.0
    total_tracks = 0.0001
    for ind in split_by_batch_size(emb_indices, mini_batch_size):
        try:
            cliques_batch = cliques[ind]
            num_other_tracks_in_clique_batch = num_other_tracks_in_clique[ind]
            emb_batch = embeddings[ind]
            similarities = np.dot(emb_batch, embeddingsT)
            top_k_indices = np.argsort(-similarities, axis=1)[:, : top_k + 1]
            top_k_indices = top_k_indices[top_k_indices != ind.reshape(-1, 1)]
            top_tracks_cliques = cliques[top_k_indices].reshape(len(ind), top_k)

            relevance = (top_tracks_cliques == cliques_batch).astype(np.float64)
            discounts = 1 / np.sqrt(np.arange(1, relevance.shape[1] + 1))
            discounts = np.tile(discounts, (len(relevance), 1))

            ideal_relevance = np.zeros_like(relevance)
            ideal_mask = np.arange(ideal_relevance.shape[1]) < num_other_tracks_in_clique_batch
            ideal_relevance[ideal_mask] = 1.0

            dcg = (relevance * discounts).sum(axis=1)
            idcg = (ideal_relevance * discounts).sum(axis=1)
            ndcg = dcg / idcg

            ndcg_sum += float(ndcg.sum())
            total_tracks += len(ind)
        except ValueError:
            logger.warning(f"Value error for calculating NDCG for mini-batch of length={len(ind)}")
    return ndcg_sum / total_tracks
