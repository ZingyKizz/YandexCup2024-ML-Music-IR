import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from csi.base.model.predict import predict
from csi.base.utils import batch_to_device
from csi.training.loop.utils import clean_old_content, split_by_batch_size

CANDIDATES_FROM_FOLDS = 200
TOP_K = 100


def make_submission(cfg, model, test_loader):
    fold_top_tracks = []
    for checkpoint_path in glob.glob(str(Path(cfg.training.checkpoint_dir) / "*")):
        model_weights = torch.load(checkpoint_path, weights_only=True, map_location="cpu")["model"]
        model.load_state_dict(model_weights, strict=True)
        fold_top_tracks.append(make_fold_submission(cfg, model, test_loader))
    submission_data = reduce_by_folds(fold_top_tracks)
    os.makedirs(cfg.submission_dir, exist_ok=True)
    clean_old_content(cfg.submission_dir)
    save_tracks_to_file(submission_data, os.path.join(cfg.submission_dir, "submission.csv"))


def save_tracks_to_file(data, output_path):
    df = pd.DataFrame(
        {
            'query_trackid': list(data.keys()),
            'track_ids': [
                ' '.join(map(lambda x: str(int(x)), track_ids)) for track_ids in data.values()
            ],
        }
    )
    df['output'] = df['query_trackid'].astype(str) + ' ' + df['track_ids'].astype(str)
    df.sort_values("query_trackid", inplace=True)
    df[['output']].to_csv(output_path, index=False, header=False)


def reduce_by_folds(fold_top_tracks):
    ranks = {}
    counter = {}
    for fold in fold_top_tracks:
        for track_id, top_tracks in fold.items():
            if track_id not in ranks:
                ranks[track_id] = defaultdict(int)
                counter[track_id] = defaultdict(int)
            for rank, recommended_track in enumerate(top_tracks):
                ranks[track_id][recommended_track] += rank
                counter[track_id][recommended_track] += 1

    candidates_with_score = {}
    for track_id, candidates_with_rank in ranks.items():
        if track_id not in candidates_with_score:
            candidates_with_score[track_id] = defaultdict(float)
        for recommended_track, total_rank in candidates_with_rank.items():
            cnt = counter[track_id][recommended_track]
            total_rank += CANDIDATES_FROM_FOLDS * (len(fold_top_tracks) - cnt)
            candidates_with_score[track_id][recommended_track] = total_rank

    res = {}
    for track_id, cws in candidates_with_score.items():
        best_tracks = [x[0] for x in sorted(cws.items(), key=lambda x: x[1])[:TOP_K]]
        res[track_id] = best_tracks

    return res


def make_fold_submission(cfg, model, test_loader):
    embeddings = []
    track_ids = []
    for batch in test_loader:
        batch = batch_to_device(batch, cfg.environment.device)
        outs = predict(model, batch)
        embs = outs["embedding"]
        track_ids.append(outs["track_id"].reshape(-1, 1))
        embeddings.append(embs)

    embeddings = torch.vstack(embeddings).detach().cpu().numpy()
    track_ids = torch.vstack(track_ids).detach().cpu().numpy()

    if cfg.nearest_neighbors_search.normalize_embeddings:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    CANDIDATES_FROM_FOLDS = 200
    res = {}

    emb_indices = np.arange(len(embeddings))
    mini_batch_size = 5000
    embeddingsT = embeddings.T
    for ind in split_by_batch_size(emb_indices, mini_batch_size):
        track_id_batch = track_ids[ind]
        emb_batch = embeddings[ind]
        similarities = np.dot(emb_batch, embeddingsT)
        top_k_indices = np.argsort(-similarities, axis=1)[:, : CANDIDATES_FROM_FOLDS + 1]
        top_k_indices = top_k_indices[top_k_indices != ind.reshape(-1, 1)]
        top_tracks = track_ids[top_k_indices].reshape(len(ind), CANDIDATES_FROM_FOLDS)
        for track_id, tracks in zip(track_id_batch.flatten(), top_tracks):
            res[int(track_id)] = tracks.tolist()
    return res
