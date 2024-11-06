import gc
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import os
import torch
from hydra.utils import call, instantiate

from csi.base.model.predict import predict
from csi.base.utils import batch_to_device, init_model, seed_everything
from csi.training.loop.utils import load_fold_checkpoint, split_by_batch_size

logger = logging.getLogger(__name__)
logger.info = print

from pathlib import Path

from tqdm import tqdm


def make_fold_submission(cfg, model, test_loader, candidates_from_fold=300):
    embeddings = []
    track_ids = []
    for batch in tqdm(test_loader, desc="Model fold inference"):
        batch = batch_to_device(batch, cfg.environment.device)
        outs = predict(model, batch)
        embs = outs["embedding"]
        track_ids.append(outs["track_id"].reshape(-1, 1))
        embeddings.append(embs)

    embeddings = torch.vstack(embeddings).detach().cpu().numpy()
    track_ids = torch.vstack(track_ids).detach().cpu().numpy()

    if cfg.nearest_neighbors_search.normalize_embeddings:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    res = {}

    emb_indices = np.arange(len(embeddings))
    mini_batch_size = 5000
    embeddingsT = embeddings.T
    for ind in tqdm(split_by_batch_size(emb_indices, mini_batch_size)):
        track_id_batch = track_ids[ind]
        emb_batch = embeddings[ind]
        similarities = np.dot(emb_batch, embeddingsT)
        top_k_indices = np.argsort(-similarities, axis=1)[:, : candidates_from_fold + 1]
        top_k_indices = top_k_indices[top_k_indices != ind.reshape(-1, 1)]
        top_tracks_similarities = np.take_along_axis(
            similarities, top_k_indices.reshape(len(ind), candidates_from_fold), axis=1
        )
        top_tracks = track_ids[top_k_indices].reshape(len(ind), candidates_from_fold)
        for track_id, tracks, sims in zip(
            track_id_batch.flatten(), top_tracks, top_tracks_similarities
        ):
            res[int(track_id)] = [(int(t), float(s)) for t, s in zip(tracks, sims)]
    return res


def get_predictions(cfg_path, candidates_from_fold=300):
    cfg = init_cfg(cfg_path)
    seed_everything(cfg.environment.seed)

    logger.info("Reading clique2tracks")

    test_dataset = call(
        cfg.test_data.dataset,
        tracks_ids=np.load(cfg.test_data.test_ids_path),
        track2clique=None,
        clique2tracks=None,
        _convert_="partial",
    )
    test_loader = instantiate(
        cfg.test_data.dataloader, test_dataset, _convert_="partial"
    )

    fold_results = []
    for fold in range(6):
        model = init_model(cfg).to(cfg.environment.device)

        if cfg.path_to_fold_checkpoints is not None:
            model = load_fold_checkpoint(model, cfg.path_to_fold_checkpoints, fold, cfg.environment.device)

        fold_results.append(
            make_fold_submission(cfg, model, test_loader, candidates_from_fold)
        )
        torch.cuda.empty_cache()
        gc.collect()
    return fold_results


def pack_to_df(predictions, metric):
    track_ids = []
    candidates_ids = []
    candidates_scores = []
    fold_indices = []
    for fold_idx, fold in enumerate(predictions):
        for track_id, cands_with_scores in fold.items():
            for cand, score in cands_with_scores:
                track_ids.append(int(track_id))
                candidates_ids.append(int(cand))
                candidates_scores.append(float(score))
                fold_indices.append(int(fold_idx))
    df = pd.DataFrame(
        {
            "track_id": track_ids,
            "candidate_id": candidates_ids,
            "candidate_score": candidates_scores,
            "fold_idx": fold_indices,
        }
    ).assign(metric=float(metric))
    return df


def init_cfg(cfg_path):
    from hydra import compose, initialize

    with initialize(version_base=None, config_path=str(Path(cfg_path).parent)):
        try:
            cfg = compose(
                config_name="config", overrides=["+read_filtered_clique2versions=null"]
            )
        except:
            cfg = compose(
                config_name="config", overrides=["read_filtered_clique2versions=null"]
            )

    checkpoints_folder = (
        "artifacts_" + str(Path(cfg_path).parent.parent.name) + "/model_checkpoints"
    )
    cfg["path_to_fold_checkpoints"] = checkpoints_folder
    cfg["read_filtered_clique2versions"] = None
    cfg["environment"]["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg["model"]["calc_pos_embeddings"] = False
    return cfg


def save_tracks_to_file(data, output_path):
    df = pd.DataFrame(
        {
            "query_trackid": list(data.keys()),
            "track_ids": [
                " ".join(map(lambda x: str(int(x)), track_ids))
                for track_ids in data.values()
            ],
        }
    )
    df["output"] = df["query_trackid"].astype(str) + " " + df["track_ids"].astype(str)
    df.sort_values("query_trackid", inplace=True)
    df[["output"]].to_csv(output_path, index=False, header=False)


def adjust_score(df):
    assert 165_510_000 == df.shape[0], f"{df.shape[0]}"
    return df.with_columns(
        (pl.col("metric") * (pl.col("candidate_score") + 1) / 2).alias(
            "candidate_score"
        )
    )


def groupby_score(df, rank_fn=None):
    res = (
        df.group_by(["track_id", "candidate_id"])
        .agg(
            [
                pl.sum("candidate_score").alias("candidate_score_sum"),
                pl.count("candidate_score").alias("candidate_cnt"),
            ]
        )
        .with_columns(
            (
                (
                    pl.col("candidate_score_sum")
                    .rank("dense", descending=True)
                    .over("track_id")
                    - 1
                )
                / 6
            ).alias("rank_sum")
        )
        .drop(["candidate_score_sum"])
    )
    res = res.with_columns(
        pl.lit(1).alias("candidate_cnt"),
    )
    if rank_fn is not None:
        res = res.with_columns(rank_fn(pl.col("rank_sum")).alias("rank_sum"))
    return res.select(["track_id", "candidate_id", "rank_sum", "candidate_cnt"])


def update_score(df1, df2):
    return (
        pl.concat([df1, df2])
        .group_by(["track_id", "candidate_id"])
        .agg(
            [
                pl.sum("rank_sum").alias("rank_sum"),
                pl.sum("candidate_cnt").alias("candidate_cnt"),
            ]
        )
        .select(["track_id", "candidate_id", "rank_sum", "candidate_cnt"])
    )


def take_top(name2metric, topk=1):
    return dict(sorted(name2metric.items(), key=lambda x: -x[1])[:topk])


if __name__ == "__main__":
    TOP = 1
    RESULTS = "results"
    FINAL_ARTIFACTS = "final_artifacts"
    POWER = 500
    model_name = "hgnetv2_b5_metric_learning_big_margin_drop_cliques_test_0_6_6folds02_19_23"
    name2metric = {model_name: 1.0}

    os.makedirs(FINAL_ARTIFACTS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    metric = name2metric[model_name]
    predictions = get_predictions(f"{FINAL_ARTIFACTS}/{model_name}/hydra/config", POWER)
    pack_to_df(predictions, metric).to_parquet(f"{RESULTS}/{model_name}.pq", index=False)

    total_df = None

    for model_name, score in tqdm(take_top(name2metric, TOP).items()):
        data = groupby_score(
            adjust_score(pl.read_parquet(f"{RESULTS}/{model_name}.pq")),
            lambda x: x,
        )
        if total_df is None:
            total_df = data
        else:
            total_df = update_score(total_df, data)

    total_df = total_df.with_columns(
        (
            (
                pl.col("rank_sum")
                + POWER * (len(name2metric) - pl.col("candidate_cnt"))
            )
            / len(name2metric)
        ).alias("mean_rank")
    )

    total_df_sorted = total_df.sort(["track_id", "mean_rank"])

    top_100_df = total_df_sorted.group_by("track_id").head(100)

    data = defaultdict(list)
    for track_id, candidate_id in top_100_df.select(
        ["track_id", "candidate_id"]
    ).iter_rows():
        data[track_id].append(candidate_id)

    save_tracks_to_file(data, "submission.csv")
