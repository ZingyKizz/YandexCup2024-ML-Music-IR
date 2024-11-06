import ast
import random
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import Dataset


def read_clique2versions(clique2versions_path: str | Path) -> dict[int, list[int]]:
    df = pd.read_csv(clique2versions_path, sep="\t", converters={"versions": ast.literal_eval})
    clique2tracks = df.set_index("clique")["versions"].to_dict()
    return clique2tracks


def filter_tracks(filtered_clique2tracks, val_clique2tracks):
    train_clique2tracks = {
        clique: track
        for clique, track in filtered_clique2tracks.items()
        if clique not in val_clique2tracks
    }

    train_track2clique = {
        track: clique for clique, tracks in train_clique2tracks.items() for track in tracks
    }
    return train_track2clique, train_clique2tracks


class CoverDataset(Dataset):
    def __init__(
        self,
        tracks_path: str | Path,
        tracks_ids: np.ndarray | list[int],
        track2clique: dict[int, int] | None = None,
        clique2tracks: dict[int, list[int]] = None,
        transformations=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tracks_path = Path(tracks_path)
        self.track_ids = np.asarray(tracks_ids)
        self.track2clique = track2clique
        self.clique2tracks = clique2tracks
        self.transformations = nn.Identity() if transformations is None else transformations

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int):
        track_id = self.track_ids[idx]
        cqt = self._load_cqt(track_id)
        clique = self.track2clique[track_id] if self.track2clique is not None else -1
        if self.clique2tracks is not None:
            pos_track_ids = [i for i in self.clique2tracks.get(clique, []) if i != track_id]
            random_pos_track_id = random.choice(pos_track_ids) if pos_track_ids else -1
            random_pos_cqt = (
                self._load_cqt(random_pos_track_id)
                if random_pos_track_id != -1
                else torch.empty(0)
            )
            num_other_tracks_in_clique = len(pos_track_ids)
        else:
            random_pos_track_id = -1
            random_pos_cqt = torch.empty(0)
            num_other_tracks_in_clique = -1
        return {
            "track_id": track_id,
            "cqt": cqt,
            "clique": clique,
            "pos_track_id": random_pos_track_id,
            "pos_cqt": random_pos_cqt,
            "num_other_tracks_in_clique": num_other_tracks_in_clique,
        }

    def _load_cqt(self, track_id: int) -> torch.Tensor:
        filename = self._make_track_path(self.tracks_path, track_id)
        cqt = np.load(str(filename))
        cqt = torch.from_numpy(cqt)
        cqt = self.transformations(cqt)
        return cqt

    @staticmethod
    def _make_track_path(tracks_path, track_id):
        return Path(tracks_path) / f"{track_id}.npy"


def _build_split(
    clique2tracks: dict[int, List[int]], cliques: List[int]
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    track2clique = {track: clique for clique in cliques for track in clique2tracks[clique]}
    clique2tracks_split = {clique: clique2tracks[clique] for clique in cliques}
    return track2clique, clique2tracks_split


def split_cliques(
    clique2tracks: dict[int, List[int]], n_splits: int = 2, val_size: float = 0.05, seed: int = 0
) -> Generator[
    tuple[
        tuple[dict[int, int], dict[int, list[int]]], tuple[dict[int, int], dict[int, list[int]]]
    ],
    None,
    None,
]:
    cliques = list(clique2tracks)
    if n_splits == 2:
        train_cliques, val_cliques = train_test_split(
            cliques, test_size=val_size, random_state=seed, shuffle=True
        )
        yield _build_split(clique2tracks, train_cliques), _build_split(clique2tracks, val_cliques)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_index, val_index in kf.split(cliques):
            train_cliques = [cliques[i] for i in train_index]
            val_cliques = [cliques[i] for i in val_index]
            yield _build_split(clique2tracks, train_cliques), _build_split(
                clique2tracks, val_cliques
            )


class MinMaxScaling:
    def __call__(self, tensor):
        return (tensor + 100.0) / 137.86386


class QuantileScaling:
    def __call__(self, tensor):
        return (tensor + 55.50517654418945) / 64.36982727050781


class ToImage:
    def __call__(self, tensor):
        return tensor.unsqueeze(0).repeat(3, 1, 1)


class Transpose:
    def __call__(self, tensor):
        return tensor.transpose(-1, -2)
