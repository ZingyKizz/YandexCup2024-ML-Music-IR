import glob
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


class RunningLoss:
    ALPHA = 0.8

    def __init__(
        self,
        name: str = "Loss",
        log_every_n_iterations: Optional[int] = None,
        total_iterations: Optional[int] = None,
    ) -> None:
        self.value = None
        self.iteration = 0
        self.log_every_n_iterations = max(
            (
                int(total_iterations * 0.005)
                if (log_every_n_iterations is None) and (total_iterations is not None)
                else log_every_n_iterations
            ),
            1,
        )
        self.total_iterations = total_iterations
        self.name = name

    def update(self, loss: torch.Tensor) -> None:
        if self.value is None:
            self.value = loss.item()
        else:
            self.value = self.ALPHA * self.value + (1 - self.ALPHA) * loss.item()

        self.iteration += 1
        should_log = (self.log_every_n_iterations is not None) and (
            self.iteration % self.log_every_n_iterations == 0
        )
        should_log |= (self.total_iterations is not None) and (
            self.iteration == self.total_iterations
        )
        should_log |= self.iteration == 1
        if should_log:
            logger.info(
                f"Iteration: {self.iteration}"
                + (f"/{self.total_iterations}" if self.total_iterations is not None else "")
                + f", {self.name}: {self.value:.4f}"
            )

    def reset(self) -> None:
        self.value = None
        self.iteration = 0


def clean_old_content(checkpoint_dir: str, prefix: str | None = None):
    dir = Path(checkpoint_dir)
    pattern = str(dir / (prefix + "*")) if prefix else str(dir / "*")
    previous_content = glob.glob(pattern)
    for file in previous_content:
        os.remove(file)


def save_checkpoint(
    checkpoint_dir,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    fold: int,
    epoch: int,
    metric_value: float,
) -> Path:
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = (
        Path(checkpoint_dir)
        / f"checkpoint__fold={fold}__epoch={epoch}__NDCG={metric_value:.8f}.pt"
    )
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, save_path)
    return save_path


def split_by_batch_size(arr, batch_size):
    nbatches = arr.shape[0] // batch_size
    if nbatches != arr.shape[0] / batch_size:
        nbatches += 1
    return np.array_split(arr, nbatches)


def load_fold_checkpoint(model, checkpoints_path, fold, map_location=None):
    checkpoints_path = Path(checkpoints_path)

    all_checkpoints = os.listdir(checkpoints_path)
    fold_checkpoint = [x for x in all_checkpoints if f"fold={fold}" in x]
    if not fold_checkpoint:
        raise ValueError(f"Checkpoint for fold={fold} not in {checkpoints_path}")
    else:
        fold_checkpoint = fold_checkpoint.pop()

    full_checkpoint_path = Path(checkpoints_path) / fold_checkpoint

    weights = torch.load(full_checkpoint_path, weights_only=True, map_location=map_location)["model"]
    model.load_state_dict(weights, strict=True)

    logger.info(f"Loaded checkpoint {full_checkpoint_path} for fold={fold}")

    return model


def freeze_layers(model, num_layers, freeze_ln=True):
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < num_layers or (name.startswith("ln") and freeze_ln):
            param.requires_grad = False
            logger.info(f"Freezed: {name}")
        else:
            logger.info(f"Unfreezed: {name}")
    return model
