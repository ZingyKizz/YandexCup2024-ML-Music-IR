from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


@contextmanager
def evaluating(model: nn.Module):
    is_train = model.training
    try:
        if is_train:
            model.eval()
        yield model
    finally:
        if is_train:
            model.train()


@torch.inference_mode()
def predict(model: nn.Module, x: Any) -> Any:
    with evaluating(model):
        outs = model(x)
    return outs
