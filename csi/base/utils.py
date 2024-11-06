import os
import random

import numpy as np
import torch
from hydra.utils import instantiate


def batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], dict):
            batch_to_device(batch[key], device)
        else:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
    return batch


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_model(cfg):
    model = instantiate(cfg.model, _convert_="partial")
    return model
