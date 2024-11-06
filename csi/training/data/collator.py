import torch
from torch.utils.data import default_collate


class AugCollator:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, batch):
        collated: dict = default_collate(batch)
        collated["og_clique"] = torch.clone(collated["clique"])
        return self.augmentations(collated)
