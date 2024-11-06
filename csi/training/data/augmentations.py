import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2


class Dummy(nn.Module):
    def forward(self, batch):
        return batch


class MixUp(v2.MixUp):
    def forward(self, batch):
        batch["cqt"], batch["clique"] = super().forward(batch["cqt"], batch["clique"])
        return batch


class CutMix(v2.CutMix):
    def forward(self, batch):
        batch["cqt"], batch["clique"] = super().forward(batch["cqt"], batch["clique"])
        return batch


class GaussianBlur(v2.GaussianBlur):
    def forward(self, batch):
        batch["cqt"] = super().forward(batch["cqt"])
        return batch


class GaussianNoise(v2.GaussianNoise):
    def forward(self, batch):
        batch["cqt"] = super().forward(batch["cqt"])
        return batch


class CoverToRandomChannel(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, batch):
        batch_size = batch["cqt"].size(0)
        random_channel_indices = torch.randint(0, self.num_channels, (batch_size,))
        batch_indices = torch.arange(batch_size)
        batch["cqt"][batch_indices, random_channel_indices] = batch["pos_cqt"][
            batch_indices, random_channel_indices
        ]
        return batch


class RandomResizedCrop(v2.RandomResizedCrop):
    def forward(self, batch):
        batch["cqt"] = super().forward(batch["cqt"])
        return batch


class TimeCutMixCustom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def apply_time(self, imgs, labels):
        b = imgs.size(0)
        t = imgs.size(-1)

        alpha = torch.rand(1)
        t_border = int(alpha * t)

        indices = torch.randperm(b)
        imgs[..., :t_border] = imgs[indices, ..., :t_border]
        labels = F.one_hot(labels, self.num_classes)
        labels = (1 - alpha) * labels + alpha * labels[indices, ...]

        return imgs, labels

    def forward(self, imgs, labels):
        imgs, labels = self.apply_time(imgs, labels)
        return imgs, labels


class TimeCutMix(TimeCutMixCustom):
    def forward(self, batch):
        batch["cqt"], batch["clique"] = super().forward(batch["cqt"], batch["clique"])
        return batch
