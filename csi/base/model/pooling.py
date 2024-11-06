import torch
import torch.nn.functional as F
from torch import nn


class GeM1D(nn.Module):
    """
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    """

    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(
            x.clamp(min=eps).pow(p),
            self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        ).pow(1.0 / p)


class AdaptiveGeM1D(nn.Module):
    def __init__(self, output_size, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.output_size = output_size

    def forward(self, x):
        input_size = x.size(2)
        stride = input_size // self.output_size
        kernel_size = input_size - (self.output_size - 1) * stride
        padding = 0
        out = F.avg_pool1d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size,
            padding=padding,
            stride=stride,
        ).pow(1.0 / self.p)
        return out


class GeM2D(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def global_avg_pooling_2d(x):
    batch_size = x.size(0)
    return torch.mean(x, dim=(2, 3)).view(batch_size, -1)
