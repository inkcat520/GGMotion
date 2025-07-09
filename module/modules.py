import torch
import math
from torch import nn
import torch.nn.functional as F


def cosine_embedding(timesteps, embedding_dim, max_period=10000):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(max_period) / (half_dim - 1)
    emb = torch.exp(-emb * torch.arange(half_dim, dtype=torch.float32))
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1, 0, 0))
    return emb.cuda()


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SmoothEmb(nn.Module):
    def __init__(self, t, c, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth = GaussianSmooth(kernel_size)
        self.emb1 = nn.Linear(t, c, bias=False)
        self.emb2 = nn.Linear(t, c, bias=False)

    def forward(self, x):
        y = x.view(-1, 1, x.shape[-1])
        y = self.smooth(y).view_as(x)

        x = self.emb1(x)
        y = self.emb2(y)
        out = x + y
        return out


class GaussianSmooth(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        k = torch.arange(self.kernel_size, device=x.device) - self.kernel_size // 2
        kernel = torch.exp(-0.5 * (k / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)
        padding = (self.kernel_size - 1) // 2
        x_padded = F.pad(x, (padding, padding), mode='replicate')
        out = F.conv1d(x_padded, kernel)
        return out

