import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from natten import na2d


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=(9, 9)
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.kernel_size = kernel_size

    def _resize(self, x, size, dtype):
        x = F.interpolate(x, size=size, mode="nearest-exact")
        x = rearrange(x, "b (n d) h w -> b h w n d", n=self.num_heads)
        return x.to(dtype)

    def forward(self, q, k, v):
        hq, wq = q.shape[-2:]
        hk, wk = k.shape[-2:]
        dilation = (hq // hk, wq // wk)
        self.dilation = dilation
        q = rearrange(q, "b (n d) h w -> b h w n d", n=self.num_heads)
        k = self._resize(k, size=(hq, wq), dtype=q.dtype)
        v = self._resize(v, size=(hq, wq), dtype=q.dtype)
        out = na2d(q, k, v, kernel_size=self.kernel_size, dilation=dilation, stride=1, backend="cutlass-fna")
        return rearrange(out, "b h w n d -> b (n d) h w")