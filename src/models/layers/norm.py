"""
norm.py — 图像 Transformer 专用 LayerNorm

为什么不直接用 nn.LayerNorm？
  nn.LayerNorm 是为 (B, seq_len, C) 设计的。
  图像特征图是 (B, C, H, W)，需要先把维度转成 (B, H*W, C)
  做完 norm 再转回来。这里统一封装这个转换。
"""

import torch
import torch.nn as nn
import numbers
from einops import rearrange

# ── 维度转换工具 ──────────────────────────────────────────────

def to_3d(x):
    """(B, C, H, W) → (B, H*W, C)"""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """(B, H*W, C) → (B, C, H, W)"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# ── 两种内部实现 ───────────────────────────────────────────────

class BiasFree_LayerNorm(nn.Module):
    """只有 scale（weight），没有 bias。
    用在 Attention 之前，去掉 bias 让 Q/K 数值更稳定。"""
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        assert len(self.normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        # x: (B, H*W, C)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """标准 LayerNorm，有 scale 也有 bias。
    用在 FFN 之前，表达能力更强。"""
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        assert len(self.normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        # x: (B, H*W, C)
        mu    = x.mean(-1, keepdim=True)
        sigma = x.var(-1,  keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

# ── 对外暴露的统一接口 ─────────────────────────────────────────

class LayerNorm(nn.Module):
    """
    图像 Transformer 用的 LayerNorm，自动处理维度转换。

    Args:
        dim:             通道数 C
        LayerNorm_type:  'WithBias'（默认）或 'BiasFree'

    Input / Output:  (B, C, H, W) → (B, C, H, W)
    """
    def __init__(self, dim: int, LayerNorm_type: str = 'WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)