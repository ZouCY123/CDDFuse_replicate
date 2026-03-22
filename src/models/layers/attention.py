"""
attention.py — MDTA (Multi-DConv Head Transposed Self-Attention) 多头转置注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention(nn.Module):
    """
    Args:
        dim:       输入通道数 C
        num_heads: 注意力头数
        bias:      卷积是否带 bias

    Input / Output:  (B, C, H, W) → (B, C, H, W)
    """
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads

        # 可学习的注意力温度，每个 head 独立，初始为 1
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q K V 生成：1×1 混通道 + 3×3 DWConv 引入空间信息
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3,
            kernel_size=3, padding=1,
            groups=dim * 3,
            bias=bias
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # 生成 Q K V，每个 shape = (B, C, H, W)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # reshape 成 (B, heads, C//heads, HW)
        # 注意：最后两维是 C//heads 和 HW，和普通 Attention 相反
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2 归一化，防止点积过大导致 softmax 饱和
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 转置注意力：(B, heads, C//heads, C//heads)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 加权聚合 V，再 reshape 回 (B, C, H, W)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return self.project_out(out)