"""
base_branch.py — BaseFeatureExtraction（Base 分支）

负责提取低频全局特征（亮度、结构、语义）。
用 AttentionBase + MLP 构成，和 Encoder 里的 TransformerBlock 结构相似，
但 QKV 生成方式略有不同。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.layers.norm import LayerNorm


class AttentionBase(nn.Module):
    """
    Base 分支专用的 Attention。
    与 MDTA 的区别：QKV 用 1×1 + 3×3 普通卷积（非深度可分离）生成。

    Input / Output:  (B, C, H, W) → (B, C, H, W)
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads

        # 可学习缩放，shape=(num_heads, 1, 1)，每个 head 独立
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1×1 升维 + 3×3 普通卷积（注意：不是深度可分离）
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 转置注意力：对通道维度做 attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.proj(out)


class Mlp(nn.Module):
    """
    Base 分支的前馈网络，结构和 FeedForward 类似但略有不同：
    DWConv 的 groups 是 hidden_features（而非 hidden_features*2），
    因为这里先升维再做 DWConv，没有后续 chunk 操作。

    Input / Output:  (B, C, H, W) → (B, C, H, W)
    """
    def __init__(self, in_features: int, ffn_expansion_factor: float = 2, bias: bool = False):
        super().__init__()
        hidden = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(in_features, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2,
            kernel_size=3, padding=1,
            groups=hidden,    # 注意：groups=hidden，不是 hidden*2
            bias=bias
        )
        self.project_out = nn.Conv2d(hidden, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtraction(nn.Module):
    """
    Base 分支：提取低频全局特征。
    结构 = Pre-Norm AttentionBase + Pre-Norm Mlp，带残差连接。

    Args:
        dim:                  通道数 C，默认 64
        num_heads:            注意力头数，默认 8
        ffn_expansion_factor: FFN 扩张比
        qkv_bias:             QKV 卷积是否带 bias

    Input / Output:  (B, C, H, W) → (B, C, H, W)
    """
    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 8,
        ffn_expansion_factor: float = 1.,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn  = AttentionBase(dim, num_heads, qkv_bias)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp   = Mlp(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x