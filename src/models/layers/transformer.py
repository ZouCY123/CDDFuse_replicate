"""
transformer.py — TransformerBlock

Pre-Norm 结构：
  x → Norm → Attention → +x → Norm → FFN → +x → 输出
"""

import torch.nn as nn
from .norm import LayerNorm
from .attention import Attention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Args:
        dim:                  通道数 C
        num_heads:            注意力头数
        ffn_expansion_factor: FFN 隐层扩张比
        bias:                 卷积 bias
        LayerNorm_type:       'WithBias' 或 'BiasFree'

    Input / Output:  (B, C, H, W) → (B, C, H, W)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float = 2,
        bias: bool = False,
        LayerNorm_type: str = 'WithBias',
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn  = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # 第一条残差
        x = x + self.ffn(self.norm2(x))    # 第二条残差
        return x