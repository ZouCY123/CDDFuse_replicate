"""
feedforward.py — GDFN 门控前馈网络
"""

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Args:
        dim:                  输入通道数 C
        ffn_expansion_factor: 隐层扩张比，默认 2
        bias:                 卷积是否带 bias

    Input / Output:  (B, C, H, W) → (B, C, H, W)
    """
    def __init__(self, dim: int, ffn_expansion_factor: float = 2, bias: bool = False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)

        # 升维到 hidden*2，为后面对半切准备
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)

        # 深度可分离卷积：groups=hidden*2，每个通道独立做 3×3
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2,
            kernel_size=3, padding=1,
            groups=hidden * 2,
            bias=bias
        )

        # 降回 dim，注意输入是 hidden（不是 hidden*2）
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)                  # (B, C, H, W) → (B, hidden*2, H, W)
        x1, x2 = self.dwconv(x).chunk(2, dim=1) # 各 (B, hidden, H, W)
        x = F.gelu(x1) * x2                     # 门控，shape 不变
        x = self.project_out(x)                 # (B, hidden, H, W) → (B, C, H, W)
        return x