"""
embed.py — OverlapPatchEmbed

用 3×3 卷积把图像映射到特征空间。
overlap 指卷积核有重叠，让相邻 patch 之间共享边界信息。

Input:   (B, in_c, H, W)   通常 in_c=1（灰度图）
Output:  (B, embed_dim, H, W)   空间尺寸不变，通道数变成 embed_dim
"""

import torch.nn as nn


class OverlapPatchEmbed(nn.Module):
    """
    Args:
        in_c:      输入通道数，灰度图为 1
        embed_dim: 输出特征维度，默认 64

    Input / Output:  (B, in_c, H, W) → (B, embed_dim, H, W)
    """
    def __init__(self, in_c: int = 1, embed_dim: int = 64, bias: bool = False):
        super().__init__()
        # padding=1 保证输出和输入空间尺寸相同
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)