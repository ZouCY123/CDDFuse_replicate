"""
cddfuse.py — Restormer_Encoder / Restormer_Decoder

Encoder 和 Decoder 是 CDDFuse 的顶层模块，
负责把所有积木组装成完整的编解码结构。
"""

import torch
import torch.nn as nn

from src.models.layers import (
    OverlapPatchEmbed,
    TransformerBlock,
)
from src.models.branches.base_branch import BaseFeatureExtraction
from src.models.branches.detail_branch import DetailFeatureExtraction


class Restormer_Encoder(nn.Module):
    """
    编码器：图像 → (base_feature, detail_feature, out_enc)

    Args:
        inp_channels:         输入通道数，灰度图为 1
        dim:                  特征维度，默认 64
        num_blocks:           每个阶段的 TransformerBlock 数量，[浅层, 深层]
        heads:                各阶段注意力头数，[浅层, 深层, Base分支]
        ffn_expansion_factor: FFN 扩张比
        bias:                 卷积 bias
        LayerNorm_type:       LayerNorm 类型

    Output:
        base_feature:   (B, dim, H, W)  低频全局特征
        detail_feature: (B, dim, H, W)  高频局部特征
        out_enc:        (B, dim, H, W)  中间特征（给 Decoder 用）
    """
    def __init__(
        self,
        inp_channels: int = 1,
        dim: int = 64,
        num_blocks: list = [4, 4],
        heads: list = [8, 8, 8],
        ffn_expansion_factor: float = 2,
        bias: bool = False,
        LayerNorm_type: str = 'WithBias',
    ):
        super().__init__()

        # 图像 → 特征图
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim)

        # TransformerBlock × num_blocks[0]：提取浅层跨模态特征
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            )
            for _ in range(num_blocks[0])
        ])

        # 双分支：从同一个特征图分别提取 base 和 detail
        self.baseFeature   = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        # 嵌入
        feat = self.patch_embed(inp_img)        # (B, dim, H, W)

        # 浅层特征提取
        out_enc = self.encoder_level1(feat)     # (B, dim, H, W)

        # 双分支分解
        base_feature   = self.baseFeature(out_enc)    # 低频
        detail_feature = self.detailFeature(out_enc)  # 高频

        return base_feature, detail_feature, out_enc