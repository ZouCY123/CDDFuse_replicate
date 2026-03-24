"""
decomp_loss.py — 特征分解损失

约束 Encoder 的双分支分解是有意义的：
  - Base 分支：VIS 和 IR 特征应相关（低频共有结构）
  - Detail 分支：VIS 和 IR 特征应不相关（高频各自细节）

公式：
    loss = cc(detail_vis, detail_ir)² / (1.01 + cc(base_vis, base_ir))
"""

import torch


def correlation_coefficient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算两个特征图之间的相关系数。

    对每个 (batch, channel) 独立计算，再取全局均值。
    值域 [-1, 1]：1 表示完全正相关，0 表示不相关，-1 表示完全负相关。

    Args:
        x: (B, C, H, W)
        y: (B, C, H, W)，shape 必须和 x 相同

    Returns:
        标量 tensor，值域 [-1, 1]
    """
    eps = torch.finfo(torch.float32).eps   # 约 1.2e-7，防止除以零

    B, C, H, W = x.shape

    # 把空间维度展平：(B, C, H*W)
    x_flat = x.reshape(B, C, -1)
    y_flat = y.reshape(B, C, -1)

    # 去均值（对空间维度）
    x_centered = x_flat - x_flat.mean(dim=-1, keepdim=True)
    y_centered = y_flat - y_flat.mean(dim=-1, keepdim=True)

    # 计算相关系数
    # numerator:   Σ(x_i * y_i)
    # denominator: sqrt(Σx_i²) * sqrt(Σy_i²)
    numerator   = (x_centered * y_centered).sum(dim=-1)
    denominator = (
        torch.sqrt((x_centered ** 2).sum(dim=-1)) *
        torch.sqrt((y_centered ** 2).sum(dim=-1))
    )

    cc = numerator / (denominator + eps)

    # 截断到 [-1, 1]，防止数值误差超出范围
    cc = torch.clamp(cc, -1.0, 1.0)

    # 对所有 (batch, channel) 取均值，得到一个标量
    return cc.mean()


def decomp_loss(
    base_vis:   torch.Tensor,
    base_ir:    torch.Tensor,
    detail_vis: torch.Tensor,
    detail_ir:  torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算特征分解损失。

    Args:
        base_vis:   VIS 的 Base 特征，(B, C, H, W)
        base_ir:    IR  的 Base 特征，(B, C, H, W)
        detail_vis: VIS 的 Detail 特征，(B, C, H, W)
        detail_ir:  IR  的 Detail 特征，(B, C, H, W)

    Returns:
        loss:  标量，用于反传
        cc_b:  Base 相关系数（监控用，训练中应趋近 1）
        cc_d:  Detail 相关系数（监控用，训练中应趋近 0）
    """
    cc_b = correlation_coefficient(base_vis,   base_ir)
    cc_d = correlation_coefficient(detail_vis, detail_ir)

    # 分子：Detail 相关性平方（希望趋近 0）
    # 分母：1.01 + Base 相关性（希望 Base 相关性大，分母大，loss 小）
    loss = cc_d ** 2 / (1.01 + cc_b)

    return loss, cc_b, cc_d