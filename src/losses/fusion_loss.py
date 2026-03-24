"""
fusion_loss.py — 融合损失（Phase II 使用）

包含两部分：
  1. 强度损失：融合图亮度 ≥ max(VIS, IR)
  2. 梯度损失：融合图边缘 ≥ max(grad(VIS), grad(IR))

总损失 = intensity_loss + 10 × gradient_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradient(nn.Module):
    """
    用 Sobel 算子计算图像梯度幅值。

    Sobel 是固定的卷积核，不需要训练，所以用
    requires_grad=False 的 Parameter 存储。

    Input / Output:  (B, 1, H, W) → (B, 1, H, W)
    """

    def __init__(self):
        super().__init__()

        # Sobel X 方向核：检测垂直边缘
        kernel_x = torch.tensor([
            [-1.,  0.,  1.],
            [-2.,  0.,  2.],
            [-1.,  0.,  1.],
        ]).reshape(1, 1, 3, 3)

        # Sobel Y 方向核：检测水平边缘
        kernel_y = torch.tensor([
            [ 1.,  2.,  1.],
            [ 0.,  0.,  0.],
            [-1., -2., -1.],
        ]).reshape(1, 1, 3, 3)

        # requires_grad=False：固定权重，不参与梯度计算
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W)，单通道图像

        Returns:
            梯度幅值 |Gx| + |Gy|，shape (B, 1, H, W)
        """
        grad_x = F.conv2d(x, self.kernel_x, padding=1)
        grad_y = F.conv2d(x, self.kernel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)


class FusionLoss(nn.Module):
    """
    融合损失，Phase II 训练时使用。

    Args:
        gradient_weight: 梯度损失的权重，默认 10

    Input:
        vis:    可见光图像，(B, 1, H, W)，值域 [0, 1]
        ir:     红外图像，  (B, 1, H, W)，值域 [0, 1]
        fused:  融合图像，  (B, 1, H, W)，值域 [0, 1]

    Returns:
        loss:            总损失（标量）
        loss_intensity:  强度损失（监控用）
        loss_gradient:   梯度损失（监控用）
    """

    def __init__(self, gradient_weight: float = 10.0):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.sobel = SobelGradient()

    def forward(
        self,
        vis:   torch.Tensor,
        ir:    torch.Tensor,
        fused: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ── 强度损失 ──────────────────────────────────────────
        # 目标：融合图亮度不低于 VIS 和 IR 中较亮的那个
        target_intensity = torch.max(vis, ir)
        loss_intensity   = F.l1_loss(fused, target_intensity)

        # ── 梯度损失 ──────────────────────────────────────────
        # 目标：融合图边缘不弱于 VIS 和 IR 中边缘更清晰的那个
        grad_vis   = self.sobel(vis)
        grad_ir    = self.sobel(ir)
        grad_fused = self.sobel(fused)

        target_gradient = torch.max(grad_vis, grad_ir)
        loss_gradient   = F.l1_loss(grad_fused, target_gradient)

        # ── 合并 ──────────────────────────────────────────────
        loss = loss_intensity + self.gradient_weight * loss_gradient

        return loss, loss_intensity, loss_gradient