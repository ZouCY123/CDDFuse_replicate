"""
src/losses/recon_loss.py — Phase I 重建损失

目标：Encoder-Decoder 能自重建输入图像。
损失 = 5 × SSIM_loss + MSE_loss + coeff_tv × gradient_loss

SSIM 衡量结构相似性，MSE 衡量像素误差，
两者结合比单独用 MSE 重建质量更好。
"""

import torch
import torch.nn as nn
import kornia.losses


class ReconLoss(nn.Module):
    """
    Phase I 重建损失。

    Args:
        ssim_weight:     SSIM 损失权重，默认 5
        gradient_weight: 梯度损失权重，默认 5

    Input:
        pred:   重建图像 (B, 1, H, W)
        target: 原始图像 (B, 1, H, W)

    Returns:
        loss, loss_ssim, loss_mse, loss_grad
    """

    def __init__(self, ssim_weight: float = 5.0, gradient_weight: float = 5.0):
        super().__init__()
        self.ssim_weight     = ssim_weight
        self.gradient_weight = gradient_weight
        self.mse  = nn.MSELoss()
        self.ssim = kornia.losses.SSIMLoss(window_size=11)

        # 用 FusionLoss 里的 Sobel 复用梯度计算
        from src.losses.fusion_loss import SobelGradient
        self.sobel = SobelGradient()

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        loss_ssim = self.ssim(pred, target)
        loss_mse  = self.mse(pred, target)
        loss_grad = nn.functional.l1_loss(self.sobel(pred), self.sobel(target))

        loss = self.ssim_weight * loss_ssim + loss_mse + self.gradient_weight * loss_grad

        return loss, loss_ssim, loss_mse, loss_grad