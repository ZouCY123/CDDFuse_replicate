"""
fusion_loss.py — 融合阶段损失函数

负责约束最终融合图像的质量，主要包括：
1. 强度损失 (Intensity Loss)
2. 纹理/梯度损失 (Gradient Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobelxy(nn.Module):
    """
    利用固定权重的卷积提取图像的 X 和 Y 方向梯度（绝对值之和）。
    优化：使用 register_buffer 替代原版硬编码的 .cuda()，使张量能自动跟随模型设备。
    """
    def __init__(self):
        super().__init__()
        kernelx = [[-1.,  0.,  1.],
                   [-2.,  0.,  2.],
                   [-1.,  0.,  1.]]
        kernely = [[ 1.,  2.,  1.],
                   [ 0.,  0.,  0.],
                   [-1., -2., -1.]]
        
        kernelx = torch.tensor(kernelx).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        kernely = torch.tensor(kernely).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        
        # 注册为 buffer，不会作为模型参数被优化，但会保存在 state_dict 中并跟随 device
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class FusionLoss(nn.Module):
    """
    融合阶段的损失函数 (Phase II)
    
    计算公式:
        L_total = L_in + 10 * L_grad
        其中:
        L_in (强度损失) = L1( max(Vis, IR), Fused )
        L_grad (梯度损失) = L1( max(grad(Vis), grad(IR)), grad(Fused) )
    """
    def __init__(self):
        super().__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        """
        Args:
            image_vis:    可见光图像 (B, C, H, W)
            image_ir:     红外图像 (B, 1, H, W)
            generate_img: 融合后的图像 (B, 1, H, W)
            
        Returns:
            loss_total: 总损失
            loss_in:    强度损失
            loss_grad:  梯度损失
        """
        # 取可见光的第一通道 (灰度图不受影响，RGB 图取第一个通道)
        image_y = image_vis[:, :1, :, :]
        
        # 1. 强度损失 (Intensity Loss)
        # 期望融合图像在像素级上保留源图像的最大强度
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        
        # 2. 梯度/纹理损失 (Gradient Loss)
        # 期望融合图像保留源图像最显著的边缘和纹理
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        
        # 3. 总损失 (官方权重: 梯度损失占优，权重为 10)
        loss_total = loss_in + 10 * loss_grad
        
        return loss_total, loss_in, loss_grad
