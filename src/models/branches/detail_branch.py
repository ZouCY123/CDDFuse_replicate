"""
detail_branch.py — DetailFeatureExtraction（Detail 分支）

用可逆神经网络（INN）提取高频局部特征（边缘、纹理）。
INN 的可逆性保证高频信息提取时不丢失。
"""

import torch
import torch.nn as nn


class InvertedResidualBlock(nn.Module):
    """
    MobileNet 风格的轻量卷积块，用作 INN 耦合层的变换函数。
    结构：pw(升维) → dw(空间混合) → pw(降维)

    Input / Output:  (B, inp, H, W) → (B, oup, H, W)
    """
    def __init__(self, inp: int, oup: int, expand_ratio: int = 2):
        super().__init__()
        hidden = int(inp * expand_ratio)
        self.block = nn.Sequential(
            # pointwise 升维
            nn.Conv2d(inp, hidden, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            # depthwise 空间卷积（ReflectionPad 比 zero-pad 边界更自然）
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden, hidden, kernel_size=3, groups=hidden, bias=False),
            nn.ReLU6(inplace=True),
            # pointwise 降维
            nn.Conv2d(hidden, oup, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class DetailNode(nn.Module):
    """
    INN 的一个耦合层（coupling layer）。
    把输入 (z1, z2) 变换成 (z1', z2')，变换严格可逆。

    变换公式：
        z2' = z2 + φ(z1)
        z1' = z1 ⊙ exp(ρ(z2')) + η(z2')

    其中 φ=theta_phi，ρ=theta_rho，η=theta_eta，
    都是 InvertedResidualBlock。

    Input:   z1 (B, 32, H, W),  z2 (B, 32, H, W)
    Output:  z1'(B, 32, H, W),  z2'(B, 32, H, W)
    """
    def __init__(self):
        super().__init__()
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        # shuffle conv：让 z1 z2 在进入耦合前先混合一次
        self.shuffleconv = nn.Conv2d(64, 64, kernel_size=1, bias=True)

    def _split(self, x):
        """沿通道对半切"""
        half = x.shape[1] // 2
        return x[:, :half], x[:, half:]

    def forward(self, z1, z2):
        # 先 shuffle，让两路特征交换信息
        z1, z2 = self._split(self.shuffleconv(torch.cat([z1, z2], dim=1)))

        # 耦合变换（顺序很重要：先更新 z2，再用更新后的 z2 更新 z1）
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    """
    Detail 分支：串联多个 DetailNode 提取高频局部特征。

    Args:
        num_layers: INN 耦合层数量，默认 3

    Input / Output:  (B, 64, H, W) → (B, 64, H, W)
    注意：输入通道必须是 64（内部对半切成两个 32）
    """
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([DetailNode() for _ in range(num_layers)])

    def forward(self, x):
        # 对半切成 z1, z2，各 32 通道
        half = x.shape[1] // 2
        z1, z2 = x[:, :half], x[:, half:]

        # 依次通过每个耦合层
        for layer in self.layers:
            z1, z2 = layer(z1, z2)

        return torch.cat([z1, z2], dim=1)