"""
patch_utils.py — Patch 提取与质量过滤

纯函数，不依赖任何外部状态。
"""

import numpy as np


def extract_patches(img: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    """
    用滑动窗口从图像中提取所有 patch。

    Args:
        img:        shape (1, H, W) 的图像，值域 [0, 1]
        patch_size: patch 的边长（正方形）
        stride:     滑动步长，步长越大 patch 越少、重叠越少

    Returns:
        shape (N, 1, patch_size, patch_size) 的数组
        N = 能切出的 patch 数量

    Example:
        img 是 (1, 640, 480)，patch_size=128，stride=200
        → 水平方向：(480-128)//200 + 1 = 2 个
        → 垂直方向：(640-128)//200 + 1 = 3 个
        → N = 2 × 3 = 6 个 patch
    """
    _, H, W = img.shape
    patches = []

    for h_start in range(0, H - patch_size + 1, stride):
        for w_start in range(0, W - patch_size + 1, stride):
            patch = img[:, h_start:h_start + patch_size, w_start:w_start + patch_size]
            patches.append(patch)

    if len(patches) == 0:
        return np.empty((0, 1, patch_size, patch_size), dtype=img.dtype)

    return np.stack(patches, axis=0)   # (N, 1, patch_size, patch_size)


def is_low_contrast(patch: np.ndarray,
                    lower_pct: float = 10.0,
                    upper_pct: float = 90.0,
                    threshold: float = 0.1) -> bool:
    """
    判断 patch 是否低对比度（例如纯天空、纯墙壁等无信息区域）。

    方法：用第 lower_pct 和 upper_pct 百分位的差值比来衡量对比度。
    差值比 < threshold 则认为低对比度，应丢弃。

    Args:
        patch:      任意 shape 的 float32 数组，值域 [0, 1]
        lower_pct:  下百分位，默认 10
        upper_pct:  上百分位，默认 90
        threshold:  对比度阈值，低于此值认为低对比度

    Returns:
        True 表示低对比度（应丢弃），False 表示正常
    """
    lo, hi = np.percentile(patch, [lower_pct, upper_pct])

    # 避免除以零：全黑图像也是低对比度
    if hi < 1e-6:
        return True

    contrast_ratio = (hi - lo) / hi
    return contrast_ratio < threshold
