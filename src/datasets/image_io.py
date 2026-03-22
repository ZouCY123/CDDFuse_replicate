"""
image_io.py — 图像读取与颜色转换

全部是无状态的纯函数，不依赖任何外部配置。
可以单独 import 使用，也方便单独测试。
"""
import os, sys
# 获取项目根目录（假设当前文件在子目录中）
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.insert(0, root_dir)

from pathlib import Path
import numpy as np
from skimage.io import imread


# 支持的图像后缀
IMAGE_EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def get_image_paths(folder: str | Path) -> list[Path]:
    """
    递归扫描文件夹，返回所有图像文件路径，按文件名排序。
    排序是关键：保证 ir/ 和 vi/ 下的文件一一对应。

    Args:
        folder: 要扫描的文件夹路径

    Returns:
        排序后的 Path 列表
    """
    folder = Path(folder)
    paths = [
        p for p in sorted(folder.rglob('*'))
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return paths

# # ===== 测试一下路径读取和排序 =====
# ir_paths  = get_image_paths('/Users/zoucy/Documents/20_Research/02_PapersWithCode/02_CDDFuse_CVPR_2023/CDDFuse_replicate/data/MSRS/test/ir')
# vis_paths = get_image_paths('/Users/zoucy/Documents/20_Research/02_PapersWithCode/02_CDDFuse_CVPR_2023/CDDFuse_replicate/data/MSRS/test/vi')

# print(f"IR  文件数: {len(ir_paths)}")
# print(f"VIS 文件数: {len(vis_paths)}")

# # 看前三对是否对齐
# for ir, vis in zip(ir_paths[:3], vis_paths[:3]):
#     print(f"  {ir.name}  ←→  {vis.name}")
# # ===============================

def read_gray(path: str | Path) -> np.ndarray:
    """
    读取图像并转为灰度，归一化到 [0, 1]。

    - 灰度图：直接归一化
    - RGB 图：转 Y 通道（亮度），再归一化
    - 其他通道数：取第一通道

    Args:
        path: 图像路径

    Returns:
        shape (H, W) 的 float32 数组，值域 [0, 1]
    """
    img = imread(str(path))

    if img.ndim == 2:
        # 已经是灰度
        gray = img.astype(np.float32) / 255.0
    elif img.ndim == 3 and img.shape[2] == 3:
        # RGB → Y（BT.601 标准）
        img = img.astype(np.float32) / 255.0
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        # 其他情况取第一通道
        gray = img[:, :, 0].astype(np.float32) / 255.0

    return gray


def add_channel_dim(img: np.ndarray) -> np.ndarray:
    """
    (H, W) → (1, H, W)，方便后续拼 batch。

    Args:
        img: shape (H, W) 的数组

    Returns:
        shape (1, H, W) 的数组
    """
    assert img.ndim == 2, f"期望 2D 数组，得到 shape={img.shape}"
    return img[np.newaxis, :, :]
