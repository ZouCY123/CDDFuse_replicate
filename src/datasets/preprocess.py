"""
preprocess.py — MSRS 数据集预处理主流程

用法：
    python -m src.datasets.preprocess --config configs/base.yaml
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .image_io import get_image_paths, read_gray, add_channel_dim
from .patch_utils import extract_patches, is_low_contrast
from ..utils.config import load_config

logger = logging.getLogger(__name__)


def preprocess(
    ir_dir:     str | Path,
    vis_dir:    str | Path,
    output:     str | Path,
    patch_size: int = 128,
    stride:     int = 200,
) -> int:
    # 函数签名保持不变，仍然可以被其他脚本直接调用
    ir_paths  = get_image_paths(ir_dir)
    vis_paths = get_image_paths(vis_dir)

    if len(ir_paths) == 0:
        raise FileNotFoundError(f"在 {ir_dir} 下找不到任何图像文件")
    if len(ir_paths) != len(vis_paths):
        raise ValueError(
            f"IR 和 VIS 图像数量不匹配：IR={len(ir_paths)}, VIS={len(vis_paths)}"
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    with h5py.File(output, 'w') as h5f:
        ir_grp  = h5f.create_group('ir_patchs')
        vis_grp = h5f.create_group('vis_patchs')

        for ir_path, vis_path in tqdm(
            zip(ir_paths, vis_paths),
            total=len(ir_paths),
            desc='预处理',
        ):
            ir_img  = add_channel_dim(read_gray(ir_path))
            vis_img = add_channel_dim(read_gray(vis_path))

            ir_patches  = extract_patches(ir_img,  patch_size, stride)
            vis_patches = extract_patches(vis_img, patch_size, stride)

            for i in range(len(ir_patches)):
                ir_p, vis_p = ir_patches[i], vis_patches[i]
                if is_low_contrast(ir_p) or is_low_contrast(vis_p):
                    continue
                key = str(total_saved)
                ir_grp.create_dataset(key,  data=ir_p,  dtype=np.float32)
                vis_grp.create_dataset(key, data=vis_p, dtype=np.float32)
                total_saved += 1

    logger.info(f"完成：共保存 {total_saved} 个 patch → {output}")
    return total_saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MSRS 数据集预处理')
    parser.add_argument(
        '--config',
        default='configs/base.yaml',
        help='配置文件路径，默认 configs/base.yaml',
    )
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args   = _parse_args()
    cfg    = load_config(args.config)

    # 从 config 读参数，传给 preprocess()
    count = preprocess(
        ir_dir     = cfg.data.ir_dir,
        vis_dir    = cfg.data.vis_dir,
        output     = cfg.data.processed_h5,
        patch_size = cfg.data.patch_size,
        stride     = cfg.data.stride,
    )
    print(f"写入 {count} 个 patch。")