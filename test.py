"""
test.py — CDDFuse 测试入口

用法：
    # 使用官方预训练权重
    python test.py --config configs/base.yaml \
                   --ckpt  pretrained/CDDFuse_IVF.pth

    # 使用自己训练的权重
    python test.py --config configs/base.yaml \
                   --ckpt  experiments/cddfuse_ivf_XXXXXXXX/checkpoints/best.pth
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.io import imsave

from src.utils.config    import load_config
from src.utils.evaluator import compute_all_metrics, METRIC_NAMES
from src.models.cddfuse  import Restormer_Encoder, Restormer_Decoder
from src.models.branches.base_branch   import BaseFeatureExtraction
from src.models.branches.detail_branch import DetailFeatureExtraction


# ── 图像读取 ──────────────────────────────────────────────────────

def read_gray_test(path: str) -> np.ndarray:
    """
    读取测试图像，返回 uint8 灰度图 shape (H, W)。
    测试时保持 uint8 是为了和论文评估指标对齐。
    """
    img = cv2.imread(str(path))
    assert img is not None, f"读取失败: {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    """(H, W) uint8 → (1, 1, H, W) float32，归一化到 [0, 1]"""
    x = torch.from_numpy(img.astype(np.float32) / 255.0)
    return x.unsqueeze(0).unsqueeze(0).to(device)


# ── 模型加载 ──────────────────────────────────────────────────────

def load_models(cfg, ckpt_path: str, device: str):
    """加载四个模块并切换到 eval 模式"""
    encoder     = Restormer_Encoder(
        inp_channels         = cfg.model.inp_channels,
        dim                  = cfg.model.dim,
        num_blocks           = cfg.model.num_blocks,
        heads                = cfg.model.heads,
        ffn_expansion_factor = cfg.model.ffn_expansion_factor,
        bias                 = cfg.model.bias,
        LayerNorm_type       = cfg.model.LayerNorm_type,
    )
    decoder     = Restormer_Decoder(
        out_channels         = cfg.model.out_channels,
        dim                  = cfg.model.dim,
        num_blocks           = cfg.model.num_blocks,
        heads                = cfg.model.heads,
        ffn_expansion_factor = cfg.model.ffn_expansion_factor,
        bias                 = cfg.model.bias,
        LayerNorm_type       = cfg.model.LayerNorm_type,
    )
    base_fuse   = BaseFeatureExtraction(
        dim       = cfg.model.dim,
        num_heads = cfg.model.heads[2],
    )
    detail_fuse = DetailFeatureExtraction(num_layers=1)  # 测试时用 1 层

    # 加载权重
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # 兼容两种存储格式
    # 格式一（自己训练的）：{'encoder': ..., 'decoder': ..., ...}
    # 格式二（官方预训练）：{'DIDF_Encoder': ..., 'DIDF_Decoder': ..., ...}
    key_map = {
        'encoder':     'DIDF_Encoder',
        'decoder':     'DIDF_Decoder',
        'base_fuse':   'BaseFuseLayer',
        'detail_fuse': 'DetailFuseLayer',
    }
    for model, key_new, key_old in [
        (encoder,     'encoder',     'DIDF_Encoder'),
        (decoder,     'decoder',     'DIDF_Decoder'),
        (base_fuse,   'base_fuse',   'BaseFuseLayer'),
        (detail_fuse, 'detail_fuse', 'DetailFuseLayer'),
    ]:
        if key_new in ckpt:
            state = ckpt[key_new]
        elif key_old in ckpt:
            state = ckpt[key_old]
        else:
            raise KeyError(f"权重文件里找不到 '{key_new}' 或 '{key_old}'")

        # 去掉 DataParallel 的 'module.' 前缀（如果有）
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)

    models = [encoder, decoder, base_fuse, detail_fuse]
    for m in models:
        m.to(device).eval()

    return encoder, decoder, base_fuse, detail_fuse


# ── 单张图像推理 ──────────────────────────────────────────────────

@torch.no_grad()
def fuse_one(ir_t, vis_t, encoder, decoder, base_fuse, detail_fuse):
    """
    单张图像融合，返回 numpy uint8 灰度图。
    """
    base_v,   detail_v,   _ = encoder(vis_t)
    base_i,   detail_i,   _ = encoder(ir_t)

    base_fused   = base_fuse(base_v   + base_i)
    detail_fused = detail_fuse(detail_v + detail_i)

    fused_t, _ = decoder(vis_t, base_fused, detail_fused)

    # 归一化到 [0, 255] 并转 uint8
    fused_t = (fused_t - fused_t.min()) / (fused_t.max() - fused_t.min() + 1e-8)
    fused_np = (fused_t.squeeze().cpu().numpy() * 255).astype(np.uint8)
    return fused_np


# ── 单个数据集评估 ────────────────────────────────────────────────

def evaluate_dataset(
    dataset_cfg,
    out_dir: Path,
    encoder, decoder, base_fuse, detail_fuse,
    device: str,
):
    """
    对一个测试数据集跑完整推理 + 评估，打印指标表格。

    Args:
        dataset_cfg: config 里 test_dirs 的一项，有 name / ir / vi 字段
        out_dir:     融合图像保存目录
    """
    ir_dir  = Path(dataset_cfg.ir)
    vis_dir = Path(dataset_cfg.vi)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 取两个文件夹都有的图像（按文件名对齐）
    ir_files  = sorted(ir_dir.glob('*'))
    vis_files = sorted(vis_dir.glob('*'))
    assert len(ir_files) == len(vis_files), \
        f"{dataset_cfg.name}: IR({len(ir_files)}) 和 VIS({len(vis_files)}) 数量不一致"

    metric_sum = {k: 0.0 for k in METRIC_NAMES}

    for ir_path, vis_path in zip(ir_files, vis_files):
        # 读图
        ir_gray  = read_gray_test(str(ir_path))
        vis_gray = read_gray_test(str(vis_path))

        # 转 tensor
        ir_t  = to_tensor(ir_gray,  device)
        vis_t = to_tensor(vis_gray, device)

        # 推理
        fused = fuse_one(ir_t, vis_t, encoder, decoder, base_fuse, detail_fuse)

        # 保存融合图
        save_path = out_dir / f"{ir_path.stem}.png"
        imsave(str(save_path), fused)

        # 计算指标（累加）
        metrics = compute_all_metrics(fused, ir_gray, vis_gray)
        for k, v in metrics.items():
            metric_sum[k] += v

    # 求均值
    n = len(ir_files)
    metric_avg = {k: v / n for k, v in metric_sum.items()}

    # 打印结果表格
    print(f"\n{'='*72}")
    print(f"  {dataset_cfg.name}")
    print(f"{'='*72}")
    header = ''.join(f'{k:>8}' for k in METRIC_NAMES)
    values = ''.join(f'{metric_avg[k]:>8.2f}' for k in METRIC_NAMES)
    print(f"{'':16}{header}")
    print(f"{'CDDFuse':16}{values}")
    print(f"{'='*72}")

    return metric_avg


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--ckpt',   required=True, help='权重文件路径')
    parser.add_argument('--gpu',    default='0')
    parser.add_argument('--out_dir', default=None,
                        help='融合图像输出目录，默认写到对应 experiments/ 子目录')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.config)

    # 推断输出目录
    if args.out_dir:
        base_out = Path(args.out_dir)
    elif 'experiments' in str(Path(args.ckpt).resolve()):
        # 自动写到 experiments/<name>/results/
        base_out = Path(args.ckpt).parent.parent / 'results'
    else:
        base_out = Path('results')

    print(f"[Test] config={args.config}")
    print(f"[Test] ckpt  ={args.ckpt}")
    print(f"[Test] output={base_out}  device={device}")

    # 加载模型
    encoder, decoder, base_fuse, detail_fuse = load_models(cfg, args.ckpt, device)

    # 逐数据集评估
    all_metrics = {}
    for ds_cfg in cfg.data.test_dirs:
        out_dir = base_out / ds_cfg.name
        metrics = evaluate_dataset(
            ds_cfg, out_dir,
            encoder, decoder, base_fuse, detail_fuse,
            device,
        )
        all_metrics[ds_cfg.name] = metrics

    print("\n测试完成！")


if __name__ == '__main__':
    main()