"""
train.py — CDDFuse 训练主入口

用法：
    python train.py --config configs/baseline/cddfuse_ivf.yaml
    python train.py --config configs/ablation/my_exp.yaml --gpu 1
"""

import argparse
import os
import sys
import random

# 添加项目根目录到 Python 路径，使得可以 import src.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config     import load_config
from src.utils.experiment import ExperimentManager
from src.datasets.dataset import MSRSDataset
from src.models.cddfuse   import Restormer_Encoder, Restormer_Decoder
from src.models.branches.base_branch   import BaseFeatureExtraction
from src.models.branches.detail_branch import DetailFeatureExtraction
from src.losses import decomp_loss, FusionLoss, ReconLoss


# ── 工具函数 ──────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizers(cfg, encoder, decoder, base_fuse, detail_fuse):
    """为四个模块分别建优化器和调度器"""
    lr           = cfg.train.lr
    weight_decay = cfg.train.weight_decay
    step_size    = cfg.train.optim_step
    gamma        = cfg.train.optim_gamma

    def make_opt(model):
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def make_sched(opt):
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=step_size, gamma=gamma
        )

    opts   = [make_opt(m)   for m in [encoder, decoder, base_fuse, detail_fuse]]
    scheds = [make_sched(o) for o in opts]
    return opts, scheds


def clip_and_step(opts, models, clip_norm):
    """梯度裁剪 + 参数更新"""
    for opt, model in zip(opts, models):
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        opt.step()


# ── Phase I：重建 + 分解 ─────────────────────────────────────────

def phase1_step(vis, ir, encoder, decoder, recon_loss_fn, cfg):
    """
    单步 Phase I 前向 + 损失计算。
    只涉及 Encoder 和 Decoder，不使用 FuseLayer。

    Returns:
        loss:     总损失，用于反传
        log_dict: 各项子损失，用于 TensorBoard
    """
    # 双路编码
    base_v, detail_v, _ = encoder(vis)
    base_i, detail_i, _ = encoder(ir)

    # 自重建：用各自的特征重建自己
    vis_hat, _ = decoder(vis, base_v, detail_v)
    ir_hat,  _ = decoder(ir,  base_i, detail_i)

    # 重建损失
    loss_recon_v, ssim_v, mse_v, grad_v = recon_loss_fn(vis_hat, vis)
    loss_recon_i, ssim_i, mse_i, grad_i = recon_loss_fn(ir_hat,  ir)

    # 分解损失
    loss_decomp, cc_b, cc_d = decomp_loss(base_v, base_i, detail_v, detail_i)

    # 总损失
    loss = (
        cfg.train.coeff_mse_vis * loss_recon_v +
        cfg.train.coeff_mse_ir  * loss_recon_i +
        cfg.train.coeff_decomp  * loss_decomp
    )

    log_dict = {
        'phase1/loss':        loss.item(),
        'phase1/recon_vis':   loss_recon_v.item(),
        'phase1/recon_ir':    loss_recon_i.item(),
        'phase1/decomp':      loss_decomp.item(),
        'phase1/cc_base':     cc_b.item(),    # 监控：应趋近 1
        'phase1/cc_detail':   cc_d.item(),    # 监控：应趋近 0
    }
    return loss, log_dict


# ── Phase II：融合 + 分解 ────────────────────────────────────────

def phase2_step(vis, ir, encoder, decoder, base_fuse, detail_fuse,
                fusion_loss_fn, cfg):
    """
    单步 Phase II 前向 + 损失计算。
    在 Phase I 基础上加入 FuseLayer。

    Returns:
        loss:     总损失
        log_dict: 各项子损失
    """
    # 双路编码
    base_v,   detail_v,   _ = encoder(vis)
    base_i,   detail_i,   _ = encoder(ir)

    # 融合：Base 和 Detail 分别融合
    base_fused   = base_fuse(base_v   + base_i)
    detail_fused = detail_fuse(detail_v + detail_i)

    # 解码得到融合图
    fused, _ = decoder(vis, base_fused, detail_fused)

    # 融合损失
    loss_fusion, loss_i, loss_g = fusion_loss_fn(vis, ir, fused)

    # 分解损失（Phase II 仍然保持特征分解的约束）
    loss_decomp, cc_b, cc_d = decomp_loss(base_v, base_i, detail_v, detail_i)

    # 总损失
    loss = loss_fusion + cfg.train.coeff_decomp * loss_decomp

    log_dict = {
        'phase2/loss':       loss.item(),
        'phase2/fusion':     loss_fusion.item(),
        'phase2/intensity':  loss_i.item(),
        'phase2/gradient':   loss_g.item(),
        'phase2/decomp':     loss_decomp.item(),
        'phase2/cc_base':    cc_b.item(),
        'phase2/cc_detail':  cc_d.item(),
    }
    return loss, log_dict


# ── 主训练循环 ───────────────────────────────────────────────────

def train(cfg):
    set_seed(cfg.experiment.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Train] {cfg.experiment.name}  device={device}")

    # ── 实验目录 + TensorBoard ────────────────────────────────
    em     = ExperimentManager(cfg)
    em.save_config(cfg)
    writer = em.get_tb_writer()

    # ── 数据 ─────────────────────────────────────────────────
    dataset = MSRSDataset(cfg.data.processed_h5)
    loader  = DataLoader(
        dataset,
        batch_size  = cfg.train.batch_size,
        shuffle     = True,
        num_workers = 0,   # Mac 本地用 0，服务器改为 4
        pin_memory  = device == 'cuda',
    )
    print(f"[Data] {len(dataset)} patches, {len(loader)} batches/epoch")

    # ── 模型 ─────────────────────────────────────────────────
    encoder     = Restormer_Encoder().to(device)
    decoder     = Restormer_Decoder().to(device)
    base_fuse   = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)

    # ── 优化器和调度器 ────────────────────────────────────────
    opts, scheds = build_optimizers(
        cfg, encoder, decoder, base_fuse, detail_fuse
    )
    opt_enc, opt_dec, opt_base, opt_detail = opts
    sch_enc, sch_dec, sch_base, sch_detail = scheds

    # ── 损失函数 ──────────────────────────────────────────────
    recon_loss_fn  = ReconLoss().to(device)
    fusion_loss_fn = FusionLoss().to(device)

    # ── 训练循环 ──────────────────────────────────────────────
    global_step = 0
    epoch_gap   = cfg.train.epoch_gap

    for epoch in range(cfg.train.num_epochs):

        is_phase1 = epoch < epoch_gap
        phase_str = 'I' if is_phase1 else 'II'

        epoch_loss = 0.0

        for vis, ir in loader:
            vis = vis.to(device)
            ir  = ir.to(device)

            # 清零梯度
            for opt in opts:
                opt.zero_grad()

            # 前向 + 损失
            if is_phase1:
                loss, log_dict = phase1_step(
                    vis, ir, encoder, decoder, recon_loss_fn, cfg
                )
            else:
                loss, log_dict = phase2_step(
                    vis, ir, encoder, decoder,
                    base_fuse, detail_fuse, fusion_loss_fn, cfg
                )

            # 反传
            loss.backward()

            # 梯度裁剪 + 更新（Phase I 只更新 enc/dec）
            clip_norm = cfg.train.clip_grad_norm
            if is_phase1:
                clip_and_step(
                    [opt_enc, opt_dec],
                    [encoder, decoder],
                    clip_norm,
                )
            else:
                clip_and_step(
                    [opt_enc, opt_dec, opt_base, opt_detail],
                    [encoder, decoder, base_fuse, detail_fuse],
                    clip_norm,
                )

            # 写 TensorBoard
            for k, v in log_dict.items():
                writer.add_scalar(k, v, global_step)

            epoch_loss  += loss.item()
            global_step += 1

        # ── epoch 结束 ────────────────────────────────────────
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1:3d}/{cfg.train.num_epochs}] "
              f"Phase {phase_str}  loss={avg_loss:.4f}")

        # 调度器步进（Phase II 才步进 base/detail 的调度器）
        sch_enc.step()
        sch_dec.step()
        if not is_phase1:
            sch_base.step()
            sch_detail.step()

        # 学习率下限保护
        for opt in opts:
            if opt.param_groups[0]['lr'] < 1e-6:
                opt.param_groups[0]['lr'] = 1e-6

        # 保存 checkpoint
        if (epoch + 1) % cfg.logging.save_freq == 0:
            em.save_checkpoint(
                state={
                    'epoch':       epoch + 1,
                    'encoder':     encoder.state_dict(),
                    'decoder':     decoder.state_dict(),
                    'base_fuse':   base_fuse.state_dict(),
                    'detail_fuse': detail_fuse.state_dict(),
                },
                epoch    = epoch + 1,
                is_best  = False,
            )

    # 保存最终模型
    em.save_checkpoint(
        state={
            'epoch':       cfg.train.num_epochs,
            'encoder':     encoder.state_dict(),
            'decoder':     decoder.state_dict(),
            'base_fuse':   base_fuse.state_dict(),
            'detail_fuse': detail_fuse.state_dict(),
        },
        epoch   = cfg.train.num_epochs,
        is_best = True,
    )

    em.close()
    print("训练完成！")


# ── 入口 ─────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--gpu',    default='0')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg  = load_config(args.config)
    train(cfg)