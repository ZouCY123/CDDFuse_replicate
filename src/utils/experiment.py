"""
src/utils/experiment.py — 实验目录管理

每次训练自动在 experiments/ 下创建带时间戳的子目录：

    experiments/
    └── cddfuse_ivf_20240315_143022/
        ├── config.yaml       ← 本次实验配置快照
        ├── checkpoints/      ← 模型权重
        ├── logs/             ← TensorBoard 日志
        └── results/          ← 测试输出（test.py 用）

用法：
    em = ExperimentManager(cfg)
    em.save_config(cfg)
    writer = em.get_tb_writer()
    em.save_checkpoint({'encoder': ...}, epoch=10, is_best=False)
    em.close()
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from tensorboardX import SummaryWriter


class ExperimentManager:

    def __init__(self, cfg):
        # 用实验名 + 时间戳命名，保证每次训练目录唯一
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name  = f"{cfg.experiment.name}_{timestamp}"

        self.exp_dir    = Path('experiments') / exp_name
        self.ckpt_dir   = self.exp_dir / 'checkpoints'
        self.log_dir    = self.exp_dir / 'logs'
        self.result_dir = self.exp_dir / 'results'

        # 一次性创建所有子目录
        for d in [self.ckpt_dir, self.log_dir, self.result_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 从 config 读保留最近几个 checkpoint，默认 3
        self.keep_last_n = cfg.logging.get('keep_last_n', 3)

        self._writer = None

        print(f"[Experiment] {self.exp_dir}")

    def save_config(self, cfg):
        """
        把本次实验的完整配置保存到实验目录。
        方便几个月后复现实验时知道当时用了什么参数。
        """
        dst = self.exp_dir / 'config.yaml'
        with open(dst, 'w', encoding='utf-8') as f:
            yaml.dump(cfg.to_dict(), f, allow_unicode=True, default_flow_style=False)

    def get_tb_writer(self) -> SummaryWriter:
        """返回 TensorBoard writer，第一次调用时创建"""
        if self._writer is None:
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        return self._writer

    def save_checkpoint(self, state: dict, epoch: int, is_best: bool = False):
        """
        保存 checkpoint，并自动清理旧文件只保留最近 N 个。

        Args:
            state:    要保存的字典（模型权重、epoch 等）
            epoch:    当前 epoch 数，用于文件命名
            is_best:  是否额外保存一份 best.pth
        """
        import torch

        path = self.ckpt_dir / f'epoch_{epoch:04d}.pth'
        torch.save(state, path)

        # 额外保存一份 best.pth（覆盖上一次的）
        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            shutil.copyfile(path, best_path)

        # 清理旧 checkpoint，只保留最近 keep_last_n 个 epoch_*.pth
        ckpts = sorted(
            self.ckpt_dir.glob('epoch_*.pth'),
            key=lambda p: int(p.stem.split('_')[1])
        )
        for old in ckpts[:-self.keep_last_n]:
            old.unlink()

    def close(self):
        """训练结束时关闭 TensorBoard writer"""
        if self._writer is not None:
            self._writer.close()