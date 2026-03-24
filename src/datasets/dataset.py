"""
dataset.py — MSRS 训练数据集
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MSRSDataset(Dataset):
    """
    MSRS 红外-可见光图像融合训练数据集。

    懒加载 + pickle 支持：
      - h5 文件在第一次 __getitem__ 时才打开
      - __getstate__ / __setstate__ 保证多进程 DataLoader 下
        文件句柄不会被 pickle，子进程各自重新打开
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._h5f    = None

        with h5py.File(h5_path, 'r') as f:
            self._keys = sorted(
                f['ir_patchs'].keys(),
                key=lambda x: int(x)
            )

    def __len__(self) -> int:
        return len(self._keys)

    def _ensure_open(self):
        if self._h5f is None:
            self._h5f = h5py.File(self.h5_path, 'r')

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        key = self._keys[index]
        ir  = np.array(self._h5f['ir_patchs'][key],  dtype=np.float32)
        vis = np.array(self._h5f['vis_patchs'][key], dtype=np.float32)
        return torch.from_numpy(vis), torch.from_numpy(ir)

    def __getstate__(self):
        """pickle 前：移除不可序列化的文件句柄"""
        state = self.__dict__.copy()
        state['_h5f'] = None
        return state

    def __setstate__(self, state):
        """unpickle 后：恢复状态，_h5f 保持 None 等待懒加载"""
        self.__dict__.update(state)

    def __del__(self):
        if self._h5f is not None:
            try:
                self._h5f.close()
            except Exception:
                pass