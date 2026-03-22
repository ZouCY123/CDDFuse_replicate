"""
src/utils/config.py — YAML 配置加载器

支持：
  1. 读取 YAML 文件
  2. _base_ 继承（子配置只写差异字段）
  3. 点号访问（cfg.data.patch_size）

用法：
    cfg = load_config('configs/base.yaml')
    print(cfg.data.patch_size)   # 128
    print(cfg.train.lr)          # 0.0001
"""

from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import yaml


# ── 核心：递归合并两个 dict ──────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """
    把 override 的字段覆盖到 base 上，递归处理嵌套 dict。

    规则：
      - override 里有的字段 → 覆盖 base
      - override 里没有的字段 → 保留 base 的值
      - 两边都是 dict → 递归合并（不是整个替换）
      - _base_ 字段本身不写入结果

    例子：
        base     = {'train': {'lr': 1e-4, 'epochs': 120}}
        override = {'train': {'lr': 1e-3}}   # 只改 lr
        结果      = {'train': {'lr': 1e-3, 'epochs': 120}}  # epochs 保留
    """
    result = deepcopy(base)
    for k, v in override.items():
        if k == '_base_':
            continue
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


# ── 加载单个 YAML 文件 ───────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


# ── 对外接口 ─────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> DotDict:
    """
    加载 YAML 配置，自动处理 _base_ 继承链，返回支持点号访问的对象。

    Args:
        config_path: YAML 文件路径

    Returns:
        DotDict 对象，支持 cfg.key.subkey 访问
    """
    config_path = Path(config_path).resolve()
    raw = _load_yaml(config_path)

    if '_base_' in raw:
        # _base_ 路径相对于当前 config 文件所在目录
        base_path = (config_path.parent / raw['_base_']).resolve()
        base_cfg  = load_config(base_path)        # 递归，支持多级继承
        merged    = _deep_merge(base_cfg._data, raw)
    else:
        merged = raw

    return DotDict(merged)


# ── 点号访问包装器 ───────────────────────────────────────────────

class DotDict:
    """
    让字典支持点号访问。

    cfg = DotDict({'data': {'patch_size': 128}})
    cfg.data.patch_size   # 128，等价于 cfg['data']['patch_size']
    """

    def __init__(self, data: dict):
        self._data = data
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, DotDict(v))
            elif isinstance(v, list):
                setattr(self, k, [
                    DotDict(item) if isinstance(item, dict) else item
                    for item in v
                ])
            else:
                setattr(self, k, v)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        """转回普通 dict，用于序列化保存"""
        return deepcopy(self._data)

    def __repr__(self) -> str:
        return f"DotDict({self._data})"