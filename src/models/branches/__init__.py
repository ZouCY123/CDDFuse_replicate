"""
branches/ 公开接口
包含了 branches 模块的核心组件，供外部调用。
"""

from .base_branch import BaseFeatureExtraction
from .detail_branch import DetailFeatureExtraction


__all__ = ['BaseFeatureExtraction', 'DetailFeatureExtraction']