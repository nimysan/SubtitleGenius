"""
Metrics module for SubtitleGenius
使用prometheus-client监控系统关键指标
"""

from .metrics_manager import MetricsManager
from .vac_metrics import register_vac_metrics

__all__ = ['MetricsManager', 'register_vac_metrics']
