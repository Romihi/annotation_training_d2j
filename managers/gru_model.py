"""
後方互換ラッパー — trajectory_models.py に移行済み

既存コードの import を壊さないために残置。
新規コードでは trajectory_models を直接使用すること。
"""

from .trajectory_models import (
    ImageEncoder,
    EgoStateEncoder,
    GRUTrajectoryModel,
)

__all__ = ['ImageEncoder', 'EgoStateEncoder', 'GRUTrajectoryModel']
