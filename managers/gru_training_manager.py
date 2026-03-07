"""
後方互換ラッパー — trajectory_training_manager.py に移行済み

既存コードの import を壊さないために残置。
新規コードでは trajectory_training_manager を直接使用すること。
"""

from .trajectory_training_manager import (
    TrajectoryTrainingManager as GRUTrainingManager,
)

__all__ = ['GRUTrainingManager']
