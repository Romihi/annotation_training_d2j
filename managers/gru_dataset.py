"""
後方互換ラッパー — trajectory_dataset.py に移行済み

既存コードの import を壊さないために残置。
新規コードでは trajectory_dataset を直接使用すること。
"""

from .trajectory_dataset import (
    TrajectorySequenceDataset as GRUSequenceDataset,
)

__all__ = ['GRUSequenceDataset']
