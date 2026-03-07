# managers/__init__.py
from .annotation_data_manager import AnnotationDataManager
from .mlflow_manager import MLflowManager,ModelType
from .datasetmanager import YOLODatasetManager

# 時系列モデル（新API）
from .trajectory_models import (
    ImageEncoder, BaseTrajectoryModel,
    GRUTrajectoryModel, TCNTrajectoryModel, CausalCNNTrajectoryModel,
    TRAJECTORY_ARCHITECTURES, create_trajectory_model,
)
from .trajectory_dataset import TrajectorySequenceDataset
from .trajectory_training_manager import TrajectoryTrainingManager

# 後方互換エイリアス
GRUSequenceDataset = TrajectorySequenceDataset
GRUTrainingManager = TrajectoryTrainingManager

__all__ = [
    'AnnotationDataManager', 'MLflowManager', 'ModelType', 'YOLODatasetManager',
    # trajectory (new)
    'ImageEncoder', 'BaseTrajectoryModel',
    'GRUTrajectoryModel', 'TCNTrajectoryModel', 'CausalCNNTrajectoryModel',
    'TRAJECTORY_ARCHITECTURES', 'create_trajectory_model',
    'TrajectorySequenceDataset', 'TrajectoryTrainingManager',
    # legacy aliases
    'GRUSequenceDataset', 'GRUTrainingManager',
]
