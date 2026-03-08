# managers/__init__.py
from .annotation_data_manager import AnnotationDataManager
from .mlflow_manager import MLflowManager,ModelType
from .datasetmanager import YOLODatasetManager

# 時系列モデル（model_catalog.py に統合済み）
from model_catalog import (
    ImageEncoder, BaseSequenceModel,
    GRUSequenceModel, TCNSequenceModel, CausalCNNSequenceModel,
    SEQUENCE_ARCHITECTURES, create_sequence_model,
)
from .sequence_dataset import SequenceDataset
from .sequence_training_manager import SequenceTrainingManager

# 後方互換エイリアス
GRUSequenceDataset = SequenceDataset
GRUTrainingManager = SequenceTrainingManager

__all__ = [
    'AnnotationDataManager', 'MLflowManager', 'ModelType', 'YOLODatasetManager',
    # sequence models
    'ImageEncoder', 'BaseSequenceModel',
    'GRUSequenceModel', 'TCNSequenceModel', 'CausalCNNSequenceModel',
    'SEQUENCE_ARCHITECTURES', 'create_sequence_model',
    'SequenceDataset', 'SequenceTrainingManager',
    # legacy aliases
    'GRUSequenceDataset', 'GRUTrainingManager',
]
