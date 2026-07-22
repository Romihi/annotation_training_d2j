# managers/__init__.py
from .annotation_data_manager import AnnotationDataManager
try:
    from .mlflow_manager import MLflowManager, ModelType
except ImportError:
    MLflowManager = None
    ModelType = None
from .datasetmanager import YOLODatasetManager
from .pose_manager import PoseSourceManager, PoseSample

# 時系列モデル（model_catalog.py に統合済み）
from model_catalog import (
    ImageEncoder, BaseSequenceModel,
    GRUSequenceModel, TCNSequenceModel, CausalCNNSequenceModel,
    SEQUENCE_ARCHITECTURES, create_sequence_model,
)
from .sequence_dataset import SequenceDataset
from .sequence_training_manager import SequenceTrainingManager

# TogiVAD（軌道語彙分類 E2E。モデル本体はリポジトリ直下の togivad パッケージ）
from .togivad_training_manager import TogivadTrainingManager, TogivadDataset

# 後方互換エイリアス
GRUSequenceDataset = SequenceDataset
GRUTrainingManager = SequenceTrainingManager

__all__ = [
    'AnnotationDataManager', 'MLflowManager', 'ModelType', 'YOLODatasetManager',
    'PoseSourceManager', 'PoseSample',
    # sequence models
    'ImageEncoder', 'BaseSequenceModel',
    'GRUSequenceModel', 'TCNSequenceModel', 'CausalCNNSequenceModel',
    'SEQUENCE_ARCHITECTURES', 'create_sequence_model',
    'SequenceDataset', 'SequenceTrainingManager',
    # togivad
    'TogivadTrainingManager', 'TogivadDataset',
    # legacy aliases
    'GRUSequenceDataset', 'GRUTrainingManager',
]
