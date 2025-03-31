"""
モデル定義ファイル - Donkeycarカスタム実装とTIMMライブラリを使用したニューラルネットワークモデルの定義
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List


import model_info
from model_info import (
    MODEL_ACCURACY_INFO, 
    MODEL_COMPUTE_INFO, 
    MODEL_PARAM_COUNTS, 
    MODEL_INPUT_SIZE, 
    get_model_accuracy,
    get_model_compute,
    get_param_count,
    get_model_input_size
)

class BaseModel(nn.Module):
    """すべてのモデルの基底クラス"""
    def __init__(self, name="base"):
        super(BaseModel, self).__init__()
        self.name = name
        
    def get_preprocess(self):
        """デフォルトの前処理を返す"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def forward(self, x):
        """順伝播処理（サブクラスでオーバーライド）"""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_info(self):
        """モデル情報を返す"""
        return {
            'name': self.name,
            'accuracy': get_model_accuracy(self.name),
            'parameters': get_param_count(self.name),
            'compute': get_model_compute(self.name),
            'input_size': get_model_input_size(self.name)
        }


class TIMMBasedModel(BaseModel):
    """TIMMライブラリを使用するモデルのベースクラス"""
    def __init__(self, name, timm_model_name=None, pretrained=True, num_outputs=2):
        super(TIMMBasedModel, self).__init__(name=name)
        
        # TIMMモデル名が指定されていない場合、モデル名をそのまま使用
        if timm_model_name is None:
            timm_model_name = name
            
        self.timm_model_name = timm_model_name
        
        # モデルの存在確認
        if timm_model_name not in timm.list_models():
            raise ValueError(f"Model '{timm_model_name}' not found in timm library")
        
        # TIMMモデルのロード
        self.base_model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=0)
        
        # 特徴量の次元を取得するためのダミー入力
        input_size = self._get_model_input_size()
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            dummy_output = self.base_model(dummy_input)
        
        # 特徴量の次元
        if isinstance(dummy_output, torch.Tensor):
            feature_dim = dummy_output.shape[1]
        else:
            # 一部のモデルは辞書を返す場合があるので対応
            feature_dim = next(iter(dummy_output.values())).shape[1] if isinstance(dummy_output, dict) else 512
        
        # 回帰器（角度と速度の予測）
        self.regressor = nn.Linear(feature_dim, num_outputs)
    
    def _get_model_input_size(self):
        """モデルの入力サイズを取得"""
        model_input_size = get_model_input_size(self.name)
        return model_input_size
    
    def forward(self, x):
        """順伝播処理"""
        features = self.base_model(x)
        
        # 特徴量がテンソルでない場合（辞書など）の対応
        if not isinstance(features, torch.Tensor):
            features = next(iter(features.values()))
            
        # 回帰出力
        output = self.regressor(features)
        return output
    
    def get_preprocess(self):
        """モデル専用の前処理を返す"""
        input_size = self._get_model_input_size()
        
        # モデルに適した前処理を定義
        # データ拡張なしのシンプルな評価用前処理
        return transforms.Compose([
            transforms.Resize((input_size[0], input_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# 各モデルの実装クラス
# 基本的にはTIMMBasedModelを継承し、必要に応じてカスタマイズ

class ResNet18Model(TIMMBasedModel):
    """TIMMベースのResNet18モデル"""
    def __init__(self, pretrained=True):
        super(ResNet18Model, self).__init__(
            name="resnet18",
            timm_model_name="resnet18",
            pretrained=pretrained
        )


class ResNet34Model(TIMMBasedModel):
    """TIMMベースのResNet34モデル"""
    def __init__(self, pretrained=True):
        super(ResNet34Model, self).__init__(
            name="resnet34",
            timm_model_name="resnet34",
            pretrained=pretrained
        )


class MobileViTXXSModel(TIMMBasedModel):
    """TIMMベースのMobileViT XXSモデル"""
    def __init__(self, pretrained=True):
        super(MobileViTXXSModel, self).__init__(
            name="mobilevit_xxs",
            timm_model_name="mobilevit_xxs",
            pretrained=pretrained
        )


class MobileViTXSModel(TIMMBasedModel):
    """TIMMベースのMobileViT XSモデル"""
    def __init__(self, pretrained=True):
        super(MobileViTXSModel, self).__init__(
            name="mobilevit_xs",
            timm_model_name="mobilevit_xs",
            pretrained=pretrained
        )


class MobileViTSModel(TIMMBasedModel):
    """TIMMベースのMobileViT Sモデル"""
    def __init__(self, pretrained=True):
        super(MobileViTSModel, self).__init__(
            name="mobilevit_s",
            timm_model_name="mobilevit_s",
            pretrained=pretrained
        )


class MobileNetV3SmallModel(TIMMBasedModel):
    """TIMMベースのMobileNetV3 Smallモデル"""
    def __init__(self, pretrained=True):
        super(MobileNetV3SmallModel, self).__init__(
            name="mobilenetv3_small_100",
            timm_model_name="mobilenetv3_small_100",
            pretrained=pretrained
        )


class MobileNetV3LargeModel(TIMMBasedModel):
    """TIMMベースのMobileNetV3 Largeモデル"""
    def __init__(self, pretrained=True):
        super(MobileNetV3LargeModel, self).__init__(
            name="mobilenetv3_large_100",
            timm_model_name="mobilenetv3_large_100",
            pretrained=pretrained
        )


class EfficientNetLite0Model(TIMMBasedModel):
    """TIMMベースのEfficientNet Lite0モデル"""
    def __init__(self, pretrained=True):
        # TIMMではefficientnet_lite0ではなくefficientnet_lite0を使用
        super(EfficientNetLite0Model, self).__init__(
            name="efficientnet_lite0",
            timm_model_name="efficientnet_lite0",
            pretrained=pretrained
        )


class EfficientNetB0Model(TIMMBasedModel):
    """TIMMベースのEfficientNet B0モデル"""
    def __init__(self, pretrained=True):
        super(EfficientNetB0Model, self).__init__(
            name="efficientnet_b0",
            timm_model_name="efficientnet_b0",
            pretrained=pretrained
        )


class ConvNextNanoModel(TIMMBasedModel):
    """TIMMベースのConvNeXt Nanoモデル"""
    def __init__(self, pretrained=True):
        super(ConvNextNanoModel, self).__init__(
            name="convnext_nano",
            timm_model_name="convnext_nano",
            pretrained=pretrained
        )


class ConvNextTinyModel(TIMMBasedModel):
    """TIMMベースのConvNeXt Tinyモデル"""
    def __init__(self, pretrained=True):
        super(ConvNextTinyModel, self).__init__(
            name="convnext_tiny",
            timm_model_name="convnext_tiny",
            pretrained=pretrained
        )


class EdgeNextXXSmallModel(TIMMBasedModel):
    """TIMMベースのEdgeNeXt XX-Smallモデル"""
    def __init__(self, pretrained=True):
        super(EdgeNextXXSmallModel, self).__init__(
            name="edgenext_xx_small",
            timm_model_name="edgenext_xx_small",
            pretrained=pretrained
        )


class EdgeNextXSmallModel(TIMMBasedModel):
    """TIMMベースのEdgeNeXt X-Smallモデル"""
    def __init__(self, pretrained=True):
        super(EdgeNextXSmallModel, self).__init__(
            name="edgenext_x_small",
            timm_model_name="edgenext_x_small",
            pretrained=pretrained
        )


class MobileOneS0Model(TIMMBasedModel):
    """TIMMベースのMobileOne S0モデル"""
    def __init__(self, pretrained=True):
        super(MobileOneS0Model, self).__init__(
            name="mobileone_s0",
            timm_model_name="mobileone_s0",
            pretrained=pretrained
        )


class MobileViTV2_050Model(TIMMBasedModel):
    """TIMMベースのMobileViT v2 050モデル"""
    def __init__(self, pretrained=True):
        super(MobileViTV2_050Model, self).__init__(
            name="mobilevitv2_050",
            timm_model_name="mobilevitv2_050",
            pretrained=pretrained
        )


class GhostNet050Model(TIMMBasedModel):
    """TIMMベースのGhostNet 050モデル"""
    def __init__(self, pretrained=True):
        super(GhostNet050Model, self).__init__(
            name="ghostnet_050",
            timm_model_name="ghostnet_050",
            pretrained=pretrained
        )


class ShuffleNetV2_x05Model(TIMMBasedModel):
    """TIMMベースのShuffleNetV2 x0.5モデル"""
    def __init__(self, pretrained=True):
        super(ShuffleNetV2_x05Model, self).__init__(
            name="shufflenetv2_x0_5",
            timm_model_name="shufflenetv2_x0_5",
            pretrained=pretrained
        )


class SwinTinyModel(TIMMBasedModel):
    """TIMMベースのSwin Transformerモデル"""
    def __init__(self, pretrained=True):
        super(SwinTinyModel, self).__init__(
            name="swin_tiny_patch4_window7_224",
            timm_model_name="swin_tiny_patch4_window7_224",
            pretrained=pretrained
        )


class SwinS3TinyModel(TIMMBasedModel):
    """TIMMベースのSwin S3 Tiny 224モデル"""
    def __init__(self, pretrained=True):
        super(SwinS3TinyModel, self).__init__(
            name="swin_s3_tiny_224",
            timm_model_name="swin_s3_tiny_224",
            pretrained=pretrained
        )


class SwinV2CRTinyNSModel(TIMMBasedModel):
    """TIMMベースのSwin V2 CR Tiny NS 224モデル"""
    def __init__(self, pretrained=True):
        super(SwinV2CRTinyNSModel, self).__init__(
            name="swinv2_cr_tiny_ns_224",
            timm_model_name="swinv2_cr_tiny_ns_224",
            pretrained=pretrained
        )

class SwinMoETinyModel(TIMMBasedModel):
    """TIMMベースのSwin MoE Tiny Patch4 Window7 224モデル"""
    def __init__(self, pretrained=True):
        super(SwinMoETinyModel, self).__init__(
            name="swin_moe_tiny_patch4_window7_224",
            timm_model_name="swin_moe_tiny_patch4_window7_224",
            pretrained=pretrained
        )

class EfficientFormerL1Model(TIMMBasedModel):
    """TIMMベースのEfficientFormer L1モデル"""
    def __init__(self, pretrained=True):
        super(EfficientFormerL1Model, self).__init__(
            name="efficientformer_l1",
            timm_model_name="efficientformer_l1",
            pretrained=pretrained
        )

class DonkeyModel(BaseModel):
    """Donkeycarで使用される標準的なモデル（カスタム実装）"""
    #def __init__(self, pretrained=False, input_size=(120, 160)):
    def __init__(self, pretrained=False, input_size=(224, 224)):
        super(DonkeyModel, self).__init__(name="donkey")
        
        # 入力サイズを保存（前処理と特徴計算で使用）
        self.input_size = input_size
        
        # 特徴抽出部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # 計算される特徴マップサイズに依存するため、ダミー入力を使って計算
        # 入力サイズに基づいてダミー入力を作成
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        dummy_output = self.features(dummy_input)
        feature_size = dummy_output.shape[1]
        
        print(f"DonkeyModel feature size: {feature_size} for input {input_size}")

        # 全結合層（Dense層として分離）
        self.dense_layers = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )        

        # 回帰器（角度と速度の予測）
        self.regressor = nn.Linear(50, 2)
    
    def forward(self, x):
        x = self.features(x)
        x = self.dense_layers(x)
        x = self.regressor(x)
        return x
    
    def get_preprocess(self):
        """Donkeycar用の前処理 - 保存されている入力サイズを使用"""
        return transforms.Compose([
            transforms.Resize(self.input_size),  # 保存された入力サイズを使用
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class DonkeyModel_FCN(BaseModel):
    """Donkeycarで使用される標準的なモデルのFCN版（カスタム実装）"""
    def __init__(self, pretrained=False, input_size=(224, 224)):
        super(DonkeyModel_FCN, self).__init__(name="donkey_fcn")
        
        # 入力サイズを保存（前処理と特徴計算で使用）
        self.input_size = input_size
        
        # 特徴抽出部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # 計算される特徴マップサイズに依存するため、ダミー入力を使って計算
        # 入力サイズに基づいてダミー入力を作成
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        dummy_output = self.features(dummy_input)
        feature_size = dummy_output.shape[1]
        
        # 回帰器（角度と速度の予測）
        self.regressor = nn.Linear(feature_size, 2)
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
    
    def get_preprocess(self):
        """Donkeycar用の前処理 - 保存されている入力サイズを使用"""
        return transforms.Compose([
            transforms.Resize(self.input_size),  # 保存された入力サイズを使用
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# 利用可能なすべてのモデルを登録する辞書
MODEL_REGISTRY = {
    # Donkeycar model
    "donkey": DonkeyModel,
    "donkey_fcn": DonkeyModel_FCN,

    # ResNet variants
    "resnet18": ResNet18Model,
    "resnet34": ResNet34Model,
    
    # MobileViT variants
    "mobilevit_xxs": MobileViTXXSModel,
    "mobilevit_xs": MobileViTXSModel,
    "mobilevit_s": MobileViTSModel,
    
    # MobileNetV3 variants
    "mobilenetv3_small_100": MobileNetV3SmallModel,
    "mobilenetv3_large_100": MobileNetV3LargeModel,
    
    # EfficientNet variants
    "efficientnet_lite0": EfficientNetLite0Model,
    "efficientnet_b0": EfficientNetB0Model,
    
    # ConvNeXt variants
    "convnext_nano": ConvNextNanoModel,
    "convnext_tiny": ConvNextTinyModel,
    
    # EdgeNeXt variants
    "edgenext_xx_small": EdgeNextXXSmallModel,
    "edgenext_x_small": EdgeNextXSmallModel,
    
    # MobileOne variants
    "mobileone_s0": MobileOneS0Model,
    
    # MobileViT v2
    "mobilevitv2_050": MobileViTV2_050Model,
    
    # GhostNet
    "ghostnet_050": GhostNet050Model,
    
    # ShuffleNetV2
    "shufflenetv2_x0_5": ShuffleNetV2_x05Model,
    
    # Swin Transformer variants
    "swin_tiny_patch4_window7_224": SwinTinyModel,
    "swin_tiny": SwinTinyModel,  # 短縮名も対応
    "swin_s3_tiny_224": SwinS3TinyModel,
    "swinv2_cr_tiny_ns_224": SwinV2CRTinyNSModel,
    "swin_moe_tiny_patch4_window7_224": SwinMoETinyModel,
    
    # EfficientFormer variants
    "efficientformer_l1": EfficientFormerL1Model,
    
}


# モデルの利用に関する関数
def get_model(model_type, pretrained=False, input_size=None):
    """モデルタイプに基づいて適切なモデルを返す
    
    Args:
        model_type: モデルの種類
        pretrained: 事前学習済みの重みを使用するかどうか
        input_size: 入力サイズ（height, width）- Noneの場合はデフォルト値を使用
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"未対応のモデルタイプ: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    
    # DonkeyModelの場合、入力サイズを渡す
    if model_type == "donkey" and input_size is not None:
        return model_class(pretrained=pretrained, input_size=input_size)
    
    # その他のモデルの場合は通常通り初期化
    return model_class(pretrained=pretrained)

def list_available_models():
    """利用可能なモデル一覧を返す"""
    return list(MODEL_REGISTRY.keys())


def list_timm_models(keyword=None):
    """TIMMライブラリで利用可能なモデルを一覧表示"""
    all_models = timm.list_models()
    
    if keyword:
        return [m for m in all_models if keyword.lower() in m.lower()]
    
    return all_models


def get_timm_model_groups():
    """TIMMモデルを軽量モデルグループ別に取得"""
    lightweight_keywords = [
        'mobilevit', 'efficientformer', 'edgenext', 'convnext', 
        'mobilenet', 'efficientnet', 'mobilenetv', 'ghostnet',
        'squeezenet', 'shufflenet', 'mnasnet', 'small', 'swin',
        'resnet18', 'resnet34'
    ]
    
    model_groups = {}
    for keyword in lightweight_keywords:
        matching_models = list_timm_models(keyword)
        if matching_models:
            model_groups[keyword] = matching_models
    
    return model_groups

    
class AnnotationDataset(torch.utils.data.Dataset):
    """アノテーションデータのためのカスタムデータセット"""
    def __init__(self, image_paths, annotations, transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # PILで画像を読み込む
        img = Image.open(img_path).convert('RGB')
        
        # 変換を適用
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                # エラーが発生した場合、明示的にNumPy変換を挟む
                img_np = np.array(img)
                img = self.transform(img_np)
        
        # angle, throttleをターゲットとして使用
        annotation = self.annotations[idx]
        target = torch.tensor([annotation["angle"], annotation["throttle"]], dtype=torch.float)
        
        return img, target