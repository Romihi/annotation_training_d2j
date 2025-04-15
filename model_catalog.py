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


# モデルのロード関数を定義して、チェックポイント形式かどうかを自動判定
def load_model_weights(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded checkpoint format model")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state_dict format model")
    return model

class BaseModel(nn.Module):
    """すべてのモデルの基底クラス"""
    def __init__(self, name="base"):
        super(BaseModel, self).__init__()
        self.name = name
        self._preprocess = None

        #以下はモデル読み込み側のロジックで実行
        # モデルを生成した時点でデバイスを決定して保存
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.to(self.device)  # モデル自体を適切なデバイスに移動

        # 初期化時に推論モードに設定
        #self.eval()
        
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


    def run(self, img_arr: np.ndarray, other_arr: np.ndarray = None):
        """
        Donkeycar parts interface to run the part in the loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        # 前処理パイプラインが初期化されていなければ作成（最初の1回だけ）
        if self._preprocess is None:
            self._preprocess = self.get_preprocess()
        
        # PILイメージに変換して前処理を適用
        pil_image = Image.fromarray(img_arr)
        tensor_image = self._preprocess(pil_image)
        tensor_image = tensor_image.unsqueeze(0)
        
        # 初期化時に決定したデバイスに直接転送
        tensor_image = tensor_image.to(self.device)
                
        # 勾配計算なしで推論を実行
        with torch.no_grad():
            # 結果は (1, 2) の形状
            result = self(tensor_image)
        
        # CPU上のNumPy配列に変換
        if result.device.type != 'cpu':
            result = result.cpu()
        result = result.numpy().reshape(-1)
        
        # 必要に応じて、出力を[-1, 1]の範囲に正規化
        #if self.name != "donkey" and self.name != "donkey_fcn":
        result = result * 2 - 1
        
        return result[0], result[1]  # angle, throttle

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
        drop = 0.2
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(24, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
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
            nn.Dropout(drop),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
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
    
# 2画像クラス
class DualInputBaseModel(BaseModel):
    """2画像入力モデルの基底クラス"""
    def __init__(self, name="dual_base", pretrained=False):
        super().__init__(name=name)
        # pretrained引数は現在は使用していないが、将来的な拡張性のために受け取る
        self.pretrained = pretrained
    
    def run(self, img_arr1: np.ndarray, img_arr2: np.ndarray):
        """2つの画像アレイに対して推論を実行"""
        if self._preprocess is None:
            self._preprocess = self.get_preprocess()

        img1 = self._preprocess(Image.fromarray(img_arr1)).unsqueeze(0).to(self.device)
        img2 = self._preprocess(Image.fromarray(img_arr2)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self(img1, img2)

        result = result.cpu().numpy().reshape(-1)
        result = result * 2 - 1  # [-1,1]スケーリング（必要なら）
        return result[0], result[1]
    
class DonkeyDualConcatModel(DualInputBaseModel):
    """2つの画像の特徴量を連結するモデル"""
    def __init__(self, input_size=(224, 224), pretrained=False):
        super().__init__(name="donkey_dual_concat", pretrained=pretrained)
        self.input_size = input_size
        
        # 特徴抽出器を作成（pretrainedを適切に渡す）
        self.feature = DonkeyModel(input_size=input_size, pretrained=pretrained).features
        self.dense = DonkeyModel(input_size=input_size, pretrained=pretrained).dense_layers
        self.regressor = nn.Linear(50 * 2, 2)

    def forward(self, x1, x2):
        f1 = self.dense(self.feature(x1))
        f2 = self.dense(self.feature(x2))
        return self.regressor(torch.cat([f1, f2], dim=1))

    def get_preprocess(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])
    
class DonkeyDual6chModel(DualInputBaseModel):
    """2つの画像をチャネル方向に結合するモデル"""
    def __init__(self, input_size=(224, 224), pretrained=False):
        super().__init__(name="donkey_dual_6ch", pretrained=pretrained)
        self.input_size = input_size
        
        # pretrainedは現在は使用しないが、将来的な拡張のために保持
        self.pretrained = pretrained
        
        self.conv = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=5, stride=2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Conv2d(24, 32, kernel_size=5, stride=2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Flatten()
        )
        dummy = torch.zeros(1, 6, *input_size)
        out_dim = self.conv(dummy).shape[1]
        self.regressor = nn.Sequential(
            nn.Linear(out_dim, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 50), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(50, 2)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # Channel方向に結合 → 6ch
        return self.regressor(self.conv(x))

    def get_preprocess(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

class DonkeyCrossAttentionModel(DualInputBaseModel):
    """クロスアテンションを使用して2つの画像を処理するモデル"""
    def __init__(self, input_size=(224, 224), pretrained=False):
        super().__init__(name="donkey_crossattn", pretrained=pretrained)
        self.input_size = input_size
        
        # 特徴抽出器を作成（pretrainedを適切に渡す）
        self.feature = DonkeyModel_FCN(input_size=input_size, pretrained=pretrained).features
        #self.feature = DonkeyModel(input_size=input_size, pretrained=pretrained).features
        
        dummy = self.feature(torch.zeros(1, 3, *input_size))
        dim = dummy.shape[1]
        self.cross_attn = SimpleCrossAttention(dim)
        self.regressor = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x1, x2):
        f1 = self.feature(x1).unsqueeze(1)  # (B, 1, C)
        f2 = self.feature(x2).unsqueeze(1)
        fused = self.cross_attn(f1, f2).squeeze(1)
        return self.regressor(fused)

    def get_preprocess(self):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

class SimpleCrossAttention(nn.Module):
    def __init__(self, dim, heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, k):
        out, _ = self.attn(q, k, k)
        return self.norm(q + out)

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
        
    # Swin Transformer variants
    "swin_tiny_patch4_window7_224": SwinTinyModel,
    "swin_tiny": SwinTinyModel,  # 短縮名も対応
    "swin_s3_tiny_224": SwinS3TinyModel,
    "swinv2_cr_tiny_ns_224": SwinV2CRTinyNSModel,
    
    # EfficientFormer variants
    "efficientformer_l1": EfficientFormerL1Model,
    
    # 2画像モデル
    "donkey_dual_concat": DonkeyDualConcatModel,
    "donkey_dual_6ch": DonkeyDual6chModel,
    "donkey_crossattn": DonkeyCrossAttentionModel,
}

def is_dual_input_model(model):
    """
    モデルが2画像入力モデルかどうかを判定する
    
    Args:
        model: 判定するモデルインスタンス
        
    Returns:
        bool: 2画像入力モデルの場合はTrue、そうでない場合はFalse
    """
    # DualInputBaseModelのインスタンスかどうかを確認
    return isinstance(model, DualInputBaseModel)

def is_dual_input_model_type(model_type):
    """
    モデルタイプが2画像入力モデルに対応するかどうかを判定する
    
    Args:
        model_type: モデルタイプの文字列
        
    Returns:
        bool: 2画像入力モデルの場合はTrue、そうでない場合はFalse
    """
    # 2画像モデルのプレフィックスまたは完全一致で判定
    dual_model_types = [
        "donkey_dual_concat",
        "donkey_dual_6ch",
        "donkey_crossattn"
    ]
    
    # リストに含まれているか、または "dual" を含む名前か
    return model_type in dual_model_types or "dual" in model_type.lower()

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

def check_all_models():
    """
    全てのモデルをチェックし、単一画像モデルと2画像モデルを分類する
    
    Returns:
        tuple: (単一画像モデルのリスト, 2画像モデルのリスト)
    """
    single_image_models = []
    dual_image_models = []
    
    for model_name, model_class in MODEL_REGISTRY.items():
        try:
            # すべてのモデルに pretrained=False を渡す（統一したインターフェース）
            model = model_class(pretrained=False)
            
            # 実際のインスタンスをチェック
            if is_dual_input_model(model):
                dual_image_models.append(model_name)
            else:
                single_image_models.append(model_name)
        except Exception as e:
            print(f"モデル {model_name} の初期化に失敗: {e}")
            
            # エラーがあっても、名前ベースで分類を試みる
            if is_dual_input_model_type(model_name):
                dual_image_models.append(f"{model_name} (名前ベースで判断)")
            else:
                single_image_models.append(f"{model_name} (初期化失敗)")
    
    return single_image_models, dual_image_models

def print_model_classification():
    """モデルの分類結果を表示する"""
    print("=== モデル分類結果 ===")
    single_models, dual_models = check_all_models()
    
    print("\n単一画像モデル:")
    for model in sorted(single_models):
        print(f"- {model}")
    
    print("\n2画像入力モデル:")
    for model in sorted(dual_models):
        print(f"- {model}")
        
    print(f"\n合計: 単一画像モデル {len(single_models)}個, 2画像モデル {len(dual_models)}個")

# メイン実行部として追加（モジュールとして実行時のみ動作）
if __name__ == "__main__":
    print_model_classification()
