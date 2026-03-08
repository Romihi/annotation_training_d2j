"""
モデル定義ファイル - Donkeycarカスタム実装とTIMMライブラリを使用したニューラルネットワークモデルの定義
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.fx
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
    get_model_input_size,
    SEQUENCE_MODEL_INFO,
    SEQUENCE_PAPER_INFO,
    get_sequence_model_info,
    get_sequence_paper_info,
    list_sequence_architectures,
)


# モデルのロード関数を定義して、チェックポイント形式かどうかを自動判定
# def load_model_weights(model, weights_path, device):
#     checkpoint = torch.load(weights_path, map_location=device)
#     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print("Loaded checkpoint format model")
#     else:
#         model.load_state_dict(checkpoint)
#         print("Loaded state_dict format model")
#     return model

def detect_num_outputs_from_checkpoint(weights_path, device='cpu'):
    """
    チェックポイントファイルから出力数を検出する

    Args:
        weights_path: 重みファイルのパス
        device: 読み込みに使用するデバイス

    Returns:
        int: 検出された出力数（検出できない場合は2）
    """
    num_outputs = 2  # デフォルト
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint

        # regressor.biasまたはregressor.weightから出力数を検出
        if isinstance(state_dict, dict):
            if 'regressor.bias' in state_dict:
                num_outputs = state_dict['regressor.bias'].shape[0]
                print(f"チェックポイントから出力数を検出 (regressor.bias): {num_outputs}")
            elif 'regressor.weight' in state_dict:
                num_outputs = state_dict['regressor.weight'].shape[0]
                print(f"チェックポイントから出力数を検出 (regressor.weight): {num_outputs}")
    except Exception as e:
        print(f"出力数の検出に失敗: {e}")

    return num_outputs


def detect_input_size_from_checkpoint(weights_path, device='cpu'):
    """
    チェックポイントファイルから入力サイズを検出する

    Args:
        weights_path: 重みファイルのパス
        device: 読み込みに使用するデバイス

    Returns:
        tuple or None: 検出された入力サイズ (height, width)、検出できない場合はNone
    """
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'input_size' in checkpoint:
            input_size = checkpoint['input_size']
            print(f"チェックポイントから入力サイズを検出: {input_size}")
            return tuple(input_size)
    except Exception as e:
        print(f"入力サイズの検出に失敗: {e}")

    return None


def load_model_weights(model, weights_path, device):
    """
    モデルの重みを読み込み、指定されたデバイスに移動する

    Args:
        model: PyTorchモデル
        weights_path: 重みファイルのパス
        device: 使用するデバイス (torch.device)

    Returns:
        重みが読み込まれ、デバイスに移動されたモデル
    """
    try:
        print(f"Loading model weights from: {weights_path}")
        print(f"Target device: {device}")
        
        # 重みを読み込み（PyTorch 2.6+対応）
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded checkpoint format model")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("Loaded state_dict format model")
            else:
                # 辞書だが特別なキーがない場合、直接state_dictとして使用
                model.load_state_dict(checkpoint)
                print("Loaded direct state_dict model")
        else:
            # 直接state_dictの場合
            model.load_state_dict(checkpoint)
            print("Loaded direct weights model")
        
        # モデルをデバイスに移動（device属性も更新される）
        model = model.to(device)
        print(f"Model moved to device: {model.device}")
        
        # 推論モードに設定
        model.eval()
        print("Model set to evaluation mode")
        
        return model
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print(f"Attempting to load with alternative method...")
        
        try:
            # 代替方法での読み込み（PyTorch 2.6+対応）
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # デバイスに移動
            model = model.to(device)
            model.eval()
            
            print("Model loaded with alternative method")
            print(f"Model device: {model.device}")
            
            return model
            
        except Exception as e2:
            print(f"All loading attempts failed: {e2}")
            raise

class BaseModel(nn.Module):
    """すべてのモデルの基底クラス"""
    def __init__(self, name="base"):
        super(BaseModel, self).__init__()
        self.name = name
        self._preprocess = None
        self.device = torch.device('cpu')  # デフォルトはCPU

    def to(self, device):
        """デバイスに移動し、device属性を更新"""
        result = super().to(device)
        if isinstance(device, torch.device):
            result.device = device
        elif isinstance(device, str):
            result.device = torch.device(device)
        else:
            # テンソルが渡された場合はそのデバイスを使用
            result.device = device.device if hasattr(device, 'device') else torch.device('cpu')
        return result

    def get_preprocess(self):
        """デフォルトの前処理を返す"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
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
        :return:            tuple of (angle, throttle) or (angle, throttle, speed)
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
            # 結果は (1, num_outputs) の形状
            result = self(tensor_image)

        # CPU上のNumPy配列に変換
        if result.device.type != 'cpu':
            result = result.cpu()
        result = result.numpy().reshape(-1)

        # num_outputsに基づいて返り値を変える
        num_outputs = getattr(self, 'num_outputs', 2)
        if num_outputs == 2:
            return result[0], result[1]  # angle, throttle
        elif num_outputs == 3:
            return result[0], result[1], result[2]  # angle, throttle, speed
        else:
            return tuple(result[:num_outputs])

class TIMMBasedModel(BaseModel):
    """TIMMライブラリを使用するモデルのベースクラス"""
    def __init__(self, name, timm_model_name=None, pretrained=True, num_outputs=2):
        super(TIMMBasedModel, self).__init__(name=name)

        # TIMMモデル名が指定されていない場合、モデル名をそのまま使用
        if timm_model_name is None:
            timm_model_name = name

        self.timm_model_name = timm_model_name
        self.num_outputs = num_outputs

        # モデルの存在確認（ベースモデル名で確認、pretrained weight指定は除外）
        base_model_name = timm_model_name.split('.')[0]
        if base_model_name not in timm.list_models():
            raise ValueError(f"Model '{base_model_name}' not found in timm library")

        # TIMMモデルのロード
        self.base_model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=0)

        # 特徴量の次元を取得するためのダミー入力
        input_size = self._get_model_input_size()
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        # BatchNormのためにevalモードで実行
        self.base_model.eval()
        with torch.no_grad():
            dummy_output = self.base_model(dummy_input)
        self.base_model.train()

        # 特徴量の次元
        if isinstance(dummy_output, torch.Tensor):
            feature_dim = dummy_output.shape[1]
        else:
            # 一部のモデルは辞書を返す場合があるので対応
            feature_dim = next(iter(dummy_output.values())).shape[1] if isinstance(dummy_output, dict) else 512

        # 回帰器（角度と速度の予測、またはangle/throttle/speedの3出力）
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
            transforms.ToTensor()
        ])

# 各モデルの実装クラス
# 基本的にはTIMMBasedModelを継承し、必要に応じてカスタマイズ

class ResNet18Model(TIMMBasedModel):
    """TIMMベースのResNet18モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(ResNet18Model, self).__init__(
            name="resnet18",
            timm_model_name="resnet18",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class ResNet34Model(TIMMBasedModel):
    """TIMMベースのResNet34モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(ResNet34Model, self).__init__(
            name="resnet34",
            timm_model_name="resnet34",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class MobileViTXXSModel(TIMMBasedModel):
    """TIMMベースのMobileViT XXSモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(MobileViTXXSModel, self).__init__(
            name="mobilevit_xxs",
            timm_model_name="mobilevit_xxs",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class MobileViTXSModel(TIMMBasedModel):
    """TIMMベースのMobileViT XSモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(MobileViTXSModel, self).__init__(
            name="mobilevit_xs",
            timm_model_name="mobilevit_xs",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class MobileViTSModel(TIMMBasedModel):
    """TIMMベースのMobileViT Sモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(MobileViTSModel, self).__init__(
            name="mobilevit_s",
            timm_model_name="mobilevit_s",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class MobileNetV3SmallModel(TIMMBasedModel):
    """TIMMベースのMobileNetV3 Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(MobileNetV3SmallModel, self).__init__(
            name="mobilenetv3_small_100",
            timm_model_name="mobilenetv3_small_100",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class MobileNetV3LargeModel(TIMMBasedModel):
    """TIMMベースのMobileNetV3 Largeモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(MobileNetV3LargeModel, self).__init__(
            name="mobilenetv3_large_100",
            timm_model_name="mobilenetv3_large_100",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class MobileNetV4ConvSmallModel(TIMMBasedModel):
    """TIMMベースのMobileNetV4 Conv Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(MobileNetV4ConvSmallModel, self).__init__(
            name="mobilenetv4_conv_small",
            timm_model_name="mobilenetv4_conv_small.e2400_r224_in1k",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class EfficientNetLite0Model(TIMMBasedModel):
    """TIMMベースのEfficientNet Lite0モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        # TIMMではefficientnet_lite0ではなくefficientnet_lite0を使用
        super(EfficientNetLite0Model, self).__init__(
            name="efficientnet_lite0",
            timm_model_name="efficientnet_lite0",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class EfficientNetB0Model(TIMMBasedModel):
    """TIMMベースのEfficientNet B0モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(EfficientNetB0Model, self).__init__(
            name="efficientnet_b0",
            timm_model_name="efficientnet_b0",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class EfficientNetV2SModel(TIMMBasedModel):
    """TIMMベースのEfficientNetV2 Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(EfficientNetV2SModel, self).__init__(
            name="efficientnetv2_s",
            timm_model_name="tf_efficientnetv2_s",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class ConvNextNanoModel(TIMMBasedModel):
    """TIMMベースのConvNeXt Nanoモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(ConvNextNanoModel, self).__init__(
            name="convnext_nano",
            timm_model_name="convnext_nano",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class ConvNextTinyModel(TIMMBasedModel):
    """TIMMベースのConvNeXt Tinyモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(ConvNextTinyModel, self).__init__(
            name="convnext_tiny",
            timm_model_name="convnext_tiny",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class EdgeNextXXSmallModel(TIMMBasedModel):
    """TIMMベースのEdgeNeXt XX-Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(EdgeNextXXSmallModel, self).__init__(
            name="edgenext_xx_small",
            timm_model_name="edgenext_xx_small",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class EdgeNextXSmallModel(TIMMBasedModel):
    """TIMMベースのEdgeNeXt X-Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(EdgeNextXSmallModel, self).__init__(
            name="edgenext_x_small",
            timm_model_name="edgenext_x_small",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class MobileOneS0Model(TIMMBasedModel):
    """TIMMベースのMobileOne S0モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(MobileOneS0Model, self).__init__(
            name="mobileone_s0",
            timm_model_name="mobileone_s0",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class MobileViTV2_050Model(TIMMBasedModel):
    """TIMMベースのMobileViT v2 050モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(MobileViTV2_050Model, self).__init__(
            name="mobilevitv2_050",
            timm_model_name="mobilevitv2_050",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class GhostNet050Model(TIMMBasedModel):
    """TIMMベースのGhostNet 050モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(GhostNet050Model, self).__init__(
            name="ghostnet_050",
            timm_model_name="ghostnet_050",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class ShuffleNetV2_x05Model(TIMMBasedModel):
    """TIMMベースのShuffleNetV2 x0.5モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(ShuffleNetV2_x05Model, self).__init__(
            name="shufflenetv2_x0_5",
            timm_model_name="shufflenetv2_x0_5",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class SwinTinyModel(TIMMBasedModel):
    """TIMMベースのSwin Transformerモデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(SwinTinyModel, self).__init__(
            name="swin_tiny_patch4_window7_224",
            timm_model_name="swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class SwinS3TinyModel(TIMMBasedModel):
    """TIMMベースのSwin S3 Tiny 224モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(SwinS3TinyModel, self).__init__(
            name="swin_s3_tiny_224",
            timm_model_name="swin_s3_tiny_224",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class SwinV2CRTinyNSModel(TIMMBasedModel):
    """TIMMベースのSwin V2 CR Tiny NS 224モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(SwinV2CRTinyNSModel, self).__init__(
            name="swinv2_cr_tiny_ns_224",
            timm_model_name="swinv2_cr_tiny_ns_224",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class SwinMoETinyModel(TIMMBasedModel):
    """TIMMベースのSwin MoE Tiny Patch4 Window7 224モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(SwinMoETinyModel, self).__init__(
            name="swin_moe_tiny_patch4_window7_224",
            timm_model_name="swin_moe_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class EfficientFormerL1Model(TIMMBasedModel):
    """TIMMベースのEfficientFormer L1モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(EfficientFormerL1Model, self).__init__(
            name="efficientformer_l1",
            timm_model_name="efficientformer_l1",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class YOLOv11nModel(TIMMBasedModel):
    """YOLOv11 Nano モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(YOLOv11nModel, self).__init__(
            name="yolo11n",
            timm_model_name="yolo11n",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class YOLOv11sModel(TIMMBasedModel):
    """YOLOv11 Small モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(YOLOv11sModel, self).__init__(
            name="yolo11s",
            timm_model_name="yolo11s",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class YOLOv11mModel(TIMMBasedModel):
    """YOLOv11 Medium モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(YOLOv11mModel, self).__init__(
            name="yolo11m",
            timm_model_name="yolo11m",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class YOLOv11lModel(TIMMBasedModel):
    """YOLOv11 Large モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(YOLOv11lModel, self).__init__(
            name="yolo11l",
            timm_model_name="yolo11l",
            pretrained=pretrained,
            num_outputs=num_outputs
        )


class YOLOv11xModel(TIMMBasedModel):
    """YOLOv11 Extra Large モデル"""
    def __init__(self, pretrained=True, num_outputs=2):
        super(YOLOv11xModel, self).__init__(
            name="yolo11x",
            timm_model_name="yolo11x",
            pretrained=pretrained,
            num_outputs=num_outputs
        )

class DonkeyModel(BaseModel):
    """Donkeycarで使用される標準的なモデル（カスタム実装）"""
    #def __init__(self, pretrained=False, input_size=(120, 160)):
    def __init__(self, pretrained=False, input_size=(224, 224), num_outputs=2):
        super(DonkeyModel, self).__init__(name="donkeycar")

        # 入力サイズを保存（前処理と特徴計算で使用）
        self.input_size = input_size
        self.num_outputs = num_outputs

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

        print(f"DonkeyModel feature size: {feature_size} for input {input_size}, num_outputs: {num_outputs}")

        # 全結合層（Dense層として分離）
        self.dense_layers = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )

        # 回帰器（角度と速度の予測、またはangle/throttle/speedの3出力）
        self.regressor = nn.Linear(50, num_outputs)

    def forward(self, x):
        x = self.features(x)
        x = self.dense_layers(x)
        x = self.regressor(x)
        return x

    def get_preprocess(self):
        """Donkeycar用の前処理 - 保存されている入力サイズを使用"""
        return transforms.Compose([
            transforms.Resize(self.input_size),  # 保存された入力サイズを使用
            transforms.ToTensor()
        ])

    def run(self, img_arr: np.ndarray, other_arr: np.ndarray = None):
        """
        Donkeycar parts interface to run the part in the loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle) or (angle, throttle, speed)
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
            result = self(tensor_image)

        # CPU上のNumPy配列に変換
        if result.device.type != 'cpu':
            result = result.cpu()
        result = result.numpy().reshape(-1)

        if self.num_outputs == 2:
            return result[0], result[1]  # angle, throttle
        else:
            return result[0], result[1], result[2]  # angle, throttle, speed

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
            transforms.ToTensor()
        ])

# 位置推論モデルのベース
class BaseLocationModel(BaseModel):
    """位置推論モデル用のベースクラス"""
    def __init__(self, name, num_classes=8):
        super(BaseLocationModel, self).__init__(name=name)
        self.num_classes = num_classes
        
        # 推論履歴管理
        self.prediction_history = []
        self.max_history = 3
        self.confirmed_class = None
    
    def get_max_probability_class(self, prob_vector):
        """最大確率のクラスを返すヘルパー関数"""
        return np.argmax(prob_vector)
    
    def get_class_with_confidence(self, prob_vector):
        """最大確率のクラスとその確率を返すヘルパー関数"""
        max_class = np.argmax(prob_vector)
        max_prob = prob_vector[max_class]
        return max_class, max_prob
    
    def _update_prediction_history(self, pred_class):
        """推論履歴を更新する内部メソッド"""
        self.prediction_history.append(pred_class)
        
        # 履歴が最大サイズを超えた場合、古いものを削除
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
    
    def get_confirmed_class(self):
        """3回同じクラスが推論されたら、そのクラスに確定するヘルパー関数"""
        if len(self.prediction_history) < self.max_history:
            return None
        
        # 最新の3回の予測が全て同じかチェック
        if all(cls == self.prediction_history[0] for cls in self.prediction_history):
            self.confirmed_class = self.prediction_history[0]
            return self.confirmed_class
        
        return None
    
    def reset_confirmation(self):
        """確定状態をリセットするヘルパー関数"""
        self.prediction_history = []
        self.confirmed_class = None


class BaseWaypointModel(BaseModel):
    """ウェイポイント推論モデル用のベースクラス"""
    def __init__(self, name, num_waypoints=4):
        super(BaseWaypointModel, self).__init__(name=name)
        self.num_waypoints = num_waypoints

    def run_regression(self, img_arr):
        """ウェイポイント推論用の共通runメソッド - x,y座標のリストを返す"""
        # 前処理パイプラインが初期化されていなければ作成
        if self._preprocess is None:
            self._preprocess = self.get_preprocess()

        # PIL Imageに変換
        if isinstance(img_arr, np.ndarray):
            img = Image.fromarray(img_arr)
        else:
            img = img_arr

        # 前処理を適用
        img_tensor = self._preprocess(img).unsqueeze(0)

        # デバイスにテンソルを移動
        img_tensor = img_tensor.to(self.device)

        # 推論実行（勾配計算無効）
        self.eval()
        with torch.no_grad():
            output = self.forward(img_tensor)

        # CPUに移動してnumpy配列に変換
        output = output.cpu().numpy().squeeze()

        # 出力をx,y座標のリストに変換
        waypoints = []
        for i in range(self.num_waypoints):
            x = output[i * 2]
            y = output[i * 2 + 1]
            waypoints.append([x, y])

        return waypoints
    
    def run_classification(self, img_arr):
        """位置推論用の共通runメソッド - 確率ベクトルを返す"""
        # 前処理パイプラインが初期化されていなければ作成
        if self._preprocess is None:
            self._preprocess = self.get_preprocess()
        
        # PILイメージに変換して前処理を適用
        pil_image = Image.fromarray(img_arr)
        tensor_image = self._preprocess(pil_image)
        tensor_image = tensor_image.unsqueeze(0)
        
        # デバイスに転送
        tensor_image = tensor_image.to(self.device)
                
        # 勾配計算なしで推論を実行
        with torch.no_grad():
            logits = self(tensor_image)
            probs = torch.softmax(logits, dim=1)
        
        # CPU上のNumPy配列に変換して確率ベクトルを返す
        probs_array = probs.cpu().numpy()[0]
        
        # 推論履歴を更新
        pred_class = np.argmax(probs_array)
        self._update_prediction_history(pred_class)
        
        return probs_array

class DonkeyWaypointModel(BaseWaypointModel):
    """Donkeycarモデルをベースとしたウェイポイント回帰用モデル"""
    def __init__(self, num_waypoints=4, pretrained=False, input_size=(224, 224)):
        super(DonkeyWaypointModel, self).__init__(name="donkey_waypoint", num_waypoints=num_waypoints)

        # 入力サイズを保存（前処理と特徴計算で使用）
        self.input_size = input_size

        # 特徴抽出部分（DonkeyModelと同じ）
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
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        dummy_output = self.features(dummy_input)
        feature_size = dummy_output.shape[1]

        print(f"DonkeyWaypointModel feature size: {feature_size} for input {input_size}")

        # 全結合層
        self.dense_layers = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )

        # 回帰器（ウェイポイント座標の予測）
        # 出力サイズは num_waypoints * 2 (x,y座標)
        self.regressor = nn.Linear(50, num_waypoints * 2)

    def forward(self, x):
        x = self.features(x)
        x = self.dense_layers(x)
        x = self.regressor(x)
        return x

    def get_preprocess(self):
        """Donkeycar用の前処理 - 保存されている入力サイズを使用"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

    def run(self, img_arr):
        """推論メソッド - BaseWaypointModelの共通メソッドを使用"""
        return self.run_regression(img_arr)


class DonkeyLocationModel(BaseLocationModel):
    """Donkeycarモデルをベースとした位置分類用モデル"""
    def __init__(self, num_classes=8, pretrained=False, input_size=(224, 224)):
        super(DonkeyLocationModel, self).__init__(name="donkey_location", num_classes=num_classes)
        
        # 入力サイズを保存（前処理と特徴計算で使用）
        self.input_size = input_size
        
        # 特徴抽出部分（DonkeyModelと同じ）
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
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        dummy_output = self.features(dummy_input)
        feature_size = dummy_output.shape[1]
        
        print(f"DonkeyLocationModel feature size: {feature_size} for input {input_size}")

        # 全結合層
        self.dense_layers = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )        

        # 分類器（位置情報の予測）
        self.classifier = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.dense_layers(x)
        x = self.classifier(x)
        return x
    
    def get_preprocess(self):
        """Donkeycar用の前処理 - 保存されている入力サイズを使用"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

    def run(self, img_arr):
        """推論メソッド - BaseLocationModelの共通メソッドを使用"""
        return self.run_classification(img_arr)


class ResNet18WaypointModel(BaseWaypointModel):
    """ResNet18をベースとしたウェイポイント回帰用モデル"""
    def __init__(self, num_waypoints=4, pretrained=True):
        super(ResNet18WaypointModel, self).__init__(name="resnet18_waypoint", num_waypoints=num_waypoints)

        # TIMMモデルのロード
        self.base_model = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)

        # 特徴量の次元を取得
        input_size = self._get_model_input_size()
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            dummy_output = self.base_model(dummy_input)

        feature_size = dummy_output.shape[1]
        print(f"ResNet18WaypointModel feature size: {feature_size}")

        # ウェイポイント回帰用のヘッド
        # 出力サイズは num_waypoints * 2 (x,y座標)
        self.waypoint_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_waypoints * 2)
        )

    def forward(self, x):
        # ResNet18で特徴抽出
        x = self.base_model(x)
        # ウェイポイント座標を回帰
        x = self.waypoint_head(x)
        return x

    def _get_model_input_size(self):
        """モデルの入力サイズを取得"""
        # ResNet18の標準入力サイズ
        return (224, 224)

    def get_preprocess(self):
        """ResNet18用の前処理"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def run(self, img_arr):
        """推論メソッド - BaseWaypointModelの共通メソッドを使用"""
        return self.run_regression(img_arr)


class ResNet18LocationModel(BaseLocationModel):
    """ResNet18をベースとした位置分類用モデル"""
    def __init__(self, num_classes=8, pretrained=True):
        super(ResNet18LocationModel, self).__init__(name="resnet18_location", num_classes=num_classes)
        
        # TIMMモデルのロード
        self.base_model = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
        
        # 特徴量の次元を取得
        input_size = self._get_model_input_size()
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            dummy_output = self.base_model(dummy_input)
        
        feature_dim = dummy_output.shape[1]
        
        # 分類器
        self.regressor = nn.Linear(feature_dim, num_classes)
    
    def _get_model_input_size(self):
        """モデルの入力サイズを取得"""
        return get_model_input_size(self.name.replace("_location", ""))
    
    def forward(self, x):
        """順伝播処理"""
        features = self.base_model(x)
        logits = self.regressor(features)
        return logits
    
    def get_preprocess(self):
        """ResNet18用の前処理"""
        input_size = self._get_model_input_size()
        return transforms.Compose([
            transforms.Resize((input_size[0], input_size[1])),
            transforms.ToTensor()
        ])

    def run(self, img_arr):
        """推論メソッド - BaseLocationModelの共通メソッドを使用"""
        return self.run_classification(img_arr)


# 利用可能なすべてのモデルを登録する辞書
MODEL_REGISTRY = {
    # Donkeycar model
    "donkeycar": DonkeyModel,
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

    # MobileNetV4 variants
    "mobilenetv4_conv_small": MobileNetV4ConvSmallModel,

    # EfficientNet variants
    "efficientnet_lite0": EfficientNetLite0Model,
    "efficientnet_b0": EfficientNetB0Model,
    "efficientnetv2_s": EfficientNetV2SModel,

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
    
    # YOLO variants
    "yolo11n": YOLOv11nModel,
    "yolo11s": YOLOv11sModel,
    "yolo11m": YOLOv11mModel,
    "yolo11l": YOLOv11lModel,
    "yolo11x": YOLOv11xModel,

    # 位置推論モデル
    "donkey_location": DonkeyLocationModel,
    "resnet18_location": ResNet18LocationModel,

    # ウェイポイント推論モデル
    "donkey_waypoint": DonkeyWaypointModel,
    "resnet18_waypoint": ResNet18WaypointModel,

}


# モデルの利用に関する関数
def get_model(model_type, pretrained=False, input_size=None, num_outputs=2):
    """モデルタイプに基づいて適切なモデルを返す

    Args:
        model_type: モデルの種類
        pretrained: 事前学習済みの重みを使用するかどうか
        input_size: 入力サイズ（height, width）- Noneの場合はデフォルト値を使用
        num_outputs: 出力数（2=angle/throttle, 3=angle/throttle/speed）
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"未対応のモデルタイプ: {model_type}")

    model_class = MODEL_REGISTRY[model_type]

    # DonkeyModel系の場合、入力サイズとnum_outputsを渡す
    if model_type in ["donkeycar", "donkey_fcn"]:
        if input_size is not None:
            return model_class(pretrained=pretrained, input_size=input_size, num_outputs=num_outputs)
        else:
            return model_class(pretrained=pretrained, num_outputs=num_outputs)
    elif model_type == "donkey_location" and input_size is not None:
        # DonkeyLocationModelの場合、num_classesも必要（デフォルト8）
        return model_class(num_classes=8, pretrained=pretrained, input_size=input_size)
    elif model_type == "donkey_waypoint" and input_size is not None:
        # DonkeyWaypointModelの場合、num_waypointsも必要（デフォルト4）
        return model_class(num_waypoints=4, pretrained=pretrained, input_size=input_size)
    elif model_type == "resnet18_waypoint":
        # ResNet18WaypointModelの場合、num_waypointsも必要（デフォルト4）
        return model_class(num_waypoints=4, pretrained=pretrained)
    elif model_type.endswith('_location'):
        # 位置推論モデルはnum_outputsを使わない
        return model_class(pretrained=pretrained)

    # TIMMベースのモデルの場合、num_outputsをサポート
    # TIMMBasedModelを継承しているモデルかチェック
    if issubclass(model_class, TIMMBasedModel):
        return model_class(pretrained=pretrained, num_outputs=num_outputs)

    # その他のモデルの場合は通常通り初期化
    return model_class(pretrained=pretrained)

def list_available_models():
    """利用可能な自動運転モデル一覧を返す（厳選されたモデルのみ）"""
    # 自動運転学習用に厳選されたモデルのみを返す
    # YOLO、位置推論、ウェイポイントモデルは除外
    allowed_models = [
        "donkeycar",
        "resnet18",
        "mobilevit_xxs",
        "mobilevitv2_050",
        "mobilenetv3_small_100",
        "mobilenetv4_conv_small",
        "efficientnet_b0",
        "efficientnetv2_s",
        "edgenext_xx_small",
        "efficientformer_l1",
    ]
    return [model for model in allowed_models if model in MODEL_REGISTRY]

def list_available_location_models():
    """利用可能な位置推論モデル一覧を返す"""
    return [model for model in MODEL_REGISTRY.keys() if model.endswith('_location')]

def list_all_available_models():
    """利用可能な全モデル一覧を返す（走行モデル + 位置推論モデル）"""
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


# =========================================================================
# 時系列軌道予測モデル
# =========================================================================

class ImageEncoder(nn.Module):
    """MobileNetV3-Smallベースの画像エンコーダ（全モデル・全画像ソース共通）"""

    def __init__(self, feat_dim=128):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=0)
        backbone_out = self.backbone.num_features  # 576
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone_out, feat_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.pool(features).flatten(1)
        return self.relu(self.fc(features))


class EgoStateEncoder(nn.Module):
    """自車状態エンコーダ [steering, throttle, vx, vy, omega] → 特徴量

    TODO: 現在 vx, vy, omega は常に0。IMU/オドメトリデータが利用可能になった場合、
          ego_dim を動的に変更する仕組みを検討する。
    """

    def __init__(self, ego_dim=5, feat_dim=32):
        super().__init__()
        self.fc = nn.Linear(ego_dim, feat_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.fc(x))


class BaseSequenceModel(nn.Module):
    """時系列モデルの共通ベース

    サブクラスは _build_temporal() と _forward_temporal() を実装する。
    """

    def __init__(self, num_image_sources, ego_dim=5, img_feat_dim=128,
                 ego_feat_dim=32, hidden_dim=256, pred_horizon=10, dropout=0.1):
        super().__init__()
        self.num_image_sources = num_image_sources
        self.pred_horizon = pred_horizon
        self.img_feat_dim = img_feat_dim
        self.hidden_dim = hidden_dim

        self.image_encoder = ImageEncoder(img_feat_dim)
        self.ego_encoder = EgoStateEncoder(ego_dim, ego_feat_dim)

        fusion_input_dim = img_feat_dim * num_image_sources + ego_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self._build_temporal(hidden_dim, dropout)

        self.head_dropout = nn.Dropout(dropout)
        self.sequence_head = nn.Linear(hidden_dim, pred_horizon * 2)

    def _build_temporal(self, hidden_dim, dropout):
        raise NotImplementedError

    def _forward_temporal(self, fused):
        raise NotImplementedError

    def forward(self, images, ego_states):
        B, T, S, C, H, W = images.shape

        images_flat = images.reshape(B * T * S, C, H, W)
        img_features = self.image_encoder(images_flat)
        img_features = img_features.reshape(B, T, S * self.img_feat_dim)

        ego_flat = ego_states.reshape(B * T, -1)
        ego_features = self.ego_encoder(ego_flat).reshape(B, T, -1)

        fused = torch.cat([img_features, ego_features], dim=-1)
        fused = self.fusion(fused.reshape(B * T, -1)).reshape(B, T, -1)

        temporal_out = self._forward_temporal(fused)

        out = self.head_dropout(temporal_out)
        output = self.sequence_head(out)
        output = torch.tanh(output)
        return output.reshape(B, self.pred_horizon, 2)


class GRUSequenceModel(BaseSequenceModel):
    """GRUベース軌道予測モデル"""

    ARCH_NAME = "gru"

    def __init__(self, num_image_sources, ego_dim=5, img_feat_dim=128,
                 ego_feat_dim=32, hidden_dim=256, num_layers=1,
                 pred_horizon=10, dropout=0.1):
        self._num_layers = num_layers
        self._dropout = dropout
        super().__init__(num_image_sources, ego_dim, img_feat_dim,
                         ego_feat_dim, hidden_dim, pred_horizon, dropout)

    def _build_temporal(self, hidden_dim, dropout):
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=self._num_layers,
            batch_first=True,
            dropout=dropout if self._num_layers > 1 else 0.0
        )

    def _forward_temporal(self, fused):
        gru_out, _ = self.gru(fused)
        return gru_out[:, -1, :]


class _TCNBlock(nn.Module):
    """TCN用の単一残差ブロック（Dilated Causal Conv1D + 残差接続）"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self._Chomp(padding), self.relu, self.dropout,
            self.conv2, self._Chomp(padding), self.relu, self.dropout,
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU(inplace=True)

    class _Chomp(nn.Module):
        def __init__(self, chomp_size):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            if self.chomp_size > 0:
                return x[:, :, :-self.chomp_size].contiguous()
            return x

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCNSequenceModel(BaseSequenceModel):
    """TCN (Temporal Convolutional Network) ベース軌道予測モデル"""

    ARCH_NAME = "tcn"

    def __init__(self, num_image_sources, ego_dim=5, img_feat_dim=128,
                 ego_feat_dim=32, hidden_dim=256, tcn_channels=None,
                 kernel_size=3, pred_horizon=10, dropout=0.1):
        self._tcn_channels = tcn_channels or [128, 128, 256]
        self._kernel_size = kernel_size
        self._dropout = dropout
        super().__init__(num_image_sources, ego_dim, img_feat_dim,
                         ego_feat_dim, hidden_dim, pred_horizon, dropout)

    def _build_temporal(self, hidden_dim, dropout):
        channels = [hidden_dim] + self._tcn_channels
        layers = []
        for i in range(len(self._tcn_channels)):
            dilation = 2 ** i
            layers.append(_TCNBlock(channels[i], channels[i + 1],
                                    self._kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)
        tcn_out_dim = self._tcn_channels[-1]
        self.tcn_proj = nn.Linear(tcn_out_dim, hidden_dim) if tcn_out_dim != hidden_dim else nn.Identity()

    def _forward_temporal(self, fused):
        x = fused.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        return self.tcn_proj(x)


class _CausalChomp(nn.Module):
    """因果畳み込みのトリミング"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class CausalCNNSequenceModel(BaseSequenceModel):
    """Causal CNN (TinyLidarNet風) 軽量軌道予測モデル"""

    ARCH_NAME = "causal_cnn"

    def __init__(self, num_image_sources, ego_dim=5, img_feat_dim=128,
                 ego_feat_dim=32, hidden_dim=256, cnn_channels=None,
                 kernel_size=3, pred_horizon=10, dropout=0.1):
        self._cnn_channels = cnn_channels or [64, 128, 256]
        self._kernel_size = kernel_size
        self._dropout = dropout
        super().__init__(num_image_sources, ego_dim, img_feat_dim,
                         ego_feat_dim, hidden_dim, pred_horizon, dropout)

    def _build_temporal(self, hidden_dim, dropout):
        channels = [hidden_dim] + self._cnn_channels
        layers = []
        for i in range(len(self._cnn_channels)):
            padding = self._kernel_size - 1
            layers.extend([
                nn.Conv1d(channels[i], channels[i + 1], self._kernel_size, padding=padding),
                _CausalChomp(padding),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        self.cnn = nn.Sequential(*layers)
        cnn_out_dim = self._cnn_channels[-1]
        self.cnn_proj = nn.Linear(cnn_out_dim, hidden_dim) if cnn_out_dim != hidden_dim else nn.Identity()

    def _forward_temporal(self, fused):
        x = fused.transpose(1, 2)
        x = self.cnn(x)
        x = x[:, :, -1]
        return self.cnn_proj(x)


SEQUENCE_ARCHITECTURES = {
    "gru": GRUSequenceModel,
    "tcn": TCNSequenceModel,
    "causal_cnn": CausalCNNSequenceModel,
}


def create_sequence_model(model_arch, num_image_sources, config):
    """アーキテクチャ名からモデルを生成する

    Args:
        model_arch: "gru" | "tcn" | "causal_cnn"
        num_image_sources: 画像ソース数
        config: dict — アーキテクチャ固有パラメータ

    Returns:
        BaseSequenceModel のサブクラスインスタンス
    """
    if model_arch not in SEQUENCE_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {model_arch}. "
                         f"Available: {list(SEQUENCE_ARCHITECTURES.keys())}")

    common = dict(
        num_image_sources=num_image_sources,
        hidden_dim=config.get('hidden_dim', 256),
        pred_horizon=config.get('pred_horizon', 10),
        dropout=config.get('dropout', 0.1),
    )

    if model_arch == "gru":
        return GRUSequenceModel(**common, num_layers=config.get('num_layers', 1))
    elif model_arch == "tcn":
        return TCNSequenceModel(
            **common,
            tcn_channels=config.get('tcn_channels', [128, 128, 256]),
            kernel_size=config.get('kernel_size', 3),
        )
    elif model_arch == "causal_cnn":
        return CausalCNNSequenceModel(
            **common,
            cnn_channels=config.get('cnn_channels', [64, 128, 256]),
            kernel_size=config.get('kernel_size', 3),
        )


SEQUENCE_MODEL_REGISTRY = SEQUENCE_ARCHITECTURES


def list_available_sequence_models():
    """利用可能な時系列軌道予測モデルのアーキテクチャ一覧を返す"""
    return list(SEQUENCE_MODEL_REGISTRY.keys())


def get_sequence_model(arch_name, num_image_sources, config):
    """時系列軌道予測モデルを生成する

    Args:
        arch_name: アーキテクチャ名 ("gru" | "tcn" | "causal_cnn")
        num_image_sources: 画像ソース数
        config: dict — アーキテクチャ固有パラメータ

    Returns:
        BaseSequenceModel のサブクラスインスタンス
    """
    return create_sequence_model(arch_name, num_image_sources, config)

    
class AnnotationDataset(torch.utils.data.Dataset):
    """アノテーションデータのためのカスタムデータセット"""
    def __init__(self, image_paths, annotations, transform=None, cache_images=False, use_speed=False, use_future=False):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        self.use_speed = use_speed
        self.use_future = use_future
        self.future_offsets = [5, 10]  # 5フレーム先と10フレーム先

    def __len__(self):
        return len(self.image_paths)

    def _get_annotation_values(self, annotation):
        """アノテーションからangle, throttle, speedを取得"""
        angle = annotation.get("angle", 0.0)
        throttle = annotation.get("throttle", 0.0)
        speed = annotation.get("speed", annotation.get("user/speed", annotation.get("pilot/speed", 0.0)))
        return angle, throttle, speed

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # キャッシュから画像を取得または読み込み
        if self.cache_images and idx in self.image_cache:
            img = self.image_cache[idx]
        else:
            # PILで画像を読み込む
            img = Image.open(img_path).convert('RGB')
            if self.cache_images:
                self.image_cache[idx] = img

        # 変換を適用
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                # エラーが発生した場合、明示的にNumPy変換を挟む
                img_np = np.array(img)
                img = self.transform(img_np)

        # 現在のアノテーション
        annotation = self.annotations[idx]
        angle, throttle, speed = self._get_annotation_values(annotation)

        # ターゲットテンソルを構築
        target_values = [angle, throttle]

        if self.use_speed:
            target_values.append(speed)

        if self.use_future:
            # 将来フレームのアノテーションを追加
            for offset in self.future_offsets:
                future_idx = idx + offset
                if future_idx < len(self.annotations):
                    future_ann = self.annotations[future_idx]
                    f_angle, f_throttle, f_speed = self._get_annotation_values(future_ann)
                else:
                    # 範囲外の場合は現在の値をコピー
                    f_angle, f_throttle, f_speed = angle, throttle, speed

                # use_speedの設定に応じて出力を調整
                if self.use_speed:
                    target_values.extend([f_angle, f_throttle, f_speed])
                else:
                    target_values.extend([f_angle, f_throttle])

        target = torch.tensor(target_values, dtype=torch.float)

        return img, target
