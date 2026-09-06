"""
モデル定義ファイル - Donkeycarカスタム実装とTIMMライブラリを使用したニューラルネットワークモデルの定義
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.fx
import timm
from PIL import Image, ImageDraw
from typing import Dict, Any, Optional, Tuple, List


from config import MAX_SPEED as _MAX_SPEED
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


def detect_multi_source_from_checkpoint(weights_path, device='cpu', verbose=False):
    """
    チェックポイントファイルからマルチソース情報を検出する

    Args:
        weights_path: 重みファイルのパス
        device: 読み込みに使用するデバイス
        verbose: True のときのみ検出結果を print する

    Returns:
        dict: {'num_sources': int, 'fusion_method': str or None, 'selected_sources': list or None, 'base_model_name': str or None}
    """
    result = {'num_sources': 1, 'fusion_method': None, 'selected_sources': None,
              'base_model_name': None, 'virtual_source_type': None, 'temporal_interval': 10}
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'num_sources' in checkpoint:
                result['num_sources'] = checkpoint['num_sources']
            if 'fusion_method' in checkpoint:
                result['fusion_method'] = checkpoint['fusion_method']
            if 'selected_sources' in checkpoint:
                result['selected_sources'] = checkpoint['selected_sources']
            if 'base_model_name' in checkpoint:
                result['base_model_name'] = checkpoint['base_model_name']
            if 'virtual_source_type' in checkpoint:
                result['virtual_source_type'] = checkpoint['virtual_source_type']
            if 'temporal_interval' in checkpoint:
                result['temporal_interval'] = checkpoint['temporal_interval']
            if verbose and result['num_sources'] > 1:
                vt = result['virtual_source_type']
                print(f"マルチソースモデル検出: {result['num_sources']}ソース, 融合: {result['fusion_method']}, "
                      f"仮想タイプ: {vt or 'なし'}, ソース: {result['selected_sources']}")
    except Exception as e:
        print(f"マルチソース情報の検出に失敗: {e}")
    return result


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
            # 学習時のspeed正規化値（保存されていれば推論・表示側で利用）
            if checkpoint.get('speed_normalize'):
                model._speed_normalize = float(checkpoint['speed_normalize'])
            # 学習時の車両マスク（保存されていれば推論時にも同じマスクを適用）
            if checkpoint.get('vehicle_mask'):
                model._vehicle_mask = [tuple(p) for p in checkpoint['vehicle_mask']]
            # 学習時の将来予測フレームオフセット（推論結果のキー・表示に利用）
            if checkpoint.get('future_offsets'):
                model._future_offsets = [int(v) for v in checkpoint['future_offsets']]
            # 学習時の画像埋込設定（推論時にも同じ合成を適用）
            if checkpoint.get('pip_embed'):
                model._pip_embed = checkpoint['pip_embed']
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

        # 初期化時に決定したデバイスに直接転送（モデルのdtypeに合わせる）
        model_dtype = next(self.parameters()).dtype
        tensor_image = tensor_image.to(device=self.device, dtype=model_dtype)

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
    """TIMMライブラリを使用するモデルのベースクラス

    input_size を指定すると、MODEL_INPUT_SIZE のデフォルトを上書きして
    任意の入力サイズで構築・推論できる（resize モード学習時の低解像対応）。
    num_classes=0 で構築された timm モデルは global pool で特徴量次元が
    入力サイズに依存しないため、低解像 input_size でも regressor の形状は不変。
    """
    def __init__(self, name, timm_model_name=None, pretrained=True, num_outputs=2,
                 input_size=None):
        super(TIMMBasedModel, self).__init__(name=name)

        # TIMMモデル名が指定されていない場合、モデル名をそのまま使用
        if timm_model_name is None:
            timm_model_name = name

        self.timm_model_name = timm_model_name
        self.num_outputs = num_outputs
        # input_size 上書き値（None なら MODEL_INPUT_SIZE のデフォルトを使用）
        self._input_size_override = tuple(input_size) if input_size is not None else None

        # モデルの存在確認（ベースモデル名で確認、pretrained weight指定は除外）
        base_model_name = timm_model_name.split('.')[0]
        if base_model_name not in timm.list_models():
            raise ValueError(f"Model '{base_model_name}' not found in timm library")

        # TIMMモデルのロード
        self.base_model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=0)

        # 特徴量の次元を取得するためのダミー入力（override があればそのサイズで構築）
        actual_size = self._get_model_input_size()
        self.input_size = tuple(actual_size)  # 推論時 / 保存時に参照される
        dummy_input = torch.zeros(1, 3, actual_size[0], actual_size[1])
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
        """モデルの入力サイズを取得（override > MODEL_INPUT_SIZE デフォルト）"""
        if getattr(self, '_input_size_override', None) is not None:
            return self._input_size_override
        return get_model_input_size(self.name)
    
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
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(ResNet18Model, self).__init__(
            name="resnet18",
            timm_model_name="resnet18",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class ResNet34Model(TIMMBasedModel):
    """TIMMベースのResNet34モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(ResNet34Model, self).__init__(
            name="resnet34",
            timm_model_name="resnet34",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class MobileViTXXSModel(TIMMBasedModel):
    """TIMMベースのMobileViT XXSモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(MobileViTXXSModel, self).__init__(
            name="mobilevit_xxs",
            timm_model_name="mobilevit_xxs",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class MobileViTXSModel(TIMMBasedModel):
    """TIMMベースのMobileViT XSモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(MobileViTXSModel, self).__init__(
            name="mobilevit_xs",
            timm_model_name="mobilevit_xs",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class MobileViTSModel(TIMMBasedModel):
    """TIMMベースのMobileViT Sモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(MobileViTSModel, self).__init__(
            name="mobilevit_s",
            timm_model_name="mobilevit_s",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class MobileNetV3SmallModel(TIMMBasedModel):
    """TIMMベースのMobileNetV3 Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(MobileNetV3SmallModel, self).__init__(
            name="mobilenetv3_small_100",
            timm_model_name="mobilenetv3_small_100",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class MobileNetV3LargeModel(TIMMBasedModel):
    """TIMMベースのMobileNetV3 Largeモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(MobileNetV3LargeModel, self).__init__(
            name="mobilenetv3_large_100",
            timm_model_name="mobilenetv3_large_100",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class MobileNetV4ConvSmallModel(TIMMBasedModel):
    """TIMMベースのMobileNetV4 Conv Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(MobileNetV4ConvSmallModel, self).__init__(
            name="mobilenetv4_conv_small",
            timm_model_name="mobilenetv4_conv_small.e2400_r224_in1k",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class EfficientNetLite0Model(TIMMBasedModel):
    """TIMMベースのEfficientNet Lite0モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        # TIMMではefficientnet_lite0ではなくefficientnet_lite0を使用
        super(EfficientNetLite0Model, self).__init__(
            name="efficientnet_lite0",
            timm_model_name="efficientnet_lite0",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class EfficientNetB0Model(TIMMBasedModel):
    """TIMMベースのEfficientNet B0モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(EfficientNetB0Model, self).__init__(
            name="efficientnet_b0",
            timm_model_name="efficientnet_b0",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class EfficientNetV2SModel(TIMMBasedModel):
    """TIMMベースのEfficientNetV2 Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(EfficientNetV2SModel, self).__init__(
            name="efficientnetv2_s",
            timm_model_name="tf_efficientnetv2_s",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class ConvNextNanoModel(TIMMBasedModel):
    """TIMMベースのConvNeXt Nanoモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(ConvNextNanoModel, self).__init__(
            name="convnext_nano",
            timm_model_name="convnext_nano",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class ConvNextTinyModel(TIMMBasedModel):
    """TIMMベースのConvNeXt Tinyモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(ConvNextTinyModel, self).__init__(
            name="convnext_tiny",
            timm_model_name="convnext_tiny",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class EdgeNextXXSmallModel(TIMMBasedModel):
    """TIMMベースのEdgeNeXt XX-Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(EdgeNextXXSmallModel, self).__init__(
            name="edgenext_xx_small",
            timm_model_name="edgenext_xx_small",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class EdgeNextXSmallModel(TIMMBasedModel):
    """TIMMベースのEdgeNeXt X-Smallモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(EdgeNextXSmallModel, self).__init__(
            name="edgenext_x_small",
            timm_model_name="edgenext_x_small",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class MobileOneS0Model(TIMMBasedModel):
    """TIMMベースのMobileOne S0モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(MobileOneS0Model, self).__init__(
            name="mobileone_s0",
            timm_model_name="mobileone_s0",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class MobileViTV2_050Model(TIMMBasedModel):
    """TIMMベースのMobileViT v2 050モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(MobileViTV2_050Model, self).__init__(
            name="mobilevitv2_050",
            timm_model_name="mobilevitv2_050",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class GhostNet050Model(TIMMBasedModel):
    """TIMMベースのGhostNet 050モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(GhostNet050Model, self).__init__(
            name="ghostnet_050",
            timm_model_name="ghostnet_050",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class ShuffleNetV2_x05Model(TIMMBasedModel):
    """TIMMベースのShuffleNetV2 x0.5モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(ShuffleNetV2_x05Model, self).__init__(
            name="shufflenetv2_x0_5",
            timm_model_name="shufflenetv2_x0_5",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class SwinTinyModel(TIMMBasedModel):
    """TIMMベースのSwin Transformerモデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(SwinTinyModel, self).__init__(
            name="swin_tiny_patch4_window7_224",
            timm_model_name="swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class SwinS3TinyModel(TIMMBasedModel):
    """TIMMベースのSwin S3 Tiny 224モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(SwinS3TinyModel, self).__init__(
            name="swin_s3_tiny_224",
            timm_model_name="swin_s3_tiny_224",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class SwinV2CRTinyNSModel(TIMMBasedModel):
    """TIMMベースのSwin V2 CR Tiny NS 224モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(SwinV2CRTinyNSModel, self).__init__(
            name="swinv2_cr_tiny_ns_224",
            timm_model_name="swinv2_cr_tiny_ns_224",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class SwinMoETinyModel(TIMMBasedModel):
    """TIMMベースのSwin MoE Tiny Patch4 Window7 224モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(SwinMoETinyModel, self).__init__(
            name="swin_moe_tiny_patch4_window7_224",
            timm_model_name="swin_moe_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class EfficientFormerL1Model(TIMMBasedModel):
    """TIMMベースのEfficientFormer L1モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(EfficientFormerL1Model, self).__init__(
            name="efficientformer_l1",
            timm_model_name="efficientformer_l1",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class YOLOv11nModel(TIMMBasedModel):
    """YOLOv11 Nano モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(YOLOv11nModel, self).__init__(
            name="yolo11n",
            timm_model_name="yolo11n",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class YOLOv11sModel(TIMMBasedModel):
    """YOLOv11 Small モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(YOLOv11sModel, self).__init__(
            name="yolo11s",
            timm_model_name="yolo11s",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class YOLOv11mModel(TIMMBasedModel):
    """YOLOv11 Medium モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(YOLOv11mModel, self).__init__(
            name="yolo11m",
            timm_model_name="yolo11m",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class YOLOv11lModel(TIMMBasedModel):
    """YOLOv11 Large モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(YOLOv11lModel, self).__init__(
            name="yolo11l",
            timm_model_name="yolo11l",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
        )


class YOLOv11xModel(TIMMBasedModel):
    """YOLOv11 Extra Large モデル"""
    def __init__(self, pretrained=True, num_outputs=2, input_size=None):
        super(YOLOv11xModel, self).__init__(
            name="yolo11x",
            timm_model_name="yolo11x",
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size,
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

        # 初期化時に決定したデバイスに直接転送（モデルのdtypeに合わせる）
        model_dtype = next(self.parameters()).dtype
        tensor_image = tensor_image.to(device=self.device, dtype=model_dtype)

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

    def run_classification(self, img_arr):
        """位置推論用の共通runメソッド - 確率ベクトルを返す"""
        # 前処理パイプラインが初期化されていなければ作成
        if self._preprocess is None:
            self._preprocess = self.get_preprocess()

        # PILイメージに変換して前処理を適用
        pil_image = Image.fromarray(img_arr)
        tensor_image = self._preprocess(pil_image)
        tensor_image = tensor_image.unsqueeze(0)

        # デバイスに転送（モデルのdtypeに合わせる）
        model_dtype = next(self.parameters()).dtype
        tensor_image = tensor_image.to(device=self.device, dtype=model_dtype)

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
    """ResNet18をベースとした位置分類用モデル

    input_size を指定すると既定の入力サイズを上書きし、実画像サイズ（や縮小サイズ）で
    構築・推論できる（自動運転モデルの TIMMBasedModel と同じ仕組み）。
    """
    def __init__(self, num_classes=8, pretrained=True, input_size=None):
        super(ResNet18LocationModel, self).__init__(name="resnet18_location", num_classes=num_classes)
        self._input_size_override = tuple(input_size) if input_size is not None else None

        # TIMMモデルのロード
        self.base_model = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)

        # 特徴量の次元を取得
        input_size = self._get_model_input_size()
        self.input_size = tuple(input_size)
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            dummy_output = self.base_model(dummy_input)

        feature_dim = dummy_output.shape[1]

        # 分類器
        self.regressor = nn.Linear(feature_dim, num_classes)

    def _get_model_input_size(self):
        """モデルの入力サイズを取得（override > MODEL_INPUT_SIZE デフォルト）"""
        if getattr(self, '_input_size_override', None) is not None:
            return self._input_size_override
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


class TIMMLocationModel(BaseLocationModel):
    """TIMMバックボーンを利用した位置分類用の汎用モデル

    name は "<timmモデル名>_location" とし、バックボーン・入力サイズ・前処理は
    "_location" を除いたtimmモデル名から解決する。分類ヘッドは既存の
    ResNet18LocationModel と同じ regressor 名で統一し、クラス数検出や
    ヘッド置き換えの既存ロジックをそのまま共通利用できるようにする。
    """
    def __init__(self, name, num_classes=8, pretrained=True, input_size=None):
        super(TIMMLocationModel, self).__init__(name=name, num_classes=num_classes)
        # input_size 上書き値（None なら MODEL_INPUT_SIZE のデフォルトを使用）。
        # timm は global pool のため特徴次元は入力サイズに依存せず、実画像サイズで構築できる
        self._input_size_override = tuple(input_size) if input_size is not None else None

        # TIMMモデルのロード（ヘッドなし）
        timm_model_name = name.replace("_location", "")
        self.base_model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=0)

        # 特徴量の次元を取得（BatchNormのためevalモードでダミー入力を通す）
        input_size = self._get_model_input_size()
        self.input_size = tuple(input_size)
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        self.base_model.eval()
        with torch.no_grad():
            dummy_output = self.base_model(dummy_input)
        self.base_model.train()
        feature_dim = dummy_output.shape[1]

        # 分類器
        self.regressor = nn.Linear(feature_dim, num_classes)

    def _get_model_input_size(self):
        """モデルの入力サイズを取得（override > MODEL_INPUT_SIZE デフォルト）"""
        if getattr(self, '_input_size_override', None) is not None:
            return self._input_size_override
        return get_model_input_size(self.name.replace("_location", ""))

    def forward(self, x):
        """順伝播処理"""
        features = self.base_model(x)
        logits = self.regressor(features)
        return logits

    def get_preprocess(self):
        """バックボーンに応じた前処理"""
        input_size = self._get_model_input_size()
        return transforms.Compose([
            transforms.Resize((input_size[0], input_size[1])),
            transforms.ToTensor()
        ])

    def run(self, img_arr):
        """推論メソッド - BaseLocationModelの共通メソッドを使用"""
        return self.run_classification(img_arr)


class MobileNetV3SmallLocationModel(TIMMLocationModel):
    """MobileNetV3-Smallをベースとした位置分類用モデル（軽量・エッジ向け）"""
    def __init__(self, num_classes=8, pretrained=True, input_size=None):
        super(MobileNetV3SmallLocationModel, self).__init__(
            name="mobilenetv3_small_100_location", num_classes=num_classes, pretrained=pretrained, input_size=input_size)


class MobileNetV4ConvSmallLocationModel(TIMMLocationModel):
    """MobileNetV4-Conv-Smallをベースとした位置分類用モデル（軽量・高精度バランス）"""
    def __init__(self, num_classes=8, pretrained=True, input_size=None):
        super(MobileNetV4ConvSmallLocationModel, self).__init__(
            name="mobilenetv4_conv_small_location", num_classes=num_classes, pretrained=pretrained, input_size=input_size)


class MobileViTXXSLocationModel(TIMMLocationModel):
    """MobileViT-XXSをベースとした位置分類用モデル（最軽量クラス・CNN+Transformer）"""
    def __init__(self, num_classes=8, pretrained=True, input_size=None):
        super(MobileViTXXSLocationModel, self).__init__(
            name="mobilevit_xxs_location", num_classes=num_classes, pretrained=pretrained, input_size=input_size)


class EfficientNetLite0LocationModel(TIMMLocationModel):
    """EfficientNet-Lite0をベースとした位置分類用モデル（エッジ最適化）"""
    def __init__(self, num_classes=8, pretrained=True, input_size=None):
        super(EfficientNetLite0LocationModel, self).__init__(
            name="efficientnet_lite0_location", num_classes=num_classes, pretrained=pretrained, input_size=input_size)


class EdgeNextXXSmallLocationModel(TIMMLocationModel):
    """EdgeNeXt-XX-Smallをベースとした位置分類用モデル（最軽量クラス・CNN+Transformer）"""
    def __init__(self, num_classes=8, pretrained=True, input_size=None):
        super(EdgeNextXXSmallLocationModel, self).__init__(
            name="edgenext_xx_small_location", num_classes=num_classes, pretrained=pretrained, input_size=input_size)


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
    "mobilenetv3_small_100_location": MobileNetV3SmallLocationModel,
    "mobilenetv4_conv_small_location": MobileNetV4ConvSmallLocationModel,
    "mobilevit_xxs_location": MobileViTXXSLocationModel,
    "efficientnet_lite0_location": EfficientNetLite0LocationModel,
    "edgenext_xx_small_location": EdgeNextXXSmallLocationModel,

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
        # 位置推論モデルはnum_outputsを使わない（input_size は実画像サイズでの構築用）
        if input_size is not None:
            return model_class(pretrained=pretrained, input_size=input_size)
        return model_class(pretrained=pretrained)

    # TIMMベースのモデルの場合、num_outputs と input_size を伝播
    # （input_size を指定すると resize モード学習で低解像 timm モデルを構築できる）
    if issubclass(model_class, TIMMBasedModel):
        if input_size is not None:
            return model_class(pretrained=pretrained, num_outputs=num_outputs, input_size=input_size)
        return model_class(pretrained=pretrained, num_outputs=num_outputs)

    # その他のモデルの場合は通常通り初期化
    return model_class(pretrained=pretrained)


def create_location_model(model_type, num_classes=8, pretrained=False, input_size=None):
    """位置推論モデルをクラス数指定付きで生成する

    get_model は num_classes を受け取らないため、保存済みチェックポイントの
    クラス数に合わせてモデルを構築する用途ではこちらを使用する。
    input_size を指定すると実画像サイズ（縮小サイズ）でモデルを構築する。
    """
    if model_type not in MODEL_REGISTRY or not model_type.endswith('_location'):
        raise ValueError(f"未対応の位置推論モデルタイプ: {model_type}")
    if input_size is not None:
        return MODEL_REGISTRY[model_type](num_classes=num_classes, pretrained=pretrained,
                                          input_size=tuple(input_size))
    return MODEL_REGISTRY[model_type](num_classes=num_classes, pretrained=pretrained)


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
    """利用可能な全モデル一覧を返す（走行モデル + 位置推論モデル + 時系列モデル）"""
    all_models = list(MODEL_REGISTRY.keys())
    for arch_name in SEQUENCE_ARCHITECTURES:
        if arch_name not in all_models:
            all_models.append(arch_name)
    return all_models


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


class MultiCameraAttentionFusion(nn.Module):
    """複数カメラ特徴量をクロスアテンションで融合
    入力: (B*T, S, feat_dim)  →  出力: (B*T, feat_dim)
    S=1 の場合は恒等変換に相当するため concat と同等。
    """

    def __init__(self, feat_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        # x: (B*T, S, feat_dim)
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        # 属性ミューテーションは ONNX dynamo export で禁止されるため
        # export / tracing / compile 中はスキップする
        if not (torch.jit.is_tracing()
                or (hasattr(torch.compiler, 'is_compiling') and torch.compiler.is_compiling())):
            self.last_attn_weights = attn_weights.detach()  # (B*T, S, S) — 可視化用
        x = self.norm(x + attn_out)   # residual + LayerNorm
        return x.mean(dim=1)          # (B*T, feat_dim)


class BaseSequenceModel(nn.Module):
    """時系列モデルの共通ベース

    サブクラスは _build_temporal() と _forward_temporal() を実装する。
    fusion_method='attention' かつ num_image_sources>1 の場合、
    クロスカメラ Attention で複数ソースを融合する。
    """

    def __init__(self, num_image_sources, ego_dim=5, img_feat_dim=128,
                 ego_feat_dim=32, hidden_dim=256, pred_horizon=10, dropout=0.1,
                 fusion_method='concat', attn_heads=4):
        super().__init__()
        self.num_image_sources = num_image_sources
        self.pred_horizon = pred_horizon
        self.img_feat_dim = img_feat_dim
        self.hidden_dim = hidden_dim

        self.image_encoder = ImageEncoder(img_feat_dim)
        self.ego_encoder = EgoStateEncoder(ego_dim, ego_feat_dim)

        use_attn = (fusion_method == 'attention' and num_image_sources > 1)
        if use_attn:
            self.camera_attention = MultiCameraAttentionFusion(img_feat_dim, attn_heads, dropout)
            fusion_input_dim = img_feat_dim + ego_feat_dim
        else:
            self.camera_attention = None
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
        img_features = self.image_encoder(images_flat)   # (B*T*S, feat_dim)

        if self.camera_attention is not None:
            img_features = img_features.reshape(B * T, S, self.img_feat_dim)
            img_features = self.camera_attention(img_features)   # (B*T, feat_dim)
            img_features = img_features.reshape(B, T, self.img_feat_dim)
        else:
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
                 pred_horizon=10, dropout=0.1, fusion_method='concat', attn_heads=4):
        self._num_layers = num_layers
        self._dropout = dropout
        super().__init__(num_image_sources, ego_dim, img_feat_dim,
                         ego_feat_dim, hidden_dim, pred_horizon, dropout,
                         fusion_method, attn_heads)

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
                 kernel_size=3, pred_horizon=10, dropout=0.1,
                 fusion_method='concat', attn_heads=4):
        self._tcn_channels = tcn_channels or [128, 128, 256]
        self._kernel_size = kernel_size
        self._dropout = dropout
        super().__init__(num_image_sources, ego_dim, img_feat_dim,
                         ego_feat_dim, hidden_dim, pred_horizon, dropout,
                         fusion_method, attn_heads)

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
                 kernel_size=3, pred_horizon=10, dropout=0.1,
                 fusion_method='concat', attn_heads=4):
        self._cnn_channels = cnn_channels or [64, 128, 256]
        self._kernel_size = kernel_size
        self._dropout = dropout
        super().__init__(num_image_sources, ego_dim, img_feat_dim,
                         ego_feat_dim, hidden_dim, pred_horizon, dropout,
                         fusion_method, attn_heads)

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
        fusion_method=config.get('fusion_method', 'concat'),
        attn_heads=config.get('attn_heads', 4),
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

    
def embed_image_pip(base_img, embed_img, rect_norm):
    """ベース画像の指定領域に別ソース画像を縮小して埋め込んだコピーを返す

    Args:
        base_img: ベースとなるPIL画像（例: cam）
        embed_img: 埋め込むPIL画像（例: lidar BEV）
        rect_norm: (x, y, w, h) 0-1の正規化座標で埋込領域を指定
    """
    if embed_img is None or not rect_norm or len(rect_norm) < 4:
        return base_img
    W, H = base_img.size
    x = int(max(0.0, min(1.0, float(rect_norm[0]))) * W)
    y = int(max(0.0, min(1.0, float(rect_norm[1]))) * H)
    w = max(1, int(float(rect_norm[2]) * W))
    h = max(1, int(float(rect_norm[3]) * H))
    # 領域が画像外にはみ出さないようクランプ
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return base_img
    base = base_img.copy()
    base.paste(embed_img.resize((w, h), Image.BILINEAR), (x, y))
    return base


def apply_vehicle_mask(img, mask_polygon):
    """車両マスク（正規化座標ポリゴン）領域を黒塗りしたコピーを返す

    学習・推論の入力画像から車体などの固定領域を無視するために使用する。
    mask_polygon: [(x, y), ...] 0-1の正規化座標。3頂点未満なら何もしない。
    """
    if not mask_polygon or len(mask_polygon) < 3:
        return img
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    draw.polygon([(x * W, y * H) for x, y in mask_polygon], fill=(0, 0, 0))
    return img


def pixelate_image(img, factor):
    """元サイズのまま内容を factor 倍の解像度に劣化させる（ピクセレーション）"""
    if factor is None or factor >= 1.0:
        return img
    W, H = img.size
    sw = max(1, int(W * factor))
    sh = max(1, int(H * factor))
    return img.resize((sw, sh), Image.NEAREST).resize((W, H), Image.NEAREST)


class PixelateTransform:
    """DataLoader ワーカーでも pickle できる pixelate 変換（transforms.Lambda の代替）"""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return pixelate_image(img, self.factor)


class AnnotationDataset(torch.utils.data.Dataset):
    """アノテーションデータのためのカスタムデータセット"""
    def __init__(self, image_paths, annotations, transform=None, cache_images=False, use_speed=False, use_future=False,
                 speed_normalize=None, mask_polygon=None, future_offsets=None,
                 pip_paths=None, pip_rect=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        self.use_speed = use_speed
        self.use_future = use_future
        self.speed_normalize = speed_normalize  # speed正規化値（None時はMAX_SPEED）
        self.mask_polygon = mask_polygon  # 車両マスク（正規化座標ポリゴン）
        self.future_offsets = list(future_offsets) if future_offsets else [5, 10]  # 将来予測のフレームオフセット
        self.pip_paths = pip_paths  # 画像埋込: image_pathsと同順の埋込画像パスリスト（Noneは埋込なし）
        self.pip_rect = pip_rect    # 画像埋込: (x, y, w, h) 正規化座標

    def __len__(self):
        return len(self.image_paths)

    def _get_annotation_values(self, annotation):
        """アノテーションからangle, throttle, speedを取得"""
        angle = annotation.get("angle", 0.0)
        throttle = annotation.get("throttle", 0.0)
        _raw_speed = annotation.get("enc/speed", annotation.get("speed", annotation.get("user/speed", annotation.get("pilot/speed", 0.0))))
        _norm = self.speed_normalize if getattr(self, 'speed_normalize', None) else _MAX_SPEED
        speed = max(0.0, min(1.0, _raw_speed / _norm)) if _norm > 0 else 0.0
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

        # 車両マスクを適用（キャッシュには元画像を保持）
        img = apply_vehicle_mask(img, self.mask_polygon)

        # 画像埋込（マスク適用後に貼り込む＝マスクで捨てた領域を埋込に再利用できる）
        if self.pip_paths is not None and self.pip_rect and idx < len(self.pip_paths):
            pip_path = self.pip_paths[idx]
            if pip_path:
                try:
                    embed_img = Image.open(pip_path).convert('RGB')
                    img = embed_image_pip(img, embed_img, self.pip_rect)
                except Exception as e:
                    print(f"画像埋込エラー ({pip_path}): {e}")

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


# =========================================================================
# マルチソース画像入力モデル
# =========================================================================

class MultiSourceModel(BaseModel):
    """任意の既存モデルを複数画像ソース入力に対応させるラッパー

    入力テンソル形状: [batch, num_sources*3, H, W]
    forward()内で各ソースの3チャネルに分割し、共有エンコーダで特徴抽出後、
    指定された融合方法で結合して出力する。
    """

    FUSION_METHODS = ['concat', 'attention']

    def __init__(self, base_model_name, num_sources=2, fusion_method='concat',
                 pretrained=True, num_outputs=2, input_size=None):
        display_name = f"multi{num_sources}_{fusion_method}_{base_model_name}"
        super().__init__(name=display_name)
        self.base_model_name = base_model_name
        self.num_sources = num_sources
        self.fusion_method = fusion_method
        self.num_outputs = num_outputs

        # ベースモデルからエンコーダを構築
        base_model_class = MODEL_REGISTRY.get(base_model_name)
        if base_model_class is None:
            raise ValueError(f"Unknown base model: {base_model_name}")

        if base_model_name in ("donkeycar", "donkey_fcn"):
            if input_size is None:
                input_size = (224, 224)
            base = base_model_class(pretrained=pretrained, input_size=input_size, num_outputs=num_outputs)
        elif issubclass(base_model_class, TIMMBasedModel):
            # resize モード時の低解像 input_size を TIMM ベースにも伝播
            if input_size is not None:
                base = base_model_class(pretrained=pretrained, num_outputs=num_outputs, input_size=input_size)
            else:
                base = base_model_class(pretrained=pretrained, num_outputs=num_outputs)
        else:
            base = base_model_class(pretrained=pretrained)

        # エンコーダと特徴次元を抽出
        if isinstance(base, TIMMBasedModel):
            self.encoder = base.base_model
            self.feature_dim = base.regressor.in_features
            self.input_size = base._get_model_input_size()
        elif isinstance(base, DonkeyModel):
            self.encoder = nn.Sequential(base.features, base.dense_layers)
            self.feature_dim = base.regressor.in_features  # 50
            self.input_size = base.input_size
        elif isinstance(base, DonkeyModel_FCN):
            self.encoder = base.features
            dummy = torch.zeros(1, 3, *(input_size or (224, 224)))
            self.feature_dim = base.features(dummy).shape[1]
            self.input_size = base.input_size
        else:
            raise ValueError(f"MultiSourceModel does not support base model type: {type(base)}")

        # 融合方法ごとのレイヤーを構築
        if fusion_method == 'concat':
            fused_dim = self.feature_dim * num_sources
            self.regressor = nn.Sequential(
                nn.Linear(fused_dim, min(256, fused_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(min(256, fused_dim), num_outputs)
            )
        elif fusion_method == 'attention':
            # アテンション融合: 各ソースの特徴にクロスアテンションを適用
            num_heads = max(1, self.feature_dim // 64)
            # feature_dimがnum_headsで割り切れることを保証
            while self.feature_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            self.attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True
            )
            self.norm = nn.LayerNorm(self.feature_dim)
            # 位置埋め込み: ソース順序（現在フレーム=0, 過去フレーム=1,2,...）を保持
            self.pos_embed = nn.Parameter(torch.randn(1, num_sources, self.feature_dim) * 0.02)
            self.regressor = nn.Sequential(
                nn.Linear(self.feature_dim, min(256, self.feature_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(min(256, self.feature_dim), num_outputs)
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}. Use: {self.FUSION_METHODS}")

        print(f"MultiSourceModel created: {display_name}")
        print(f"  encoder feature_dim={self.feature_dim}, num_sources={num_sources}, fusion={fusion_method}")
        print(f"  input_size={self.input_size}, num_outputs={num_outputs}")

    def _encode(self, x):
        """エンコーダで特徴抽出（テンソル形式を保証）"""
        features = self.encoder(x)
        if not isinstance(features, torch.Tensor):
            features = next(iter(features.values()))
        return features

    def forward(self, x):
        """順伝播 - 入力 [batch, num_sources*3, H, W] を分割してエンコード・融合"""
        features_list = []
        for i in range(self.num_sources):
            src = x[:, i*3:(i+1)*3, :, :]
            feat = self._encode(src)
            features_list.append(feat)

        if self.fusion_method == 'concat':
            fused = torch.cat(features_list, dim=1)
        elif self.fusion_method == 'attention':
            # [batch, num_sources, feature_dim]
            seq = torch.stack(features_list, dim=1)
            seq = seq + self.pos_embed  # 位置情報を加算してソース順序を保持
            attn_out, attn_weights = self.attention(seq, seq, seq,
                                                    need_weights=True, average_attn_weights=True)
            # 属性ミューテーションは ONNX dynamo export で禁止されるため
            # export / tracing / compile 中はスキップする
            if not (torch.jit.is_tracing()
                    or (hasattr(torch.compiler, 'is_compiling') and torch.compiler.is_compiling())):
                self.last_attn_weights = attn_weights.detach()  # (batch, S, S) — 可視化用
            norm_out = self.norm(seq + attn_out)
            fused = norm_out[:, 0, :]  # インデックス0=現在フレームの出力を使用

        return self.regressor(fused)

    def get_preprocess(self):
        """各ソース画像に対する前処理を返す（個別適用後にチャネル連結する）"""
        return transforms.Compose([
            transforms.Resize((self.input_size[0], self.input_size[1])),
            transforms.ToTensor()
        ])

    def run_multi(self, *img_arrs):
        """複数画像で推論を実行

        Args:
            *img_arrs: num_sources個のuint8 numpy配列
        Returns:
            tuple: (angle, throttle, ...)
        """
        if len(img_arrs) != self.num_sources:
            raise ValueError(
                f"Expected {self.num_sources} images, got {len(img_arrs)}"
            )

        if self._preprocess is None:
            self._preprocess = self.get_preprocess()

        tensors = []
        for img_arr in img_arrs:
            pil_image = Image.fromarray(img_arr)
            tensor = self._preprocess(pil_image)
            tensors.append(tensor)

        # チャネル連結: [num_sources*3, H, W] -> [1, num_sources*3, H, W]
        stacked = torch.cat(tensors, dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self(stacked)

        if result.device.type != 'cpu':
            result = result.cpu()
        result = result.numpy().reshape(-1)

        return tuple(result[:self.num_outputs])


class MultiSourceDataset(torch.utils.data.Dataset):
    """複数画像ソースのデータセット - チャネル連結テンソルを返す

    grouped_image_paths: [[source1_path, source2_path, ...], ...]
    各サンプルで全ソースの画像をチャネル方向に連結した [num_sources*3, H, W] テンソルを返す
    """

    def __init__(self, grouped_image_paths, annotations, num_sources,
                 transform=None, use_speed=False, use_future=False, speed_normalize=None,
                 mask_polygon=None, future_offsets=None):
        self.grouped_paths = grouped_image_paths
        self.annotations = annotations
        self.num_sources = num_sources
        self.transform = transform
        self.use_speed = use_speed
        self.use_future = use_future
        self.speed_normalize = speed_normalize  # speed正規化値（None時はMAX_SPEED）
        self.mask_polygon = mask_polygon  # 車両マスク（正規化座標ポリゴン）
        self.future_offsets = list(future_offsets) if future_offsets else [5, 10]

    def __len__(self):
        return len(self.grouped_paths)

    def _get_annotation_values(self, annotation):
        """アノテーションからangle, throttle, speedを取得"""
        angle = annotation.get("angle", 0.0)
        throttle = annotation.get("throttle", 0.0)
        _raw_speed = annotation.get("enc/speed", annotation.get("speed", annotation.get("user/speed", annotation.get("pilot/speed", 0.0))))
        _norm = self.speed_normalize if getattr(self, 'speed_normalize', None) else _MAX_SPEED
        speed = max(0.0, min(1.0, _raw_speed / _norm)) if _norm > 0 else 0.0
        return angle, throttle, speed

    def __getitem__(self, idx):
        paths = self.grouped_paths[idx]

        # 各ソース画像を読み込み・変換
        images = []
        for path in paths:
            img = Image.open(path).convert('RGB')
            img = apply_vehicle_mask(img, self.mask_polygon)
            if self.transform:
                try:
                    img = self.transform(img)
                except Exception:
                    img = self.transform(np.array(img))
            images.append(img)

        # チャネル方向に連結: [num_sources*3, H, W]
        stacked = torch.cat(images, dim=0)

        # ターゲットテンソルを構築（AnnotationDatasetと同じロジック）
        annotation = self.annotations[idx]
        angle, throttle, speed = self._get_annotation_values(annotation)
        target_values = [angle, throttle]

        if self.use_speed:
            target_values.append(speed)

        if self.use_future:
            for offset in self.future_offsets:
                future_idx = idx + offset
                if future_idx < len(self.annotations):
                    future_ann = self.annotations[future_idx]
                    f_angle, f_throttle, f_speed = self._get_annotation_values(future_ann)
                else:
                    f_angle, f_throttle, f_speed = angle, throttle, speed
                if self.use_speed:
                    target_values.extend([f_angle, f_throttle, f_speed])
                else:
                    target_values.extend([f_angle, f_throttle])

        target = torch.tensor(target_values, dtype=torch.float)
        return stacked, target


class VirtualSourceDataset(torch.utils.data.Dataset):
    """単一カメラ画像から複数の仮想ソースを生成するデータセット

    virtual_type:
        'crop'     : 水平方向の空間クロップ [左/中/右] (A)
        'scale'    : 中央スケールピラミッド [全体/2x/4x zoom] (B)
        'temporal' : 現在＋過去フレームスタック [t, t-1, t-2] (C)
    """

    _SOURCE_NAMES = {
        'crop':     {2: ['left', 'right'],
                     3: ['left', 'center', 'right'],
                     4: ['left', 'center_l', 'center_r', 'right']},
        'scale':    {2: ['full', 'zoom2x'],
                     3: ['full', 'zoom2x', 'zoom4x'],
                     4: ['full', 'zoom2x', 'zoom4x', 'zoom8x']},
        'temporal': {2: ['t+0', 't-1'],
                     3: ['t+0', 't-1', 't-2'],
                     4: ['t+0', 't-1', 't-2', 't-3']},
    }

    @staticmethod
    def source_names(virtual_type: str, num_sources: int, temporal_interval: int = 10) -> list:
        if virtual_type == 'temporal':
            return ['t+0'] + [f't-{i * temporal_interval}' for i in range(1, num_sources)]
        names = VirtualSourceDataset._SOURCE_NAMES.get(virtual_type, {})
        return names.get(num_sources, [f'{virtual_type}_{i}' for i in range(num_sources)])

    def __init__(self, image_paths, annotations, num_virtual_sources=3,
                 virtual_type='crop', transform=None, use_speed=False, use_future=False,
                 temporal_interval: int = 10, speed_normalize=None, mask_polygon=None,
                 future_offsets=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.num_virtual_sources = num_virtual_sources
        self.virtual_type = virtual_type
        self.transform = transform
        self.use_speed = use_speed
        self.use_future = use_future
        self.temporal_interval = temporal_interval
        self.speed_normalize = speed_normalize  # speed正規化値（None時はMAX_SPEED）
        self.mask_polygon = mask_polygon  # 車両マスク（正規化座標ポリゴン）
        self.future_offsets = list(future_offsets) if future_offsets else [5, 10]

    def __len__(self):
        return len(self.image_paths)

    def _get_annotation_values(self, annotation):
        angle = annotation.get("angle", 0.0)
        throttle = annotation.get("throttle", 0.0)
        _raw_speed = annotation.get("enc/speed", annotation.get("speed", annotation.get("user/speed", annotation.get("pilot/speed", 0.0))))
        _norm = self.speed_normalize if getattr(self, 'speed_normalize', None) else _MAX_SPEED
        speed = max(0.0, min(1.0, _raw_speed / _norm)) if _norm > 0 else 0.0
        return angle, throttle, speed

    def _spatial_crops(self, img):
        """水平方向に N 分割（隣接スライス間 15% オーバーラップ）"""
        W, H = img.size
        n = self.num_virtual_sources
        step = W / n
        overlap = step * 0.15
        crops = []
        for i in range(n):
            x0 = max(0, int(step * i - overlap))
            x1 = min(W, int(step * (i + 1) + overlap))
            crops.append(img.crop((x0, 0, x1, H)))
        return crops

    def _scale_pyramid(self, img):
        """中央クロップを段階的に拡大してリサイズ (55% ずつ縮小)"""
        W, H = img.size
        sources = [img]
        scale = 0.55
        for _ in range(self.num_virtual_sources - 1):
            cw = max(1, int(W * scale))
            ch = max(1, int(H * scale))
            x0 = (W - cw) // 2
            y0 = (H - ch) // 2
            cropped = img.crop((x0, y0, x0 + cw, y0 + ch))
            sources.append(cropped.resize((W, H), Image.LANCZOS))
            scale *= 0.55
        return sources

    def _temporal_stack(self, idx):
        """現在フレームと過去 N-1 フレームを取得（先頭でクランプ）"""
        sources = []
        for k in range(self.num_virtual_sources):
            prev_idx = max(0, idx - k * self.temporal_interval)
            frame = Image.open(self.image_paths[prev_idx]).convert('RGB')
            sources.append(apply_vehicle_mask(frame, self.mask_polygon))
        return sources

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        # 車両マスクは元画像座標で適用（crop/scaleの仮想ソースにも正しく反映される）
        img = apply_vehicle_mask(img, self.mask_polygon)

        if self.virtual_type == 'crop':
            source_imgs = self._spatial_crops(img)
        elif self.virtual_type == 'scale':
            source_imgs = self._scale_pyramid(img)
        elif self.virtual_type == 'temporal':
            source_imgs = self._temporal_stack(idx)
        else:
            source_imgs = [img] * self.num_virtual_sources

        tensors = []
        for s_img in source_imgs:
            if self.transform:
                try:
                    t = self.transform(s_img)
                except Exception:
                    import torchvision.transforms as _T
                    t = self.transform(_T.ToTensor()(s_img))
            else:
                import torchvision.transforms as _T
                t = _T.ToTensor()(s_img)
            tensors.append(t)

        stacked = torch.cat(tensors, dim=0)

        annotation = self.annotations[idx]
        angle, throttle, speed = self._get_annotation_values(annotation)
        target_values = [angle, throttle]

        if self.use_speed:
            target_values.append(speed)

        if self.use_future:
            for offset in self.future_offsets:
                future_idx = min(idx + offset, len(self.annotations) - 1)
                future_ann = self.annotations[future_idx]
                f_angle, f_throttle, f_speed = self._get_annotation_values(future_ann)
                if self.use_speed:
                    target_values.extend([f_angle, f_throttle, f_speed])
                else:
                    target_values.extend([f_angle, f_throttle])

        return stacked, torch.tensor(target_values, dtype=torch.float)


def create_multi_source_model(base_model_name, num_sources=2, fusion_method='concat',
                              pretrained=True, num_outputs=2, input_size=None):
    """マルチソースモデルのファクトリ関数"""
    return MultiSourceModel(
        base_model_name=base_model_name,
        num_sources=num_sources,
        fusion_method=fusion_method,
        pretrained=pretrained,
        num_outputs=num_outputs,
        input_size=input_size
    )


# ---------------------------------------------------------------------------
# 位置推論モデル: 複数画像入力 + クラス分類 / 座標・姿勢回帰
# ---------------------------------------------------------------------------

# 出力ヘッドの正規順序。output_mode は存在するヘッド名をこの順で '_' 連結した文字列
# （例: 'class', 'pose', 'class_pose', 'grid', 'class_grid', 'pose_grid', 'class_pose_grid'）
#   class: 位置クラス分類 / pose: 座標・姿勢回帰 / grid: x,y を格子に離散化した格子分類
LOCATION_HEAD_ORDER = ('class', 'pose', 'grid')
LOCATION_OUTPUT_MODES = ('class', 'pose', 'class_pose', 'grid', 'class_grid', 'pose_grid',
                         'class_pose_grid')


def location_heads(output_mode):
    """output_mode → 含まれるヘッド名のタプル（LOCATION_HEAD_ORDER 順）"""
    parts = set(str(output_mode or 'class').split('_'))
    unknown = parts - set(LOCATION_HEAD_ORDER)
    if unknown:
        raise ValueError(f"Unknown output_mode: {output_mode}. Use: {LOCATION_OUTPUT_MODES}")
    return tuple(h for h in LOCATION_HEAD_ORDER if h in parts)


def make_output_mode(use_class=False, use_pose=False, use_grid=False):
    """ヘッドの有無から output_mode 文字列を組み立てる（何も無ければ 'class'）"""
    flags = {'class': use_class, 'pose': use_pose, 'grid': use_grid}
    heads = [h for h in LOCATION_HEAD_ORDER if flags[h]]
    return '_'.join(heads) if heads else 'class'


# --- 格子分類（x, y を格子セルに離散化） -----------------------------------

def make_grid_config(pose_targets, cell_size=0.5, margin_ratio=0.02):
    """[x, y, theta] リストから格子定義を作る

    Returns: {'x_min', 'y_min', 'cell_size', 'nx', 'ny', 'num_cells', 'occupied': {cell: count}}
    セル index = iy * nx + ix（ix: x方向、iy: y方向）
    """
    arr = np.asarray(pose_targets, dtype=np.float64).reshape(-1, 3)
    cell = float(cell_size)
    x_min, x_max = float(arr[:, 0].min()), float(arr[:, 0].max())
    y_min, y_max = float(arr[:, 1].min()), float(arr[:, 1].max())
    mx = max((x_max - x_min) * margin_ratio, cell * 0.05)
    my = max((y_max - y_min) * margin_ratio, cell * 0.05)
    x_min -= mx
    y_min -= my
    nx = max(1, int(math.ceil((x_max + mx - x_min) / cell)))
    ny = max(1, int(math.ceil((y_max + my - y_min) / cell)))
    cfg = {'x_min': x_min, 'y_min': y_min, 'cell_size': cell, 'nx': nx, 'ny': ny,
           'num_cells': nx * ny}
    occupied = {}
    for x, y in arr[:, :2]:
        c = grid_cell_index(x, y, cfg)
        occupied[c] = occupied.get(c, 0) + 1
    cfg['occupied'] = occupied
    return cfg


def grid_cell_index(x, y, grid_config):
    """座標 → セル index（範囲外は端のセルにクランプ）"""
    cell = grid_config['cell_size']
    ix = int((x - grid_config['x_min']) / cell)
    iy = int((y - grid_config['y_min']) / cell)
    ix = min(max(ix, 0), grid_config['nx'] - 1)
    iy = min(max(iy, 0), grid_config['ny'] - 1)
    return iy * grid_config['nx'] + ix


def grid_cell_center(cell, grid_config):
    """セル index → セル中心座標 (x, y)"""
    nx = grid_config['nx']
    ix, iy = int(cell) % nx, int(cell) // nx
    c = grid_config['cell_size']
    return (grid_config['x_min'] + (ix + 0.5) * c, grid_config['y_min'] + (iy + 0.5) * c)


def grid_topn(probs, grid_config, n=10):
    """確率ベクトル → 上位 n セル [{'cell', 'ix', 'iy', 'prob', 'x', 'y'}, ...]（確率降順）"""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    n = max(1, min(int(n), probs.shape[0]))
    order = np.argsort(probs)[::-1][:n]
    nx = grid_config['nx']
    items = []
    for cell in order:
        cx, cy = grid_cell_center(int(cell), grid_config)
        items.append({'cell': int(cell), 'ix': int(cell) % nx, 'iy': int(cell) // nx,
                      'prob': float(probs[cell]), 'x': cx, 'y': cy})
    return items


def grid_weighted_position(top_items, n=None):
    """上位セルの中心を確率で重み付け平均した座標 (x, y)。n で使用する上位件数を絞る"""
    items = list(top_items)[: (int(n) if n else None)]
    if not items:
        return None
    w = sum(max(it['prob'], 0.0) for it in items)
    if w <= 0:
        return items[0]['x'], items[0]['y']
    return (sum(it['x'] * it['prob'] for it in items) / w,
            sum(it['y'] * it['prob'] for it in items) / w)


def grid_position_errors(logits_or_probs, true_xy, grid_config, top_n=3):
    """バッチの格子出力から (Top1 中心の位置誤差[m], Top-N 重み付き位置誤差[m]) を返す"""
    p = np.asarray(logits_or_probs, dtype=np.float64)
    p = p.reshape(-1, p.shape[-1])
    xy = np.asarray(true_xy, dtype=np.float64).reshape(-1, 2)
    e1, ew = [], []
    for row, (tx, ty) in zip(p, xy):
        top = grid_topn(row, grid_config, n=max(1, top_n))
        e1.append(math.hypot(top[0]['x'] - tx, top[0]['y'] - ty))
        wx, wy = grid_weighted_position(top)
        ew.append(math.hypot(wx - tx, wy - ty))
    return np.asarray(e1), np.asarray(ew)


def location_virtual_sources(img, virtual_type, num_sources):
    """単一画像から仮想ソース画像のリストを生成する（crop / scale）

    VirtualSourceDataset と同じ分割規則。temporal は呼び出し側で
    過去フレームのパスを組にして渡すため、ここでは扱わない。
    """
    W, H = img.size
    if virtual_type == 'crop':
        step = W / num_sources
        overlap = step * 0.15
        crops = []
        for i in range(num_sources):
            x0 = max(0, int(step * i - overlap))
            x1 = min(W, int(step * (i + 1) + overlap))
            crops.append(img.crop((x0, 0, x1, H)))
        return crops
    if virtual_type == 'scale':
        sources = [img]
        scale = 0.55
        for _ in range(num_sources - 1):
            cw = max(1, int(W * scale))
            ch = max(1, int(H * scale))
            x0 = (W - cw) // 2
            y0 = (H - ch) // 2
            sources.append(img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.LANCZOS))
            scale *= 0.55
        return sources
    return [img] * num_sources


def split_location_outputs(outputs, output_mode):
    """位置モデルの forward 出力を (class_logits, pose, grid_logits) に分解する。無いものは None。

    forward はヘッドが1つならテンソル、複数なら LOCATION_HEAD_ORDER 順のタプルを返す。
    """
    heads = location_heads(output_mode)
    if len(heads) == 1:
        values = (outputs,)
    else:
        values = tuple(outputs)
    by_head = dict(zip(heads, values))
    return by_head.get('class'), by_head.get('pose'), by_head.get('grid')


def normalize_pose_targets(poses, pose_norm, include_heading=True):
    """[x, y, theta] の配列を学習用ベクトルへ正規化する

    x, y は pose_norm の min/max で [-1, 1] へ、theta は (cos, sin) へ変換する。
    Returns: np.ndarray [N, 4] (include_heading) または [N, 2]
    """
    arr = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    x_rng = max(pose_norm['x_max'] - pose_norm['x_min'], 1e-6)
    y_rng = max(pose_norm['y_max'] - pose_norm['y_min'], 1e-6)
    xn = 2.0 * (arr[:, 0] - pose_norm['x_min']) / x_rng - 1.0
    yn = 2.0 * (arr[:, 1] - pose_norm['y_min']) / y_rng - 1.0
    cols = [xn, yn]
    if include_heading:
        cols.extend([np.cos(arr[:, 2]), np.sin(arr[:, 2])])
    return np.stack(cols, axis=1).astype(np.float32)


def denormalize_pose_output(vec, pose_norm, include_heading=True):
    """モデル出力ベクトル → (x[m], y[m], theta[rad] or None)"""
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    x_rng = pose_norm['x_max'] - pose_norm['x_min']
    y_rng = pose_norm['y_max'] - pose_norm['y_min']
    x = (vec[0] + 1.0) / 2.0 * x_rng + pose_norm['x_min']
    y = (vec[1] + 1.0) / 2.0 * y_rng + pose_norm['y_min']
    theta = None
    if include_heading and vec.shape[0] >= 4:
        theta = float(np.arctan2(vec[3], vec[2]))
    return float(x), float(y), theta


def pose_errors(pred_vec, target_vec, pose_norm, include_heading=True):
    """正規化ベクトル同士から (位置誤差[m], 方位誤差[rad] or None) を計算する（バッチ対応）"""
    pred = np.asarray(pred_vec, dtype=np.float64)
    tgt = np.asarray(target_vec, dtype=np.float64)
    pred = pred.reshape(-1, pred.shape[-1])
    tgt = tgt.reshape(-1, tgt.shape[-1])
    x_rng = pose_norm['x_max'] - pose_norm['x_min']
    y_rng = pose_norm['y_max'] - pose_norm['y_min']
    dx = (pred[:, 0] - tgt[:, 0]) / 2.0 * x_rng
    dy = (pred[:, 1] - tgt[:, 1]) / 2.0 * y_rng
    pos_err = np.hypot(dx, dy)
    head_err = None
    if include_heading and pred.shape[1] >= 4 and tgt.shape[1] >= 4:
        th_p = np.arctan2(pred[:, 3], pred[:, 2])
        th_t = np.arctan2(tgt[:, 3], tgt[:, 2])
        head_err = np.abs((th_p - th_t + np.pi) % (2 * np.pi) - np.pi)
    return pos_err, head_err


class MultiSourceLocationModel(BaseLocationModel):
    """位置推論モデルの汎用ラッパー（複数画像入力・複数出力ヘッド）

    - 入力: [batch, num_sources*3, H, W]（MultiSourceModel と同じチャネル連結形式）
      num_sources=1 のときは通常の [batch, 3, H, W]
    - 出力（output_mode）:
        'class'      : logits [B, num_classes]
        'pose'       : pose   [B, pose_dim]   (x, y, cos, sin) 正規化値
        'class_pose' : (logits, pose)
    - エンコーダは既存の "<backbone>_location" モデル（base_model_name）から抽出し、
      ソース間で共有する。num_sources=1 のときは state_dict のキーが
      既存の単一入力位置モデル（base_model.* / regressor.*）と互換になる。
    """

    FUSION_METHODS = ('concat', 'attention')

    def __init__(self, base_model_name, num_sources=1, fusion_method='concat',
                 num_classes=8, output_mode='class', pose_dim=4, pretrained=True,
                 input_size=None, num_grid_classes=0):
        heads = location_heads(output_mode)   # 不正な output_mode はここで ValueError
        if 'grid' in heads and int(num_grid_classes or 0) <= 0:
            raise ValueError("格子分類には num_grid_classes（格子セル数）が必要です。")
        if fusion_method not in self.FUSION_METHODS:
            raise ValueError(f"Unknown fusion method: {fusion_method}. Use: {self.FUSION_METHODS}")
        if base_model_name not in MODEL_REGISTRY or not base_model_name.endswith('_location'):
            raise ValueError(f"未対応の位置推論モデルタイプ: {base_model_name}")

        display_name = (base_model_name if num_sources == 1
                        else f"multi{num_sources}_{fusion_method}_{base_model_name}")
        super().__init__(name=display_name, num_classes=num_classes)
        self.base_model_name = base_model_name
        self.num_sources = num_sources
        self.fusion_method = fusion_method
        self.output_mode = output_mode
        self.heads = heads
        self.pose_dim = pose_dim
        self.num_grid_classes = int(num_grid_classes or 0)

        base_cls = MODEL_REGISTRY[base_model_name]
        if base_model_name == 'donkey_location':
            base = base_cls(num_classes=num_classes, pretrained=pretrained,
                            input_size=input_size or (224, 224))
            self.base_model = nn.Sequential(base.features, base.dense_layers)
            self.feature_dim = base.classifier.in_features
            self.input_size = tuple(base.input_size)
        else:
            # input_size を渡すと実画像サイズ（縮小サイズ）で構築（timm は global pool のため
            # 特徴次元は入力サイズに依存しない）
            base = base_cls(num_classes=num_classes, pretrained=pretrained, input_size=input_size)
            self.base_model = base.base_model
            self.feature_dim = base.regressor.in_features
            self.input_size = tuple(base._get_model_input_size())

        # --- 融合 ---
        if num_sources == 1:
            fused_dim = self.feature_dim
        elif fusion_method == 'concat':
            fused_dim = self.feature_dim * num_sources
        else:
            num_heads = max(1, self.feature_dim // 64)
            while self.feature_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            self.attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)
            self.norm = nn.LayerNorm(self.feature_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, num_sources, self.feature_dim) * 0.02)
            fused_dim = self.feature_dim
        self.fused_dim = fused_dim

        # --- 出力ヘッド ---
        # クラス分類ヘッド（既存モデルと同じ regressor 名。単一入力時は Linear で互換）
        if 'class' in heads:
            if num_sources == 1:
                self.regressor = nn.Linear(fused_dim, num_classes)
            else:
                hidden = min(256, fused_dim)
                self.regressor = nn.Sequential(
                    nn.Linear(fused_dim, hidden), nn.ReLU(inplace=True),
                    nn.Dropout(0.2), nn.Linear(hidden, num_classes))
        # 座標・姿勢回帰ヘッド
        if 'pose' in heads:
            hidden = min(256, fused_dim)
            self.pose_head = nn.Sequential(
                nn.Linear(fused_dim, hidden), nn.ReLU(inplace=True),
                nn.Dropout(0.2), nn.Linear(hidden, pose_dim))
        # 格子分類ヘッド（x, y を格子セルに離散化したクラス分類）
        if 'grid' in heads:
            hidden = min(256, fused_dim)
            self.grid_head = nn.Sequential(
                nn.Linear(fused_dim, hidden), nn.ReLU(inplace=True),
                nn.Dropout(0.2), nn.Linear(hidden, self.num_grid_classes))

        fusion_desc = fusion_method if num_sources > 1 else '-'
        print(f"MultiSourceLocationModel created: {display_name} "
              f"(sources={num_sources}, fusion={fusion_desc}, output={output_mode}, "
              f"feature_dim={self.feature_dim}, input_size={self.input_size})")

    def _encode(self, x):
        features = self.base_model(x)
        if not isinstance(features, torch.Tensor):
            features = next(iter(features.values()))
        return features

    def _fuse(self, x):
        if self.num_sources == 1:
            return self._encode(x)
        feats = [self._encode(x[:, i * 3:(i + 1) * 3, :, :]) for i in range(self.num_sources)]
        if self.fusion_method == 'concat':
            return torch.cat(feats, dim=1)
        seq = torch.stack(feats, dim=1) + self.pos_embed
        attn_out, attn_weights = self.attention(seq, seq, seq, need_weights=True,
                                                average_attn_weights=True)
        if not (torch.jit.is_tracing()
                or (hasattr(torch.compiler, 'is_compiling') and torch.compiler.is_compiling())):
            self.last_attn_weights = attn_weights.detach()
        return self.norm(seq + attn_out)[:, 0, :]

    def forward(self, x):
        """ヘッドが1つならテンソル、複数なら LOCATION_HEAD_ORDER 順のタプルを返す"""
        fused = self._fuse(x)
        outs = []
        for head in self.heads:
            if head == 'class':
                outs.append(self.regressor(fused))
            elif head == 'pose':
                outs.append(self.pose_head(fused))
            else:
                outs.append(self.grid_head(fused))
        return outs[0] if len(outs) == 1 else tuple(outs)

    def _get_model_input_size(self):
        return self.input_size

    def get_preprocess(self):
        """各ソース画像に個別適用する前処理（適用後にチャネル連結する）"""
        return transforms.Compose([
            transforms.Resize((self.input_size[0], self.input_size[1])),
            transforms.ToTensor()
        ])

    def run(self, *img_arrs, virtual_type=None):
        """複数画像で推論を実行し、{'probs': ndarray|None, 'pose_vec': ndarray|None} を返す

        virtual_type='crop'/'scale' の場合は1枚の画像から仮想ソースを生成する。
        pose_vec は正規化値のため、denormalize_pose_output で座標へ戻す。
        """
        if self._preprocess is None:
            self._preprocess = self.get_preprocess()
        pil_images = [Image.fromarray(a) if isinstance(a, np.ndarray) else a for a in img_arrs]
        if virtual_type in ('crop', 'scale') and len(pil_images) == 1 and self.num_sources > 1:
            pil_images = location_virtual_sources(pil_images[0], virtual_type, self.num_sources)
        if len(pil_images) != self.num_sources:
            raise ValueError(f"Expected {self.num_sources} images, got {len(pil_images)}")
        tensors = [self._preprocess(img) for img in pil_images]
        stacked = torch.cat(tensors, dim=0).unsqueeze(0)
        model_dtype = next(self.parameters()).dtype
        stacked = stacked.to(device=self.device, dtype=model_dtype)
        with torch.no_grad():
            outputs = self(stacked)
        logits, pose, grid = split_location_outputs(outputs, self.output_mode)
        result = {'probs': None, 'pose_vec': None, 'grid_probs': None}
        if logits is not None:
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            result['probs'] = probs
            self._update_prediction_history(int(np.argmax(probs)))
        if pose is not None:
            result['pose_vec'] = pose.float().cpu().numpy()[0]
        if grid is not None:
            result['grid_probs'] = torch.softmax(grid, dim=1).cpu().numpy()[0]
        return result


def create_multi_source_location_model(base_model_name, num_sources=1, fusion_method='concat',
                                       num_classes=8, output_mode='class', pose_dim=4,
                                       pretrained=True, input_size=None, num_grid_classes=0):
    """位置推論ラッパーモデルのファクトリ関数"""
    return MultiSourceLocationModel(
        base_model_name=base_model_name, num_sources=num_sources, fusion_method=fusion_method,
        num_classes=num_classes, output_mode=output_mode, pose_dim=pose_dim,
        pretrained=pretrained, input_size=input_size, num_grid_classes=num_grid_classes)
