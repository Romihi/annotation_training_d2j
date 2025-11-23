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
    get_model_input_size
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
        #result = result * 2 - 1
        
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
            transforms.ToTensor()
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


class YOLOv11nModel(TIMMBasedModel):
    """YOLOv11 Nano モデル"""
    def __init__(self, pretrained=True):
        super(YOLOv11nModel, self).__init__(
            name="yolo11n",
            timm_model_name="yolo11n",
            pretrained=pretrained
        )


class YOLOv11sModel(TIMMBasedModel):
    """YOLOv11 Small モデル"""
    def __init__(self, pretrained=True):
        super(YOLOv11sModel, self).__init__(
            name="yolo11s",
            timm_model_name="yolo11s",
            pretrained=pretrained
        )


class YOLOv11mModel(TIMMBasedModel):
    """YOLOv11 Medium モデル"""
    def __init__(self, pretrained=True):
        super(YOLOv11mModel, self).__init__(
            name="yolo11m",
            timm_model_name="yolo11m",
            pretrained=pretrained
        )


class YOLOv11lModel(TIMMBasedModel):
    """YOLOv11 Large モデル"""
    def __init__(self, pretrained=True):
        super(YOLOv11lModel, self).__init__(
            name="yolo11l",
            timm_model_name="yolo11l",
            pretrained=pretrained
        )


class YOLOv11xModel(TIMMBasedModel):
    """YOLOv11 Extra Large モデル"""
    def __init__(self, pretrained=True):
        super(YOLOv11xModel, self).__init__(
            name="yolo11x",
            timm_model_name="yolo11x",
            pretrained=pretrained
        )

class DonkeyModel(BaseModel):
    """Donkeycarで使用される標準的なモデル（カスタム実装）"""
    #def __init__(self, pretrained=False, input_size=(120, 160)):
    def __init__(self, pretrained=False, input_size=(224, 224)):
        super(DonkeyModel, self).__init__(name="donkeycar")
        
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
            transforms.ToTensor()
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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    
    # DonkeyModel系の場合、入力サイズを渡す
    if model_type in ["donkeycar", "donkey_fcn"] and input_size is not None:
        return model_class(pretrained=pretrained, input_size=input_size)
    elif model_type == "donkey_location" and input_size is not None:
        # DonkeyLocationModelの場合、num_classesも必要（デフォルト8）
        return model_class(num_classes=8, pretrained=pretrained, input_size=input_size)
    elif model_type == "donkey_waypoint" and input_size is not None:
        # DonkeyWaypointModelの場合、num_waypointsも必要（デフォルト4）
        return model_class(num_waypoints=4, pretrained=pretrained, input_size=input_size)
    elif model_type == "resnet18_waypoint":
        # ResNet18WaypointModelの場合、num_waypointsも必要（デフォルト4）
        return model_class(num_waypoints=4, pretrained=pretrained)
        return model_class(num_classes=8, pretrained=pretrained, input_size=input_size)
    
    # その他のモデルの場合は通常通り初期化
    return model_class(pretrained=pretrained)

def list_available_models():
    """利用可能な走行モデル一覧を返す（位置推論モデルは除く）"""
    # 位置推論モデルを除いたモデルのみを返す
    return [model for model in MODEL_REGISTRY.keys() if not model.endswith('_location')]

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

    
class AnnotationDataset(torch.utils.data.Dataset):
    """アノテーションデータのためのカスタムデータセット"""
    def __init__(self, image_paths, annotations, transform=None, cache_images=False):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        
    def __len__(self):
        return len(self.image_paths)
    
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
        
        # angle, throttleをターゲットとして使用
        annotation = self.annotations[idx]
        target = torch.tensor([annotation["angle"], annotation["throttle"]], dtype=torch.float)
        
        return img, target
