"""
GradCAM (Gradient-weighted Class Activation Mapping) ユーティリティ
自動運転モデルの判断根拠を可視化するための機能

pytorch-grad-camライブラリを使用
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple, Dict, Any, List

# pytorch-grad-camのインポートを試みる
GRADCAM_AVAILABLE = False
try:
    from pytorch_grad_cam import GradCAM as PytorchGradCAM
    from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM, LayerCAM
    from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    pass


class RegressionOutputTarget:
    """回帰モデルの特定出力インデックスをターゲットにするクラス"""
    def __init__(self, output_index):
        self.output_index = output_index

    def __call__(self, model_output):
        # model_output: (batch, num_outputs)
        return model_output[:, self.output_index]


class GradCAM:
    """
    GradCAMによるモデルの注目領域可視化

    pytorch-grad-camライブラリを使用した実装
    回帰モデル（自動運転モデル）に対応
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: PyTorchモデル
            target_layer: 可視化対象のレイヤー（Noneの場合は自動検出）
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self._cam_instance = None

    def _find_target_layers(self):
        """モデルの最後の畳み込み層を自動検出してリストで返す"""
        target_layer = None

        # TIMMベースのモデル
        if hasattr(self.model, 'base_model'):
            base = self.model.base_model
            # 最後の畳み込み層を探す
            for name, module in base.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module

        # DonkeyModelの場合
        elif hasattr(self.model, 'features'):
            for module in self.model.features.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module

        # その他のモデル
        else:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module

        if target_layer is None:
            raise ValueError("畳み込み層が見つかりません")

        return [target_layer]

    def _get_cam_instance(self, method='gradcam'):
        """CAMインスタンスを取得（キャッシュ）"""
        if not GRADCAM_AVAILABLE:
            raise ImportError(
                "pytorch-grad-cam がインストールされていません。\n"
                "pip install grad-cam でインストールしてください。"
            )

        target_layers = [self.target_layer] if self.target_layer else self._find_target_layers()

        # メソッドに応じたCAMクラスを選択
        if method == 'gradcam++':
            cam_class = GradCAMPlusPlus
        elif method == 'eigencam':
            cam_class = EigenCAM
        elif method == 'layercam':
            cam_class = LayerCAM
        else:
            cam_class = PytorchGradCAM

        return cam_class(model=self.model, target_layers=target_layers)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_output_index: int = 0,
        normalize: bool = True,
        method: str = 'gradcam'
    ) -> np.ndarray:
        """
        GradCAMヒートマップを生成

        Args:
            input_tensor: 入力画像テンソル (1, C, H, W)
            target_output_index: 注目する出力インデックス
                - 0: angle
                - 1: throttle
                - 2: speed (3出力以上の場合)
            normalize: ヒートマップを0-1に正規化するか
            method: 使用するCAM手法 ('gradcam', 'gradcam++', 'eigencam', 'layercam')

        Returns:
            ヒートマップ (H, W) 形式のnumpy配列
        """
        if not GRADCAM_AVAILABLE:
            # フォールバック：簡易CAM
            return self._fallback_cam(input_tensor, target_output_index, normalize)

        try:
            # CAMインスタンスを取得
            cam = self._get_cam_instance(method)

            # 回帰モデル用のターゲット
            targets = [RegressionOutputTarget(target_output_index)]

            # GradCAMを計算
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # バッチの最初の画像のCAMを取得
            heatmap = grayscale_cam[0, :]

            # 正規化（pytorch-grad-camは既に0-1に正規化されている）
            if normalize and heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            return heatmap

        except Exception as e:
            print(f"pytorch-grad-cam エラー: {e}, フォールバックCAMを使用")
            return self._fallback_cam(input_tensor, target_output_index, normalize)

    def _fallback_cam(
        self,
        input_tensor: torch.Tensor,
        target_output_index: int = 0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        フォールバック用の簡易CAM（勾配を使わない）
        """
        device = input_tensor.device
        self.model = self.model.to(device)
        self.model.eval()

        target_layers = [self.target_layer] if self.target_layer else self._find_target_layers()
        target_layer = target_layers[0]

        # アクティベーションを保存
        activations = None
        handle = None

        def save_activation(module, input, output):
            nonlocal activations
            activations = output.detach().clone()

        try:
            handle = target_layer.register_forward_hook(save_activation)

            with torch.no_grad():
                output = self.model(input_tensor)

                if activations is None:
                    raise RuntimeError("アクティベーションが取得できませんでした")

                batch, channels, h, w = activations.shape
                weights = activations.mean(dim=(2, 3))

                cam = torch.zeros(h, w, device=device)
                for i in range(channels):
                    cam += weights[0, i] * activations[0, i]

                cam = F.relu(cam)
                cam = cam.cpu().numpy()

                if normalize and cam.max() > 0:
                    cam = (cam - cam.min()) / (cam.max() - cam.min())

                return cam

        finally:
            if handle:
                handle.remove()

    def remove_hooks(self):
        """互換性のためのダミーメソッド"""
        pass

    def generate_multi_output_cam(
        self,
        input_tensor: torch.Tensor,
        output_indices: List[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        複数の出力に対するGradCAMを生成

        Args:
            input_tensor: 入力画像テンソル
            output_indices: 可視化する出力インデックスのリスト

        Returns:
            出力名をキー、ヒートマップを値とする辞書
        """
        output_names = ['angle', 'throttle', 'speed',
                       'future_5_angle', 'future_5_throttle', 'future_5_speed',
                       'future_10_angle', 'future_10_throttle', 'future_10_speed']

        if output_indices is None:
            # デフォルトはangle, throttleのみ
            output_indices = [0, 1]

        results = {}
        for idx in output_indices:
            if idx < len(output_names):
                name = output_names[idx]
            else:
                name = f'output_{idx}'

            try:
                cam = self.generate_cam(input_tensor.clone(), target_output_index=idx)
                results[name] = cam
            except Exception as e:
                print(f"GradCAM生成エラー (出力{idx}): {e}")
                continue

        return results


def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    ヒートマップにカラーマップを適用

    Args:
        heatmap: 正規化されたヒートマップ (H, W), 値は0-1
        colormap: OpenCVのカラーマップ定数

    Returns:
        カラーマップが適用された画像 (H, W, 3) BGR形式
    """
    # 0-255にスケール
    heatmap_uint8 = np.uint8(255 * heatmap)

    # カラーマップを適用
    colored = cv2.applyColorMap(heatmap_uint8, colormap)

    return colored


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    元画像にヒートマップをオーバーレイ

    Args:
        image: 元画像 (H, W, 3) RGB形式
        heatmap: 正規化されたヒートマップ (H, W)
        alpha: ヒートマップの透明度 (0-1)
        colormap: OpenCVのカラーマップ定数

    Returns:
        オーバーレイ画像 (H, W, 3) RGB形式
    """
    # ヒートマップを画像サイズにリサイズ
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # カラーマップを適用 (BGR)
    heatmap_colored = apply_colormap(heatmap_resized, colormap)

    # BGRからRGBに変換
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # オーバーレイ
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def generate_gradcam_visualization(
    model,
    image_path: str,
    transform,
    target_output: str = 'angle',
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    GradCAM可視化の完全なパイプライン

    Args:
        model: PyTorchモデル
        image_path: 画像ファイルパス
        transform: 前処理変換
        target_output: 注目する出力 ('angle', 'throttle', 'speed')
        alpha: ヒートマップの透明度
        colormap: カラーマップ
        device: 使用するデバイス

    Returns:
        (オーバーレイ画像, ヒートマップ, メタ情報)のタプル
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 出力インデックスのマッピング
    output_map = {
        'angle': 0,
        'throttle': 1,
        'speed': 2,
        'future_5_angle': 3,
        'future_5_throttle': 4,
        'future_5_speed': 5,
        'future_10_angle': 6,
        'future_10_throttle': 7,
        'future_10_speed': 8
    }

    target_idx = output_map.get(target_output, 0)

    # 画像を読み込み
    original_image = Image.open(image_path).convert('RGB')
    original_np = np.array(original_image)

    # 前処理
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # GradCAMインスタンス作成
    gradcam = GradCAM(model)

    # ヒートマップ生成
    heatmap = gradcam.generate_cam(input_tensor, target_output_index=target_idx)

    # オーバーレイ画像作成
    overlay = overlay_heatmap(original_np, heatmap, alpha=alpha, colormap=colormap)

    # 推論結果も取得
    with torch.no_grad():
        output = model(input_tensor)
        output_values = output[0].cpu().numpy()

    # メタ情報
    meta = {
        'target_output': target_output,
        'target_index': target_idx,
        'output_values': output_values,
        'heatmap_min': float(heatmap.min()),
        'heatmap_max': float(heatmap.max()),
        'heatmap_mean': float(heatmap.mean())
    }

    return overlay, heatmap, meta


def generate_comparison_visualization(
    model,
    image_path: str,
    transform,
    outputs: List[str] = None,
    alpha: float = 0.5,
    device: torch.device = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    複数出力の比較可視化を生成

    Args:
        model: PyTorchモデル
        image_path: 画像ファイルパス
        transform: 前処理変換
        outputs: 比較する出力リスト (デフォルト: ['angle', 'throttle'])
        alpha: ヒートマップの透明度
        device: 使用するデバイス

    Returns:
        出力名をキー、(オーバーレイ, ヒートマップ)タプルを値とする辞書
    """
    if outputs is None:
        outputs = ['angle', 'throttle']

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}

    for output_name in outputs:
        try:
            overlay, heatmap, _ = generate_gradcam_visualization(
                model=model,
                image_path=image_path,
                transform=transform,
                target_output=output_name,
                alpha=alpha,
                device=device
            )
            results[output_name] = (overlay, heatmap)
        except Exception as e:
            print(f"GradCAM生成エラー ({output_name}): {e}")
            continue

    return results


def create_side_by_side_visualization(
    original_image: np.ndarray,
    overlays: Dict[str, np.ndarray],
    labels: Dict[str, str] = None
) -> np.ndarray:
    """
    元画像と複数のGradCAMオーバーレイを横並びで表示

    Args:
        original_image: 元画像 (H, W, 3)
        overlays: 出力名をキー、オーバーレイ画像を値とする辞書
        labels: 出力名をキー、表示ラベルを値とする辞書

    Returns:
        横並び画像 (H, W*N, 3)
    """
    if labels is None:
        labels = {
            'angle': 'Angle',
            'throttle': 'Throttle',
            'speed': 'Speed'
        }

    images = [original_image]

    for name, overlay in overlays.items():
        # ラベルを追加
        labeled = overlay.copy()
        label_text = labels.get(name, name)

        # テキストを描画
        cv2.putText(
            labeled,
            label_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        images.append(labeled)

    # 横に連結
    return np.hstack(images)
