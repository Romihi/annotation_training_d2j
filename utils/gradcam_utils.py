"""
GradCAM (Gradient-weighted Class Activation Mapping) ユーティリティ
自動運転モデルの判断根拠を可視化するための機能

pytorch-grad-camライブラリを使用

ViT系モデル対応:
- MobileViT, Swin Transformer, EfficientFormer等のTransformerベースモデルに対応
- 適切なターゲットレイヤーを自動検出
- reshape_transformを使用してAttentionマップを空間的に再構成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple, Dict, Any, List, Callable

# pytorch-grad-camのインポートを試みる
GRADCAM_AVAILABLE = False
SCORECAM_AVAILABLE = False
try:
    from pytorch_grad_cam import GradCAM as PytorchGradCAM
    from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM, LayerCAM, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
    SCORECAM_AVAILABLE = True
except ImportError:
    try:
        # ScoreCAMがない古いバージョンの場合
        from pytorch_grad_cam import GradCAM as PytorchGradCAM
        from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM, LayerCAM
        from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        GRADCAM_AVAILABLE = True
    except ImportError:
        pass


# ViT系モデルの識別用キーワード
VIT_MODEL_KEYWORDS = [
    'vit', 'swin', 'mobilevit', 'efficientformer', 'deit', 'beit',
    'crossvit', 'levit', 'poolformer', 'pvt', 'tnt', 'twins'
]

# ハイブリッドモデル（Conv + Attention）のキーワード
HYBRID_MODEL_KEYWORDS = [
    'edgenext', 'convnext', 'coatnet', 'maxvit', 'mobilevitv2'
]


def is_vit_model(model) -> bool:
    """モデルがViT系（Transformerベース）かどうかを判定"""
    model_name = getattr(model, 'name', '').lower()

    # モデル名でチェック
    for keyword in VIT_MODEL_KEYWORDS:
        if keyword in model_name:
            return True

    # TIMMベースモデルの場合、timm_model_nameもチェック
    if hasattr(model, 'timm_model_name'):
        timm_name = model.timm_model_name.lower()
        for keyword in VIT_MODEL_KEYWORDS:
            if keyword in timm_name:
                return True

    # 構造的にTransformerブロックがあるかチェック
    if hasattr(model, 'base_model'):
        base = model.base_model
        # blocksやlayersという名前のTransformerブロックを持つか
        if hasattr(base, 'blocks') or hasattr(base, 'layers'):
            # さらにMultiheadAttentionを含むかチェック
            for module in base.modules():
                if isinstance(module, nn.MultiheadAttention):
                    return True
                # timm特有のAttentionモジュール名
                module_name = module.__class__.__name__.lower()
                if 'attention' in module_name or 'msa' in module_name:
                    return True

    return False


def is_hybrid_model(model) -> bool:
    """モデルがハイブリッド（Conv + Attention）かどうかを判定"""
    model_name = getattr(model, 'name', '').lower()

    for keyword in HYBRID_MODEL_KEYWORDS:
        if keyword in model_name:
            return True

    if hasattr(model, 'timm_model_name'):
        timm_name = model.timm_model_name.lower()
        for keyword in HYBRID_MODEL_KEYWORDS:
            if keyword in timm_name:
                return True

    return False


def get_model_architecture_type(model) -> str:
    """モデルのアーキテクチャタイプを判定"""
    if is_vit_model(model):
        return 'vit'
    elif is_hybrid_model(model):
        return 'hybrid'
    else:
        return 'cnn'


def create_reshape_transform(model, height: int = 14, width: int = 14) -> Optional[Callable]:
    """
    ViT系モデル用のreshape_transform関数を作成

    Transformerの出力は (batch, num_tokens, channels) 形式なので、
    これを (batch, channels, height, width) に変換する必要がある

    Args:
        model: 対象モデル
        height: 出力の高さ（パッチ数）
        width: 出力の幅（パッチ数）

    Returns:
        reshape_transform関数、またはCNNモデルの場合はNone
    """
    arch_type = get_model_architecture_type(model)

    if arch_type == 'cnn':
        return None

    def reshape_transform_vit(tensor):
        """
        ViT用のreshape transform
        入力: (batch, num_tokens, channels) - CLSトークンを含む場合あり
        出力: (batch, channels, height, width)
        """
        if len(tensor.shape) == 4:
            # 既に4D（CNNライクな出力）の場合はそのまま返す
            return tensor

        if len(tensor.shape) == 3:
            batch, num_tokens, channels = tensor.shape

            # CLSトークンがある場合（最初のトークン）は除外
            # 一般的なViTは num_tokens = (H/patch) * (W/patch) + 1
            expected_spatial_tokens = height * width

            if num_tokens == expected_spatial_tokens + 1:
                # CLSトークンを除外
                tensor = tensor[:, 1:, :]
            elif num_tokens > expected_spatial_tokens:
                # その他の特殊トークンも除外
                tensor = tensor[:, -expected_spatial_tokens:, :]

            # (batch, H*W, C) -> (batch, C, H, W)
            tensor = tensor.permute(0, 2, 1)
            tensor = tensor.reshape(batch, channels, height, width)

        return tensor

    def reshape_transform_swin(tensor):
        """
        Swin Transformer用のreshape transform
        Swinは階層的な構造を持ち、各ステージで異なる解像度
        """
        if len(tensor.shape) == 4:
            return tensor

        if len(tensor.shape) == 3:
            batch, num_tokens, channels = tensor.shape

            # 空間サイズを推定（正方形と仮定）
            spatial_size = int(np.sqrt(num_tokens))
            if spatial_size * spatial_size != num_tokens:
                # 正方形でない場合、与えられたheight, widthを使用
                spatial_size = height

            tensor = tensor.permute(0, 2, 1)
            tensor = tensor.reshape(batch, channels, spatial_size, spatial_size)

        return tensor

    # モデルタイプに応じて適切なtransformを選択
    model_name = getattr(model, 'name', '').lower()
    if hasattr(model, 'timm_model_name'):
        model_name = model.timm_model_name.lower()

    if 'swin' in model_name:
        return reshape_transform_swin
    else:
        return reshape_transform_vit


class RegressionOutputTarget:
    """
    回帰モデルの特定出力インデックスをターゲットにするクラス

    Args:
        output_index: 対象の出力インデックス (0=angle, 1=throttle, 2=speed)
        direction: 可視化する勾配の方向
            - 'positive': 出力を増加させる方向の寄与（デフォルト）
                         angle: 右に切る根拠、throttle: 加速の根拠
            - 'negative': 出力を減少させる方向の寄与
                         angle: 左に切る根拠、throttle: 減速の根拠
            - 'absolute': 正負両方の寄与（絶対値）
    """
    def __init__(self, output_index, direction='positive'):
        self.output_index = output_index
        self.direction = direction

    def __call__(self, model_output):
        # model_output: (batch, num_outputs) または (num_outputs,)
        # pytorch-grad-camは内部でバッチ次元を削除する場合がある
        if len(model_output.shape) == 1:
            # 1次元の場合（バッチ次元なし）
            value = model_output[self.output_index]
        else:
            # 2次元の場合（バッチ次元あり）
            value = model_output[:, self.output_index]

        # 方向に応じて値を変換
        if self.direction == 'negative':
            # 負の寄与を可視化するため、出力を反転
            # これにより「出力を減少させる」特徴が正の勾配として現れる
            return -value
        elif self.direction == 'absolute':
            # 正負両方の寄与を見るため絶対値を使用
            # 注意: これは近似的な可視化であり、正確ではない
            import torch
            return torch.abs(value) if hasattr(value, 'abs') else abs(value)
        else:
            # 'positive': デフォルト、そのまま返す
            return value


class GradCAM:
    """
    GradCAMによるモデルの注目領域可視化

    pytorch-grad-camライブラリを使用した実装
    回帰モデル（自動運転モデル）に対応

    ViT系モデル対応:
    - Transformerベースモデルの適切なターゲットレイヤーを自動検出
    - reshape_transformを適用してAttentionマップを空間的に再構成
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
        self._arch_type = get_model_architecture_type(model)
        self._reshape_transform = None

    def _find_target_layers_cnn(self, base_model=None):
        """CNNモデルの最後の畳み込み層を検出"""
        target_layer = None
        search_model = base_model if base_model else self.model

        for name, module in search_model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module

        return target_layer

    def _find_target_layers_vit(self, base_model):
        """
        ViT系モデルの適切なターゲットレイヤーを検出

        ViTでは最後のTransformerブロックのnorm層またはFFN入力をターゲットにする
        これにより画像パッチに対応したトークンの情報を取得できる
        """
        model_name = getattr(self.model, 'name', '').lower()
        if hasattr(self.model, 'timm_model_name'):
            model_name = self.model.timm_model_name.lower()

        target_layer = None

        # MobileViT系
        if 'mobilevit' in model_name:
            # MobileViTは最後のConv層を使う（ハイブリッド構造）
            # stages -> 最後のMobileViTBlockのconv_proj
            if hasattr(base_model, 'stages'):
                for stage in reversed(list(base_model.stages)):
                    for name, module in stage.named_modules():
                        if isinstance(module, nn.Conv2d):
                            target_layer = module
                            break
                    if target_layer:
                        break

            # フォールバック: 最後のConv2d
            if target_layer is None:
                target_layer = self._find_target_layers_cnn(base_model)

        # Swin Transformer系
        elif 'swin' in model_name:
            # Swinは最後のステージのnorm層をターゲットに
            if hasattr(base_model, 'layers'):
                # swin_transformer構造: layers[-1].blocks[-1].norm2
                last_layer = base_model.layers[-1]
                if hasattr(last_layer, 'blocks') and len(last_layer.blocks) > 0:
                    last_block = last_layer.blocks[-1]
                    if hasattr(last_block, 'norm2'):
                        target_layer = last_block.norm2
                    elif hasattr(last_block, 'norm1'):
                        target_layer = last_block.norm1

            # 代替: norm属性を探す
            if target_layer is None and hasattr(base_model, 'norm'):
                target_layer = base_model.norm

        # EfficientFormer系
        elif 'efficientformer' in model_name:
            # EfficientFormerはstages構造を持つ
            if hasattr(base_model, 'stages'):
                # 最後のステージから探索
                for stage_idx in range(len(base_model.stages) - 1, -1, -1):
                    stage = base_model.stages[stage_idx]
                    # Meta4D または Meta3D ブロックの最後
                    for name, module in reversed(list(stage.named_modules())):
                        if hasattr(module, 'norm') and isinstance(module.norm, nn.LayerNorm):
                            target_layer = module.norm
                            break
                        elif isinstance(module, nn.LayerNorm):
                            target_layer = module
                            break
                    if target_layer:
                        break

            # フォールバック
            if target_layer is None:
                target_layer = self._find_target_layers_cnn(base_model)

        # 一般的なViT (DeiT, BEiT等)
        else:
            # blocks[-1].norm2 または blocks[-1].norm1 をターゲットに
            if hasattr(base_model, 'blocks'):
                if len(base_model.blocks) > 0:
                    last_block = base_model.blocks[-1]
                    # FFN入力のnorm層 (記事の推奨)
                    if hasattr(last_block, 'norm2'):
                        target_layer = last_block.norm2
                    elif hasattr(last_block, 'norm1'):
                        target_layer = last_block.norm1
                    elif hasattr(last_block, 'ln_2'):
                        target_layer = last_block.ln_2
                    elif hasattr(last_block, 'ln_1'):
                        target_layer = last_block.ln_1

            # normで探索
            if target_layer is None and hasattr(base_model, 'norm'):
                target_layer = base_model.norm

        return target_layer

    def _find_target_layers_hybrid(self, base_model):
        """
        ハイブリッドモデル（Conv + Attention）の適切なターゲットレイヤーを検出

        EdgeNeXtやConvNeXtなどは畳み込み主体だが一部Attentionを含む
        """
        model_name = getattr(self.model, 'name', '').lower()
        if hasattr(self.model, 'timm_model_name'):
            model_name = self.model.timm_model_name.lower()

        target_layer = None

        # EdgeNeXt
        if 'edgenext' in model_name:
            # EdgeNeXtは最後のステージの畳み込み層が適切
            if hasattr(base_model, 'stages'):
                last_stage = base_model.stages[-1]
                for name, module in last_stage.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
            # フォールバック
            if target_layer is None:
                target_layer = self._find_target_layers_cnn(base_model)

        # ConvNeXt
        elif 'convnext' in model_name:
            # ConvNeXtは純粋なCNN構造なのでConv2dを使用
            if hasattr(base_model, 'stages'):
                last_stage = base_model.stages[-1]
                for name, module in last_stage.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
            if target_layer is None:
                target_layer = self._find_target_layers_cnn(base_model)

        # MobileViTv2
        elif 'mobilevitv2' in model_name:
            # MobileViTv2も最後の畳み込み層
            target_layer = self._find_target_layers_cnn(base_model)

        else:
            # その他のハイブリッドモデルはCNN方式でフォールバック
            target_layer = self._find_target_layers_cnn(base_model)

        return target_layer

    def _find_target_layers(self):
        """モデルのアーキテクチャに応じた適切なターゲットレイヤーを自動検出"""
        target_layer = None

        # TIMMベースのモデル
        if hasattr(self.model, 'base_model'):
            base = self.model.base_model

            if self._arch_type == 'vit':
                target_layer = self._find_target_layers_vit(base)
            elif self._arch_type == 'hybrid':
                target_layer = self._find_target_layers_hybrid(base)
            else:
                target_layer = self._find_target_layers_cnn(base)

        # DonkeyModelの場合
        elif hasattr(self.model, 'features'):
            target_layer = self._find_target_layers_cnn(self.model.features)

        # その他のモデル
        else:
            target_layer = self._find_target_layers_cnn()

        if target_layer is None:
            raise ValueError(
                f"ターゲットレイヤーが見つかりません "
                f"(arch_type: {self._arch_type})"
            )

        return [target_layer]

    def _estimate_spatial_size(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        """入力テンソルからViTの空間サイズ（パッチ数）を推定"""
        # 一般的なViTのパッチサイズは16x16
        _, _, h, w = input_tensor.shape
        patch_size = 16  # デフォルト

        # モデル名からパッチサイズを推定
        model_name = getattr(self.model, 'name', '').lower()
        if hasattr(self.model, 'timm_model_name'):
            model_name = self.model.timm_model_name.lower()

        if 'patch16' in model_name:
            patch_size = 16
        elif 'patch14' in model_name:
            patch_size = 14
        elif 'patch8' in model_name:
            patch_size = 8
        elif 'patch4' in model_name:
            patch_size = 4

        # Swinの場合はwindow sizeも考慮
        if 'swin' in model_name:
            # Swinは階層的なので最後のステージは縮小されている
            # 224入力、patch4の場合: 56 -> 28 -> 14 -> 7
            patch_size = 32  # 最終ステージの実効パッチサイズ

        spatial_h = h // patch_size
        spatial_w = w // patch_size

        return spatial_h, spatial_w

    def _get_cam_instance(self, method='gradcam', input_tensor: torch.Tensor = None):
        """CAMインスタンスを取得"""
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
        elif method == 'scorecam':
            if not SCORECAM_AVAILABLE:
                print("ScoreCAMは利用できません。GradCAMにフォールバックします。")
                cam_class = PytorchGradCAM
            else:
                cam_class = ScoreCAM
        else:
            cam_class = PytorchGradCAM

        # ViT系モデルの場合、reshape_transformを設定
        reshape_transform = None
        if self._arch_type == 'vit' and input_tensor is not None:
            spatial_h, spatial_w = self._estimate_spatial_size(input_tensor)
            reshape_transform = create_reshape_transform(
                self.model, height=spatial_h, width=spatial_w
            )
            self._reshape_transform = reshape_transform

        return cam_class(
            model=self.model,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        )

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_output_index: int = 0,
        normalize: bool = True,
        method: str = 'gradcam',
        direction: str = 'positive'
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
            method: 使用するCAM手法 ('gradcam', 'gradcam++', 'eigencam', 'layercam', 'scorecam')
            direction: 可視化する勾配の方向
                - 'positive': 出力を増加させる方向（右に切る/加速の根拠）
                - 'negative': 出力を減少させる方向（左に切る/減速の根拠）
                - 'absolute': 正負両方の寄与

        Returns:
            ヒートマップ (H, W) 形式のnumpy配列
        """
        if not GRADCAM_AVAILABLE:
            # フォールバック：簡易CAM
            return self._fallback_cam(input_tensor, target_output_index, normalize, direction)

        try:
            # CAMインスタンスを取得（ViT用にinput_tensorも渡す）
            cam = self._get_cam_instance(method, input_tensor=input_tensor)

            # 回帰モデル用のターゲット（方向を指定）
            targets = [RegressionOutputTarget(target_output_index, direction=direction)]

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
            import traceback
            traceback.print_exc()
            return self._fallback_cam(input_tensor, target_output_index, normalize, direction)

    def _fallback_cam(
        self,
        input_tensor: torch.Tensor,
        target_output_index: int = 0,
        normalize: bool = True,
        direction: str = 'positive'
    ) -> np.ndarray:
        """
        フォールバック用の簡易CAM（勾配を使わない）
        ViT系モデルにも対応

        注意: フォールバックCAMは勾配を使用しないため、
        direction引数は効果がありません（アクティベーションベースの可視化）
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

                # アクティベーションの形状を確認
                if len(activations.shape) == 4:
                    # CNN: (batch, channels, h, w)
                    batch, channels, h, w = activations.shape
                    weights = activations.mean(dim=(2, 3))

                    cam = torch.zeros(h, w, device=device)
                    for i in range(channels):
                        cam += weights[0, i] * activations[0, i]

                elif len(activations.shape) == 3:
                    # ViT: (batch, num_tokens, channels)
                    batch, num_tokens, channels = activations.shape

                    # 空間サイズを推定
                    spatial_h, spatial_w = self._estimate_spatial_size(input_tensor)

                    # CLSトークンがある場合は除外
                    expected_spatial_tokens = spatial_h * spatial_w
                    if num_tokens == expected_spatial_tokens + 1:
                        # CLSトークンを除外
                        spatial_tokens = activations[:, 1:, :]
                    elif num_tokens > expected_spatial_tokens:
                        spatial_tokens = activations[:, -expected_spatial_tokens:, :]
                    else:
                        spatial_tokens = activations

                    # (batch, H*W, C) -> (batch, C, H, W)
                    spatial_tokens = spatial_tokens.permute(0, 2, 1)
                    try:
                        spatial_tokens = spatial_tokens.reshape(batch, channels, spatial_h, spatial_w)
                    except RuntimeError:
                        # サイズが合わない場合は正方形と仮定
                        actual_tokens = spatial_tokens.shape[2]
                        side = int(np.sqrt(actual_tokens))
                        spatial_tokens = spatial_tokens[:, :, :side*side].reshape(batch, channels, side, side)
                        spatial_h, spatial_w = side, side

                    # 重み付きCAM
                    weights = spatial_tokens.mean(dim=(2, 3))
                    cam = torch.zeros(spatial_h, spatial_w, device=device)
                    for i in range(channels):
                        cam += weights[0, i] * spatial_tokens[0, i]

                else:
                    raise RuntimeError(f"サポートされていないアクティベーション形状: {activations.shape}")

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


def apply_bidirectional_colormap(
    positive_heatmap: np.ndarray,
    negative_heatmap: np.ndarray
) -> np.ndarray:
    """
    正負両方向のヒートマップを赤青カラーマップで合成

    Args:
        positive_heatmap: 正の寄与のヒートマップ (H, W), 値は0-1
        negative_heatmap: 負の寄与のヒートマップ (H, W), 値は0-1

    Returns:
        合成されたカラー画像 (H, W, 3) RGB形式
        - 赤: 正の寄与（出力を増加させる方向）
        - 青: 負の寄与（出力を減少させる方向）
        - 紫: 両方の寄与が重なる部分
    """
    h, w = positive_heatmap.shape

    # RGB画像を作成
    result = np.zeros((h, w, 3), dtype=np.float32)

    # 赤チャンネル: 正の寄与
    result[:, :, 0] = positive_heatmap

    # 青チャンネル: 負の寄与
    result[:, :, 2] = negative_heatmap

    # 緑チャンネル: 両方が重なる部分を少し明るく（オプション）
    # result[:, :, 1] = np.minimum(positive_heatmap, negative_heatmap) * 0.3

    # 0-255にスケール
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return result


def generate_bidirectional_cam(
    gradcam_instance,
    input_tensor,
    target_output_index: int = 0,
    method: str = 'gradcam'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    正負両方向のGradCAMヒートマップを生成

    Args:
        gradcam_instance: GradCAMインスタンス
        input_tensor: 入力画像テンソル (1, C, H, W)
        target_output_index: 注目する出力インデックス
        method: 使用するCAM手法

    Returns:
        (positive_heatmap, negative_heatmap, combined_rgb) のタプル
        - positive_heatmap: 正の寄与 (H, W)
        - negative_heatmap: 負の寄与 (H, W)
        - combined_rgb: 赤青合成画像 (H, W, 3) RGB形式
    """
    # 正の寄与（出力を増加させる方向）
    positive_heatmap = gradcam_instance.generate_cam(
        input_tensor,
        target_output_index=target_output_index,
        method=method,
        direction='positive'
    )

    # 負の寄与（出力を減少させる方向）
    negative_heatmap = gradcam_instance.generate_cam(
        input_tensor,
        target_output_index=target_output_index,
        method=method,
        direction='negative'
    )

    # 赤青カラーマップで合成
    combined_rgb = apply_bidirectional_colormap(positive_heatmap, negative_heatmap)

    return positive_heatmap, negative_heatmap, combined_rgb


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
