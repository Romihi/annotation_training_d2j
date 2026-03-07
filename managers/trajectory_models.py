"""
時系列モデル定義

共通コンポーネント:
  ImageEncoder: MobileNetV3-Small → 128次元特徴量
  EgoStateEncoder: 5次元状態 → 32次元特徴量

アーキテクチャ選択:
  GRU          — nn.GRU ベース
  TCN          — Dilated Causal Conv1D スタック
  CausalCNN    — 軽量 Causal Conv1D (TinyLidarNet風)
"""

import torch
import torch.nn as nn
import timm


# =============================================================================
# 共通エンコーダ
# =============================================================================

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
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, feat_dim)
        """
        features = self.backbone.forward_features(x)
        features = self.pool(features).flatten(1)
        return self.relu(self.fc(features))


class EgoStateEncoder(nn.Module):
    """自車状態エンコーダ [steering, throttle, vx, vy, omega] → 特徴量"""

    def __init__(self, ego_dim=5, feat_dim=32):
        super().__init__()
        self.fc = nn.Linear(ego_dim, feat_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.fc(x))


# =============================================================================
# 共通ベースクラス
# =============================================================================

class BaseTrajectoryModel(nn.Module):
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

        # 共通エンコーダ
        self.image_encoder = ImageEncoder(img_feat_dim)
        self.ego_encoder = EgoStateEncoder(ego_dim, ego_feat_dim)

        # Fusion: concat → Linear → ReLU
        fusion_input_dim = img_feat_dim * num_image_sources + ego_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # サブクラスで時系列バックボーンを構築
        self._build_temporal(hidden_dim, dropout)

        # 共通 Trajectory Head
        self.head_dropout = nn.Dropout(dropout)
        self.trajectory_head = nn.Linear(hidden_dim, pred_horizon * 2)

    def _build_temporal(self, hidden_dim, dropout):
        """サブクラスで時系列レイヤーを構築する"""
        raise NotImplementedError

    def _forward_temporal(self, fused):
        """サブクラスで時系列処理を実行する

        Args:
            fused: (B, T, hidden_dim)
        Returns:
            (B, hidden_dim)  — 最終的な集約表現
        """
        raise NotImplementedError

    def forward(self, images, ego_states):
        """
        Args:
            images:     (B, T, num_sources, 3, H, W)
            ego_states: (B, T, 5)
        Returns:
            trajectories: (B, pred_horizon, 2)
        """
        B, T, S, C, H, W = images.shape

        # 画像エンコード: (B*T*S, 3, H, W) → (B, T, S*img_feat_dim)
        images_flat = images.reshape(B * T * S, C, H, W)
        img_features = self.image_encoder(images_flat)
        img_features = img_features.reshape(B, T, S * self.img_feat_dim)

        # Ego エンコード: (B*T, 5) → (B, T, ego_feat_dim)
        ego_flat = ego_states.reshape(B * T, -1)
        ego_features = self.ego_encoder(ego_flat).reshape(B, T, -1)

        # Fusion: (B, T, hidden_dim)
        fused = torch.cat([img_features, ego_features], dim=-1)
        fused = self.fusion(fused.reshape(B * T, -1)).reshape(B, T, -1)

        # 時系列バックボーン: (B, T, hidden_dim) → (B, hidden_dim)
        temporal_out = self._forward_temporal(fused)

        # Trajectory Head
        out = self.head_dropout(temporal_out)
        trajectory = self.trajectory_head(out)
        trajectory = torch.tanh(trajectory)
        return trajectory.reshape(B, self.pred_horizon, 2)


# =============================================================================
# GRU バックボーン
# =============================================================================

class GRUTrajectoryModel(BaseTrajectoryModel):
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


# =============================================================================
# TCN バックボーン
# =============================================================================

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
        """因果畳み込みのために末尾をトリミング"""
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


class TCNTrajectoryModel(BaseTrajectoryModel):
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
        # TCN出力次元が hidden_dim と異なる場合のプロジェクション
        tcn_out_dim = self._tcn_channels[-1]
        self.tcn_proj = nn.Linear(tcn_out_dim, hidden_dim) if tcn_out_dim != hidden_dim else nn.Identity()

    def _forward_temporal(self, fused):
        # (B, T, hidden_dim) → (B, hidden_dim, T) for Conv1d
        x = fused.transpose(1, 2)
        x = self.tcn(x)
        # 最終タイムステップを使用: (B, C, T) → (B, C)
        x = x[:, :, -1]
        return self.tcn_proj(x)


# =============================================================================
# CausalCNN バックボーン (TinyLidarNet風)
# =============================================================================

class CausalCNNTrajectoryModel(BaseTrajectoryModel):
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
            padding = self._kernel_size - 1  # causal padding
            layers.extend([
                nn.Conv1d(channels[i], channels[i + 1], self._kernel_size, padding=padding),
                _CausalChomp(padding),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        self.cnn = nn.Sequential(*layers)
        # 出力次元プロジェクション
        cnn_out_dim = self._cnn_channels[-1]
        self.cnn_proj = nn.Linear(cnn_out_dim, hidden_dim) if cnn_out_dim != hidden_dim else nn.Identity()

    def _forward_temporal(self, fused):
        # (B, T, hidden_dim) → (B, hidden_dim, T) for Conv1d
        x = fused.transpose(1, 2)
        x = self.cnn(x)
        # 最終タイムステップ: (B, C, T) → (B, C)
        x = x[:, :, -1]
        return self.cnn_proj(x)


class _CausalChomp(nn.Module):
    """因果畳み込みのトリミング"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


# =============================================================================
# ファクトリ関数
# =============================================================================

TRAJECTORY_ARCHITECTURES = {
    "gru": GRUTrajectoryModel,
    "tcn": TCNTrajectoryModel,
    "causal_cnn": CausalCNNTrajectoryModel,
}


def create_trajectory_model(model_arch, num_image_sources, config):
    """アーキテクチャ名からモデルを生成する

    Args:
        model_arch: "gru" | "tcn" | "causal_cnn"
        num_image_sources: 画像ソース数
        config: dict — アーキテクチャ固有パラメータを含む

    Returns:
        BaseTrajectoryModel のサブクラスインスタンス
    """
    if model_arch not in TRAJECTORY_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {model_arch}. "
                         f"Available: {list(TRAJECTORY_ARCHITECTURES.keys())}")

    common = dict(
        num_image_sources=num_image_sources,
        hidden_dim=config.get('hidden_dim', 256),
        pred_horizon=config.get('pred_horizon', 10),
        dropout=config.get('dropout', 0.1),
    )

    if model_arch == "gru":
        return GRUTrajectoryModel(
            **common,
            num_layers=config.get('num_layers', 1),
        )
    elif model_arch == "tcn":
        return TCNTrajectoryModel(
            **common,
            tcn_channels=config.get('tcn_channels', [128, 128, 256]),
            kernel_size=config.get('kernel_size', 3),
        )
    elif model_arch == "causal_cnn":
        return CausalCNNTrajectoryModel(
            **common,
            cnn_channels=config.get('cnn_channels', [64, 128, 256]),
            kernel_size=config.get('kernel_size', 3),
        )
