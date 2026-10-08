"""
LiDAR Policy（2D LiDAR 点群 → angle/throttle の End-to-End 運転モデル）の
モデル定義と前処理。設計は dev/SPEC_lidar_policy.md（段階1: 1D-CNN + FC ヘッド）。

入力は **生の距離スキャン [mm]** を K フレーム積層した (B, K, N_raw) と
自車状態 (B, S)。前処理（間引き・正規化・無効値マスク）は ScanPreprocessor と
して**モデル内部**に持ち、学習・GUI 推論・ONNX（実機 planner）で同一コードが
走るようにする（planner 側に前処理を複製しない）。

出力:
    action : (B, 2) = [angle(tanh, togikaidrive 規約 +1=右), throttle(sigmoid)]
    traj   : (B, H, 2) 補助ヘッド（ego 座標 +X前方/+Y左 [m]）。cfg.use_traj のとき
"""

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# プリセット: バックボーンの段構成 [(channels, kernel, stride, dilation), ...]
# base: ResNet1D-lite（≈0.8M params）/ tiny: TinyLidarNet 相当の Conv1d 5段
LIDAR_POLICY_PRESETS = {
    "base": {
        "stem": (32, 7, 2),
        "stages": [(32, 5, 1, 1), (64, 5, 2, 2), (128, 3, 2, 4), (256, 3, 2, 8)],
        "blocks_per_stage": 2,
        "hidden_dim": 256,
    },
    "tiny": {
        "stem": (24, 11, 4),
        "stages": [(36, 7, 4, 1), (48, 5, 2, 1), (64, 3, 1, 1), (64, 3, 1, 1)],
        "blocks_per_stage": 1,
        "hidden_dim": 128,
    },
}

# 間引きビン数の選択肢（1081 点スキャン基準の目安: 全点/半分/1/4/RL互換）
LIDAR_POLICY_BIN_CHOICES = (1081, 541, 271, 108)


@dataclass
class LidarPolicyConfig:
    preset: str = "base"
    num_beams_raw: int = 1081        # 記録スキャンの点数（manifest lidar_data_points）
    num_bins: int = 541              # 前処理後のビン数
    downsample_mode: str = "minpool"  # "minpool"（細い障害物を残す）/ "index"（RL互換の等間隔抽出）
    stack_frames: int = 3            # 積層する過去フレーム数 K（1=現フレームのみ）
    use_valid_ch: bool = True        # 無効ビーム(0/範囲外)マスクを ch として追加
    max_range_mm: float = 10000.0    # 正規化上限（実機 config.LIDAR_MAX_DISTANCE と合わせる）
    min_range_mm: float = 50.0       # これ未満は無効扱い
    angle_start_deg: float = -135.0  # manifest の lidar_angle_start/end（左右反転拡張の対称性用）
    angle_end_deg: float = 135.0
    clockwise: bool = False
    state_dim: int = 1               # [speed/max_speed]
    max_speed: float = 5.0           # 速度正規化（実機 config.F1TENTH_MAX_SPEED と合わせる）
    hidden_dim: int = 256
    dropout: float = 0.1
    use_traj: bool = True            # 補助軌道ヘッド
    horizon: int = 10                # 軌道点数 H
    dt: float = 0.1                  # 軌道の時間刻み [s]（horizon*dt 秒先まで）
    steer_sign: int = 1              # +1: 出力 angle は togikaidrive 規約（+1=右）

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def in_channels(self):
        return self.stack_frames * (2 if self.use_valid_ch else 1)


class ScanPreprocessor(nn.Module):
    """生スキャン [mm] (B, K, N_raw) → (B, K*C, num_bins) 正規化テンソル。

    - 無効値: r < min_range または r > max_range（0 は無効の記録規約）
    - minpool: 無効を +inf 扱いで区間最小をとる（細い障害物・近い壁を優先）。
      区間が全て無効なら無効
    - index : planner plan=="rl" と同じ linspace 等間隔抽出（RL 互換）
    - range_norm = clip(r, 0, max) / max、無効ビンは 1.0（=最遠）で埋める
    - valid ch  = 有効 1 / 無効 0
    ONNX でも同一グラフとして書き出される（MaxPool1d / index_select のみ）。
    """

    def __init__(self, cfg: LidarPolicyConfig):
        super().__init__()
        self.cfg = cfg
        n_raw, n_bins = cfg.num_beams_raw, cfg.num_bins
        if cfg.downsample_mode == "index":
            idx = np.linspace(0, n_raw - 1, n_bins).round().astype(np.int64)
            self.register_buffer("index", torch.from_numpy(idx), persistent=False)
            self.kernel = 1
            self.pad = 0
        else:
            self.kernel = int(math.ceil(n_raw / n_bins))
            self.pad = self.kernel * n_bins - n_raw
            self.index = None

    def forward(self, scan_mm: torch.Tensor):
        cfg = self.cfg
        x = scan_mm.float()
        valid = (x >= cfg.min_range_mm) & (x <= cfg.max_range_mm)
        # 無効は +max に置換（minpool で選ばれない / 正規化後 1.0 = 最遠）
        big = torch.full_like(x, cfg.max_range_mm * 4.0)
        x = torch.where(valid, x, big)
        if self.index is not None:
            x = x.index_select(-1, self.index)
        else:
            if self.pad > 0:
                x = F.pad(x, (0, self.pad), value=cfg.max_range_mm * 4.0)
            x = -F.max_pool1d(-x, self.kernel, stride=self.kernel)
        v = (x <= cfg.max_range_mm).float()
        r = torch.clamp(x, 0.0, cfg.max_range_mm) / cfg.max_range_mm
        if cfg.use_valid_ch:
            # (B, K, N) ×2 → (B, 2K, N)  フレーム毎に [range, valid] を並べる
            out = torch.stack([r, v], dim=2).flatten(1, 2)
        else:
            out = r
        return out


class _ResBlock1d(nn.Module):
    def __init__(self, cin, cout, k, stride, dilation):
        super().__init__()
        # 奇数カーネル前提: 2*pad == dilation*(k-1) で本流と skip の出力長が一致し、
        # ONNX トレース時に長さ合わせの分岐が不要になる
        assert k % 2 == 1, "kernel size must be odd"
        pad = dilation * (k - 1) // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=pad,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                      nn.BatchNorm1d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        s = x if self.skip is None else self.skip(x)
        return F.relu(y + s)


class LidarPolicyNet(nn.Module):
    """1D-CNN エンコーダ + FC ヘッド（段階1）。forward は生スキャン [mm] を受ける。"""

    def __init__(self, cfg: LidarPolicyConfig):
        super().__init__()
        self.cfg = cfg
        preset = LIDAR_POLICY_PRESETS[cfg.preset]
        self.pre = ScanPreprocessor(cfg)

        c0, k0, s0 = preset["stem"]
        layers = [nn.Conv1d(cfg.in_channels, c0, k0, stride=s0, padding=k0 // 2, bias=False),
                  nn.BatchNorm1d(c0), nn.ReLU(inplace=True)]
        cin = c0
        for (c, k, s, d) in preset["stages"]:
            for b in range(preset["blocks_per_stage"]):
                layers.append(_ResBlock1d(cin, c, k, s if b == 0 else 1, d))
                cin = c
        self.backbone = nn.Sequential(*layers)
        feat_dim = cin * 2                                   # avg + max pool

        hidden = cfg.hidden_dim
        self.state_mlp = nn.Sequential(nn.Linear(cfg.state_dim, 32), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Linear(feat_dim + 32, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(cfg.dropout))
        self.head_steer = nn.Linear(hidden, 1)
        self.head_speed = nn.Linear(hidden, 1)
        self.head_traj = nn.Linear(hidden, cfg.horizon * 2) if cfg.use_traj else None

    def encode(self, scan_mm, state):
        x = self.pre(scan_mm)                                # (B, C, N)
        f = self.backbone(x)                                 # (B, C', N')
        f = torch.cat([f.mean(-1), f.amax(-1)], dim=1)       # (B, 2C')
        s = self.state_mlp(state.float())
        return self.fuse(torch.cat([f, s], dim=1))

    def forward(self, scan_mm, state):
        h = self.encode(scan_mm, state)
        action = torch.cat([torch.tanh(self.head_steer(h)),
                            torch.sigmoid(self.head_speed(h))], dim=1)
        if self.head_traj is not None:
            traj = self.head_traj(h).view(-1, self.cfg.horizon, 2)
            return action, traj
        return action


class FlatObsWrapper(nn.Module):
    """RL 互換エクスポート: obs (B, num_bins + 1) = [scan_norm..., vel_norm] を受ける。

    planner plan=="rl" / rl_policy.RLPolicy と同じ観測形式（正規化済み・
    downsample_mode="index"・stack_frames=1・valid ch なし のモデルでのみ意味がある）。
    前処理をスキップしてバックボーンに直接入れる。
    """

    def __init__(self, net: LidarPolicyNet):
        super().__init__()
        self.net = net

    def forward(self, obs):
        n = self.net.cfg.num_bins
        scan_norm = obs[:, :n].unsqueeze(1)                  # (B, 1, N)
        state = obs[:, n:]
        f = self.net.backbone(scan_norm)
        f = torch.cat([f.mean(-1), f.amax(-1)], dim=1)
        s = self.net.state_mlp(state.float())
        h = self.net.fuse(torch.cat([f, s], dim=1))
        return torch.cat([torch.tanh(self.net.head_steer(h)),
                          torch.sigmoid(self.net.head_speed(h))], dim=1)


def build_lidar_policy(cfg: LidarPolicyConfig) -> LidarPolicyNet:
    if cfg.preset not in LIDAR_POLICY_PRESETS:
        raise ValueError(f"unknown preset: {cfg.preset}")
    if cfg.stack_frames < 1 or cfg.num_bins < 8:
        raise ValueError("stack_frames >= 1, num_bins >= 8 が必要です")
    return LidarPolicyNet(cfg)


def flip_scan_lr(scan_mm: np.ndarray, cfg: LidarPolicyConfig) -> np.ndarray:
    """左右反転拡張: スキャンを角度対称に反転（angle_start = -angle_end のとき
    単純な逆順）。操舵符号・軌道 y の反転は呼び出し側で行う。"""
    return np.ascontiguousarray(scan_mm[..., ::-1])
