"""
LiDAR Policy 用データセット。

セッションの lidar/{_index}_lidar_distance_array_.npy（生距離 [mm], int16）を
K フレーム積層して返し、教師は運転アノテーション (angle, throttle)、任意で
pose_manager 由来の将来軌道 (H,2)（補助ヘッド）。

画像パス <session>/images/{_index}_..jpg からセッションと _index を導出する規約は
TogivadDataset._lidar_occ と同じ。前フレームは「同一セッション かつ npy が存在」の
ときだけ使い、無ければ現フレームを複製する（先頭・欠損・別セッション境界）。
"""

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .lidar_policy_models import LidarPolicyConfig, flip_scan_lr


def scan_path_for_image(img_path):
    """画像パス → (session_dir, lidar npy パス)。npy が無ければ (session, None)。"""
    session = os.path.dirname(os.path.dirname(img_path))
    prefix = os.path.basename(img_path).split('_')[0]
    npy = os.path.join(session, 'lidar', f'{prefix}_lidar_distance_array_.npy')
    return session, (npy if os.path.exists(npy) else None)


def load_lidar_meta(session_dir):
    """manifest.json 3行目の LiDAR 角度規約（togivad.dataset._load_lidar_meta と同じ）。"""
    meta = {"angle_start": -135.0, "angle_end": 135.0, "clockwise": False,
            "data_points": None}
    try:
        with open(os.path.join(session_dir, "manifest.json")) as f:
            lines = f.read().splitlines()
        d = json.loads(lines[2])
        if "lidar_angle_start" in d:
            meta.update(angle_start=float(d["lidar_angle_start"]),
                        angle_end=float(d["lidar_angle_end"]),
                        clockwise=bool(d.get("lidar_clockwise", False)))
        if "lidar_data_points" in d:
            meta["data_points"] = int(d["lidar_data_points"])
    except Exception:
        pass
    return meta


def load_session_modes(session_dir):
    """catalog_*.catalog から {_index: user/mode} を読む（教師種別フィルタ用）。"""
    modes = {}
    for path in sorted(glob.glob(os.path.join(session_dir, "catalog_*.catalog"))):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "_index" in row:
                        modes[int(row["_index"])] = str(row.get("user/mode", ""))
        except Exception:
            continue
    return modes


class LidarPolicyDataset(Dataset):
    """(scan_stack (K, N_raw) float32[mm], state (S,), action (2,), traj (H,2), traj_mask)"""

    def __init__(self, valid_indexes, annotations, images, cfg: LidarPolicyConfig,
                 pose_manager=None, pose_source="pose", exclude=None,
                 mode_filter="all", augment=False, mirror=False,
                 beam_dropout=0.05, range_noise_mm=20.0, roll_beams=2):
        self.cfg = cfg
        self.images = images
        self.augment = augment
        self.mirror = mirror
        self.beam_dropout = float(beam_dropout)
        self.range_noise_mm = float(range_noise_mm)
        self.roll_beams = int(roll_beams)
        self._scan_cache = {}                     # npy path -> np.ndarray(int16)
        self._mode_cache = {}                     # session -> {_index: mode}
        exclude = exclude or set()

        # 積層に使う前フレームのパス列（先頭=最古 … 末尾=現フレーム）
        self.samples = []                         # [(idx, [npy...K], state, action, traj|None)]
        self.n_no_scan = 0
        self.n_mode_skipped = 0
        for idx in sorted(valid_indexes):
            if idx in exclude:
                continue
            ann = annotations.get(idx)
            if ann is None or idx >= len(images):
                continue
            session, npy = scan_path_for_image(images[idx])
            if npy is None:
                self.n_no_scan += 1
                continue
            if mode_filter != "all" and not self._mode_ok(session, images[idx], mode_filter):
                self.n_mode_skipped += 1
                continue
            paths = [npy]
            for k in range(1, cfg.stack_frames):
                j = idx - k
                prev = None
                if j >= 0 and j < len(images):
                    s2, p2 = scan_path_for_image(images[j])
                    if s2 == session:
                        prev = p2
                paths.insert(0, prev if prev is not None else paths[0])
            speed = float(ann.get("speed", ann.get("pose/speed", 0.0)) or 0.0)
            state = np.array([np.clip(abs(speed) / cfg.max_speed, 0.0, 1.0)], np.float32)
            action = np.array([float(ann.get("angle", 0.0) or 0.0),
                               float(ann.get("throttle", 0.0) or 0.0)], np.float32)
            traj = None
            if cfg.use_traj and pose_manager is not None:
                try:
                    traj = pose_manager.compute_future_trajectory(
                        idx, horizon=cfg.horizon, dt=cfg.dt, exclude=exclude,
                        prefer=pose_source)
                except Exception:
                    traj = None
                if traj is not None:
                    traj = np.asarray(traj, np.float32)
                    if traj.shape != (cfg.horizon, 2):
                        traj = None
            self.samples.append((idx, paths, state, action, traj))

    def _mode_ok(self, session, img_path, mode_filter):
        modes = self._mode_cache.get(session)
        if modes is None:
            modes = load_session_modes(session)
            self._mode_cache[session] = modes
        try:
            entry = int(os.path.basename(img_path).split('_')[0])
        except ValueError:
            return True
        mode = modes.get(entry, "")
        if mode_filter == "auto":
            return mode.startswith("auto") or mode == "local"
        if mode_filter == "user":
            return mode == "user"
        return True

    # --- 統計（損失重み・表示用） ---
    def steer_values(self):
        return np.asarray([a[0] for _, _, _, a, _ in self.samples], np.float32)

    def has_traj(self):
        return any(t is not None for *_, t in self.samples)

    def indexes(self):
        return [s[0] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def _load_scan(self, path):
        arr = self._scan_cache.get(path)
        if arr is None:
            try:
                arr = np.load(path).astype(np.float32).reshape(-1)
            except Exception:
                arr = np.zeros(self.cfg.num_beams_raw, np.float32)
            n = self.cfg.num_beams_raw
            if arr.shape[0] != n:
                # 点数が違うセッションは線形補間で揃える（0=無効はそのまま伝播しうる）
                arr = np.interp(np.linspace(0, arr.shape[0] - 1, n),
                                np.arange(arr.shape[0]), arr).astype(np.float32)
            self._scan_cache[path] = arr
        return arr

    def __getitem__(self, i):
        idx, paths, state, action, traj = self.samples[i]
        scan = np.stack([self._load_scan(p) for p in paths], 0)       # (K, N)
        action = action.copy()
        traj_t = (np.zeros((self.cfg.horizon, 2), np.float32) if traj is None
                  else traj.copy())
        mask = 0.0 if traj is None else 1.0

        if self.augment:
            scan = scan.copy()
            if self.range_noise_mm > 0:
                valid = scan > 0
                scan[valid] += np.random.normal(0.0, self.range_noise_mm,
                                                valid.sum()).astype(np.float32)
            if self.beam_dropout > 0:
                drop = np.random.rand(*scan.shape) < self.beam_dropout
                scan[drop] = 0.0                                        # 無効化
            if self.roll_beams > 0:
                r = np.random.randint(-self.roll_beams, self.roll_beams + 1)
                if r:
                    scan = np.roll(scan, r, axis=-1)
            if self.mirror and np.random.rand() < 0.5:
                scan = flip_scan_lr(scan, self.cfg)
                action[0] = -action[0]
                traj_t[:, 1] = -traj_t[:, 1]

        return {
            "scan": torch.from_numpy(np.ascontiguousarray(scan)),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "traj": torch.from_numpy(traj_t),
            "traj_mask": torch.tensor(mask, dtype=torch.float32),
            "index": idx,
        }


def block_split_indices(n, val_split, block_size=200):
    """連続ブロック単位の train/val 分割（時系列相関によるリークを抑える）。

    サンプル列を block_size ごとに区切り、1/val_split ブロックに 1 つを val に
    割り当てる。返り値は (train_idx, val_idx) の位置インデックス配列。
    """
    if n <= 1:
        return list(range(n)), []
    val_split = min(max(val_split, 0.01), 0.9)
    every = max(2, int(round(1.0 / val_split)))
    n_blocks = max(1, int(np.ceil(n / block_size)))
    train_idx, val_idx = [], []
    for b in range(n_blocks):
        lo, hi = b * block_size, min(n, (b + 1) * block_size)
        # 先頭ブロックは train に固定し、以降 every 個に 1 つを val
        target = val_idx if (b % every == every - 1) else train_idx
        target.extend(range(lo, hi))
    if not val_idx:                       # ブロック数が少なく val が空 → 末尾を val
        cut = max(1, int(n * val_split))
        train_idx, val_idx = list(range(0, n - cut)), list(range(n - cut, n))
    return train_idx, val_idx
