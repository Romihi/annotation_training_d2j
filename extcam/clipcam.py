"""クリップごとのカメラ（較正＋手ぶれ補正）。Qt 非依存。

較正ファイルの探し方:
  1. extcam/clip_NN.calib.json … そのクリップ専用（他のカメラから取り込んだ動画。importer が書く）
  2. extcam/calib.json         … 走行のカメラ（extcam）。クリップ専用が無いときに使う

手ぶれ補正: extcam/clip_NN.stab.npy（N×3×3、フレーム i の画素 → 基準フレームの画素のホモグラフィ。NaN は補正なし）。
較正は基準フレームの画素で行うので、手持ちのクリップでは「フレームの画素 ⇔ 基準フレームの画素」を挟んで投影・逆投影する。
固定カメラ（extcam）では stab が無く、すべて恒等変換になる。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .geometry import CameraCalib, VehicleKpts, load_setup, save_setup

EXT_DIR = "extcam"


def run_setup_path(run_dir: str) -> str:
    return os.path.join(run_dir, EXT_DIR, "calib.json")


def clip_setup_path(run_dir: str, num: int) -> str:
    return os.path.join(run_dir, EXT_DIR, f"clip_{int(num):02d}.calib.json")


def stab_path(run_dir: str, num: int) -> str:
    return os.path.join(run_dir, EXT_DIR, f"clip_{int(num):02d}.stab.npy")


def setup_path_for(run_dir: str, num: int | None) -> str:
    """そのクリップが使う較正ファイル（専用があればそれ）。"""
    if num is not None:
        p = clip_setup_path(run_dir, num)
        if os.path.isfile(p):
            return p
    return run_setup_path(run_dir)


def _apply_h(H, uv):
    uv = np.asarray(uv, float).reshape(-1, 2)
    if H is None or not len(uv):
        return uv.copy()
    return cv2.perspectiveTransform(uv.reshape(-1, 1, 2), H).reshape(-1, 2)


class ClipCamera:
    """CameraCalib（基準フレームの画素）＋フレームごとの手ぶれ補正。"""

    def __init__(self, calib: CameraCalib, stab: np.ndarray | None = None, ref_index: int | None = None):
        self.calib = calib
        self.stab = stab
        self.ref_index = ref_index

    @property
    def handheld(self) -> bool:
        return self.stab is not None

    def H(self, i):
        """フレーム i → 基準フレーム。無い・壊れているなら None（恒等とみなす）。"""
        if self.stab is None or i is None or not (0 <= int(i) < len(self.stab)):
            return None
        h = self.stab[int(i)]
        return None if not np.isfinite(h).all() else h

    def to_ref(self, uv, i):
        return _apply_h(self.H(i), uv)

    def to_frame(self, uv, i):
        h = self.H(i)
        return _apply_h(None if h is None else np.linalg.inv(h), uv)

    def project(self, pts_map, i=None):
        """地図 → フレーム i の画素（i=None なら基準フレームの画素）。"""
        uv = self.calib.project(pts_map)
        return uv if i is None else self.to_frame(uv, i)

    def in_front(self, pts_map):
        return self.calib.in_front(pts_map)

    def backproject(self, uv, z=0.0, i=None):
        return self.calib.backproject(uv if i is None else self.to_ref(uv, i), z)


def load_clip_camera(run_dir: str, num: int | None):
    """返り値: (ClipCamera|None, VehicleKpts, pairs, map_dir, 使った較正ファイルのパス)"""
    path = setup_path_for(run_dir, num)
    if not os.path.isfile(path):
        return None, VehicleKpts(), [], None, path
    calib, veh, pairs, map_dir = load_setup(path)
    ref_index = None
    try:
        with open(path, encoding="utf-8") as f:
            ref_index = json.load(f).get("ref_index")
    except (OSError, ValueError):
        pass
    stab = None
    if num is not None and path == clip_setup_path(run_dir, num) and os.path.isfile(stab_path(run_dir, num)):
        stab = np.load(stab_path(run_dir, num))
    return (ClipCamera(calib, stab, ref_index) if calib else None), veh, pairs, map_dir, path


def save_clip_setup(path: str, calib: CameraCalib | None, veh: VehicleKpts, pairs: list, map_dir: str | None,
                    extra: dict | None = None):
    """save_setup に、クリップ専用の付帯情報（ref_index・取り込み元など）を足して書く。

    既存ファイルの付帯情報は引き継ぐ（GUI の自動保存で ref_index や source が消えないように）。
    """
    keep = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            keep = {k: v for k, v in old.items() if k not in _STD_KEYS}
        except (OSError, ValueError):
            keep = {}
    keep.update(extra or {})
    save_setup(path, calib, veh, pairs, map_dir)
    if keep:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d.update(keep)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)


_STD_KEYS = {"version", "camera", "vehicle_kpts", "kpt_names", "pairs", "map_dir"}
