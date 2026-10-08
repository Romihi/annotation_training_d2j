"""走行フォルダの地図・コース外形・車両の自己位置・車載画像を読む。Qt 非依存。

地図は course_editor と同じもの: <map_dir>/<name>.yaml（ROS map_server 形式）と map_bounds.json（outer / holes）。
map_dir は走行フォルダの map_ref.json の map_dir（リポジトリ相対）から探す。走行フォルダ内の map/ スナップショットは
地図画像だけで外形を持たないことがあるので、外形は map_dir 側を優先する。
"""
from __future__ import annotations

import bisect
import glob
import json
import math
import os

import numpy as np

# 比較に使う自己位置ソースの優先順（catalog のキー接頭辞）
POSE_SOURCES = ("fused", "slam", "aruco", "vslam", "pose")


def repo_root_of(run_dir: str) -> str:
    """data/data_<TS> → リポジトリのルート（2 つ上）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))


def resolve_map_dir(run_dir: str) -> str | None:
    ref = os.path.join(run_dir, "map_ref.json")
    cands = []
    if os.path.isfile(ref):
        try:
            with open(ref, encoding="utf-8") as f:
                md = json.load(f).get("map_dir")
            if md:
                cands.append(md if os.path.isabs(md) else os.path.join(repo_root_of(run_dir), md))
        except (OSError, ValueError):
            pass
    cands.append(os.path.join(run_dir, "map"))
    for c in cands:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.yaml")):
            return os.path.abspath(c)
    return None


class CourseMap:
    """地図画像（任意）とコース外形。座標はすべて map フレーム [m]。"""

    def __init__(self, map_dir: str | None):
        self.map_dir = map_dir
        self.image = None           # グレースケール (H, W)
        self.resolution = None
        self.origin = None          # [x, y, yaw]
        self.outer = []             # [[x, y], ...]
        self.holes = []             # [[[x, y], ...], ...]
        if not map_dir:
            return
        self._load_image()
        self._load_bounds()

    def _load_image(self):
        import yaml   # 遅延 import（PyYAML はツールの依存に含まれる）
        ymls = sorted(glob.glob(os.path.join(self.map_dir, "*.yaml")))
        for y in ymls:
            try:
                with open(y, encoding="utf-8") as f:
                    meta = yaml.safe_load(f)
                img_path = os.path.join(self.map_dir, meta["image"])
                import cv2
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if int(meta.get("negate", 0)):
                    img = 255 - img
                self.image = img
                self.resolution = float(meta["resolution"])
                self.origin = [float(v) for v in meta["origin"]]
                return
            except Exception:   # noqa: BLE001
                continue

    def _load_bounds(self):
        p = os.path.join(self.map_dir, "map_bounds.json")
        if not os.path.isfile(p):
            return
        with open(p, encoding="utf-8") as f:
            b = json.load(f)
        self.outer = [list(map(float, q)) for q in b.get("outer") or []]
        self.holes = [[list(map(float, q)) for q in h] for h in b.get("holes") or []]

    def extent(self):
        """matplotlib の imshow 用 [x0, x1, y0, y1]。"""
        if self.image is None:
            return None
        h, w = self.image.shape[:2]
        x0, y0 = self.origin[0], self.origin[1]
        return [x0, x0 + w * self.resolution, y0, y0 + h * self.resolution]

    def vertices(self) -> np.ndarray:
        """外形と穴の頂点すべて（較正の対応点の候補）。"""
        pts = list(self.outer) + [q for h in self.holes for q in h]
        return np.array(pts, float).reshape(-1, 2)

    def polylines(self) -> list:
        """閉じた折れ線のリスト（描画・投影の確認用）。"""
        out = []
        for poly in [self.outer] + list(self.holes):
            if len(poly) >= 2:
                out.append(np.array(poly + [poly[0]], float))
        return out

    def snap(self, x: float, y: float, radius: float = 0.3):
        """近い頂点があればそこへ吸着する。"""
        v = self.vertices()
        if not len(v):
            return x, y, False
        d = np.hypot(v[:, 0] - x, v[:, 1] - y)
        k = int(d.argmin())
        if d[k] <= radius:
            return float(v[k, 0]), float(v[k, 1]), True
        return x, y, False


class RunLog:
    """catalog から車両の自己位置と車載画像のパスを時刻つきで読む（比較・仮ラベル用）。"""

    def __init__(self, run_dir: str, source: str | None = None):
        self.run_dir = os.path.abspath(run_dir)
        self.t = []                 # 秒（Jetson の時刻）
        self.idx = []
        self.xyyaw = []             # [(x, y, yaw) | None]
        self.status = []
        self.images = []            # 車載画像（cam0 優先）のファイル名
        self.sources_found = set()
        self.source = source
        self._load()

    def _load(self):
        rows = []
        for p in sorted(glob.glob(os.path.join(self.run_dir, "catalog_*.catalog")),
                        key=lambda s: int(os.path.basename(s).split("_")[1].split(".")[0])):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        rows = [r for r in rows if r.get("_timestamp_ms") is not None]
        rows.sort(key=lambda r: r["_timestamp_ms"])
        avail = [s for s in POSE_SOURCES if any(f"{s}/x" in r for r in rows[:50])]
        self.sources_found = set(avail)
        src = self.source if self.source in avail else (avail[0] if avail else None)
        self.source = src
        for r in rows:
            self.t.append(r["_timestamp_ms"] / 1000.0)
            self.idx.append(r.get("_index"))
            p = None
            if src and f"{src}/x" in r:
                try:
                    p = (float(r[f"{src}/x"]), float(r[f"{src}/y"]), float(r[f"{src}/theta"]))
                except (TypeError, ValueError):
                    p = None
            self.xyyaw.append(p)
            self.status.append(r.get(f"{src}/status", "") if src else "")
            img = r.get("cam0/image_array") or r.get("cam/image_array") or r.get("cam1/image_array")
            self.images.append(img)

    def __len__(self):
        return len(self.t)

    def nearest(self, t: float, max_dt: float = 0.2):
        """時刻 t に最も近い記録の番号（max_dt を超えたら None）。"""
        if not self.t:
            return None
        k = bisect.bisect_left(self.t, t)
        best = None
        for j in (k - 1, k):
            if 0 <= j < len(self.t) and (best is None or abs(self.t[j] - t) < abs(self.t[best] - t)):
                best = j
        return best if best is not None and abs(self.t[best] - t) <= max_dt else None

    def pose_at(self, t: float, max_dt: float = 0.2):
        """時刻 t の姿勢を前後 2 点の線形補間で（向きは最短回りで補間）。"""
        if not self.t:
            return None
        k = bisect.bisect_left(self.t, t)
        if k <= 0 or k >= len(self.t):
            j = self.nearest(t, max_dt)
            return self.xyyaw[j] if j is not None else None
        a, b = self.xyyaw[k - 1], self.xyyaw[k]
        ta, tb = self.t[k - 1], self.t[k]
        if a is None or b is None or tb - ta > max_dt * 2:
            j = self.nearest(t, max_dt)
            return self.xyyaw[j] if j is not None else None
        w = (t - ta) / (tb - ta) if tb > ta else 0.0
        dyaw = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
        return (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + dyaw * w)

    def image_path(self, j: int):
        if j is None or not (0 <= j < len(self.images)) or not self.images[j]:
            return None
        return os.path.join(self.run_dir, "images", self.images[j])

    def trajectory(self) -> np.ndarray:
        pts = [p for p in self.xyyaw if p is not None]
        return np.array(pts, float).reshape(-1, 3)
