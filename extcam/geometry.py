"""カメラ較正（地図 ⇔ 画像）と、キーポイントからの車両姿勢。Qt 非依存。

座標系: 地図は ROS の map フレーム（m、+z 上）。車両は base_link = 後軸中心（+x 前、+y 左）。
カメラは固定。床（z=0）上の 4 点以上の対応から、内部パラメータ（未知なら焦点距離だけ推定）と
外部パラメータ（R, t）を求める。外部まで求めるので、床以外の高さ z の平面へも投影できる
（車両のルーフは z = h の平面にある → ホモグラフィ 1 枚では足りない理由）。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

KPT_NAMES = ["front_left", "front_right", "rear_left", "rear_right"]
# 左右反転したときの入れ替え（YOLO pose の flip_idx）
FLIP_IDX = [1, 0, 3, 2]


# ---------------------------------------------------------------------------
# 車両のキーポイント（ルーフ 4 隅）の幾何
# ---------------------------------------------------------------------------

@dataclass
class VehicleKpts:
    """ルーフ 4 隅の base_link 座標と高さ。bbox の推定にはボディの箱も使う。"""
    front_x: float = 0.31       # ルーフ前端の x [m]（後軸から前へ）
    rear_x: float = -0.05       # ルーフ後端の x [m]
    half_w: float = 0.10        # ルーフの半幅 [m]
    height: float = 0.12        # ルーフの高さ [m]（キーポイントはこの平面）
    body_front_x: float = 0.31  # ボディの箱（仮ラベルの bbox 用）
    body_rear_x: float = -0.05
    body_half_w: float = 0.12

    @classmethod
    def from_vehicle_json(cls, path: str) -> "VehicleKpts":
        """togikaidrive の vehicle.json（body.length_m / width_m / wheelbase_m）から既定値を作る。"""
        with open(path, encoding="utf-8") as f:
            body = (json.load(f) or {}).get("body") or {}
        length = float(body.get("length_m") or 0.36)
        width = float(body.get("width_m") or 0.24)
        rear_oh = body.get("rear_overhang_m")
        rear = -float(rear_oh) if rear_oh is not None else -0.05
        return cls(front_x=rear + length, rear_x=rear, half_w=width * 0.4, height=0.12,
                   body_front_x=rear + length, body_rear_x=rear, body_half_w=width / 2)

    def body_points(self) -> np.ndarray:
        """キーポイントの base_link 座標 (4, 2)。並びは KPT_NAMES。"""
        return np.array([[self.front_x, self.half_w], [self.front_x, -self.half_w],
                         [self.rear_x, self.half_w], [self.rear_x, -self.half_w]], float)

    def box_corners(self) -> np.ndarray:
        """ボディの箱の 8 隅の base_link 座標 (8, 3)。"""
        xs = (self.body_front_x, self.body_rear_x)
        ys = (self.body_half_w, -self.body_half_w)
        zs = (0.0, self.height)
        return np.array([[x, y, z] for x in xs for y in ys for z in zs], float)


def body_to_map(pts_body: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(pts_body, float)
    out = np.empty_like(p)
    out[:, 0] = x + c * p[:, 0] - s * p[:, 1]
    out[:, 1] = y + s * p[:, 0] + c * p[:, 1]
    if p.shape[1] > 2:
        out[:, 2:] = p[:, 2:]
    return out


def pose_from_points(body_pts: np.ndarray, map_pts: np.ndarray):
    """対応する 2 点以上から 2D 剛体変換（回転＋並進）を最小二乗で求める（Kabsch）。

    返り値: (x, y, yaw, rms) … base_link の地図座標。点が足りなければ None。
    """
    b = np.asarray(body_pts, float)[:, :2]
    m = np.asarray(map_pts, float)[:, :2]
    if len(b) < 2:
        return None
    cb, cm = b.mean(0), m.mean(0)
    B, M = b - cb, m - cm
    hxy = (B * M).sum()                       # Σ(bx mx + by my)
    cross = (B[:, 0] * M[:, 1] - B[:, 1] * M[:, 0]).sum()
    yaw = math.atan2(cross, hxy)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    t = cm - rot @ cb
    resid = m - (b @ rot.T + t)
    return float(t[0]), float(t[1]), yaw, float(np.sqrt((resid ** 2).sum(1).mean()))


# ---------------------------------------------------------------------------
# カメラ
# ---------------------------------------------------------------------------

def _focal_from_homography(H: np.ndarray, cx: float, cy: float):
    """床の平面のホモグラフィ 1 枚から焦点距離を推定する（主点＝既知、正方画素、スキューなし）。

    ω = K^-T K^-1 = diag(a, a, 1)（a = 1/f²）として、r1⊥r2 と |r1|=|r2| の 2 式を最小二乗で解く。
    """
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    h = T @ H
    h1, h2 = h[:, 0], h[:, 1]
    A = np.array([h1[0] * h2[0] + h1[1] * h2[1],
                  h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    B = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    denom = float(A @ A)
    if denom <= 0:
        return None
    a = -float(A @ B) / denom
    if a <= 0:
        return None
    return 1.0 / math.sqrt(a)


@dataclass
class CameraCalib:
    image_size: tuple                      # (w, h)
    K: list                                # 3x3
    dist: list = field(default_factory=lambda: [0.0] * 5)
    rvec: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tvec: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    reproj_rms_px: float | None = None
    n_points: int = 0
    intrinsics_source: str = "estimated"   # estimated / given

    # --- 推定 ---------------------------------------------------------------
    @classmethod
    def fit(cls, img_pts, map_pts, image_size, K=None, dist=None, refine_focal=True) -> "CameraCalib":
        """床（z=0）上の対応点から較正する。K が無ければ焦点距離を推定する（4 点以上、6 点以上を推奨）。"""
        ip = np.asarray(img_pts, np.float64).reshape(-1, 2)
        mp = np.asarray(map_pts, np.float64).reshape(-1, 2)
        if len(ip) < 4 or len(ip) != len(mp):
            raise ValueError("対応点は 4 組以上必要です")
        w, h = image_size
        dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64).ravel()
        obj = np.hstack([mp, np.zeros((len(mp), 1))]).astype(np.float64)
        given = K is not None
        if given:
            K = np.asarray(K, np.float64)
        else:
            und = ip
            H, _ = cv2.findHomography(mp, und, 0)
            if H is None:
                raise ValueError("ホモグラフィを求められません（点が一直線上にある等）")
            f = _focal_from_homography(H, w / 2.0, h / 2.0)
            if f is None or not (0.2 * w < f < 20 * w):
                f = 1.0 * w          # 推定できない配置（正面からほぼ真っ直ぐ等）→ 画角約 53° の仮値
            K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
            if refine_focal and len(ip) >= 6:
                flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
                         cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
                         cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
                try:
                    _, K2, _, _, _ = cv2.calibrateCamera(
                        [obj.astype(np.float32)], [ip.astype(np.float32)], (int(w), int(h)),
                        K.copy(), np.zeros(5), flags=flags)
                    if 0.2 * w < K2[0, 0] < 20 * w:
                        K = K2
                except cv2.error:
                    pass
        ok, rvec, tvec = cv2.solvePnP(obj, ip, K, dist, flags=cv2.SOLVEPNP_IPPE if len(ip) >= 4
                                      else cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("solvePnP に失敗しました")
        ok, rvec, tvec = cv2.solvePnP(obj, ip, K, dist, rvec, tvec, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        cal = cls(image_size=(int(w), int(h)), K=K.tolist(), dist=dist.tolist(),
                  rvec=rvec.ravel().tolist(), tvec=tvec.ravel().tolist(), n_points=len(ip),
                  intrinsics_source="given" if given else "estimated")
        cal.reproj_rms_px = float(np.sqrt((cal.reproj_errors(ip, mp) ** 2).mean()))
        return cal

    # --- 投影 ---------------------------------------------------------------
    def _arrays(self):
        return (np.asarray(self.K, np.float64), np.asarray(self.dist, np.float64),
                np.asarray(self.rvec, np.float64), np.asarray(self.tvec, np.float64))

    def project(self, pts_map) -> np.ndarray:
        """地図座標 (N, 2|3)（z 省略時は 0）→ 画素 (N, 2)。"""
        p = np.asarray(pts_map, np.float64)
        if p.ndim == 1:
            p = p[None]
        if p.shape[1] == 2:
            p = np.hstack([p, np.zeros((len(p), 1))])
        if len(p) == 0:
            return np.zeros((0, 2))
        K, d, r, t = self._arrays()
        uv, _ = cv2.projectPoints(p, r, t, K, d)
        return uv.reshape(-1, 2)

    def in_front(self, pts_map) -> np.ndarray:
        """カメラの前方にある点か（後ろの点を投影すると像が反転して紛らわしい）。"""
        p = np.asarray(pts_map, np.float64)
        if p.shape[1] == 2:
            p = np.hstack([p, np.zeros((len(p), 1))])
        K, d, r, t = self._arrays()
        R, _ = cv2.Rodrigues(r)
        return (p @ R.T + t)[:, 2] > 1e-3

    def backproject(self, uv, z: float = 0.0) -> np.ndarray:
        """画素 (N, 2) → 高さ z の水平面上の地図座標 (N, 2)。平面と交わらない画素は nan。"""
        K, d, r, t = self._arrays()
        u = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(u, K, d).reshape(-1, 2)
        R, _ = cv2.Rodrigues(r)
        C = -R.T @ t
        dirs = np.hstack([n, np.ones((len(n), 1))]) @ R       # = (R^T d_cam)^T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - C[2]) / dirs[:, 2]
        out = C[:2] + s[:, None] * dirs[:, :2]
        out[(s <= 0) | ~np.isfinite(s)] = np.nan
        return out

    def reproj_errors(self, img_pts, map_pts) -> np.ndarray:
        return np.linalg.norm(self.project(map_pts) - np.asarray(img_pts, float).reshape(-1, 2), axis=1)

    def camera_position(self) -> np.ndarray:
        _, _, r, t = self._arrays()
        R, _ = cv2.Rodrigues(r)
        return (-R.T @ t).ravel()

    def ground_resolution(self, pt_map, z: float = 0.0) -> float:
        """地図上のその点での 1 画素あたりの距離 [m/px]（遠い側ほど粗い）。"""
        p = np.array([[pt_map[0], pt_map[1], z]], float)
        uv = self.project(p)[0]
        q = self.backproject(np.array([[uv[0] + 1.0, uv[1]], [uv[0], uv[1] + 1.0]]), z)
        return float(np.nanmax(np.linalg.norm(q - p[:, :2], axis=1)))

    # --- 保存 ---------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        d["camera_position_m"] = self.camera_position().round(4).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalib":
        keys = {"image_size", "K", "dist", "rvec", "tvec", "reproj_rms_px", "n_points", "intrinsics_source"}
        return cls(**{k: v for k, v in d.items() if k in keys})


# ---------------------------------------------------------------------------
# 較正ファイル（カメラ＋車両キーポイント＋対応点）
# ---------------------------------------------------------------------------

def save_setup(path: str, calib: CameraCalib | None, vehicle: VehicleKpts, pairs: list,
               map_dir: str | None = None):
    data = {"version": 1, "camera": calib.to_dict() if calib else None,
            "vehicle_kpts": asdict(vehicle), "kpt_names": KPT_NAMES,
            "pairs": pairs, "map_dir": map_dir}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def load_setup(path: str):
    """返り値: (CameraCalib|None, VehicleKpts, pairs, map_dir)"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cam = CameraCalib.from_dict(d["camera"]) if d.get("camera") else None
    veh = VehicleKpts(**{k: v for k, v in (d.get("vehicle_kpts") or {}).items()
                         if k in VehicleKpts.__dataclass_fields__})
    return cam, veh, d.get("pairs") or [], d.get("map_dir")


def kpts_from_pose(calib: CameraCalib, veh: VehicleKpts, x: float, y: float, yaw: float):
    """車両の姿勢から、画像上のキーポイント (4, 3)[u, v, vis] と bbox [x1, y1, x2, y2] を作る（仮ラベル用）。"""
    kb = veh.body_points()
    km = body_to_map(np.hstack([kb, np.full((4, 1), veh.height)]), x, y, yaw)
    uv = calib.project(km)
    w, h = calib.image_size
    vis = np.where((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                   & calib.in_front(km), 2, 0)
    box = calib.project(body_to_map(veh.box_corners(), x, y, yaw))
    x1, y1 = np.clip(box.min(0), 0, [w - 1, h - 1])
    x2, y2 = np.clip(box.max(0), 0, [w - 1, h - 1])
    kpts = np.hstack([uv, vis[:, None]])
    return kpts, [float(x1), float(y1), float(x2), float(y2)]


def pose_from_kpts(calib: CameraCalib, veh: VehicleKpts, kpts, min_conf: float = 0.5):
    """画像上のキーポイント (4, 3)[u, v, conf|vis] → base_link の (x, y, yaw, rms, n)。2 点未満なら None。"""
    k = np.asarray(kpts, float).reshape(-1, 3)
    use = k[:, 2] >= min_conf
    if use.sum() < 2:
        return None
    mp = calib.backproject(k[use, :2], z=veh.height)
    ok = np.isfinite(mp).all(1)
    if ok.sum() < 2:
        return None
    r = pose_from_points(veh.body_points()[use][ok], mp[ok])
    if r is None:
        return None
    return (*r, int(ok.sum()))
