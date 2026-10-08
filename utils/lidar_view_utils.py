"""
LiDAR ビュー（画像ソース '__lidar__'）用のデータ準備。

描画は main.py の ImageLabel.draw_lidar_view が行い、ここではセッションの
lidar/{idx}_lidar_distance_array_.npy の読み込み・角度変換・統計・モデル入力
（ScanPreprocessor 通過後のビン列）を返す。パス規約とフレーム積層の扱いは
managers/lidar_policy_dataset と同じ（学習が見ているものをそのまま可視化する）。
"""

import math
import os

import numpy as np

from managers.lidar_policy_dataset import load_lidar_meta, scan_path_for_image


def collect_lidar_frame(images, index, stack_frames=1, annotations=None):
    """現フレーム（と過去 K-1 フレーム）の生スキャンを読む。

    Returns:
        dict | None — {
          "scans": [np.ndarray(N,) float32 mm ...]  先頭=最古 … 末尾=現フレーム,
          "duplicated": int   過去フレームが無く現フレームを複製した数,
          "meta": {angle_start, angle_end, clockwise, data_points},
          "session": str, "npy": str,
        }  現フレームの npy が無ければ None
    """
    if images is None or index is None or not (0 <= index < len(images)):
        return None
    session, npy = scan_path_for_image(images[index])
    if npy is None:
        return None
    paths = [npy]
    dup = 0
    for k in range(1, max(1, int(stack_frames))):
        j = index - k
        prev = None
        if 0 <= j < len(images):
            s2, p2 = scan_path_for_image(images[j])
            if s2 == session:
                prev = p2
        if prev is None:
            dup += 1
            prev = paths[0]
        paths.insert(0, prev)
    scans = []
    for p in paths:
        try:
            scans.append(np.load(p).astype(np.float32).reshape(-1))
        except Exception:
            scans.append(np.zeros(1081, np.float32))
    meta = load_lidar_meta(session)
    if meta.get("data_points") is None:
        meta["data_points"] = int(scans[-1].shape[0])
    return {"scans": scans, "duplicated": dup, "meta": meta,
            "session": session, "npy": npy}


def scan_angles(n, meta):
    """ビーム index → 角度 [rad]（ロボット座標: 0=前方, CCW 正=左）。"""
    sweep = np.linspace(meta["angle_start"], meta["angle_end"], n)
    return np.deg2rad(-sweep if meta.get("clockwise") else sweep)


def scan_to_xy(scan_mm, meta, min_range_mm=50.0, max_range_mm=None):
    """生スキャン → (x_fwd, y_left)[m], valid mask, r[m], angle[rad]。"""
    r = np.asarray(scan_mm, np.float64) / 1000.0
    ang = scan_angles(r.shape[0], meta)
    valid = r * 1000.0 >= min_range_mm
    if max_range_mm is not None:
        valid &= r * 1000.0 <= max_range_mm
    return r * np.cos(ang), r * np.sin(ang), valid, r, ang


def scan_stats(scan_mm, meta, min_range_mm=50.0, max_range_mm=10000.0):
    """情報表示用の統計: 有効率、前方(±15°)・左右(±90°±15°)の最小距離 [m]。"""
    r = np.asarray(scan_mm, np.float64)
    ang = scan_angles(r.shape[0], meta)
    valid = (r >= min_range_mm) & (r <= max_range_mm)

    def _min_in(lo_deg, hi_deg):
        m = valid & (ang >= math.radians(lo_deg)) & (ang <= math.radians(hi_deg))
        return float(r[m].min() / 1000.0) if m.any() else float("nan")

    return {
        "valid_ratio": float(valid.mean()) if r.size else 0.0,
        "front_min_m": _min_in(-15, 15),
        "left_min_m": _min_in(75, 105),
        "right_min_m": _min_in(-105, -75),
        "n_beams": int(r.size),
    }


def model_input_bins(scans, cfg):
    """学習時と同じ ScanPreprocessor を通した (num_bins,) の距離 [m] と有効 mask、
    各ビンの角度 [rad] を返す（現フレーム=末尾のみ）。

    cfg: managers.lidar_policy_models.LidarPolicyConfig
    """
    import torch
    from managers.lidar_policy_models import ScanPreprocessor

    n_raw = int(cfg.num_beams_raw)
    cur = np.asarray(scans[-1], np.float32)
    if cur.shape[0] != n_raw:
        cur = np.interp(np.linspace(0, cur.shape[0] - 1, n_raw),
                        np.arange(cur.shape[0]), cur).astype(np.float32)
    pre = ScanPreprocessor(cfg)
    with torch.no_grad():
        out = pre(torch.from_numpy(cur)[None, None])[0]      # (C, num_bins)
    r_norm = out[0].numpy()
    valid = (out[1].numpy() > 0.5) if cfg.use_valid_ch else (r_norm < 0.999)
    r_m = r_norm * cfg.max_range_mm / 1000.0
    meta = {"angle_start": cfg.angle_start_deg, "angle_end": cfg.angle_end_deg,
            "clockwise": cfg.clockwise}
    if cfg.downsample_mode == "index":
        idx = np.linspace(0, n_raw - 1, cfg.num_bins).round().astype(int)
        ang = scan_angles(n_raw, meta)[idx]
    else:
        k = int(math.ceil(n_raw / cfg.num_bins))
        centers = np.minimum(np.arange(cfg.num_bins) * k + (k - 1) / 2.0, n_raw - 1)
        ang = np.interp(centers, np.arange(n_raw), scan_angles(n_raw, meta))
    return r_m, valid, ang


# ---------------------------------------------------------------- 測距ゾーン
# catalog の lidar/{zone} または ultrasonic/{zone}（[mm]）。togikaidrive の
# config ZONE_NAMES と同じ 5 ゾーン。表示は monitor.py の drawSensorFan と同じ
# 扇形（距離で着色: <300 赤 / <600 黄 / それ以外 緑）。
ZONE_NAMES = ("RrLH", "FrLH", "FrFR", "FrRH", "RrRH")
ZONE_PREFIXES = ("lidar", "ultrasonic")

# UST20: config_default.py の ZONE_INDEX（LIDAR_ANGLE_STEP=4 → 1°=4 index）
_UST20_ZONE_INDEX = {"RrLH": (720, 960), "FrLH": (600, 720), "FrFR": (480, 600),
                     "FrRH": (360, 480), "RrRH": (120, 360)}
# 超音波 / 未知 LiDAR: monitor.py の sensorConfig 既定（中心角[deg], 幅[deg]）
_DEFAULT_ZONE_GEOM = {"FrFR": (0.0, 30.0), "FrLH": (45.0, 30.0), "FrRH": (-45.0, 30.0),
                      "RrLH": (90.0, 30.0), "RrRH": (-90.0, 30.0)}
# TMINI(400点): ZONE_INDEX は 50 点=45° 幅で隙間なく配置
_TMINI_ZONE_GEOM = {k: (c, 45.0) for k, (c, _) in _DEFAULT_ZONE_GEOM.items()}


def zone_readings(annotation):
    """annotations[idx] から測距ゾーン値を読む → (prefix, {zone: mm}) / (None, {})。"""
    if not annotation:
        return None, {}
    for prefix in ZONE_PREFIXES:
        vals = {}
        for z in ZONE_NAMES:
            v = annotation.get(f"{prefix}/{z}")
            if isinstance(v, (int, float)):
                vals[z] = float(v)
        if vals:
            return prefix, vals
    return None, {}


def zone_geometry(prefix, meta):
    """ゾーン名 → (中心角[rad], 幅[rad])。角度はロボット座標（0=前方, CCW 正=左）。

    lidar かつ UST20（1081点）は config の ZONE_INDEX から角度規約（meta）で
    変換する（monitor.py の _compute_zone_angles と同じ考え方）。それ以外は
    monitor.py の既定配置（前/±45°/±90°）。
    """
    n = int(meta.get("data_points") or 0)
    geom = {}
    if prefix == "lidar" and n == 1081:
        ang = scan_angles(n, meta)
        for z, (lo, hi) in _UST20_ZONE_INDEX.items():
            lo_i, hi_i = max(0, lo), min(n - 1, hi - 1)
            a0, a1 = float(ang[lo_i]), float(ang[hi_i])
            geom[z] = ((a0 + a1) / 2.0, abs(a1 - a0))
        return geom
    table = _TMINI_ZONE_GEOM if (prefix == "lidar" and n == 400) else _DEFAULT_ZONE_GEOM
    return {z: (math.radians(c), math.radians(w)) for z, (c, w) in table.items()}


def zone_color_rgb(distance_mm):
    """monitor.py getDistanceColor と同じ閾値（<300 赤 / <600 黄 / 緑）。"""
    if distance_mm < 300:
        return (255, 51, 51)
    if distance_mm < 600:
        return (255, 204, 0)
    return (51, 200, 90)
