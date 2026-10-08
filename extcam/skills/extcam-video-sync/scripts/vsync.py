#!/usr/bin/env python3
"""任意のカメラ（スマホ・手持ち可）で撮ったコースの動画を、走行記録（自己位置）と時刻合わせする。

依存: numpy, opencv-python(-headless), scipy。リポジトリにもネットにも依存しない（Claude.ai でも動く）。

    python3 vsync.py prepare  VIDEO --work W                  # フレーム時刻・手ぶれ補正・基準フレーム・メタデータの手がかり
    python3 vsync.py mask     --work W [--ignore-top 0.4] [--ignore-rect x0,y0,x1,y1 ...]   # 柵（赤・白）のマスク
    python3 vsync.py calibrate --work W --bounds map_bounds.json [--cam-box x0,x1,y0,y1]   # 柵 × 外形で自動較正
    python3 vsync.py detect   --work W                          # 動く車の画像位置（コースの範囲だけ）
    python3 vsync.py sync     --work W --runs RUN_or_CSV ...    # 走行ごとに時刻差を 1 次元で探す
    python3 vsync.py verify   --work W --run RUN_or_CSV --offset T   # 通過時刻の照合と重ね描き

走行の入力は、togikaidrive の走行フォルダ（catalog_*.catalog を読む）か、export_runs.py が作る
<名前>.traj.csv（t,x,y,theta）。t は車両 Jetson の UNIX 秒。結果の offset は「動画の t=0 の Jetson 時刻」。

座標: 地図は ROS の map フレーム [m]。車の位置は base_link（後軸）から前へ --fwd、高さ --car-h の点
（画像で検出する塊の中心に近い所）として投影する。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dtm
import glob
import json
import math
import os
import re
import struct
import sys

import cv2
import numpy as np

PROC_W = 640                       # 手ぶれ補正・検出は縮小して行う（座標は元の解像度で保存）


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------

def _jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ts(t):
    return dtm.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def mp4_creation_time(path):
    """mp4/mov の mvhd の作成時刻（UTC の UNIX 秒）。書き出し時刻のことが多いので候補を絞るヒントにだけ使う。"""
    try:
        with open(path, "rb") as f:
            data = f.read(8 * 1024 * 1024)
        k = data.find(b"mvhd")
        if k < 0:
            with open(path, "rb") as f:
                f.seek(max(0, os.path.getsize(path) - 8 * 1024 * 1024))
                data = f.read()
            k = data.find(b"mvhd")
        if k < 0:
            return None
        ver = data[k + 4]
        sec = struct.unpack(">Q", data[k + 8:k + 16])[0] if ver == 1 else struct.unpack(">I", data[k + 8:k + 12])[0]
        return sec - 2082844800 if sec > 2082844800 else None
    except OSError:
        return None


def quicktime_creationdate(path):
    """iPhone 等の com.apple.quicktime.creationdate（撮影開始、タイムゾーン付き、秒単位）。

    オリジナルの .MOV には入っていて撮影時刻として信用できる（2026-10-08 検証: 映像から求めた時刻と秒で一致）。
    共有・書き出しした mp4 では消えることが多い。moov は末尾にあることが多いので先頭と末尾を読む。
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(16 * 1024 * 1024)
            f.seek(max(0, size - 16 * 1024 * 1024))
            tail = f.read()
    except OSError:
        return None
    for blob in (tail, head):
        k = blob.find(b"com.apple.quicktime.creationdate")
        if k < 0:
            continue
        m = re.search(rb"(20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)([+-]\d\d:?\d\d|Z)", blob[k:k + 4096])
        if m:
            s = m.group(1).decode() + m.group(2).decode().replace("Z", "+0000")
            s = re.sub(r"([+-]\d\d):(\d\d)$", r"\1\2", s)
            fmt = "%Y-%m-%dT%H:%M:%S.%f%z" if "." in s else "%Y-%m-%dT%H:%M:%S%z"
            return dtm.datetime.strptime(s, fmt).timestamp()
    return None


def filename_time_hint(path):
    """ファイル名の数値を時刻として読めるか（Apple の基準時刻 2001-01-01 からの秒・UNIX 秒・UNIX ミリ秒）。"""
    m = re.search(r"(\d{9,13}(?:\.\d+)?)", os.path.basename(path))
    if not m:
        return []
    v = float(m.group(1))
    out = []
    for name, t in (("cf_absolute", v + 978307200), ("unix", v), ("unix_ms", v / 1000)):
        if 1.5e9 < t < 2.2e9:
            out.append({"kind": name, "t": t, "local": _ts(t)})
    return out


# ---------------------------------------------------------------------------
# 走行（自己位置）
# ---------------------------------------------------------------------------

POSE_SOURCES = ("fused", "slam", "aruco", "vslam", "pose")


def load_traj(src):
    """走行フォルダ or traj.csv → (名前, T[s], X[N,3]=x,y,theta)。"""
    if os.path.isdir(src):
        rows = []
        cats = sorted(glob.glob(os.path.join(src, "catalog_*.catalog")),
                      key=lambda s: int(re.findall(r"(\d+)\.catalog$", s)[0]))
        for p in cats:
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
        src_key = next((s for s in POSE_SOURCES if any(f"{s}/x" in r for r in rows[:200])), None)
        T, X = [], []
        for r in rows:
            if src_key and f"{src_key}/x" in r and r.get(f"{src_key}/status", "ok") == "ok" \
                    and r.get("_timestamp_ms") is not None:
                T.append(r["_timestamp_ms"] / 1000.0)
                X.append((r[f"{src_key}/x"], r[f"{src_key}/y"], r[f"{src_key}/theta"]))
        name = os.path.basename(os.path.normpath(src))
    else:
        T, X = [], []
        with open(src, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                T.append(float(r["t"]))
                X.append((float(r["x"]), float(r["y"]), float(r["theta"])))
        name = os.path.basename(src).replace(".traj.csv", "")
    if not T:
        return name, np.zeros(0), np.zeros((0, 3))
    o = np.argsort(T)
    return name, np.asarray(T)[o], np.asarray(X, float)[o]


def _is_run(p):
    return (os.path.isdir(p) and bool(glob.glob(os.path.join(p, "catalog_*.catalog")))) or \
        (os.path.isfile(p) and p.endswith(".traj.csv"))


def expand_runs(items):
    """走行フォルダ（catalog あり）/ *.traj.csv / それらを含むフォルダ / glob。zip などは飛ばす。"""
    out = []
    for it in items:
        cands = sorted(glob.glob(it)) or [it]
        for c in cands:
            if _is_run(c):
                out.append(c)
            elif os.path.isdir(c):
                out += [p for p in sorted(glob.glob(os.path.join(c, "data_*")) + glob.glob(os.path.join(c, "*.traj.csv")))
                        if _is_run(p)]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# カメラ
# ---------------------------------------------------------------------------

class Cam:
    def __init__(self, K, R, t, dist=None):
        self.K, self.R, self.t = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float).ravel()
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()

    @classmethod
    def load(cls, path):
        d = _jload(path)
        return cls(d["K"], d["R"], d["t"], d.get("dist"))

    def project(self, P):
        P = np.asarray(P, float)
        pc = P @ self.R.T + self.t
        rv, _ = cv2.Rodrigues(self.R)
        uv, _ = cv2.projectPoints(P.reshape(-1, 1, 3), rv, self.t, self.K, self.dist)
        uv = uv.reshape(-1, 2)
        uv[pc[:, 2] <= 0.05] = np.nan
        return uv

    def backproject(self, uv, z):
        n = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        C = -self.R.T @ self.t
        d = np.hstack([n, np.ones((len(n), 1))]) @ self.R
        s = (z - C[2]) / d[:, 2]
        return C[:2] + s[:, None] * d[:, :2]


def car_points(T, X, ts, fwd, h):
    th = np.unwrap(X[:, 2])
    x, y, a = np.interp(ts, T, X[:, 0]), np.interp(ts, T, X[:, 1]), np.interp(ts, T, th)
    return np.stack([x + fwd * np.cos(a), y + fwd * np.sin(a), np.full(len(x), h)], 1)


def outline_points(bounds, z=0.05, step=0.05, outer_only=False):
    polys = [bounds["outer"]] + ([] if outer_only else list(bounds.get("holes") or []))
    pts = []
    for poly in polys:
        p = np.array(list(poly) + [poly[0]], float)
        for k in range(len(p) - 1):
            n = max(2, int(np.linalg.norm(p[k + 1] - p[k]) / step))
            for s in np.linspace(0, 1, n, endpoint=False):
                pts.append([*(p[k] + (p[k + 1] - p[k]) * s), z])
    return np.array(pts)


# ---------------------------------------------------------------------------
# prepare: フレーム時刻・手ぶれ補正
# ---------------------------------------------------------------------------

def cmd_prepare(a):
    os.makedirs(a.work, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"動画を開けません: {a.video}（HEVC なら ffmpeg で H.264 に変換）")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    small, ts = [], []
    full_size = None
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if full_size is None:
            full_size = (f.shape[1], f.shape[0])
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        small.append(cv2.resize(f, (PROC_W, int(PROC_W * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA))
        i += 1
    n = len(small)
    if n < 10:
        sys.exit("フレームが少なすぎます")
    # OpenCV の時刻は最後のフレームで 0 に戻ることがある → 単調でない所はフレーム番号/fps で埋める
    ts = np.array(ts)
    bad = np.r_[False, np.diff(ts) <= 0] | ~np.isfinite(ts)
    if bad.mean() > 0.2:
        ts = np.arange(n) / fps
    else:
        ts[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], ts[~bad])
    ts = ts - ts[0]
    scale = full_size[0] / PROC_W
    ref_i = int(a.ref) if a.ref is not None else n // 2
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    g_ref = cv2.cvtColor(small[ref_i], cv2.COLOR_BGR2GRAY)
    kr, dr = orb.detectAndCompute(g_ref, None)
    Hs, inl = [], []
    for f in small:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        k, d = orb.detectAndCompute(g, None)
        H, ni = None, 0
        if d is not None and len(k) > 50:
            m = [p[0] for p in bf.knnMatch(d, dr, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(m) > 30:
                src = np.float32([k[q.queryIdx].pt for q in m])
                dst = np.float32([kr[q.trainIdx].pt for q in m])
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
                ni = int(mask.sum()) if mask is not None else 0
        Hs.append(H if H is not None and ni >= 40 else None)
        inl.append(ni)
    # 縮小画像の H を元解像度へ: H_full = S H S^-1
    S = np.diag([scale, scale, 1.0])
    Hfull = np.array([S @ H @ np.linalg.inv(S) if H is not None else np.full((3, 3), np.nan) for H in Hs])
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_i)
    ok, ref_full = cap.read()
    if not ok:
        ref_full = cv2.resize(small[ref_i], full_size)
    cv2.imwrite(os.path.join(a.work, "ref.jpg"), ref_full)
    np.savez_compressed(os.path.join(a.work, "frames.npz"), small=np.array(small), ts=ts, H=Hfull)
    # 確認用の一覧（9 枚）
    idx = np.linspace(0, n - 1, 9).astype(int)
    tiles = [cv2.putText(cv2.resize(small[j], (426, 240)), f"t={ts[j]:.1f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                         0.8, (0, 255, 255), 2) for j in idx]
    sheet = np.vstack([np.hstack(tiles[k:k + 3]) for k in (0, 3, 6)])
    cv2.imwrite(os.path.join(a.work, "contact.jpg"), sheet)
    hints = {"quicktime_creationdate": None, "mp4_creation_time": None, "filename": filename_time_hint(a.video)}
    qt = quicktime_creationdate(a.video)
    if qt:
        hints["quicktime_creationdate"] = {"t": qt, "local": _ts(qt),
                                           "note": "撮影開始（秒単位）。sync --around にそのまま使える"}
    ct = mp4_creation_time(a.video)
    if ct:
        hints["mp4_creation_time"] = {"t": ct, "local": _ts(ct)}
    meta = {"video": os.path.abspath(a.video), "n_frames": n, "duration_s": float(ts[-1]), "fps": fps,
            "size": list(full_size), "ref_index": ref_i, "stab_fail": int(sum(H is None for H in Hs)),
            "stab_inliers_min": int(min(inl)), "stab_inliers_median": float(np.median(inl)), "time_hints": hints}
    _jdump(meta, os.path.join(a.work, "video.json"))
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"→ {a.work}/contact.jpg（どんな画か）と ref.jpg（基準フレーム）を見る")


# ---------------------------------------------------------------------------
# mask: 柵（赤・白）
# ---------------------------------------------------------------------------

def _parse_rects(rs):
    return [tuple(int(float(v)) for v in r.split(",")) for r in rs or []]


def sdr_from_hlg(img, gamma=1.6, sat=1.6):
    """iPhone の HDR（HLG・Dolby Vision）を tone map せずに 8 bit で読むと、白っぽく彩度が浅い。
    柵の色の判定用に、ガンマで中間調を沈め、彩度を持ち上げる（見た目の正確さは要らない）。"""
    f = (img.astype(np.float32) / 255.0) ** gamma
    hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def barrier_mask(img, ignore_top, rects, red=True, white=True, hdr=False):
    if hdr:
        img = sdr_from_hlg(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    if red:
        m |= cv2.inRange(hsv, (0, 90, 70), (10, 255, 255)) | cv2.inRange(hsv, (165, 90, 70), (180, 255, 255))
    if white:
        m |= cv2.inRange(hsv, (0, 0, 170), (180, 40, 255))
    m[: int(img.shape[0] * ignore_top)] = 0
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = 0
    return m


def cmd_mask(a):
    ref = cv2.imread(os.path.join(a.work, "ref.jpg"))
    m = barrier_mask(ref, a.ignore_top, _parse_rects(a.ignore_rect), not a.no_red, not a.no_white, a.hdr)
    cv2.imwrite(os.path.join(a.work, "barrier.png"), m)
    vis = ref.copy()
    vis[m > 0] = (0, 255, 0)
    cv2.imwrite(os.path.join(a.work, "barrier_vis.jpg"), cv2.resize(vis, (960, int(960 * ref.shape[0] / ref.shape[1]))))
    cfg = {"ignore_top": a.ignore_top, "ignore_rect": a.ignore_rect or [], "hdr": a.hdr}
    _jdump(cfg, os.path.join(a.work, "mask.json"))
    print(f"柵の画素 {int((m > 0).sum())} → {a.work}/barrier_vis.jpg を見て、天井・壁・机・床の白線が多く混ざっていれば"
          " --ignore-top / --ignore-rect で除く")


# ---------------------------------------------------------------------------
# calibrate: 外形 × 柵の重なりでカメラを推定
# ---------------------------------------------------------------------------

def _cam_from(p, W, H):
    cx, cy, cz, tx, ty, roll, f, k1 = p
    C = np.array([cx, cy, cz])
    z = np.array([tx, ty, 0.0]) - C
    z /= np.linalg.norm(z)
    xr = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(xr) < 1e-6:
        return None
    xr /= np.linalg.norm(xr)
    yr = np.cross(z, xr)
    cr, sr = math.cos(roll), math.sin(roll)
    R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.vstack([xr, yr, z])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return Cam(K, R, -R @ C, [k1, 0, 0, 0, 0])


def cmd_calibrate(a):
    from scipy.optimize import minimize
    ref = cv2.imread(os.path.join(a.work, "ref.jpg"))
    H, W = ref.shape[:2]
    barrier = cv2.imread(os.path.join(a.work, "barrier.png"), 0)
    if barrier is None:
        sys.exit("先に mask を実行")
    dist = cv2.distanceTransform(255 - barrier, cv2.DIST_L2, 5)
    bpix = np.argwhere(barrier > 0)[::7][:, ::-1]
    bb = np.array([*bpix.min(0), *bpix.max(0)], float)      # 柵の外接矩形 [u0, v0, u1, v1]
    bounds = _jload(a.bounds)
    OUT = outline_points(bounds, z=a.barrier_z)
    allxy = np.array(list(bounds["outer"]), float)
    x0, y0 = allxy.min(0)
    x1, y1 = allxy.max(0)
    span = max(x1 - x0, y1 - y0)
    if a.cam_box:
        bx0, bx1, by0, by1 = map(float, a.cam_box.split(","))
    else:
        bx0, bx1, by0, by1 = x0 - 0.4 * span, x1 + 0.4 * span, y0 - 0.4 * span, y1 + 0.4 * span
    top = int(H * (_jload(os.path.join(a.work, "mask.json")).get("ignore_top", 0.0)))

    def cost(p, detail=False):
        cam = _cam_from(p, W, H)
        if cam is None or not (a.f_min * 0.7 < p[6] < a.f_max * 1.3) or abs(p[7]) > 0.5 or p[2] < 0.3:
            return 1e3
        uv = cam.project(OUT)
        ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= top) & (uv[:, 1] < H - 1)
        if ok.sum() < 150:
            return 1e3
        q = uv[ok]
        d1 = float(np.mean(np.minimum(dist[q[:, 1].astype(int), q[:, 0].astype(int)], 30)))
        m = np.zeros((H, W), np.uint8)
        m[q[:, 1].astype(int), q[:, 0].astype(int)] = 255
        d2map = cv2.distanceTransform(255 - m, cv2.DIST_L2, 3)
        d2 = float(np.mean(np.minimum(d2map[bpix[:, 1], bpix[:, 0]], 30)))   # 柵 → 外形（潰れた解を防ぐ）
        ob = np.array([*q.min(0), *q.max(0)])
        iw = max(0.0, min(ob[2], bb[2]) - max(ob[0], bb[0]))
        ih = max(0.0, min(ob[3], bb[3]) - max(ob[1], bb[1]))
        ua = (ob[2] - ob[0]) * (ob[3] - ob[1]) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - iw * ih
        iou = iw * ih / max(ua, 1.0)                                         # 外形と柵の広がりが合っているか
        if detail:
            return d1, d2, int(ok.sum()), iou
        return d1 + d2 + 30.0 * (1.0 - iou)

    rng = np.random.default_rng(a.seed)
    cands = []
    for _ in range(a.samples):
        # カメラはコースの外（箱の中で外形の外側）、注視点はコースの中
        while True:
            cx, cy = rng.uniform(bx0, bx1), rng.uniform(by0, by1)
            if cv2.pointPolygonTest(allxy.astype(np.float32), (float(cx), float(cy)), False) < 0 or a.cam_inside:
                break
        p = [cx, cy, rng.uniform(a.h_min, a.h_max), rng.uniform(x0, x1), rng.uniform(y0, y1),
             rng.uniform(-0.1, 0.1), rng.uniform(a.f_min, a.f_max), 0.0]
        cands.append((cost(p), p))
    if a.init:
        ini = _jload(a.init)
        if "params" not in ini:
            sys.exit(f"{a.init} に params がありません（この版の calibrate が書いた calib.json を使う）")
        p0 = list(ini["params"])
        w0 = (ini.get("image_size") or [W, H])[0]
        p0[6] *= W / float(w0)                      # 解像度が違えば焦点距離を換算
        cands.append((cost(p0) - 1e-6, p0))         # 初期値を必ず詰める候補に入れる
    cands.sort(key=lambda c: c[0])
    res = []
    for c0, p0 in cands[: a.starts]:
        r = minimize(cost, p0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-3})
        res.append((r.fun, list(r.x)))
    res.sort(key=lambda c: c[0])
    best = res[0][1]
    d1, d2, n, iou = cost(best, detail=True)
    cam = _cam_from(best, W, H)
    out = {"K": cam.K.tolist(), "R": cam.R.tolist(), "t": cam.t.tolist(), "dist": cam.dist.tolist(),
           "params": [float(v) for v in best],
           "camera_position_m": [round(v, 3) for v in best[:3]], "f_px": round(best[6], 1), "k1": round(best[7], 4),
           "outline_to_barrier_px": round(d1, 2), "barrier_to_outline_px": round(d2, 2), "n_outline_px": n,
           "extent_iou": round(iou, 3), "cost": round(res[0][0], 2),
           "runners_up": [round(r[0], 2) for r in res[1:5]], "image_size": [W, H], "bounds": os.path.abspath(a.bounds)}
    _jdump(out, os.path.join(a.work, "calib.json"))
    vis = ref.copy()
    uv = cam.project(OUT)
    for p in uv[np.isfinite(uv).all(1)]:
        if 0 <= p[0] < W and 0 <= p[1] < H:
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(a.work, "calib_overlay.jpg"), cv2.resize(vis, (960, int(960 * H / W))))
    print(json.dumps({k: out[k] for k in ("camera_position_m", "f_px", "k1", "outline_to_barrier_px",
                                          "barrier_to_outline_px", "extent_iou", "cost", "runners_up")},
                     ensure_ascii=False))
    print(f"→ {a.work}/calib_overlay.jpg を見て、緑の外形が柵に重なっているか確認する（ずれていれば --cam-box で範囲を絞る）")


def cmd_mapplot(a):
    """地図の外形（頂点番号つき）と走行の軌跡を描く。カメラがコースのどちら側にいるかを決める材料。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = _jload(a.bounds)
    fig, ax = plt.subplots(figsize=(8, 8))
    for pi, poly in enumerate([b["outer"]] + list(b.get("holes") or [])):
        p = np.array(list(poly) + [poly[0]], float)
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.5, color="C%d" % pi)
        for j, q in enumerate(poly):
            ax.annotate(f"{pi}.{j}", q, fontsize=7, xytext=(2, 2), textcoords="offset points", color="C%d" % pi)
    for src in a.runs or []:
        name, T, X = load_traj(src)
        if len(T):
            ax.plot(X[:, 0], X[:, 1], "-", lw=0.6, alpha=0.6, label=name)
    if os.path.isfile(os.path.join(a.work or "", "calib.json")):
        c = _jload(os.path.join(a.work, "calib.json"))["camera_position_m"]
        ax.plot(c[0], c[1], "r^", ms=12)
        ax.annotate("camera", c[:2], color="r")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if a.runs:
        ax.legend(fontsize=7)
    out = a.out or os.path.join(a.work or ".", "map.png")
    fig.savefig(out, dpi=90, bbox_inches="tight")
    print(f"→ {out}（x 右・y 上の地図。基準フレームと見比べ、カメラのいる側を --cam-box x0,x1,y0,y1 で与える）")


# ---------------------------------------------------------------------------
# detect: 動く車（背景差分、コースの範囲だけ）
# ---------------------------------------------------------------------------

def cmd_detect(a):
    z = np.load(os.path.join(a.work, "frames.npz"))
    small, Hfull, ts = z["small"], z["H"], z["ts"]
    meta = _jload(os.path.join(a.work, "video.json"))
    W, H = meta["size"]
    sw, sh = small.shape[2], small.shape[1]
    scale = W / sw
    S = np.diag([scale, scale, 1.0])
    Hs = [None if np.isnan(h).any() else np.linalg.inv(S) @ h @ S for h in Hfull]
    cam = Cam.load(os.path.join(a.work, "calib.json"))
    bounds = _jload(_jload(os.path.join(a.work, "calib.json"))["bounds"])
    o = outline_points(bounds, z=0.0, outer_only=True)
    uv = cam.project(o)
    uv = uv[np.isfinite(uv).all(1)]
    course = np.zeros((H, W), np.uint8)
    cv2.fillPoly(course, [uv.astype(np.int32)], 255)
    course = cv2.resize(cv2.erode(course, np.ones((7, 7), np.uint8)), (sw, sh))
    warped = [cv2.warpPerspective(f, h, (sw, sh)) if h is not None else None for f, h in zip(small, Hs)]
    stack = np.array([w for w in warped[::5] if w is not None])
    bg = np.median(stack, axis=0).astype(np.uint8)
    cents, prev = [], None
    for i, w in enumerate(warped):
        if w is None:
            cents.append(None)
            continue
        valid = cv2.erode(cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), Hs[i], (sw, sh)), np.ones((9, 9), np.uint8))
        d = cv2.absdiff(w, bg).max(axis=2)
        d[(valid == 0) | (course == 0)] = 0
        _, m = cv2.threshold(d, a.diff_thr, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        nl, _, st, cc = cv2.connectedComponentsWithStats(m)
        best = None
        for j in range(1, nl):
            ar = st[j, cv2.CC_STAT_AREA]
            if ar < a.min_area or ar > a.max_area:
                continue
            sc = ar - (0 if prev is None else 3 * float(np.hypot(*(cc[j] - prev))))
            if best is None or sc > best[0]:
                best = (sc, cc[j], ar)
        if best:
            prev = best[1]
            cents.append([float(best[1][0] * scale), float(best[1][1] * scale), int(best[2])])
        else:
            cents.append(None)
    _jdump({"ts": ts.tolist(), "cents": cents}, os.path.join(a.work, "det.json"))
    ref = cv2.imread(os.path.join(a.work, "ref.jpg"))
    for i, c in enumerate(cents):
        if c:
            col = cv2.applyColorMap(np.uint8([[int(255 * i / len(cents))]]), cv2.COLORMAP_JET)[0, 0].tolist()
            cv2.circle(ref, (int(c[0]), int(c[1])), 5, col, -1)
    cv2.imwrite(os.path.join(a.work, "det_track.jpg"), cv2.resize(ref, (960, int(960 * H / W))))
    nd = sum(c is not None for c in cents)
    print(f"検出 {nd}/{len(cents)} フレーム → {a.work}/det_track.jpg（色=時刻、青→赤）で車の通り道になっているか確認する")


# ---------------------------------------------------------------------------
# sync / verify
# ---------------------------------------------------------------------------

def _load_det(work):
    d = _jload(os.path.join(work, "det.json"))
    ts = np.array(d["ts"])
    det = np.array([c is not None for c in d["cents"]])
    uv = np.array([c[:2] if c else [np.nan, np.nan] for c in d["cents"]], float)
    return ts, det, uv


def _visible_map(det, uv, shape, r=35):
    vis = np.zeros(shape, np.uint8)
    for ok, p in zip(det, uv):
        if ok:
            cv2.circle(vis, (int(p[0]), int(p[1])), r, 255, -1)
    return vis


def _score(cam, T, X, ts, det, uv, vis, off, fwd, h, tol):
    tt = ts + off
    ok = (tt >= T[0]) & (tt <= T[-1])
    if ok.mean() < 0.6:
        return None
    pr = np.full((len(ts), 2), np.nan)
    pr[ok] = cam.project(car_points(T, X, tt[ok], fwd, h))
    Hh, Ww = vis.shape
    inb = ok & np.isfinite(pr).all(1) & (pr[:, 0] >= 0) & (pr[:, 0] < Ww) & (pr[:, 1] >= 0) & (pr[:, 1] < Hh)
    px = np.where(inb, pr[:, 0], 0).astype(int)
    py = np.where(inb, pr[:, 1], 0).astype(int)
    expect = inb & (vis[py, px] > 0)
    e = np.linalg.norm(pr - uv, axis=1)
    hit = det & ok & (e < tol)
    tot = (det & ok).sum() + expect.sum()
    med = float(np.nanmedian(e[det & ok])) if (det & ok).any() else 1e9
    return {"F": float(2 * hit.sum() / max(tot, 1)), "hit": int(hit.sum()), "det": int((det & ok).sum()),
            "expected": int(expect.sum()), "miss": int((expect & ~det).sum()), "med_px": med}


def cmd_sync(a):
    cam = Cam.load(os.path.join(a.work, "calib.json"))
    ts, det, uv = _load_det(a.work)
    meta = _jload(os.path.join(a.work, "video.json"))
    vis = _visible_map(det, uv, (meta["size"][1], meta["size"][0]))
    dur = ts[-1]
    rows = []
    for src in expand_runs(a.runs):
        name, T, X = load_traj(src)
        if len(T) < 50:
            continue
        lo, hi = T[0] - dur * 0.4, T[-1] - dur * 0.6
        if a.around is not None:
            lo, hi = max(lo, a.around - a.window), min(hi, a.around + a.window)
        best = None
        for off in np.arange(lo, hi, a.step):
            s = _score(cam, T, X, ts, det, uv, vis, off, a.fwd, a.car_h, a.tol)
            if s and (best is None or s["F"] > best[1]["F"]):
                best = (off, s)
        if best:
            rows.append({"run": os.path.abspath(src), "name": name, "offset": best[0], **best[1]})
    if not rows:
        sys.exit("どの走行とも重なりません（--runs と動画の長さを確認）")
    rows.sort(key=lambda r: -r["F"])
    top = rows[0]
    name, T, X = load_traj(top["run"])
    fine = []
    for off in np.arange(top["offset"] - 0.3, top["offset"] + 0.3, 1 / 120):
        s = _score(cam, T, X, ts, det, uv, vis, off, a.fwd, a.car_h, a.tol)
        if s:
            fine.append((s["F"], -s["med_px"], off, s))
    fine.sort(reverse=True)
    off = fine[0][2]
    curve = []
    for d in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
        s = _score(cam, T, X, ts, det, uv, vis, off + d, a.fwd, a.car_h, a.tol)
        curve.append({"d_s": d, "med_px": round(s["med_px"], 1) if s else None})
    out = {"best": {"run": top["run"], "name": name, "offset": off, "video_t0_local": _ts(off), **fine[0][3]},
           "margin_F_vs_2nd": round(top["F"] - rows[1]["F"], 3) if len(rows) > 1 else None,
           "ranking": [{k: (round(v, 3) if isinstance(v, float) and k != "offset" else v) for k, v in r.items()}
                       for r in rows[:8]], "offset_curve": curve}
    _jdump(out, os.path.join(a.work, "sync.json"))
    print(json.dumps({"best": out["best"], "margin_F_vs_2nd": out["margin_F_vs_2nd"],
                      "top5": [(r["name"], round(r["F"], 3), round(r["offset"], 2)) for r in rows[:5]],
                      "offset_curve": curve}, ensure_ascii=False, indent=1))
    print("→ 次に verify で通過時刻の照合と重ね描きを確認する（2 位との差が小さいときは必須）")


def _passes(mask, ts, min_len=5):
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        if not m and s is not None:
            if i - s > min_len:
                out.append([round(float(ts[s]), 2), round(float(ts[min(i, len(ts) - 1)]), 2)])
            s = None
    return out


def cmd_verify(a):
    cam = Cam.load(os.path.join(a.work, "calib.json"))
    ts, det, uv = _load_det(a.work)
    meta = _jload(os.path.join(a.work, "video.json"))
    W, H = meta["size"]
    name, T, X = load_traj(a.run)
    off = float(a.offset)
    pr = cam.project(car_points(T, X, ts + off, a.fwd, a.car_h))
    # 検出が多く出る帯（見通しのよいレーン）を自動で選び、そこを通った時刻を比べる
    vis = _visible_map(det, uv, (H, W), r=25)
    inb = np.isfinite(pr).all(1) & (pr[:, 0] >= 0) & (pr[:, 0] < W) & (pr[:, 1] >= 0) & (pr[:, 1] < H)
    px, py = np.where(inb, pr[:, 0], 0).astype(int), np.where(inb, pr[:, 1], 0).astype(int)
    exp = inb & (vis[py, px] > 0)
    p_det, p_prj = _passes(det, ts), _passes(exp, ts)
    pairs = []
    for s, e in p_det:
        if p_prj:
            j = int(np.argmin([abs(s - q[0]) for q in p_prj]))
            pairs.append({"video": [s, e], "projected": p_prj[j], "d_start_s": round(p_prj[j][0] - s, 2)})
    m = det & np.isfinite(pr).all(1)
    e = np.linalg.norm(pr[m] - uv[m], axis=1)
    gm = np.linalg.norm(cam.backproject(pr[m], a.car_h) - cam.backproject(uv[m], a.car_h), axis=1)
    res = {"run": name, "run_path": os.path.abspath(a.run), "offset": off, "video_t0_local": _ts(off), "passes": pairs,
           "px_err": {k: round(float(np.percentile(e, q)), 1) for k, q in (("p50", 50), ("p75", 75), ("p90", 90))},
           "ground_err_m": {k: round(float(np.percentile(gm, q)), 2) for k, q in (("p50", 50), ("p75", 75), ("p90", 90))}}
    _jdump(res, os.path.join(a.work, "verify.json"))
    ref = cv2.imread(os.path.join(a.work, "ref.jpg"))
    for i in range(0, len(ts), 2):
        col = cv2.applyColorMap(np.uint8([[int(255 * i / len(ts))]]), cv2.COLORMAP_JET)[0, 0].tolist()
        if inb[i]:
            cv2.drawMarker(ref, (int(pr[i, 0]), int(pr[i, 1])), col, cv2.MARKER_TILTED_CROSS, 10, 2)
        if det[i]:
            cv2.circle(ref, (int(uv[i, 0]), int(uv[i, 1])), 5, col, -1)
    cv2.imwrite(os.path.join(a.work, "verify_overlay.jpg"), cv2.resize(ref, (960, int(960 * H / W))))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"→ {a.work}/verify_overlay.jpg: ●=検出 ×=自己位置の投影（同じ色=同じ時刻）。通過時刻の差 d_start_s が ±0.2 s 以内なら一致")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("video")
    p.add_argument("--work", required=True)
    p.add_argument("--ref", type=int, default=None, help="基準フレーム（既定は中央）")
    p = sub.add_parser("mask")
    p.add_argument("--work", required=True)
    p.add_argument("--ignore-top", type=float, default=0.4, help="上からこの割合を除く（天井・壁）")
    p.add_argument("--ignore-rect", action="append", help="x0,y0,x1,y1（元の解像度）を除く。複数可")
    p.add_argument("--no-red", action="store_true")
    p.add_argument("--no-white", action="store_true")
    p.add_argument("--hdr", action="store_true", help="iPhone の HDR 動画（HLG / Dolby Vision、白っぽく写る）の色を戻してから判定")
    p = sub.add_parser("calibrate")
    p.add_argument("--work", required=True)
    p.add_argument("--bounds", required=True, help="map_bounds.json（outer / holes、map フレーム）")
    p.add_argument("--cam-box", default=None, help="カメラ位置の探索範囲 x0,x1,y0,y1 [m]")
    p.add_argument("--cam-inside", action="store_true", help="カメラがコースの内側にあってもよい")
    p.add_argument("--h-min", type=float, default=1.0)
    p.add_argument("--h-max", type=float, default=3.5)
    p.add_argument("--f-min", type=float, default=400)
    p.add_argument("--f-max", type=float, default=1500)
    p.add_argument("--barrier-z", type=float, default=0.05, help="柵の見える帯の高さ [m]")
    p.add_argument("--init", default=None, help="前回の calib.json を初期値にする（同じ場所から撮った動画・解像度違いの同じ動画）")
    p.add_argument("--samples", type=int, default=6000)
    p.add_argument("--starts", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("mapplot")
    p.add_argument("--bounds", required=True)
    p.add_argument("--work", default=None)
    p.add_argument("--runs", nargs="*")
    p.add_argument("--out", default=None)
    p = sub.add_parser("detect")
    p.add_argument("--work", required=True)
    p.add_argument("--diff-thr", type=int, default=40)
    p.add_argument("--min-area", type=int, default=25)
    p.add_argument("--max-area", type=int, default=4000)
    for name in ("sync", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--fwd", type=float, default=0.13, help="後軸から車体中心まで [m]")
        p.add_argument("--car-h", type=float, default=0.06, help="車体中心の高さ [m]")
        if name == "sync":
            p.add_argument("--runs", nargs="+", required=True, help="走行フォルダ / *.traj.csv / それらを含むフォルダ")
            p.add_argument("--step", type=float, default=1 / 15)
            p.add_argument("--tol", type=float, default=40.0, help="一致とみなす距離 [px]")
            p.add_argument("--around", type=float, default=None, help="時刻の見当（UNIX 秒）があれば ±window だけ探す")
            p.add_argument("--window", type=float, default=600.0)
        else:
            p.add_argument("--run", required=True)
            p.add_argument("--offset", required=True)
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "mask": cmd_mask, "calibrate": cmd_calibrate, "mapplot": cmd_mapplot,
     "detect": cmd_detect,
     "sync": cmd_sync, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
