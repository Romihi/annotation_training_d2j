"""YOLO11n-pose ＋ BoT-SORT で clip を追跡し、地図座標の位置・向きへ変換して平滑化する。Qt 非依存。

    python -m extcam.tracker run      data/data_<TS> --clip 0 --model best.pt   # 推論 → raw → 平滑化
    python -m extcam.tracker smooth   data/data_<TS> --clip 0                   # 手直し（edits）を反映して平滑化だけ
    python -m extcam.tracker prelabel data/data_<TS> --clip 0 --from pose|model [--model best.pt] --step 30

出力（走行フォルダの extcam/）:
  track_NN.raw.csv     検出ごと: frame, t, id, conf, bbox, キーポイント, 地図座標の生の姿勢
  track_NN.csv         平滑化後: frame, t_jetson, id, x, y, yaw, v, yaw_rate, meas, outlier（base_link＝後軸中心）
  track_NN.edits.json  手直し（除外する ID・ID の統合・除外するフレーム）。GUI が書く
  track_NN.summary.json  ID ごとの件数と、車両の自己位置との比較（RMSE・向きの誤差）

BoT-SORT は ID 付けだけに使う（カメラ固定なので GMC なし、似た車どうしで効かない ReID もなし）。
位置・向きの推定は地図座標の等速モデルで行う。オフラインなので前後両方向の平滑化（RTS）を使う。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from extcam.clip import Clip                                         # noqa: E402
from extcam.clipcam import load_clip_camera                          # noqa: E402
from extcam.geometry import KPT_NAMES, kpts_from_pose, pose_from_kpts  # noqa: E402
from extcam.labels import LabelStore                                 # noqa: E402
from extcam.runlog import RunLog                                     # noqa: E402

NK = len(KPT_NAMES)


def setup_path(run_dir: str) -> str:
    """走行のカメラ（sidecam）の較正。クリップ専用の較正は clipcam.setup_path_for。"""
    return os.path.join(run_dir, "extcam", "calib.json")


def _camera(run_dir: str, clip_num: int):
    cc, veh, _, _, path = load_clip_camera(run_dir, clip_num)
    if cc is None:
        raise ValueError(f"較正がありません（{path}）")
    return cc, veh


def _paths(clip: Clip) -> dict:
    b = os.path.join(clip.ext_dir, f"track_{clip.num:02d}")
    return {"raw": b + ".raw.csv", "track": b + ".csv", "edits": b + ".edits.json",
            "summary": b + ".summary.json", "tracker_yaml": os.path.join(clip.ext_dir, "botsort_extcam.yaml")}


def write_tracker_yaml(path: str, fps: float = 60.0, high: float = 0.25, low: float = 0.1,
                       buffer_s: float = 1.0):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""tracker_type: botsort
track_high_thresh: {high}
track_low_thresh: {low}
new_track_thresh: {high}
track_buffer: {int(round(fps * buffer_s))}
match_thresh: 0.8
fuse_score: True
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: False
model: auto
""")
    return path


# ---------------------------------------------------------------------------
# 推論（YOLO → BoT-SORT）
# ---------------------------------------------------------------------------

RAW_FIELDS = (["frame", "t_jetson", "t_cam", "id", "conf", "x1", "y1", "x2", "y2"] +
              [f"k{j}{c}" for j in range(NK) for c in ("u", "v", "c")] +
              ["px", "py", "pyaw", "prms", "pn"])


def run_inference(run_dir: str, clip_num: int, model_path: str, conf: float = 0.25,
                  imgsz: int = 1280, device=None, kpt_conf: float = 0.5, progress=None,
                  max_frames: int | None = None) -> str:
    from ultralytics import YOLO
    clip = Clip(run_dir, clip_num)
    cc, veh = _camera(run_dir, clip_num)
    p = _paths(clip)
    n = len(clip)
    fps = (n - 1) / (clip.t_jetson[-1] - clip.t_jetson[0]) if n > 1 and clip.t_jetson else 60.0
    write_tracker_yaml(p["tracker_yaml"], fps=fps)
    model = YOLO(model_path)
    kw = dict(persist=True, tracker=p["tracker_yaml"], conf=conf, imgsz=imgsz, verbose=False)
    if device not in (None, ""):
        kw["device"] = device
    t0 = time.time()
    with open(p["raw"], "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(RAW_FIELDS)
        for i in range(n if max_frames is None else min(n, max_frames)):
            img = clip.read(i)
            if img is None:
                break
            res = model.track(img, **kw)[0]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            ids = boxes.id.int().tolist() if boxes.id is not None else [-1] * len(boxes)
            xyxy = boxes.xyxy.cpu().numpy()
            cf = boxes.conf.cpu().numpy()
            kp = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((len(boxes), NK, 3))
            for b in range(len(boxes)):
                k = kp[b]
                kr = k.copy()
                kr[:, :2] = cc.to_ref(k[:, :2], i)      # 手持ちなら基準フレームの画素へ（固定カメラは恒等）
                pose = pose_from_kpts(cc.calib, veh, kr, min_conf=kpt_conf)
                px = py = pyaw = prms = ""
                pn = 0
                if pose is not None:
                    px, py, pyaw, prms, pn = (round(pose[0], 4), round(pose[1], 4), round(pose[2], 4),
                                              round(pose[3], 4), pose[4])
                wr.writerow([i, f"{clip.time_of(i):.4f}" if clip.time_of(i) else "",
                             f"{clip.t_cam[i]:.4f}" if i < len(clip.t_cam) else "", ids[b],
                             round(float(cf[b]), 4), *[round(float(v), 1) for v in xyxy[b]],
                             *[round(float(v), 3) for v in k.reshape(-1)], px, py, pyaw, prms, pn])
            if progress and (i % 30 == 0 or i == n - 1):
                progress(i + 1, n, time.time() - t0)
    return p["raw"]


# ---------------------------------------------------------------------------
# 手直しと平滑化
# ---------------------------------------------------------------------------

def load_raw(path: str) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_edits(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"exclude_ids": [], "merge": {}, "exclude_frames": []}


def save_edits(path: str, edits: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(edits, f, ensure_ascii=False, indent=1)


def apply_edits(rows: list, edits: dict) -> list:
    """ID の統合（merge: {"旧": 新}）→ 除外 ID → 除外フレーム（[[id, frame], ...] か [frame, ...]）。

    除外フレームの id は**統合前の元の ID**（GUI の表は元の ID で並ぶ）。除外 ID は統合の前後どちらでも効く。
    """
    merge = {int(k): int(v) for k, v in (edits.get("merge") or {}).items()}
    excl_ids = set(int(x) for x in edits.get("exclude_ids") or [])
    excl_frames = set()
    excl_any = set()
    for e in edits.get("exclude_frames") or []:
        if isinstance(e, (list, tuple)):
            excl_frames.add((int(e[0]), int(e[1])))
        else:
            excl_any.add(int(e))
    out = []
    for r in rows:
        raw_id = tid = int(r["id"])
        for _ in range(8):          # 統合の連鎖（a→b→c）を辿る
            if tid in merge:
                tid = merge[tid]
        fr = int(r["frame"])
        if raw_id in excl_ids or tid in excl_ids or fr in excl_any or (raw_id, fr) in excl_frames:
            continue
        rr = dict(r)
        rr["id"] = tid
        out.append(rr)
    return out


def _rts_cv(t, z, R, q_acc, gate2=16.0, init_vel_var=4.0):
    """等速モデルのカルマンフィルタ＋RTS 平滑化。z: (N, d) 観測、R: (N,) 観測分散。

    返り値: (平滑化した状態 (N, 2d) [位置..., 速度...], 外れ値フラグ (N,))
    """
    n, d = z.shape
    I = np.eye(2 * d)
    xs_f = np.zeros((n, 2 * d))
    Ps_f = np.zeros((n, 2 * d, 2 * d))
    xs_p = np.zeros_like(xs_f)
    Ps_p = np.zeros_like(Ps_f)
    Fs = np.zeros((n, 2 * d, 2 * d))
    outlier = np.zeros(n, bool)
    H = np.hstack([np.eye(d), np.zeros((d, d))])
    x = np.concatenate([z[0], np.zeros(d)])
    P = np.diag([R[0]] * d + [init_vel_var] * d)
    for k in range(n):
        dt = 0.0 if k == 0 else max(1e-4, t[k] - t[k - 1])
        F = np.eye(2 * d)
        F[:d, d:] = np.eye(d) * dt
        G = np.vstack([np.eye(d) * dt * dt / 2, np.eye(d) * dt])
        Q = G @ G.T * q_acc ** 2
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        Fs[k] = F
        xs_p[k], Ps_p[k] = x, P
        Rk = np.eye(d) * R[k]
        y = z[k] - H @ x
        S = H @ P @ H.T + Rk
        m2 = float(y @ np.linalg.solve(S, y))
        if k > 0 and m2 > gate2:
            outlier[k] = True           # 観測を使わず予測のまま進める
        else:
            Kg = P @ H.T @ np.linalg.inv(S)
            x = x + Kg @ y
            P = (I - Kg @ H) @ P
        xs_f[k], Ps_f[k] = x, P
    xs = xs_f.copy()
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ Fs[k + 1].T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
    return xs, outlier


def smooth_rows(rows: list, sigma_pos: float = 0.03, sigma_yaw: float = 0.06,
                q_acc: float = 6.0, q_yaw_acc: float = 40.0, max_gap_s: float = 0.5) -> list:
    """ID ごとに、時刻の空白（max_gap_s 超）で区切った区間ごとに平滑化する。"""
    by_id = {}
    for r in rows:
        if r.get("px") in ("", None):
            continue
        by_id.setdefault(int(r["id"]), []).append(r)
    out = []
    for tid, rs in by_id.items():
        # 同じフレームに同じ ID が 2 つ来ることは無い前提（BoT-SORT）。念のため conf の高い方
        best = {}
        for r in rs:
            fr = int(r["frame"])
            if fr not in best or float(r["conf"]) > float(best[fr]["conf"]):
                best[fr] = r
        rs = [best[k] for k in sorted(best)]
        t = np.array([float(r["t_jetson"]) for r in rs])
        segs, s0 = [], 0
        for k in range(1, len(rs)):
            if t[k] - t[k - 1] > max_gap_s:
                segs.append((s0, k))
                s0 = k
        segs.append((s0, len(rs)))
        for a, b in segs:
            tt = t[a:b]
            xy = np.array([[float(r["px"]), float(r["py"])] for r in rs[a:b]])
            rms = np.array([float(r["prms"] or 0) for r in rs[a:b]])
            yaw = np.unwrap(np.array([float(r["pyaw"]) for r in rs[a:b]]))
            Rp = sigma_pos ** 2 + rms ** 2
            sp, out_p = _rts_cv(tt, xy, Rp, q_acc)
            sy, out_y = _rts_cv(tt, yaw[:, None], np.full(len(tt), sigma_yaw ** 2), q_yaw_acc)
            for k, r in enumerate(rs[a:b]):
                yw = math.atan2(math.sin(sy[k, 0]), math.cos(sy[k, 0]))
                out.append({"frame": int(r["frame"]), "t_jetson": float(r["t_jetson"]), "id": tid,
                            "x": sp[k, 0], "y": sp[k, 1], "yaw": yw,
                            "v": math.hypot(sp[k, 2], sp[k, 3]), "yaw_rate": sy[k, 1],
                            "conf": float(r["conf"]), "raw_x": xy[k, 0], "raw_y": xy[k, 1],
                            "raw_yaw": math.atan2(math.sin(yaw[k]), math.cos(yaw[k])),
                            "outlier": int(out_p[k] or out_y[k]), "segment": a})
    out.sort(key=lambda r: (r["frame"], r["id"]))
    return out


TRACK_FIELDS = ["frame", "t_jetson", "id", "x", "y", "yaw", "v", "yaw_rate", "conf",
                "raw_x", "raw_y", "raw_yaw", "outlier", "segment"]


def write_track(path: str, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=TRACK_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                         if k in TRACK_FIELDS})


def load_track(path: str) -> list:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append({k: (int(v) if k in ("frame", "id", "outlier", "segment") else float(v))
                        for k, v in r.items()})
    return out


def compare(track: list, runlog: RunLog | None) -> dict:
    """ID ごとの件数と、車両の自己位置（catalog）との差。自己位置が無ければ件数だけ。"""
    by_id = {}
    for r in track:
        by_id.setdefault(r["id"], []).append(r)
    summary = {"ids": {}, "pose_source": runlog.source if runlog else None}
    for tid, rs in sorted(by_id.items(), key=lambda kv: -len(kv[1])):
        e = {"n": len(rs), "t0": rs[0]["t_jetson"], "t1": rs[-1]["t_jetson"],
             "outliers": int(sum(r["outlier"] for r in rs))}
        if runlog is not None and len(runlog):
            dpos, dyaw, dlat, dlon = [], [], [], []
            for r in rs:
                p = runlog.pose_at(r["t_jetson"])
                if p is None:
                    continue
                dx, dy = r["x"] - p[0], r["y"] - p[1]
                dpos.append(math.hypot(dx, dy))
                c, s = math.cos(p[2]), math.sin(p[2])
                dlon.append(c * dx + s * dy)            # 車両の前後方向（+ = 外部カメラの方が前）
                dlat.append(-s * dx + c * dy)           # 横方向（+ = 外部カメラの方が左）
                dyaw.append(math.degrees(math.atan2(math.sin(r["yaw"] - p[2]), math.cos(r["yaw"] - p[2]))))
            if dpos:
                a = np.array
                e.update({"n_matched": len(dpos), "pos_rmse_m": float(np.sqrt((a(dpos) ** 2).mean())),
                          "pos_p50_m": float(np.median(dpos)), "lon_mean_m": float(np.mean(dlon)),
                          "lat_mean_m": float(np.mean(dlat)), "yaw_err_p50_deg": float(np.median(np.abs(dyaw))),
                          "yaw_mean_deg": float(np.mean(dyaw))})
        summary["ids"][str(tid)] = e
    return summary


def smooth_clip(run_dir: str, clip_num: int, **kw) -> dict:
    clip = Clip(run_dir, clip_num)
    p = _paths(clip)
    rows = apply_edits(load_raw(p["raw"]), load_edits(p["edits"]))
    track = smooth_rows(rows, **kw)
    write_track(p["track"], track)
    try:
        rl = RunLog(run_dir)
    except Exception:   # noqa: BLE001
        rl = None
    summ = compare(track, rl)
    summ.update({"clip": clip.name, "n_raw": len(rows), "saved": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(p["summary"], "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    return summ


# ---------------------------------------------------------------------------
# 仮ラベル
# ---------------------------------------------------------------------------

def prelabel_from_pose(run_dir: str, clip_num: int, indices, overwrite: bool = False,
                       max_dt: float = 0.1) -> int:
    """車両の自己位置を画像へ投影して仮ラベルを作る（モデルが無い最初の 1 歩）。手で直す前提。"""
    clip = Clip(run_dir, clip_num)
    cc, veh = _camera(run_dir, clip_num)
    w, h = cc.calib.image_size
    rl = RunLog(run_dir)
    store = LabelStore(clip)
    n = 0
    for i in indices:
        if store.is_confirmed(i) or (store.get(i) is not None and not overwrite):
            continue
        t = clip.time_of(i)
        p = rl.pose_at(t, max_dt=max_dt) if t is not None else None
        if p is None:
            continue
        kpts, bbox = kpts_from_pose(cc.calib, veh, *p)
        if cc.handheld:                           # 基準フレームの画素 → このフレームの画素
            kpts[:, :2] = cc.to_frame(kpts[:, :2], i)
            inside = (kpts[:, 0] >= 0) & (kpts[:, 0] < w) & (kpts[:, 1] >= 0) & (kpts[:, 1] < h)
            kpts[:, 2] = np.where(inside & (kpts[:, 2] > 0), 2, 0)
            x1, y1, x2, y2 = bbox
            corners = cc.to_frame(np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], float), i)
            (bx1, by1), (bx2, by2) = np.clip(corners.min(0), 0, [w - 1, h - 1]), np.clip(corners.max(0), 0, [w - 1, h - 1])
            bbox = [float(bx1), float(by1), float(bx2), float(by2)]
        if (kpts[:, 2] > 0).sum() < 2:
            continue
        store.set(i, [{"bbox": bbox, "kpts": kpts.round(1).tolist(), "src": "pose"}])
        n += 1
    store.save()
    return n


def prelabel_from_model(run_dir: str, clip_num: int, indices, model_path: str, conf: float = 0.4,
                        imgsz: int = 1280, device=None, overwrite: bool = False, progress=None) -> int:
    from ultralytics import YOLO
    clip = Clip(run_dir, clip_num)
    clip.extract(indices)
    store = LabelStore(clip)
    model = YOLO(model_path)
    kw = dict(conf=conf, imgsz=imgsz, verbose=False)
    if device not in (None, ""):
        kw["device"] = device
    n = 0
    idx = list(indices)
    for k, i in enumerate(idx):
        if store.is_confirmed(i) or (store.get(i) is not None and not overwrite):
            continue            # 人が確認したフレームは上書きしない
        img = clip.cached_frame(i)
        if img is None:
            continue
        res = model.predict(img, **kw)[0]
        insts = []
        if res.boxes is not None:
            kp = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
            for b in range(len(res.boxes)):
                kk = kp[b] if kp is not None else np.zeros((NK, 3))
                vis = np.where(kk[:, 2] >= 0.5, 2, 0)
                insts.append({"bbox": [round(float(v), 1) for v in res.boxes.xyxy[b].tolist()],
                              "kpts": np.hstack([kk[:, :2].round(1), vis[:, None]]).tolist(),
                              "src": "model", "conf": round(float(res.boxes.conf[b]), 3)})
        store.set(i, insts)
        n += 1
        if progress:
            progress(k + 1, len(idx))
    store.save()
    return n


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="外部カメラ動画の車両追跡")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="推論 → raw → 平滑化")
    r.add_argument("run_dir")
    r.add_argument("--clip", type=int, default=0)
    r.add_argument("--model", required=True)
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--imgsz", type=int, default=1280)
    r.add_argument("--device", default=None)
    r.add_argument("--max-frames", type=int, default=None)
    s = sub.add_parser("smooth", help="edits を反映して平滑化だけやり直す")
    s.add_argument("run_dir")
    s.add_argument("--clip", type=int, default=0)
    pl = sub.add_parser("prelabel", help="仮ラベルを作る")
    pl.add_argument("run_dir")
    pl.add_argument("--clip", type=int, default=0)
    pl.add_argument("--from", dest="src", choices=["pose", "model"], default="pose")
    pl.add_argument("--model", default=None)
    pl.add_argument("--step", type=int, default=30)
    pl.add_argument("--count", type=int, default=0)
    pl.add_argument("--overwrite", action="store_true")
    pl.add_argument("--imgsz", type=int, default=1280)
    pl.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    def prog(k, n, el=None):
        print(f"progress {k}/{n}" + (f" {el:.1f}s" if el is not None else ""), flush=True)

    if a.cmd == "run":
        run_inference(a.run_dir, a.clip, a.model, a.conf, a.imgsz, a.device, progress=prog,
                      max_frames=a.max_frames)
        summ = smooth_clip(a.run_dir, a.clip)
        print("SUMMARY " + json.dumps(summ, ensure_ascii=False), flush=True)
    elif a.cmd == "smooth":
        summ = smooth_clip(a.run_dir, a.clip)
        print("SUMMARY " + json.dumps(summ, ensure_ascii=False), flush=True)
    elif a.cmd == "prelabel":
        clip = Clip(a.run_dir, a.clip)
        idx = clip.sample_indices(step=a.step, count=a.count)
        if a.src == "pose":
            n = prelabel_from_pose(a.run_dir, a.clip, idx, overwrite=a.overwrite)
        else:
            if not a.model:
                ap.error("--from model には --model が必要です")
            n = prelabel_from_model(a.run_dir, a.clip, idx, a.model, imgsz=a.imgsz, device=a.device,
                                    overwrite=a.overwrite, progress=prog)
        print(f"PRELABEL {n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
