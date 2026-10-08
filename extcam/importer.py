"""他のカメラ（スマホ・手持ち可）の動画を、走行と時刻合わせして走行フォルダの extcam/ へ取り込む。Qt 非依存。

    python -m extcam.importer analyze VIDEO --data-root data [--hdr] [--cam-box x0,x1,y0,y1] [--init calib.json]
    python -m extcam.importer write   --work data/extcam_imports/<動画名>

analyze は extcam/skills/extcam-video-sync/scripts/vsync.py（スキルと同じ正本）の各段を順に呼び、
最後に合否の目安を自動で判定して <work>/import_summary.json に書く。各段の画像（contact / barrier_vis / map /
calib_overlay / det_track / verify_overlay）は人（または Claude）が見て確認する。手順と判断基準は SKILL.md。

write は、時刻合わせした走行の時間範囲（前後に余白）だけを切り出して書く:
  extcam/clip_NN.mkv（ffmpeg が無ければ .avi）  動画（必要なら縮小）
  extcam/clip_NN.frames.csv   frame, t_cam（動画内の時刻）, t_jetson（= t_cam + offset）, key
  extcam/clip_NN.calib.json   このクリップ専用の較正（基準フレームの画素）。ref_index・取り込み元・時刻合わせの結果つき
  extcam/clip_NN.stab.npy     手持ちのとき、フレーム → 基準フレームのホモグラフィ（固定カメラなら作らない）
  extcam/clip_NN.import.json  取り込みの記録（合否の目安・元動画・ハッシュ）
  extcam/clips.json           クリップ一覧へ追記（sidecam の取り込みと同じファイル）
"""
from __future__ import annotations

import argparse
import collections
import datetime as dtm
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.dirname(_HERE)
if _TOOL not in sys.path:
    sys.path.insert(0, _TOOL)

from extcam.clip import list_clips                                     # noqa: E402
from extcam.clipcam import clip_setup_path, run_setup_path, save_clip_setup, stab_path  # noqa: E402
from extcam.geometry import CameraCalib, VehicleKpts, load_setup       # noqa: E402
from extcam.runlog import repo_root_of, resolve_map_dir                # noqa: E402

VSYNC_PATH = os.path.join(_HERE, "skills", "extcam-video-sync", "scripts", "vsync.py")
STEPS = ["prepare", "mask", "mapplot", "calibrate", "detect", "sync", "verify"]

# 合否の目安（SKILL.md と同じ）
CHECK = {"calib_d1": 6.0, "calib_d2": 16.0, "calib_iou": 0.95, "sync_margin": 0.1, "sync_med_px": 40.0,
         "curve_ratio": 1.5, "hint_tol_s": 2.0, "pass_tol_s": 0.3}


def load_vsync():
    spec = importlib.util.spec_from_file_location("extcam_vsync_core", VSYNC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _jdump(obj, p):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def default_work(data_root: str, video: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(video))[0])
    return os.path.join(os.path.abspath(data_root), "extcam_imports", stem)


# ---------------------------------------------------------------------------
# 走行の候補・地図
# ---------------------------------------------------------------------------

def _run_ts_from_name(name: str):
    m = re.match(r"data_(\d{8})_(\d{6})", name)
    if not m:
        return None
    try:
        return dtm.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return None


def candidate_runs(data_root: str, hint_t: float | None = None, date: str | None = None,
                   hours: float = 6.0) -> list:
    """走行の候補。撮影時刻のヒントがあれば前後 hours 時間、日付指定ならその日（フォルダ名で粗く絞る）。

    フォルダ名の時刻は Jetson の時計ずれで実時刻と数分ずれることがあるので、幅を広めに取る。
    """
    runs = sorted(d for d in glob.glob(os.path.join(data_root, "data_*"))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "catalog_*.catalog")))
    if date:
        runs = [r for r in runs if os.path.basename(r).startswith(f"data_{date}")]
    elif hint_t is not None:
        runs = [r for r in runs if (_run_ts_from_name(os.path.basename(r)) or 0) and
                abs(_run_ts_from_name(os.path.basename(r)) - hint_t) <= hours * 3600]
    else:
        raise ValueError("撮影時刻のヒントが無いので、--date YYYYMMDD か --runs で走行を絞ってください")
    return runs


def guess_bounds(runs: list) -> str | None:
    """候補の走行で最も多く使われた地図の map_bounds.json。"""
    cnt = collections.Counter()
    for r in runs:
        md = resolve_map_dir(r)
        if md and os.path.isfile(os.path.join(md, "map_bounds.json")):
            cnt[md] += 1
    if not cnt:
        return None
    return os.path.join(cnt.most_common(1)[0][0], "map_bounds.json")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def evaluate(work: str) -> dict:
    """各段の出力から合否の目安を判定する。"""
    res = {"checks": {}, "ok": True}

    def put(name, ok, detail):
        res["checks"][name] = {"ok": bool(ok), "detail": detail}
        res["ok"] = res["ok"] and bool(ok)

    vj = os.path.join(work, "video.json")
    hint = None
    if os.path.isfile(vj):
        v = _jload(vj)
        res["video"] = {k: v.get(k) for k in ("video", "n_frames", "duration_s", "size", "stab_fail")}
        q = (v.get("time_hints") or {}).get("quicktime_creationdate")
        hint = q["t"] if q else None
        res["hint_t"] = hint
        put("stabilization", v.get("stab_fail", 1) <= 0.1 * max(1, v.get("n_frames", 1)),
            f"手ぶれ補正の失敗 {v.get('stab_fail')} / {v.get('n_frames')}")
    cj = os.path.join(work, "calib.json")
    if os.path.isfile(cj):
        c = _jload(cj)
        put("calibration", c["outline_to_barrier_px"] < CHECK["calib_d1"] and c["barrier_to_outline_px"] < CHECK["calib_d2"]
            and c.get("extent_iou", 1.0) > CHECK["calib_iou"],
            f"外形→柵 {c['outline_to_barrier_px']} px / 柵→外形 {c['barrier_to_outline_px']} px / IoU {c.get('extent_iou')}"
            "（calib_overlay.jpg で柵に沿っているか目視）")
    sj = os.path.join(work, "sync.json")
    if os.path.isfile(sj):
        s = _jload(sj)
        b = s["best"]
        res["sync"] = {k: b[k] for k in ("run", "name", "offset", "video_t0_local", "F", "med_px")}
        res["sync"]["margin"] = s.get("margin_F_vs_2nd")
        curve = {c["d_s"]: c["med_px"] for c in s.get("offset_curve") or []}
        c0 = curve.get(0.0) or 1e9
        sharp = min(curve.get(-0.1) or 0, curve.get(0.1) or 0) / max(c0, 1e-6)
        if hint is not None:
            put("sync_vs_hint", abs(b["offset"] - hint) <= CHECK["hint_tol_s"],
                f"撮影時刻のメタデータとの差 {b['offset'] - hint:+.2f} s（秒単位の値なので ±1 s は正常）")
        margin = s.get("margin_F_vs_2nd")
        put("sync_unique", margin is None or margin >= CHECK["sync_margin"] or hint is not None,
            f"2 位との差 {margin}（候補が 1 本だけなら None）")
        put("sync_sharp", sharp >= CHECK["curve_ratio"] and b["med_px"] < CHECK["sync_med_px"],
            f"±0.1 s で誤差 {sharp:.1f} 倍、中央値 {b['med_px']:.1f} px")
    wj = os.path.join(work, "verify.json")
    if os.path.isfile(wj):
        w = _jload(wj)
        main = [p for p in w.get("passes") or [] if p["video"][1] - p["video"][0] >= 0.8]
        ok = bool(main) and all(abs(p["d_start_s"]) <= CHECK["pass_tol_s"] for p in main)
        put("passes", ok, f"主な通過の開始時刻の差 {[p['d_start_s'] for p in main]} s / 位置 p50 {w['ground_err_m']['p50']} m")
        res["ground_err_m"] = w["ground_err_m"]
    return res


def analyze(video: str, data_root: str, work: str | None = None, steps=None, runs=None, date=None,
            bounds=None, hdr=False, ignore_top=0.4, ignore_rect=None, cam_box=None, init=None,
            use_hint=True, hint_window=60.0, extra_calib=None) -> dict:
    vs = load_vsync()
    work = work or default_work(data_root, video)
    os.makedirs(work, exist_ok=True)
    steps = steps or STEPS
    cfg_path = os.path.join(work, "import_config.json")
    cfg = _jload(cfg_path) if os.path.isfile(cfg_path) else {}
    cfg.update({"video": os.path.abspath(video), "data_root": os.path.abspath(data_root)})

    def call(argv):
        print("$ vsync " + " ".join(str(a) for a in argv), flush=True)
        vs.main([str(a) for a in argv])

    if "prepare" in steps:
        call(["prepare", video, "--work", work])
    hint = None
    vj = os.path.join(work, "video.json")
    if os.path.isfile(vj):
        q = (_jload(vj).get("time_hints") or {}).get("quicktime_creationdate")
        hint = q["t"] if q else None
    if runs:
        run_list = [r for p in runs for r in (sorted(glob.glob(p)) or [p])]
    else:
        run_list = candidate_runs(data_root, hint if use_hint else None, date)
    cfg["runs"] = run_list
    bounds = bounds or cfg.get("bounds") or guess_bounds(run_list)
    if not bounds and any(s in steps for s in ("mapplot", "calibrate")):
        raise ValueError("コース外形（map_bounds.json）が見つかりません。--bounds で指定してください")
    cfg["bounds"] = bounds
    _jdump(cfg, cfg_path)
    print(f"走行の候補 {len(run_list)} 本 / 外形 {bounds}" + (f" / 撮影時刻 {dtm.datetime.fromtimestamp(hint)}" if hint else ""),
          flush=True)
    if "mask" in steps:
        argv = ["mask", "--work", work, "--ignore-top", ignore_top]
        for r in ignore_rect or []:
            argv += ["--ignore-rect", r]
        if hdr:
            argv.append("--hdr")
        call(argv)
    if "mapplot" in steps:
        call(["mapplot", "--bounds", bounds, "--work", work, "--runs", *run_list[:1]])
    if "calibrate" in steps:
        argv = ["calibrate", "--work", work, "--bounds", bounds]
        if cam_box:
            argv += ["--cam-box", cam_box]
        if init:
            argv += ["--init", init]
        argv += list(extra_calib or [])
        call(argv)
    if "detect" in steps:
        call(["detect", "--work", work])
    if "sync" in steps:
        argv = ["sync", "--work", work, "--runs", *run_list]
        if hint is not None and use_hint:
            argv += ["--around", hint, "--window", hint_window]
        call(argv)
    if "verify" in steps:
        s = _jload(os.path.join(work, "sync.json"))["best"]
        call(["verify", "--work", work, "--run", s["run"], "--offset", s["offset"]])
    summ = evaluate(work)
    summ["work"] = work
    _jdump(summ, os.path.join(work, "import_summary.json"))
    return summ


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def find_ffmpeg(explicit: str | None = None):
    if explicit:
        return explicit
    if os.environ.get("EXTCAM_FFMPEG"):
        return os.environ["EXTCAM_FFMPEG"]
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:   # noqa: BLE001
        return None


def _file_hash(path: str) -> str:
    """先頭・末尾 4 MB とサイズのハッシュ（大きな動画を全部読まない、同じ動画の判別用）。"""
    h = hashlib.sha1()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
        f.seek(max(0, size - (4 << 20)))
        h.update(f.read())
    h.update(str(size).encode())
    return h.hexdigest()


def _catalog_range(run_dir: str):
    vs = load_vsync()
    _, T, _ = vs.load_traj(run_dir)
    return (float(T[0]), float(T[-1])) if len(T) else (None, None)


def resolve_run(work: str, run: str) -> str:
    """走行フォルダを絶対パスに。相対パス（古い sync.json）は作業ディレクトリではなく取り込み設定の data_root を基準にする。"""
    if os.path.isabs(run):
        return run
    cfg = os.path.join(work, "import_config.json")
    if os.path.isfile(cfg):
        root = _jload(cfg).get("data_root")
        if root and os.path.isdir(os.path.join(root, os.path.basename(os.path.normpath(run)))):
            return os.path.join(root, os.path.basename(os.path.normpath(run)))
    raise ValueError(f"走行フォルダを特定できません: {run}（--run で絶対パスを指定してください）")


def write(work: str, run: str | None = None, offset: float | None = None, margin_s: float = 3.0,
          max_width: int = 1920, ffmpeg: str | None = None, trim: bool = True, force: bool = False) -> dict:
    vs = load_vsync()
    summ = _jload(os.path.join(work, "import_summary.json")) if os.path.isfile(os.path.join(work, "import_summary.json")) else {}
    if not force and summ and not summ.get("ok", False):
        bad = [k for k, v in summ.get("checks", {}).items() if not v["ok"]]
        raise ValueError(f"合否の目安を満たしていません {bad}。確認したうえで書くなら --force")
    sync = _jload(os.path.join(work, "sync.json"))["best"]
    run = resolve_run(work, run or sync["run"])
    offset = float(offset if offset is not None else sync["offset"])
    meta = _jload(os.path.join(work, "video.json"))
    cal = _jload(os.path.join(work, "calib.json"))
    z = np.load(os.path.join(work, "frames.npz"))
    ts, Hs = z["ts"], z["H"]
    n = len(ts)
    W0, H0 = meta["size"]
    # 書き出す範囲: 走行の記録がある時間 ± 余白
    i0, i1 = 0, n
    if trim:
        t0, t1 = _catalog_range(run)
        if t0 is not None:
            tj = ts + offset
            keep = np.where((tj >= t0 - margin_s) & (tj <= t1 + margin_s))[0]
            if len(keep) == 0:
                raise ValueError("動画と走行の記録が重なっていません")
            i0, i1 = int(keep[0]), int(keep[-1]) + 1
    scale = min(1.0, max_width / float(W0))
    W, H = int(round(W0 * scale)) // 2 * 2, int(round(H0 * scale)) // 2 * 2
    ext_dir = os.path.join(run, "extcam")
    os.makedirs(ext_dir, exist_ok=True)
    num = max([c for c, _ in list_clips(run)], default=-1) + 1
    base = os.path.join(ext_dir, f"clip_{num:02d}")
    ff = find_ffmpeg(ffmpeg)
    if ff:
        video_out = base + ".mkv"
        proc = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                 "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{meta.get('fps') or 30:.3f}",
                                 "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                 "-g", "30", "-bf", "0", "-pix_fmt", "yuv420p", video_out], stdin=subprocess.PIPE)
        writer = None
    else:
        print("警告: ffmpeg が見つからないので MJPG の .avi で書きます（容量が数倍）。"
              "ffmpeg を入れるか、環境変数 EXTCAM_FFMPEG / --ffmpeg で場所を指定してください", flush=True)
        video_out = base + ".avi"
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"MJPG"), float(meta.get("fps") or 30), (W, H))
        proc = None
    cap = cv2.VideoCapture(meta["video"])
    k = 0
    written = 0
    while k < i1:
        ok, f = cap.read()
        if not ok:
            break
        if k >= i0:
            if scale != 1.0:
                f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
            if proc:
                proc.stdin.write(np.ascontiguousarray(f).tobytes())
            else:
                writer.write(f)
            written += 1
        k += 1
    cap.release()
    if proc:
        proc.stdin.close()
        proc.wait()
    else:
        writer.release()
    i1 = i0 + written
    with open(base + ".frames.csv", "w", encoding="utf-8") as fh:
        fh.write("frame,t_cam,t_jetson,key\n")
        for j, i in enumerate(range(i0, i1)):
            fh.write(f"{j},{ts[i]:.4f},{ts[i] + offset:.4f},\n")
    # 手ぶれ補正（元解像度）→ 書き出した解像度へ: H' = S H S^-1。動きがほぼ無ければ固定カメラとして作らない
    S = np.diag([W / float(W0), H / float(H0), 1.0])
    Si = np.linalg.inv(S)
    sub = np.array([S @ h @ Si if np.isfinite(h).all() else np.full((3, 3), np.nan) for h in Hs[i0:i1]])
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], np.float32).reshape(-1, 1, 2)
    moves = [float(np.abs(cv2.perspectiveTransform(corners, h) - corners).max()) for h in sub if np.isfinite(h).all()]
    handheld = bool(moves) and max(moves) > 2.0
    if handheld:
        np.save(stab_path(run, num), sub)
    # 較正: vsync の K, R, t（元解像度・基準フレーム）→ CameraCalib（書き出した解像度）
    K = np.array(cal["K"], float)
    K[0] *= W / float(W0)
    K[1] *= H / float(H0)
    rvec, _ = cv2.Rodrigues(np.array(cal["R"], float))
    calib = CameraCalib(image_size=(W, H), K=K.tolist(), dist=list(cal.get("dist") or [0] * 5),
                        rvec=rvec.ravel().tolist(), tvec=list(np.array(cal["t"], float).ravel()),
                        reproj_rms_px=None, n_points=0, intrinsics_source="vsync")
    veh = VehicleKpts()
    if os.path.isfile(run_setup_path(run)):
        veh = load_setup(run_setup_path(run))[1]
    else:
        vj = os.path.join(repo_root_of(run), "vehicle.json")
        if os.path.isfile(vj):
            veh = VehicleKpts.from_vehicle_json(vj)
    ref = meta.get("ref_index")
    src = {"type": "import", "video": meta["video"], "sha1_head_tail": _file_hash(meta["video"]),
           "frames": [i0, i1], "scale": scale, "handheld": handheld,
           "time_hints": meta.get("time_hints"), "imported": time.strftime("%Y-%m-%d %H:%M:%S")}
    syncrec = {"run": os.path.basename(os.path.normpath(run)), "offset": offset,
               "video_t0_local": dtm.datetime.fromtimestamp(offset).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
               "F": sync.get("F"), "med_px": sync.get("med_px"), "checks": summ.get("checks"),
               "ground_err_m": summ.get("ground_err_m")}
    save_clip_setup(clip_setup_path(run, num), calib, veh, [], os.path.dirname(cal["bounds"]) if cal.get("bounds") else None,
                    extra={"ref_index": (ref - i0) if ref is not None and i0 <= ref < i1 else None,
                           "source": src, "sync": syncrec})
    for name in ("calib_overlay.jpg", "verify_overlay.jpg"):
        p = os.path.join(work, name)
        if os.path.isfile(p):
            shutil.copy(p, base + "." + name)
    rec = {"clip": f"clip_{num:02d}", "video": os.path.basename(video_out), "n_frames": written,
           "source": src, "sync": syncrec, "work": os.path.abspath(work)}
    _jdump(rec, base + ".import.json")
    cj = os.path.join(ext_dir, "clips.json")
    clips = _jload(cj) if os.path.isfile(cj) else {"session": os.path.basename(os.path.normpath(run)), "clips": []}
    clips.setdefault("clips", []).append({"video": os.path.basename(video_out), "frames": os.path.basename(base + ".frames.csv"),
                                          "n_frames": written, "source": "import", "range_jetson":
                                          [float(ts[i0] + offset), float(ts[i1 - 1] + offset)],
                                          "calib": os.path.basename(clip_setup_path(run, num)),
                                          "stab": os.path.basename(stab_path(run, num)) if handheld else None})
    _jdump(clips, cj)
    print(f"→ {video_out}（{written} フレーム、{W}x{H}、{'手持ち（手ぶれ補正あり）' if handheld else '固定カメラ'}）", flush=True)
    return rec


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("analyze", help="時刻合わせ（vsync の各段）と合否の目安")
    a_.add_argument("video")
    a_.add_argument("--data-root", required=True, help="走行フォルダの親（data/）")
    a_.add_argument("--work", default=None, help="作業フォルダ（既定 <data-root>/extcam_imports/<動画名>）")
    a_.add_argument("--steps", default=",".join(STEPS), help="実行する段（カンマ区切り）: " + ",".join(STEPS))
    a_.add_argument("--runs", nargs="*", default=None, help="走行を明示（既定は撮影時刻か --date で絞る）")
    a_.add_argument("--date", default=None, help="YYYYMMDD（撮影時刻のヒントが無いとき）")
    a_.add_argument("--bounds", default=None, help="map_bounds.json（既定は候補の走行の地図から）")
    a_.add_argument("--hdr", action="store_true")
    a_.add_argument("--ignore-top", type=float, default=0.4)
    a_.add_argument("--ignore-rect", action="append", default=None)
    a_.add_argument("--cam-box", default=None)
    a_.add_argument("--init", default=None)
    a_.add_argument("--no-hint", action="store_true", help="撮影時刻のヒントを使わず全候補を探す")
    w_ = sub.add_parser("write", help="走行フォルダの extcam/ へ書き出す")
    w_.add_argument("--work", required=True)
    w_.add_argument("--run", default=None)
    w_.add_argument("--offset", type=float, default=None)
    w_.add_argument("--margin", type=float, default=3.0)
    w_.add_argument("--max-width", type=int, default=1920)
    w_.add_argument("--ffmpeg", default=None)
    w_.add_argument("--no-trim", action="store_true")
    w_.add_argument("--force", action="store_true", help="合否の目安を満たさなくても書く")
    a = ap.parse_args(argv)
    if a.cmd == "analyze":
        summ = analyze(a.video, a.data_root, a.work, [s for s in a.steps.split(",") if s], a.runs, a.date,
                       a.bounds, a.hdr, a.ignore_top, a.ignore_rect, a.cam_box, a.init, not a.no_hint)
        print("SUMMARY " + json.dumps(summ, ensure_ascii=False), flush=True)
        return 0 if summ.get("ok") else 2
    rec = write(a.work, a.run, a.offset, a.margin, a.max_width, a.ffmpeg, not a.no_trim, a.force)
    print("WRITTEN " + json.dumps(rec, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
