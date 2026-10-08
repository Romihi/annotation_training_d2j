"""extcam の中核（Qt・GPU・動画なし）の机上テスト。

    python -m pytest dev/test_extcam.py -q -p no:anyio -p no:cacheprovider
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extcam import tracker as trk                                                  # noqa: E402
from extcam.clip import Clip                                                       # noqa: E402
from extcam.geometry import (CameraCalib, VehicleKpts, body_to_map, kpts_from_pose,  # noqa: E402
                             load_setup, pose_from_kpts, pose_from_points, save_setup)
from extcam.labels import LabelStore, export_yolo_pose                             # noqa: E402

W, H = 1280, 720


def _true_camera(f=760.0, pos=(-2.5, 3.0, 3.2), target=(4.3, 2.8, 0.0)):
    cam = np.array(pos, float)
    z = np.array(target, float) - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    rvec, _ = cv2.Rodrigues(R)
    return CameraCalib(image_size=(W, H), K=[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
                       rvec=rvec.ravel().tolist(), tvec=(-R @ cam).tolist())


def _floor_points():
    return np.array([[0, 0], [2, -1.5], [5, -2], [7.5, 0], [8, 4], [6, 7], [3, 7.5], [1, 4], [4, 2], [5.5, 5]], float)


def test_calibration_recovers_focal_and_pose():
    true = _true_camera()
    mp = _floor_points()
    rng = np.random.default_rng(0)
    ip = true.project(mp) + rng.normal(0, 0.5, (len(mp), 2))
    cal = CameraCalib.fit(ip, mp, (W, H))
    assert abs(cal.K[0][0] - 760) < 15
    assert np.linalg.norm(cal.camera_position() - true.camera_position()) < 0.05
    assert cal.reproj_rms_px < 1.0
    # 屋根の高さの平面への逆投影
    pts = np.array([[2.0, 1.0, 0.12], [6.0, 5.0, 0.12]])
    back = cal.backproject(true.project(pts), z=0.12)
    assert np.abs(back - pts[:, :2]).max() < 0.02


def test_calibration_needs_four_points():
    with pytest.raises(ValueError):
        CameraCalib.fit([[0, 0]] * 3, [[0, 0]] * 3, (W, H))


def test_pose_from_points_and_kpts_roundtrip():
    veh = VehicleKpts()
    b = veh.body_points()
    m = body_to_map(b, 3.0, 2.0, 0.7)
    x, y, yaw, rms = pose_from_points(b, m)
    assert (x, y) == pytest.approx((3.0, 2.0), abs=1e-9) and yaw == pytest.approx(0.7)
    # 2 点（前左・後左）だけでも向きまで決まる
    x2, y2, yaw2, _ = pose_from_points(b[[0, 2]], m[[0, 2]])
    assert yaw2 == pytest.approx(0.7) and x2 == pytest.approx(3.0)
    true = _true_camera()
    k, bbox = kpts_from_pose(true, veh, 4.0, 1.0, -2.0)
    assert (k[:, 2] == 2).all() and bbox[0] < k[:, 0].min() and bbox[2] > k[:, 0].max()
    px, py, pyaw, prms, n = pose_from_kpts(true, veh, k)
    assert (px, py) == pytest.approx((4.0, 1.0), abs=1e-6) and pyaw == pytest.approx(-2.0) and n == 4
    k[1, 2] = k[3, 2] = 0                   # 右側が隠れても 2 点で求まる
    assert pose_from_kpts(true, veh, k)[2] == pytest.approx(-2.0)
    k[2, 2] = 0
    assert pose_from_kpts(true, veh, k) is None


def test_setup_save_load(tmp_path):
    cal = _true_camera()
    veh = VehicleKpts(height=0.15)
    p = tmp_path / "calib.json"
    save_setup(str(p), cal, veh, [{"img": [1, 2], "map": [3, 4]}], "/maps/x")
    c2, v2, pairs, md = load_setup(str(p))
    assert c2.K == cal.K and v2.height == 0.15 and pairs[0]["map"] == [3, 4] and md == "/maps/x"


def _rows_on_circle(n=300, dt=1 / 60, r=2.0, v=2.0, noise=0.03, seed=0, tid=1, outliers=()):
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for i in range(n):
        th = v * i * dt / r
        x, y, yaw = r * math.cos(th), r * math.sin(th), th + math.pi / 2
        truth.append((x, y, yaw))
        nx, ny = x + rng.normal(0, noise), y + rng.normal(0, noise)
        if i in outliers:
            nx += 1.0
        rows.append({"frame": str(i), "t_jetson": f"{1000 + i * dt:.5f}", "id": str(tid), "conf": "0.9",
                     "px": str(nx), "py": str(ny), "pyaw": str(yaw + rng.normal(0, 0.05)), "prms": "0.0"})
    return rows, np.array(truth)


def test_smoothing_reduces_noise_and_rejects_outliers():
    rows, truth = _rows_on_circle(outliers=(100, 101))
    out = trk.smooth_rows(rows)
    assert len(out) == len(rows)
    xy = np.array([[r["x"], r["y"]] for r in out])
    raw = np.array([[float(r["px"]), float(r["py"])] for r in rows])
    err_s = np.linalg.norm(xy - truth[:, :2], axis=1)
    err_r = np.linalg.norm(raw - truth[:, :2], axis=1)
    keep = np.ones(len(rows), bool)
    keep[[100, 101]] = False
    assert np.sqrt((err_s[keep] ** 2).mean()) < 0.6 * np.sqrt((err_r[keep] ** 2).mean())
    assert out[100]["outlier"] == 1 and err_s[100] < 0.1
    v = np.array([r["v"] for r in out[20:-20]])
    assert np.median(v) == pytest.approx(2.0, abs=0.1)
    yaw_err = [abs(math.atan2(math.sin(r["yaw"] - t[2]), math.cos(r["yaw"] - t[2]))) for r, t in zip(out, truth)]
    assert np.median(yaw_err) < 0.03


def test_smoothing_splits_on_gap():
    rows, _ = _rows_on_circle(n=100)
    for r in rows[50:]:
        r["t_jetson"] = f"{float(r['t_jetson']) + 2.0:.5f}"     # 2 s の空白
    out = trk.smooth_rows(rows)
    assert len({r["segment"] for r in out}) == 2


def test_apply_edits_merge_exclude():
    a, _ = _rows_on_circle(n=10, tid=1)
    b, _ = _rows_on_circle(n=10, tid=2)
    c, _ = _rows_on_circle(n=10, tid=3)
    rows = a + b + c
    out = trk.apply_edits(rows, {"merge": {"2": 1}, "exclude_ids": [3], "exclude_frames": [[1, 0], 5]})
    ids = {r["id"] for r in out}
    assert ids == {1}
    frames = sorted(int(r["frame"]) for r in out)
    assert 5 not in frames and frames.count(0) == 1        # (1, 0) は除外、元 ID 2 の frame 0 は残る


def _fake_clip(tmp_path, n=12):
    run = tmp_path / "data_20261008_000000"
    ext = run / "extcam"
    (ext / "frames" / "clip_00").mkdir(parents=True)
    (ext / "clip_00.mkv").write_bytes(b"")
    with open(ext / "clip_00.frames.csv", "w") as f:
        f.write("frame,t_cam,t_jetson,key\n")
        for i in range(n):
            f.write(f"{i},{100 + i / 60:.4f},{100 + i / 60:.4f},{int(i % 60 == 0)}\n")
    for i in range(n):          # 動画の代わりに書き出し済みのフレーム
        img = np.full((H, W, 3), 80, np.uint8)
        cv2.imwrite(str(ext / "frames" / "clip_00" / f"{i:06d}.jpg"), img)
    return str(run)


def test_label_store_and_export(tmp_path):
    run = _fake_clip(tmp_path)
    clip = Clip(run, 0)
    assert len(clip) == 12 and clip.index_at(100 + 5.2 / 60) == 5
    st = LabelStore(clip)
    inst = {"bbox": [100, 100, 200, 160], "kpts": [[120, 110, 2], [180, 110, 2], [120, 150, 1], [-5, 150, 2]],
            "src": "manual"}
    for i in range(6):
        st.set(i, [inst], confirmed=True)
    st.set(6, [inst])                       # 未確認
    st.confirm(7)                           # 車なしの負例
    st.save()
    st2 = LabelStore(clip)
    assert st2.is_confirmed(3) and not st2.is_confirmed(6) and st2.get(7) == []
    ds = export_yolo_pose([run], str(tmp_path / "ds"), val_ratio=0.2)
    assert ds["n_train"] + ds["n_val"] == 7
    txts = sorted((tmp_path / "ds" / "labels").rglob("*.txt"))
    lines = [t.read_text().strip() for t in txts]
    assert lines.count("") == 1                                   # 負例
    vals = [ln for ln in lines if ln][0].split()
    assert len(vals) == 5 + 4 * 3 and vals[0] == "0"
    assert [float(v) for v in vals[-3:]] == [0.0, 0.0, 0.0]       # 画面外の点は可視性 0
    y = (tmp_path / "ds" / "data.yaml").read_text()
    assert "kpt_shape: [4, 3]" in y and "flip_idx: [1, 0, 3, 2]" in y
    ds2 = export_yolo_pose([run], str(tmp_path / "ds2"), include_unconfirmed=True)
    assert ds2["n_train"] + ds2["n_val"] == 8


def test_prelabel_from_pose(tmp_path):
    run = _fake_clip(tmp_path)
    cal = _true_camera()
    veh = VehicleKpts()
    save_setup(trk.setup_path(run), cal, veh, [], None)
    with open(os.path.join(run, "catalog_0.catalog"), "w") as f:
        for k in range(30):
            t = 100 + k * 0.01
            f.write(json.dumps({"_index": k, "_timestamp_ms": int(t * 1000), "fused/x": 3.0 + t - 100,
                                "fused/y": 1.0, "fused/theta": 0.0, "fused/status": "ok"}) + "\n")
    n = trk.prelabel_from_pose(run, 0, range(12))
    assert n == 12
    st = LabelStore(Clip(run, 0))
    assert not st.is_confirmed(0)                                 # 仮ラベルは未確認
    k = np.array(st.get(6)[0]["kpts"])
    p = pose_from_kpts(cal, veh, k)
    assert p[0] == pytest.approx(3.0 + 6 / 60, abs=0.01) and p[2] == pytest.approx(0.0, abs=0.01)
    st.set(3, [], confirmed=True)
    st.save()
    trk.prelabel_from_pose(run, 0, range(12), overwrite=True)
    assert LabelStore(Clip(run, 0)).get(3) == []                  # 確認済みは上書きしない


# --- 取り込み（他のカメラ・手持ち）: clipcam / importer -------------------------------

def _shift_stab(n, dx=3.0, dy=-2.0):
    """フレーム i は基準フレームから (i*dx, i*dy) ずれた画（手持ちの揺れの代わり）。H はフレーム → 基準。"""
    return np.array([[[1, 0, -i * dx], [0, 1, -i * dy], [0, 0, 1]] for i in range(n)], float)


def test_clipcam_precedence_roundtrip_and_extras(tmp_path):
    from extcam.clipcam import clip_setup_path, load_clip_camera, save_clip_setup, stab_path
    run = _fake_clip(tmp_path)
    cal = _true_camera()
    save_setup(trk.setup_path(run), _true_camera(f=500.0), VehicleKpts(), [], None)     # 走行の（extcam の）較正
    cc, _, _, _, path = load_clip_camera(run, 0)
    assert path.endswith("calib.json") and not cc.handheld and abs(cc.calib.K[0][0] - 500) < 1e-9
    save_clip_setup(clip_setup_path(run, 0), cal, VehicleKpts(), [], None, extra={"ref_index": 5, "source": {"type": "import"}})
    np.save(stab_path(run, 0), _shift_stab(12))
    cc, _, _, _, path = load_clip_camera(run, 0)
    assert path.endswith("clip_00.calib.json") and cc.handheld and cc.ref_index == 5 and abs(cc.calib.K[0][0] - 760) < 1e-9
    uv = np.array([[100.0, 200.0], [640.0, 360.0]])
    assert np.allclose(cc.to_ref(cc.to_frame(uv, 7), 7), uv)
    assert np.allclose(cc.to_ref(uv, 4), uv - [12.0, -8.0])
    save_clip_setup(clip_setup_path(run, 0), cal, VehicleKpts(height=0.2), [{"img": [1, 2], "map": [3, 4]}], None)
    d = json.load(open(clip_setup_path(run, 0)))
    assert d["ref_index"] == 5 and d["source"]["type"] == "import" and d["vehicle_kpts"]["height"] == 0.2


def test_prelabel_and_pose_through_stabilization(tmp_path):
    from extcam.clipcam import clip_setup_path, load_clip_camera, save_clip_setup, stab_path
    run = _fake_clip(tmp_path)
    cal, veh = _true_camera(), VehicleKpts()
    save_clip_setup(clip_setup_path(run, 0), cal, veh, [], None, extra={"ref_index": 0})
    np.save(stab_path(run, 0), _shift_stab(12))
    with open(os.path.join(run, "catalog_0.catalog"), "w") as f:
        for k in range(30):
            t = 100 + k * 0.01
            f.write(json.dumps({"_index": k, "_timestamp_ms": int(t * 1000), "fused/x": 3.0, "fused/y": 1.0,
                                "fused/theta": 0.5, "fused/status": "ok"}) + "\n")
    assert trk.prelabel_from_pose(run, 0, range(12)) == 12
    st = LabelStore(Clip(run, 0))
    k_ref, _ = kpts_from_pose(cal, veh, 3.0, 1.0, 0.5)
    k9 = np.array(st.get(9)[0]["kpts"])
    assert np.allclose(k9[:, :2], k_ref[:, :2] + [27.0, -18.0], atol=0.2)     # フレーム 9 は (27, -18) ずれた画
    cc = load_clip_camera(run, 0)[0]
    kr = k9.copy()
    kr[:, :2] = cc.to_ref(k9[:, :2], 9)
    p = pose_from_kpts(cc.calib, veh, kr)
    assert p[0] == pytest.approx(3.0, abs=0.01) and p[1] == pytest.approx(1.0, abs=0.01) and p[2] == pytest.approx(0.5, abs=0.01)


def test_importer_candidates_and_run_resolution(tmp_path):
    import datetime as dtm
    from extcam import importer
    root = tmp_path / "data"
    for name in ("data_20261007_214644", "data_20261007_120000", "data_20261005_210000", "data_20261007_220000"):
        d = root / name
        d.mkdir(parents=True)
        if name != "data_20261007_220000":                       # catalog の無いフォルダは候補にしない
            (d / "catalog_0.catalog").write_text("{}\n")
    hint = dtm.datetime(2026, 10, 7, 21, 48, 34).timestamp()
    near = [os.path.basename(r) for r in importer.candidate_runs(str(root), hint)]
    assert near == ["data_20261007_214644"]
    assert len(importer.candidate_runs(str(root), None, date="20261007")) == 2
    with pytest.raises(ValueError):
        importer.candidate_runs(str(root))
    work = tmp_path / "work"
    work.mkdir()
    (work / "import_config.json").write_text(json.dumps({"data_root": str(root)}))
    assert importer.resolve_run(str(work), "../data/data_20261007_214644") == str(root / "data_20261007_214644")
    with pytest.raises(ValueError):
        importer.resolve_run(str(work), "../elsewhere/data_19990101_000000")
