"""YOLO11n-pose をキーポイント（ルーフ 4 隅）で学習する。GUI からは別プロセスで呼ぶ（Windows の DLL 競合と GPU メモリを避ける）。

    python -m extcam.train --runs data/data_A data/data_B --out models/extcam/run1 --epochs 100 --imgsz 1280

データセットは <out>/dataset に書き出し、重みは <out>/weights/best.pt。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from extcam.labels import export_yolo_pose   # noqa: E402


def train(run_dirs, out_dir, model="yolo11n-pose.pt", epochs=100, imgsz=1280, batch=8,
          device=None, patience=30, workers=2, val_ratio=0.15, include_unconfirmed=False) -> dict:
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ds = export_yolo_pose(run_dirs, os.path.join(out_dir, "dataset"), val_ratio=val_ratio,
                          include_unconfirmed=include_unconfirmed,
                          progress=lambda k, n: print(f"export {k}/{n}", flush=True) if k % 50 == 0 or k == n else None)
    print(f"データセット: train {ds['n_train']} / val {ds['n_val']} → {ds['data_yaml']}", flush=True)
    from ultralytics import YOLO   # 遅延 import（起動を軽くする）
    y = YOLO(model)
    kw = dict(data=ds["data_yaml"], epochs=int(epochs), imgsz=int(imgsz), batch=int(batch),
              project=out_dir, name="train", exist_ok=True, patience=int(patience),
              workers=int(workers), verbose=True,
              # 固定カメラ・横からの俯角なので、上下反転と大きな回転は現実に無い絵を作るだけ
              flipud=0.0, fliplr=0.5, degrees=0.0, mosaic=1.0, hsv_v=0.5)
    if device not in (None, ""):
        kw["device"] = device
    t0 = time.time()
    y.train(**kw)
    best = os.path.join(out_dir, "train", "weights", "best.pt")
    info = {"best": best if os.path.isfile(best) else None, "dataset": ds, "model": model,
            "epochs": epochs, "imgsz": imgsz, "runs": [os.path.abspath(r) for r in run_dirs],
            "elapsed_s": round(time.time() - t0, 1), "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "extcam_train.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print("BEST", info["best"], flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLO11n-pose をルーフ 4 点で学習する")
    ap.add_argument("--runs", nargs="+", required=True, help="ラベル済みの走行フォルダ（複数可）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11n-pose.pt", help="初期重み（続きから学習するなら前回の best.pt）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="車が小さく写るので 1280 を既定にする")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--include-unconfirmed", action="store_true",
                    help="人が確認していない仮ラベル（自己位置の投影・モデル）も使う")
    a = ap.parse_args(argv)
    info = train(a.runs, a.out, a.model, a.epochs, a.imgsz, a.batch, a.device, a.patience, a.workers,
                 include_unconfirmed=a.include_unconfirmed)
    return 0 if info["best"] else 1


if __name__ == "__main__":
    sys.exit(main())
