# -*- coding: utf-8 -*-
"""TogivadTrainingManager のヘッドレス統合テスト。

main.py の train_sequence_model ダイアログが渡すのと同じデータ構造
（annotations / images / source_images_map / PoseSourceManager）を
記録セッションから直接構築し、GUI なしで学習〜単一フレーム予測まで検証する。

使い方（annotation_training_d2j ディレクトリで）:
    PYTHONUTF8=1 python dev/test_togivad_training.py <data_dir> [--epochs 2]
"""
import argparse
import json
import os
import sys
import tempfile
from glob import glob

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# Windows+CUDA の import 順序制約（main.py 冒頭と同じ）:
# pandas → torch を PyQt5 より先に import する。managers は内部で
# PyQt5 → torch の順に import するため、先に torch を読み込んでおかないと
# c10.dll の初期化に失敗する（WinError 1114）
import pandas  # noqa: F401,E402
import torch   # noqa: F401,E402

from managers.pose_manager import PoseSourceManager
from managers.togivad_training_manager import TogivadTrainingManager


def load_session(data_dir):
    """catalog → main.py 相当の annotations / images / source_images_map"""
    rows = []
    for p in sorted(glob(os.path.join(data_dir, "catalog_*.catalog"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("_timestamp_ms", 0))

    pose_manager = PoseSourceManager()
    annotations, images, cam_images = {}, [], []
    img_dir = os.path.join(data_dir, "images")
    for idx, r in enumerate(rows):
        path = os.path.join(img_dir, r.get("cam/image_array", ""))
        images.append(path)
        cam_images.append(path)
        ann = {"angle": float(r.get("user/angle", 0.0)),
               "throttle": float(r.get("user/throttle", 0.0))}
        if r.get("enc/speed") is not None:
            ann["speed"] = float(r["enc/speed"])
        # main.py と同様、数値センサ値（pose/speed 等）も保持する
        for k, v in r.items():
            if isinstance(v, (int, float)) and not k.startswith("_") \
                    and k not in ann:
                ann[k] = v
        annotations[idx] = ann
        pose_manager.ingest_entry(idx, r)
    return annotations, images, {"cam": cam_images}, pose_manager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--subset", type=int, default=1200,
                    help="使用フレーム数（0=全フレーム）")
    args = ap.parse_args()

    annotations, images, source_images_map, pose_manager = \
        load_session(args.data_dir)
    print(f"frames={len(images)} pose_sources={pose_manager.available_sources()}")

    valid_indexes = list(range(len(images)))
    if args.subset:
        valid_indexes = valid_indexes[:args.subset]

    def progress(current, total, message=None):
        if message and ("完了" in message or "構築" in message
                        or "早期終了" in message):
            print(f"  [{current}/{total}] {message.splitlines()[0]}")
        return True

    out_dir = tempfile.mkdtemp(prefix="togivad_test_models_")
    manager = TogivadTrainingManager(out_dir, mlflow_manager=None)

    # (pose_source, pred_seconds, pred_points) の組み合わせを検証。
    # 3つ目は既定(1.0s/20点)と異なるホライズン(2.0秒先/10点=200ms間隔)で、
    # 予測秒数・点数の調整が学習・保存・予測に反映されることを確認する。
    cases = [
        ("pose", 1.0, 20),
        ("slam", 1.0, 20),
        ("pose", 2.0, 10),
    ]
    for pose_source, pred_seconds, pred_points in cases:
        print(f"\n=== pose_source={pose_source} "
              f"pred={pred_seconds}s/{pred_points}pts ===")
        result = manager.train(
            valid_indexes=valid_indexes,
            annotations=annotations,
            images=images,
            source_images_map=source_images_map,
            selected_sources=["cam"],
            pose_manager=pose_manager,
            config={
                "pose_source": pose_source,
                "pred_seconds": pred_seconds,
                "pred_points": pred_points,
                "vocab_k": 64,
                "vocab_from_logs": True,
                "ego_dropout": 0.3,
                "epochs": args.epochs,
                "batch_size": 32,
                "learning_rate": 3e-4,
                "val_split": 0.2,
                "use_early_stopping": True,
                "patience": 10,
            },
            progress_callback=progress)
        assert result["status"] == "completed", result
        print(f"  status={result['status']} samples={result['total_sequences']} "
              f"best_val_CE={result['best_val_loss']:.4f} "
              f"top1={result['val_top1']:.1%} ADE={result['val_ade_m']:.3f}m")
        assert os.path.exists(result["model_path"])
        curve = os.path.splitext(result["model_path"])[0] + "_training_curve.png"
        assert os.path.exists(curve), "学習曲線PNGが無い"

        # 単一フレーム予測（マップビュー重畳用 API）の煙テスト
        model, cfg, vocab, meta = TogivadTrainingManager.load_model(
            result["model_path"])
        assert meta["pose_source"] == pose_source
        # 予測秒数・点数が cfg / 語彙形状に反映されていること
        assert cfg.horizon == pred_points, (cfg.horizon, pred_points)
        assert abs(cfg.dt - pred_seconds / pred_points) < 1e-9
        assert vocab.shape[1] == pred_points, vocab.shape
        pred = manager.predict_current(
            model, cfg, vocab, ["cam"], valid_indexes[len(valid_indexes) // 2],
            images, source_images_map, annotations,
            pose_manager=pose_manager, pose_source=pose_source)
        assert pred is not None and pred["best"].shape == (cfg.horizon, 2)
        print(f"  predict_current OK: top1確率={pred['probs'][0]:.3f} "
              f"終端点=({pred['best'][-1][0]:.2f}, {pred['best'][-1][1]:.2f})m")

    print("\nALL OK")


if __name__ == "__main__":
    main()
