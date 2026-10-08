"""キーポイントのラベル（走行フォルダの extcam/labels/clip_NN.json）と YOLO pose 形式への書き出し。Qt 非依存。

1 インスタンス = {"bbox": [x1, y1, x2, y2], "kpts": [[u, v, vis] × 4], "src": "manual|pose|model"}
vis は YOLO と同じ 0=画面外・ラベルなし / 1=隠れている（位置は推定）/ 2=見えている。
人が確認したフレームは confirmed に入る（編集した・「確認済み」を押した）。学習に使うのは既定で確認済みだけ。
確認済みで空リストのフレームは「車がいない」負例として学習に入る。
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time

import cv2

from .clip import Clip, list_clips
from .geometry import FLIP_IDX, KPT_NAMES

CLASS_NAMES = ["car"]


class LabelStore:
    def __init__(self, clip: Clip):
        self.clip = clip
        self.path = os.path.join(clip.ext_dir, "labels", f"{clip.name}.json")
        self.frames: dict[int, list] = {}
        self.confirmed: set[int] = set()
        self.last_frame: int | None = None      # 最後に開いていたフレーム（再開用）
        self.dirty = False
        self._last_dirty = False
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.frames = {int(k): v for k, v in (d.get("frames") or {}).items()}
            self.confirmed = set(int(k) for k in d.get("confirmed") or [])
            lf = d.get("last_frame")
            self.last_frame = int(lf) if lf is not None else None

    def get(self, i: int):
        return self.frames.get(int(i))

    def set(self, i: int, instances: list, confirmed: bool | None = None):
        self.frames[int(i)] = instances
        if confirmed is True:
            self.confirmed.add(int(i))
        elif confirmed is False:
            self.confirmed.discard(int(i))
        self.dirty = True

    def confirm(self, i: int, value: bool = True):
        if int(i) not in self.frames:
            self.frames[int(i)] = []
        (self.confirmed.add if value else self.confirmed.discard)(int(i))
        self.dirty = True

    def is_confirmed(self, i: int) -> bool:
        return int(i) in self.confirmed

    def remove(self, i: int):
        if self.frames.pop(int(i), None) is not None:
            self.confirmed.discard(int(i))
            self.dirty = True

    def set_last(self, i: int):
        """再開位置を覚える。ラベルの変更（dirty）とは分けて扱う（開くたびに一覧を作り直さないため）。"""
        if self.last_frame != int(i):
            self.last_frame = int(i)
            self._last_dirty = True

    def needs_save(self) -> bool:
        return self.dirty or self._last_dirty

    def labeled_indices(self) -> list:
        return sorted(self.frames)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "kpt_names": KPT_NAMES, "classes": CLASS_NAMES,
                       "clip": self.clip.name, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "last_frame": self.last_frame,
                       "confirmed": sorted(self.confirmed),
                       "frames": {str(k): v for k, v in sorted(self.frames.items())}},
                      f, ensure_ascii=False, indent=0)
        os.replace(tmp, self.path)
        self.dirty = False
        self._last_dirty = False


def _yolo_line(inst: dict, w: int, h: int) -> str | None:
    x1, y1, x2, y2 = inst["bbox"]
    x1, x2 = sorted((max(0.0, x1), min(w - 1.0, x2)))
    y1, y2 = sorted((max(0.0, y1), min(h - 1.0, y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    vals = [0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
    for u, v, vis in inst["kpts"]:
        vis = int(vis)
        if vis <= 0 or not (0 <= u < w and 0 <= v < h):
            vals += [0.0, 0.0, 0]
        else:
            vals += [u / w, v / h, vis]
    return " ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in vals)


def export_yolo_pose(run_dirs: list, out_dir: str, val_ratio: float = 0.15, seed: int = 0,
                     progress=None, include_unconfirmed: bool = False) -> dict:
    """ラベル済みフレームを YOLO pose のデータセットにする。返り値: {"data_yaml", "n_train", "n_val"}。

    検証用は**クリップ単位ではなくフレーム単位**で分ける（クリップが少ないうちは仕方ない）。
    隣のフレームはほぼ同じ絵なので、検証の数値は楽観的に出る点に注意。
    """
    items = []
    for rd in run_dirs:
        for num, _ in list_clips(rd):
            clip = Clip(rd, num)
            store = LabelStore(clip)
            for i in store.labeled_indices():
                if include_unconfirmed or store.is_confirmed(i):
                    items.append((clip, i, store.get(i)))
    if not items:
        raise ValueError("学習に使えるフレームがありません（確認済みのラベルが 0 件）")
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) >= 5 else 0
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    counts = {"train": 0, "val": 0}
    for k, (clip, i, insts) in enumerate(items):
        split = "val" if k < n_val else "train"
        img = clip.cached_frame(i)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = f"{os.path.basename(clip.run_dir)}_{clip.name}_{i:06d}"
        cv2.imwrite(os.path.join(out_dir, "images", split, stem + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = [ln for ln in (_yolo_line(x, w, h) for x in insts or []) if ln]
        with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        counts[split] += 1
        if progress:
            progress(k + 1, len(items))
    if counts["val"] == 0:      # 少なすぎるときは学習データを検証にも使う（数値は参考程度）
        val_rel = "images/train"
    else:
        val_rel = "images/val"
    data_yaml = os.path.join(out_dir, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: {val_rel}\n"
                f"kpt_shape: [{len(KPT_NAMES)}, 3]\nflip_idx: {FLIP_IDX}\n"
                f"# kpts: {', '.join(KPT_NAMES)}\n"
                f"names:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(CLASS_NAMES)))
    return {"data_yaml": data_yaml, "n_train": counts["train"], "n_val": counts["val"]}
