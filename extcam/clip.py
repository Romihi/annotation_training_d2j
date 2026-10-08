"""走行フォルダの extcam/clip_NN.mkv と clip_NN.frames.csv を読む。Qt 非依存。

frames.csv（togikaidrive の runtime/ext_camera.py が書く）: frame, t_cam, t_jetson, key
t_jetson は車両 Jetson の時刻（catalog の _timestamp_ms / 1000 と同じ時計）。
"""
from __future__ import annotations

import bisect
import csv
import glob
import os
import re

import cv2
import numpy as np

EXT_DIR = "extcam"
VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")   # sidecam は mkv。取り込み（importer）は ffmpeg が無いと avi


def list_clips(run_dir: str) -> list:
    """[(番号, mkv のパス), ...]"""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, EXT_DIR, "clip_*.*"))):
        m = re.search(r"clip_(\d+)(\.[A-Za-z0-9]+)$", p)
        if m and m.group(2).lower() in VIDEO_EXTS and int(m.group(1)) not in out:
            out[int(m.group(1))] = p
    return sorted(out.items())


def clip_video_path(ext_dir: str, num: int) -> str:
    for e in VIDEO_EXTS:
        p = os.path.join(ext_dir, f"clip_{int(num):02d}{e}")
        if os.path.isfile(p):
            return p
    return os.path.join(ext_dir, f"clip_{int(num):02d}.mkv")


class Clip:
    def __init__(self, run_dir: str, num: int):
        self.run_dir = os.path.abspath(run_dir)
        self.num = int(num)
        self.ext_dir = os.path.join(self.run_dir, EXT_DIR)
        self.video_path = clip_video_path(self.ext_dir, self.num)
        self.frames_path = os.path.join(self.ext_dir, f"clip_{self.num:02d}.frames.csv")
        self.t_cam, self.t_jetson = [], []
        if os.path.isfile(self.frames_path):
            with open(self.frames_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.t_cam.append(float(row["t_cam"]))
                    self.t_jetson.append(float(row.get("t_jetson") or row["t_cam"]))
        self._cap = None
        self._pos = -1          # 直前に読んだフレーム番号
        self._size = None

    @property
    def name(self) -> str:
        return f"clip_{self.num:02d}"

    def __len__(self):
        if self.t_jetson:
            return len(self.t_jetson)
        cap = self._open()
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _open(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise IOError(f"動画を開けません: {self.video_path}")
            self._pos = -1
        return self._cap

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def image_size(self):
        if self._size is None:
            cap = self._open()
            self._size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return self._size

    def read(self, i: int):
        """フレーム i（BGR）。番号が frames.csv の行と必ず一致するよう、シークせず順に読む。

        clip は concat で時刻が振り直された可変フレームレートの動画なので、OpenCV の
        CAP_PROP_POS_FRAMES（時刻から番号を逆算する）はずれることがある。後ろへ戻るときは
        開き直して grab（表示用の変換をしない）で読み進める。20 s・60 fps で数秒。
        """
        i = int(i)
        if i <= self._pos:
            self.close()
        cap = self._open()
        while self._pos < i - 1:
            if not cap.grab():
                return None
            self._pos += 1
        ok, frame = cap.read()
        if not ok:
            return None
        self._pos += 1
        return frame

    def frame_cache_path(self, i: int) -> str:
        return os.path.join(self.ext_dir, "frames", self.name, f"{int(i):06d}.jpg")

    def cached_frame(self, i: int):
        """ラベル付け用に JPEG で書き出してあればそれを、無ければ動画から読む（結果は書き出す）。"""
        p = self.frame_cache_path(i)
        if os.path.isfile(p):
            img = cv2.imread(p)
            if img is not None:
                return img
        img = self.read(i)
        if img is not None:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return img

    def extract(self, indices, progress=None) -> int:
        """指定フレームを順に読み、JPEG のキャッシュへ書き出す（シークを避けて 1 回で読む）。"""
        want = sorted(set(int(i) for i in indices))
        todo = [i for i in want if not os.path.isfile(self.frame_cache_path(i))]
        if not todo:
            return 0
        cap = cv2.VideoCapture(self.video_path)
        n, k = 0, 0
        try:
            os.makedirs(os.path.dirname(self.frame_cache_path(0)), exist_ok=True)
            while k < len(todo):
                ok, img = cap.read()
                if not ok:
                    break
                if n == todo[k]:
                    cv2.imwrite(self.frame_cache_path(n), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    k += 1
                    if progress:
                        progress(k, len(todo))
                n += 1
        finally:
            cap.release()
        return k

    def time_of(self, i: int):
        return self.t_jetson[i] if 0 <= i < len(self.t_jetson) else None

    def index_at(self, t_jetson: float) -> int:
        """時刻に最も近いフレーム番号。"""
        ts = self.t_jetson
        if not ts:
            return 0
        k = bisect.bisect_left(ts, t_jetson)
        if k <= 0:
            return 0
        if k >= len(ts):
            return len(ts) - 1
        return k if ts[k] - t_jetson < t_jetson - ts[k - 1] else k - 1

    def sample_indices(self, step: int = 0, count: int = 0) -> list:
        """ラベル付け候補: step フレームおき、または全体から count 枚を等間隔。"""
        n = len(self)
        if count and count > 0:
            return sorted(set(np.linspace(0, n - 1, min(count, n)).round().astype(int).tolist()))
        step = max(1, int(step or 30))
        return list(range(0, n, step))
