"""外部カメラウィンドウの「0. 取り込み」タブ: 他のカメラの動画を走行と時刻合わせして extcam/ へ書き出す。

処理は python -m extcam.importer（analyze の各段 / write）を別プロセスで順に動かす。各段の画像を右で見て、
柵のマスク・カメラの側（地図で範囲をドラッグ）・較正の重なり・検出・時刻合わせを人が確認しながら進める。
手順と判断基準は extcam/skills/extcam-video-sync/SKILL.md。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                             QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from . import importer
from .runlog import CourseMap
from .widgets import ImageView, MapView

IMAGES = [("contact.jpg", "動画の一覧"), ("ref.jpg", "基準フレーム"), ("barrier_vis.jpg", "柵のマスク"),
          ("map.png", "地図（頂点番号）"), ("calib_overlay.jpg", "較正の重なり"), ("det_track.jpg", "車の検出"),
          ("verify_overlay.jpg", "時刻合わせの検証")]
STEP_BUTTONS = [("prepare", "① 下ごしらえ"), ("mask", "② 柵のマスク"), ("mapplot", "③ 地図"),
                ("calibrate", "④ 較正"), ("detect", "⑤ 車の検出"), ("sync", "⑥ 時刻探索"), ("verify", "⑦ 検証")]


class ImportTab(QWidget):
    def __init__(self, ctx, open_run):
        super().__init__()
        from .window import ProcessRunner   # 循環 import を避けて遅延
        self.ctx = ctx
        self.open_run = open_run            # 書き出した走行を開く（ExtcamWindow.open_run）
        self.queue = []
        self.cam_box_drag = None
        lay = QHBoxLayout(self)

        # --- 左: 入力と操作 ---------------------------------------------------
        left = QVBoxLayout()
        files = QGroupBox("入力")
        f = QFormLayout(files)
        self.video = QLineEdit()
        self.data_root = QLineEdit()
        self.work = QLineEdit()
        self.work.setPlaceholderText("空なら <データ>/extcam_imports/<動画名>")
        for label, edit, fn in (("動画", self.video, self._pick_video), ("データ（data/）", self.data_root, self._pick_data),
                                ("作業フォルダ", self.work, self._pick_work)):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            b = QPushButton("…")
            b.setMaximumWidth(32)
            b.clicked.connect(fn)
            row.addWidget(b)
            f.addRow(label, row)
        left.addWidget(files)

        opt = QGroupBox("設定（SKILL.md の手順 2〜6）")
        o = QFormLayout(opt)
        self.hdr = QCheckBox("HDR 動画（iPhone の HLG / Dolby Vision。白っぽく写る）")
        self.ignore_top = QDoubleSpinBox()
        self.ignore_top.setRange(0.0, 0.9)
        self.ignore_top.setSingleStep(0.02)
        self.ignore_top.setValue(0.4)
        self.ignore_rect = QLineEdit()
        self.ignore_rect.setPlaceholderText("x0,y0,x1,y1; …（元の解像度、机・モニター等を除く）")
        self.cam_box = QLineEdit()
        self.cam_box.setPlaceholderText("x0,x1,y0,y1 [m]（右の「地図」タブでドラッグして指定）")
        self.init = QLineEdit()
        self.init.setPlaceholderText("前回の calib.json（同じ場所から撮った動画）")
        b_init = QPushButton("…")
        b_init.setMaximumWidth(32)
        b_init.clicked.connect(self._pick_init)
        hi = QHBoxLayout()
        hi.addWidget(self.init, 1)
        hi.addWidget(b_init)
        self.use_hint = QCheckBox("撮影時刻（iPhone のオリジナル）で走行を絞る")
        self.use_hint.setChecked(True)
        self.date = QLineEdit()
        self.date.setPlaceholderText("撮影時刻が無いとき: YYYYMMDD")
        self.runs = QLineEdit()
        self.runs.setPlaceholderText("走行を明示するとき（空白区切り、glob 可）")
        o.addRow(self.hdr)
        o.addRow("上を除く割合", self.ignore_top)
        o.addRow("除く矩形", self.ignore_rect)
        o.addRow("カメラの範囲", self.cam_box)
        o.addRow("較正の初期値", hi)
        o.addRow(self.use_hint)
        o.addRow("日付", self.date)
        o.addRow("走行", self.runs)
        left.addWidget(opt)

        steps = QGroupBox("手順（各段の画像を右で確認してから次へ）")
        s = QVBoxLayout(steps)
        row = QHBoxLayout()
        for k, (step, label) in enumerate(STEP_BUTTONS):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, st=step: self.run_steps([st]))
            row.addWidget(b)
            if k == 3:
                s.addLayout(row)
                row = QHBoxLayout()
        s.addLayout(row)
        row2 = QHBoxLayout()
        b_all = QPushButton("▶ ②〜⑦ をまとめて")
        b_all.clicked.connect(lambda: self.run_steps([st for st, _ in STEP_BUTTONS[1:]]))
        b_stop = QPushButton("中止")
        b_stop.clicked.connect(self._stop)
        row2.addWidget(b_all)
        row2.addWidget(b_stop)
        s.addLayout(row2)
        row3 = QHBoxLayout()
        self.force = QCheckBox("目安を満たさなくても書く")
        b_write = QPushButton("⑧ 走行フォルダへ書き出す")
        b_write.clicked.connect(self.write)
        row3.addWidget(b_write)
        row3.addWidget(self.force)
        s.addLayout(row3)
        left.addWidget(steps)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setFont(QFont("Monospace", 9))
        left.addWidget(self.log, 1)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(560)
        lay.addWidget(lw)

        # --- 右: 画像・地図・判定 ------------------------------------------------
        right = QTabWidget()
        img_page = QWidget()
        il = QVBoxLayout(img_page)
        self.img_combo = QComboBox()
        for fn, label in IMAGES:
            self.img_combo.addItem(label, fn)
        self.img_combo.currentIndexChanged.connect(self._show_image)
        il.addWidget(self.img_combo)
        self.img = ImageView()
        il.addWidget(self.img, 1)
        right.addTab(img_page, "画像")
        map_page = QWidget()
        ml = QVBoxLayout(map_page)
        ml.addWidget(QLabel("カメラのいる側を、地図上で 2 点クリック（対角）して範囲にする。赤い ▲ は較正で求めたカメラ位置。"))
        self.map = MapView()
        self.map.draw_extra = self._draw_map
        self.map.clicked.connect(self._map_click)
        ml.addWidget(self.map, 1)
        right.addTab(map_page, "地図（カメラの範囲）")
        self.verdict = QTextEdit()
        self.verdict.setReadOnly(True)
        right.addTab(self.verdict, "判定")
        self.right = right
        lay.addWidget(right, 1)

        self.runner = ProcessRunner(self.log, on_done=self._step_done)
        ctx.listeners.append(self._ctx_changed)

    # --- 入力 --------------------------------------------------------------
    def _ctx_changed(self):
        if not self.data_root.text() and self.ctx.run_dir:
            self.data_root.setText(os.path.dirname(self.ctx.run_dir))

    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "動画", os.path.expanduser("~"), "動画 (*.mp4 *.mov *.MOV *.MP4 *.mkv *.avi)")
        if p:
            self.video.setText(p)
            if self.data_root.text():
                self.work.setText(importer.default_work(self.data_root.text(), p))
            self.refresh()

    def _pick_data(self):
        d = QFileDialog.getExistingDirectory(self, "走行フォルダの親（data/）", self.data_root.text() or "")
        if d:
            self.data_root.setText(d)

    def _pick_work(self):
        d = QFileDialog.getExistingDirectory(self, "作業フォルダ（前回の続き）", self.work.text() or self.data_root.text())
        if d:
            self.work.setText(d)
            cfg = os.path.join(d, "import_config.json")
            if os.path.isfile(cfg):
                with open(cfg, encoding="utf-8") as f:
                    c = json.load(f)
                self.video.setText(c.get("video", self.video.text()))
                self.data_root.setText(c.get("data_root", self.data_root.text()))
            self.refresh()

    def _pick_init(self):
        p, _ = QFileDialog.getOpenFileName(self, "前回の calib.json", self.data_root.text(), "JSON (*.json)")
        if p:
            self.init.setText(p)

    def work_dir(self):
        if self.work.text().strip():
            return self.work.text().strip()
        if self.video.text() and self.data_root.text():
            return importer.default_work(self.data_root.text(), self.video.text())
        return None

    # --- 実行 --------------------------------------------------------------
    def _analyze_args(self, steps):
        a = [self.video.text(), "--data-root", self.data_root.text(), "--work", self.work_dir(),
             "--steps", ",".join(steps), "--ignore-top", f"{self.ignore_top.value():.3f}"]
        for r in [r.strip() for r in self.ignore_rect.text().split(";") if r.strip()]:
            a += ["--ignore-rect", r]
        if self.hdr.isChecked():
            a.append("--hdr")
        if self.cam_box.text().strip():
            a += ["--cam-box", self.cam_box.text().strip()]
        if self.init.text().strip():
            a += ["--init", self.init.text().strip()]
        if not self.use_hint.isChecked():
            a.append("--no-hint")
        if self.date.text().strip():
            a += ["--date", self.date.text().strip()]
        if self.runs.text().strip():
            a += ["--runs", *self.runs.text().split()]
        return a

    def run_steps(self, steps):
        if not self.video.text() or not self.data_root.text():
            QMessageBox.warning(self, "取り込み", "動画とデータ（data/）を選んでください")
            return
        if self.runner.running():
            QMessageBox.warning(self, "取り込み", "前の段がまだ動いています")
            return
        self.queue = list(steps)
        self._next()

    def _next(self):
        if not self.queue:
            return
        st = self.queue.pop(0)
        self.log.appendPlainText(f"--- {dict(STEP_BUTTONS).get(st, st)} ---")
        self.runner.start("extcam.importer", ["analyze", *self._analyze_args([st])])
        self.current = st

    def _stop(self):
        self.queue = []
        self.runner.stop()

    def _step_done(self, code):
        self.refresh(select=getattr(self, "current", None))
        if code not in (0, 2):              # 2 = 目安を満たさない（解析自体は終わっている）
            self.queue = []
            self.log.appendPlainText("この段で止まりました。ログと画像を確認してください")
            return
        if self.queue:
            self._next()

    def write(self):
        w = self.work_dir()
        if not w or not os.path.isfile(os.path.join(w, "sync.json")):
            QMessageBox.warning(self, "書き出し", "先に ⑥ 時刻探索 まで進めてください")
            return
        try:
            with open(os.path.join(w, "sync.json"), encoding="utf-8") as f:
                best = json.load(f)["best"]
            run = importer.resolve_run(w, best["run"])
        except Exception as e:   # noqa: BLE001
            QMessageBox.warning(self, "書き出し", f"書き出し先の走行を特定できません: {e}")
            return
        if QMessageBox.question(self, "書き出し", f"次の走行フォルダの extcam/ へ書き出します。\n\n{run}\n"
                                f"動画の t=0 = {best.get('video_t0_local')}\n\nよいですか？") != QMessageBox.Yes:
            return
        args = ["write", "--work", w, "--run", run]
        if self.force.isChecked():
            args.append("--force")
        self.current = "write"
        self.runner.on_done = self._write_done
        self.runner.start("extcam.importer", args)

    def _write_done(self, code):
        self.runner.on_done = self._step_done
        if code != 0:
            QMessageBox.warning(self, "書き出し", "書き出せませんでした（ログを確認。目安を満たしていなければ「目安を満たさなくても書く」）")
            return
        try:
            with open(os.path.join(self.work_dir(), "sync.json"), encoding="utf-8") as f:
                run = importer.resolve_run(self.work_dir(), json.load(f)["best"]["run"])
        except (OSError, ValueError, KeyError):
            return
        if QMessageBox.question(self, "書き出し", f"{run} に書き出しました。開いて較正・ラベルへ進みますか？") == QMessageBox.Yes:
            self.open_run(os.path.abspath(run), select_last_clip=True)

    # --- 表示 --------------------------------------------------------------
    def refresh(self, select=None):
        w = self.work_dir()
        order = {"prepare": "contact.jpg", "mask": "barrier_vis.jpg", "mapplot": "map.png", "calibrate": "calib_overlay.jpg",
                 "detect": "det_track.jpg", "sync": "verify_overlay.jpg", "verify": "verify_overlay.jpg"}
        if select in order:
            k = [fn for fn, _ in IMAGES].index(order[select])
            self.img_combo.setCurrentIndex(k)
        self._show_image()
        self._load_course()
        if w and os.path.isdir(w):
            self._show_verdict(importer.evaluate(w))
        if select == "mapplot":
            self.right.setCurrentIndex(1)

    def _show_image(self, *_):
        w = self.work_dir()
        fn = self.img_combo.currentData()
        p = os.path.join(w, fn) if w and fn else None
        img = cv2.imread(p) if p and os.path.isfile(p) else None
        self.img.set_image(img, keep_view=False)

    def _load_course(self):
        w = self.work_dir()
        cfg = os.path.join(w, "import_config.json") if w else None
        bounds = None
        if cfg and os.path.isfile(cfg):
            with open(cfg, encoding="utf-8") as f:
                bounds = json.load(f).get("bounds")
        if bounds and os.path.isfile(bounds):
            md = os.path.dirname(bounds)
            if self.map.course is None or self.map.course.map_dir != md:
                self.map.set_course(CourseMap(md))
            else:
                self.map.redraw()

    def _show_verdict(self, summ):
        lines = []
        for name, c in (summ.get("checks") or {}).items():
            mark = "✔" if c["ok"] else "✖"
            col = "#108010" if c["ok"] else "#c02020"
            lines.append(f"<p style='color:{col}'><b>{mark} {name}</b>: {c['detail']}</p>")
        s = summ.get("sync")
        if s:
            lines.insert(0, f"<h3>{s['name']}  動画の t=0 = {s['video_t0_local']}</h3>"
                            f"<p>一致 F={s['F']:.3f}、中央値 {s['med_px']:.1f} px、2 位との差 {s.get('margin')}</p>")
        if summ.get("hint_t"):
            import datetime as dtm
            lines.insert(0, f"<p>撮影時刻（メタデータ）: {dtm.datetime.fromtimestamp(summ['hint_t'])}</p>")
        lines.append("<p><b>総合: " + ("合格（⑧ で書き出せる）" if summ.get("ok") else "要確認") + "</b></p>")
        self.verdict.setHtml("".join(lines))

    # --- 地図でカメラの範囲 --------------------------------------------------
    def _map_click(self, x, y, button):
        if button != 1:
            return
        if self.cam_box_drag is None:
            self.cam_box_drag = (x, y)
        else:
            x0, y0 = self.cam_box_drag
            self.cam_box.setText(f"{min(x0, x):.2f},{max(x0, x):.2f},{min(y0, y):.2f},{max(y0, y):.2f}")
            self.cam_box_drag = None
        self.map.redraw()

    def _draw_map(self, ax):
        txt = self.cam_box.text().strip()
        if txt:
            try:
                x0, x1, y0, y1 = map(float, txt.split(","))
                ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=True, alpha=0.15, color="#d02020"))
            except ValueError:
                pass
        if self.cam_box_drag:
            ax.plot(*self.cam_box_drag, "x", color="#d02020", ms=10)
        w = self.work_dir()
        cj = os.path.join(w, "calib.json") if w else None
        if cj and os.path.isfile(cj):
            with open(cj, encoding="utf-8") as f:
                cp = json.load(f).get("camera_position_m")
            if cp:
                ax.plot(cp[0], cp[1], "^", color="#d02020", ms=11)
                ax.annotate(f"カメラ h={cp[2]:.1f}m", cp[:2], xytext=(6, -12), textcoords="offset points", color="#d02020")
