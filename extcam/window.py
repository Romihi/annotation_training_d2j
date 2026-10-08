"""外部カメラ（sidecam）の走行動画: 較正・ラベル付け・学習・追跡の別ウィンドウ。

main.py のツールバー「🎥」から開く。単体でも起動できる:
    python -m extcam.window [data/data_<TS>]

学習と追跡（推論）は別プロセス（python -m extcam.train / extcam.tracker）で動かし、ログをこの画面に流す。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QKeySequence
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                             QPlainTextEdit, QProgressDialog, QPushButton, QShortcut, QSlider, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

_TOOL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOL_ROOT not in sys.path:
    sys.path.insert(0, _TOOL_ROOT)

from extcam import tracker as trk                                       # noqa: E402
from extcam.clip import Clip, list_clips                                # noqa: E402
from extcam.clipcam import load_clip_camera, save_clip_setup, _apply_h   # noqa: E402
from extcam.geometry import (KPT_NAMES, CameraCalib, VehicleKpts,       # noqa: E402
                             body_to_map, kpts_from_pose, load_setup, save_setup)
from extcam.labels import LabelStore                                    # noqa: E402
from extcam.runlog import CourseMap, RunLog, repo_root_of, resolve_map_dir  # noqa: E402
from extcam.widgets import ImageView, MapView, pen                      # noqa: E402

KPT_COLORS = ["#ff3030", "#ff9a00", "#30a0ff", "#30e070"]   # 前左・前右・後左・後右
KPT_SHORT = ["前左", "前右", "後左", "後右"]
ID_COLORS = ["#ff4040", "#40c0ff", "#ffd030", "#a060ff", "#40ff80", "#ff80c0", "#ffffff"]


def _id_color(tid: int) -> str:
    return ID_COLORS[int(tid) % len(ID_COLORS)]


class Context:
    """ウィンドウ全体で共有する状態（走行フォルダ・クリップ・較正・地図・自己位置）。"""

    def __init__(self):
        self.run_dir = None
        self.clip: Clip | None = None
        self.calib: CameraCalib | None = None
        self.veh = VehicleKpts()
        self.pairs = []
        self.course: CourseMap | None = None
        self.runlog: RunLog | None = None
        self.stab = None                # 手持ちクリップのフレーム → 基準フレームの変換（N×3×3）。固定カメラは None
        self.ref_index = None           # 手持ちクリップの基準フレーム（較正はこの画素で持つ）
        self.setup_file = None          # 使っている較正ファイル（クリップ専用 or 走行の calib.json）
        self.listeners = []

    def setup_path(self):
        if self.setup_file:
            return self.setup_file
        return trk.setup_path(self.run_dir) if self.run_dir else None

    def _load_camera(self):
        """クリップに合う較正を読む（clip_NN.calib.json があればそれ、無ければ走行の calib.json）。"""
        veh_json = os.path.join(repo_root_of(self.run_dir), "vehicle.json")
        default_veh = VehicleKpts.from_vehicle_json(veh_json) if os.path.isfile(veh_json) else VehicleKpts()
        cc, veh, pairs, md, path = load_clip_camera(self.run_dir, self.clip.num if self.clip else None)
        self.setup_file = path
        exists = os.path.isfile(path)
        self.calib = cc.calib if cc else None
        self.stab = cc.stab if cc else None
        self.ref_index = cc.ref_index if cc else None
        self.veh = veh if exists else default_veh
        self.pairs = pairs if exists else []
        return md

    # 手持ちクリップの座標変換（固定カメラでは恒等）。較正・対応点は基準フレームの画素で持つ
    def H(self, i):
        if self.stab is None or i is None or not (0 <= int(i) < len(self.stab)):
            return None
        h = self.stab[int(i)]
        return h if np.isfinite(h).all() else None

    def to_ref(self, uv, i):
        return _apply_h(self.H(i), uv)

    def to_frame(self, uv, i):
        h = self.H(i)
        return _apply_h(None if h is None else np.linalg.inv(h), uv)

    def load_run(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)
        clips = list_clips(self.run_dir)
        self.clip = Clip(self.run_dir, clips[0][0]) if clips else None
        map_dir = resolve_map_dir(self.run_dir)
        md = self._load_camera()
        if md and os.path.isdir(md):
            map_dir = md
        self.course = CourseMap(map_dir)
        try:
            self.runlog = RunLog(self.run_dir)
        except Exception as e:   # noqa: BLE001
            print("自己位置を読めません:", e)
            self.runlog = None
        self.notify()

    def set_clip(self, num: int):
        if self.clip is not None:
            self.clip.close()
        self.clip = Clip(self.run_dir, num)
        md = self._load_camera()
        if md and os.path.isdir(md) and (self.course is None or self.course.map_dir != md):
            self.course = CourseMap(md)
        self.notify()

    def save_setup(self):
        save_clip_setup(self.setup_path(), self.calib, self.veh, self.pairs,
                        self.course.map_dir if self.course else None)

    def notify(self):
        for fn in self.listeners:
            fn()


class ProcessRunner:
    """python -m extcam.xxx を QProcess で動かし、出力をテキスト欄へ流す。"""

    def __init__(self, log: QPlainTextEdit, on_line=None, on_done=None):
        self.log, self.on_line, self.on_done = log, on_line, on_done
        self.proc = None

    def running(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, module: str, args: list):
        if self.running():
            QMessageBox.warning(None, "実行中", "前の処理がまだ動いています")
            return
        self.proc = QProcess()
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", _TOOL_ROOT + os.pathsep + env.value("PYTHONPATH", ""))
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONUTF8", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(_TOOL_ROOT)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.log.appendPlainText(f"$ python -m {module} {' '.join(args)}")
        self.proc.start(sys.executable, ["-m", module] + [str(a) for a in args])

    def stop(self):
        if self.running():
            self.proc.kill()

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.replace("\r", "\n").splitlines():
            if not line.strip():
                continue
            self.log.appendPlainText(line)
            if self.on_line:
                self.on_line(line)

    def _finished(self, code, _status):
        self.log.appendPlainText(f"（終了コード {code}）")
        if self.on_done:
            self.on_done(code)


def _frame_slider_row(parent_layout, on_change):
    row = QHBoxLayout()
    slider = QSlider(Qt.Horizontal)
    spin = QSpinBox()
    spin.setMaximumWidth(90)
    tlabel = QLabel("")
    slider.valueChanged.connect(spin.setValue)
    spin.valueChanged.connect(slider.setValue)
    spin.valueChanged.connect(on_change)
    row.addWidget(QLabel("フレーム"))
    row.addWidget(slider, 1)
    row.addWidget(spin)
    row.addWidget(tlabel)
    parent_layout.addLayout(row)
    return slider, spin, tlabel


# ===========================================================================
# 1. 較正
# ===========================================================================

class CalibTab(QWidget):
    """画像の点 → 地図の点（外形の頂点へ吸着）の順にクリックして対応点を作り、カメラを較正する。"""

    def __init__(self, ctx: Context):
        super().__init__()
        self.ctx = ctx
        self.pending_img = None
        self.frame_img = None
        lay = QVBoxLayout(self)
        help_ = QLabel("① 画像で床の上の点（コース外形の角など）を左クリック → ② 地図で同じ点を左クリック（頂点に吸着）。"
                       "6 組以上を画面全体に散らすと焦点距離まで安定して求まる。右ドラッグで移動・ホイールで拡大。")
        help_.setWordWrap(True)
        lay.addWidget(help_)
        self.slider, self.spin, self.tlabel = _frame_slider_row(lay, self.load_frame)
        split = QSplitter(Qt.Horizontal)
        self.img = ImageView()
        self.img.overlays.append(self._draw_img)
        self.img.pressed.connect(self._img_click)
        self.map = MapView()
        self.map.draw_extra = self._draw_map
        self.map.clicked.connect(self._map_click)
        split.addWidget(self.img)
        split.addWidget(self.map)
        split.setSizes([700, 500])
        lay.addWidget(split, 1)

        bottom = QHBoxLayout()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["u", "v", "x [m]", "y [m]", "誤差 [px]"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMaximumHeight(170)
        bottom.addWidget(self.table, 2)

        side = QVBoxLayout()
        b_del = QPushButton("選択した対応点を削除")
        b_del.clicked.connect(self._delete_pair)
        b_solve = QPushButton("較正する")
        b_solve.clicked.connect(self.solve)
        b_save = QPushButton("保存")
        b_save.clicked.connect(self._save)
        self.b_save = b_save
        b_import = QPushButton("他の走行の較正を読み込む…")
        b_import.clicked.connect(self._import)
        b_intr = QPushButton("内部パラメータ（K, dist）を読み込む…")
        b_intr.clicked.connect(self._load_intrinsics)
        self.result = QLabel("未較正")
        self.result.setWordWrap(True)
        for w in (b_del, b_solve, b_save, b_import, b_intr, self.result):
            side.addWidget(w)
        bottom.addLayout(side, 1)

        veh_box = QGroupBox("車両のキーポイント（ルーフ 4 隅、base_link＝後軸中心）")
        form = QFormLayout(veh_box)
        self.veh_spins = {}
        for key, label in (("front_x", "前端 x [m]"), ("rear_x", "後端 x [m]"), ("half_w", "半幅 [m]"),
                           ("height", "高さ h [m]"), ("body_half_w", "ボディ半幅 [m]")):
            sp = QDoubleSpinBox()
            sp.setRange(-1.0, 1.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.005)
            sp.valueChanged.connect(self._veh_changed)
            form.addRow(label, sp)
            self.veh_spins[key] = sp
        bottom.addWidget(veh_box, 1)
        lay.addLayout(bottom)
        self.intrinsics = None          # (K, dist) を読み込んだら固定で使う
        self.saved_label = QLabel("")
        self.saved_label.setStyleSheet("color: #808080;")
        side.addWidget(self.saved_label)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._autosave)
        ctx.listeners.append(self.refresh)

    def _autosave(self):
        """対応点・較正・車両寸法を変えるたびに calib.json へ書く（「保存」を押し忘れても消えない）。"""
        self._save_timer.stop()
        if self.ctx.run_dir is None:
            return
        try:
            self.ctx.save_setup()
            self.saved_label.setText("自動保存 " + time.strftime("%H:%M:%S"))
        except Exception as e:   # noqa: BLE001
            self.saved_label.setText(f"自動保存に失敗: {e}")

    # --- 状態の反映 --------------------------------------------------------
    def refresh(self):
        self._save_timer.stop()         # 前の走行への保存待ちを、読み込んだ別の走行へ書かない
        c = self.ctx
        n = len(c.clip) if c.clip else 0
        self.slider.setRange(0, max(0, n - 1))
        self.spin.setRange(0, max(0, n - 1))
        if c.stab is not None and c.ref_index is not None and 0 <= c.ref_index < n:
            self.spin.blockSignals(True)          # 手持ちのクリップは基準フレームを開く（較正はその画素）
            self.spin.setValue(int(c.ref_index))
            self.slider.setValue(int(c.ref_index))
            self.spin.blockSignals(False)
        for k, sp in self.veh_spins.items():
            sp.blockSignals(True)
            sp.setValue(getattr(c.veh, k))
            sp.blockSignals(False)
        self.map.set_course(c.course)
        if c.setup_path():
            self.b_save.setText(f"保存（{os.path.basename(c.setup_path())}）")
        self._fill_table()
        self._show_result()
        self.load_frame(self.spin.value())

    def load_frame(self, i):
        c = self.ctx
        if not c.clip:
            self.img.set_image(None)
            return
        self.frame_img = c.clip.read(int(i))
        self.img.set_image(self.frame_img)
        t = c.clip.time_of(int(i))
        self.tlabel.setText(time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}" if t else "")

    def _veh_changed(self):
        for k, sp in self.veh_spins.items():
            setattr(self.ctx.veh, k, sp.value())
        if self.ctx.veh.body_front_x < self.ctx.veh.front_x:
            self.ctx.veh.body_front_x = self.ctx.veh.front_x
        self.ctx.veh.body_rear_x = min(self.ctx.veh.body_rear_x, self.ctx.veh.rear_x)
        self.img.update()
        self._save_timer.start()

    def _fill_table(self):
        pairs = self.ctx.pairs
        errs = None
        if self.ctx.calib is not None and pairs:
            errs = self.ctx.calib.reproj_errors([p["img"] for p in pairs], [p["map"] for p in pairs])
        self.table.setRowCount(len(pairs))
        for r, p in enumerate(pairs):
            vals = [p["img"][0], p["img"][1], p["map"][0], p["map"][1], errs[r] if errs is not None else None]
            for col, v in enumerate(vals):
                it = QTableWidgetItem("" if v is None else f"{v:.2f}" if col >= 2 else f"{v:.1f}")
                if col == 4 and v is not None and v > 3:
                    it.setForeground(QColor("#d02020"))
                self.table.setItem(r, col, it)

    def _show_result(self):
        cal = self.ctx.calib
        if cal is None:
            self.result.setText(f"未較正（対応点 {len(self.ctx.pairs)} 組）")
            return
        pos = cal.camera_position()
        rms = f"{cal.reproj_rms_px:.2f} px（{cal.n_points} 組）" if cal.reproj_rms_px is not None else f"—（{cal.intrinsics_source}）"
        txt = (f"再投影誤差 {rms}\n焦点距離 {cal.K[0][0]:.0f} px（{cal.intrinsics_source}）\n"
               f"カメラ位置 ({pos[0]:.2f}, {pos[1]:.2f}, 高さ {pos[2]:.2f}) m")
        if self.ctx.course and len(self.ctx.course.vertices()):
            v = self.ctx.course.vertices()
            res = [cal.ground_resolution(p, self.ctx.veh.height) for p in v[cal.in_front(v)]]
            if res:
                txt += f"\n1 px の距離: 近 {min(res) * 100:.1f} 〜 遠 {max(res) * 100:.1f} cm"
        self.result.setText(txt)

    # --- 操作 --------------------------------------------------------------
    def _img_click(self, x, y, button, _mods):
        if button != int(Qt.LeftButton):
            return
        rx, ry = self.ctx.to_ref([[x, y]], self.spin.value())[0]     # 対応点は基準フレームの画素で持つ
        self.pending_img = (float(rx), float(ry))
        self.img.update()
        self.map.redraw()

    def _map_click(self, x, y, button):
        if button != 1 or self.pending_img is None:
            return
        if self.ctx.course:
            x, y, _ = self.ctx.course.snap(x, y, radius=0.3)
        self.ctx.pairs.append({"img": [float(self.pending_img[0]), float(self.pending_img[1])],
                               "map": [float(x), float(y)]})
        self.pending_img = None
        self._fill_table()
        self._show_result()
        self.img.update()
        self.map.redraw()
        self._autosave()

    def _delete_pair(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.ctx.pairs[r]
        self._fill_table()
        self.img.update()
        self.map.redraw()
        if rows:
            self._autosave()

    def solve(self):
        c = self.ctx
        if len(c.pairs) < 4:
            QMessageBox.warning(self, "較正", "対応点が 4 組以上必要です（6 組以上を推奨）")
            return
        if c.clip is None:
            return
        K, dist = self.intrinsics if self.intrinsics else (None, None)
        try:
            c.calib = CameraCalib.fit([p["img"] for p in c.pairs], [p["map"] for p in c.pairs],
                                      c.clip.image_size, K=K, dist=dist)
        except Exception as e:   # noqa: BLE001
            QMessageBox.warning(self, "較正", f"較正できません: {e}")
            return
        self._fill_table()
        self._show_result()
        self.img.update()
        self.map.redraw()
        self._autosave()

    def _save(self):
        if self.ctx.run_dir is None:
            return
        self.ctx.save_setup()
        QMessageBox.information(self, "保存", f"保存しました: {self.ctx.setup_path()}")

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "calib.json を選ぶ", self.ctx.run_dir or "", "JSON (*.json)")
        if not path:
            return
        cam, veh, pairs, _ = load_setup(path)
        self.ctx.calib, self.ctx.veh, self.ctx.pairs = cam, veh, pairs
        self.refresh()
        self._autosave()

    def _load_intrinsics(self):
        path, _ = QFileDialog.getOpenFileName(self, "K と dist を含む JSON", "", "JSON (*.json)")
        if not path:
            return
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d = d.get("camera", d)
        self.intrinsics = (np.array(d["K"], float), np.array(d.get("dist") or [0] * 5, float))
        QMessageBox.information(self, "内部パラメータ", "読み込みました。次の「較正する」から固定値として使います。")

    # --- 重ね描き ----------------------------------------------------------
    def _draw_img(self, p, tw):
        c = self.ctx
        if c.calib is not None and c.course is not None:
            p.setPen(pen("#40ff60", 1.5))
            for pl in c.course.polylines():
                dense = np.vstack([np.linspace(pl[k], pl[k + 1], 12) for k in range(len(pl) - 1)])
                ok = c.calib.in_front(dense)
                uv = c.to_frame(c.calib.project(dense), self.spin.value())
                for k in range(len(uv) - 1):
                    if ok[k] and ok[k + 1]:
                        p.drawLine(tw(*uv[k]), tw(*uv[k + 1]))
        fi = self.spin.value()
        for k, pr in enumerate(c.pairs):
            q = tw(*c.to_frame([pr["img"]], fi)[0])
            p.setPen(pen("#ffd000", 2))
            p.drawEllipse(q, 5, 5)
            p.drawText(QPointF(q.x() + 7, q.y() - 7), str(k + 1))
            if c.calib is not None:
                pj = c.to_frame(c.calib.project([pr["map"]]), fi)[0]
                p.setPen(pen("#ff40ff", 1.5))
                p.drawLine(q, tw(*pj))
        if self.pending_img:
            p.setPen(pen("#ff3030", 2.5))
            q = tw(*c.to_frame([self.pending_img], fi)[0])
            p.drawLine(q.x() - 9, q.y(), q.x() + 9, q.y())
            p.drawLine(q.x(), q.y() - 9, q.x(), q.y() + 9)

    def _draw_map(self, ax):
        for k, pr in enumerate(self.ctx.pairs):
            ax.plot(*pr["map"], "o", ms=8, mfc="none", mec="#e0a000", mew=2, zorder=5)
            ax.annotate(str(k + 1), pr["map"], xytext=(5, 5), textcoords="offset points", color="#a06000")
        cal = self.ctx.calib
        if cal is not None:
            cp = cal.camera_position()
            ax.plot(cp[0], cp[1], "^", ms=10, color="#d02020", zorder=6)
            ax.annotate(f"カメラ h={cp[2]:.1f}m", cp[:2], xytext=(6, -12), textcoords="offset points",
                        color="#d02020")
        if self.pending_img:
            ax.set_title("地図で同じ点をクリック", color="#d02020")
        else:
            ax.set_title("")


# ===========================================================================
# 2. ラベル
# ===========================================================================

class LabelTab(QWidget):
    """候補フレームに箱＋4 点を付ける。自己位置かモデルから仮ラベルを作り、人が直して確認済みにする。

    操作: 空いた所を左ドラッグ → 箱。続けて左クリックで 前左→前右→後左→後右 の順に点を置く。
    点の近くを左ドラッグで移動。1〜4 で次に置く点を選ぶ。V で選んだ点の可視性（2 見える→1 隠れ→0 なし）。
    Delete で選んだ車を削除。Enter で確認済みにして次へ。N で「車なし」として確認済み。
    """

    def __init__(self, ctx: Context, runner_log: QPlainTextEdit):
        super().__init__()
        self.ctx = ctx
        self.store: LabelStore | None = None
        self.cur = None                 # 現在のフレーム番号
        self.insts = []                 # 編集中のインスタンス
        self.sel = None                 # 選択中のインスタンス番号
        self.next_kpt = 0
        self.drag = None                # ("box", x0, y0) / ("kpt", inst, k)
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        row = QHBoxLayout()
        self.step = QSpinBox()
        self.step.setRange(1, 600)
        self.step.setValue(30)
        b_make = QPushButton("候補を作る")
        b_make.clicked.connect(self.make_candidates)
        row.addWidget(QLabel("間隔"))
        row.addWidget(self.step)
        row.addWidget(b_make)
        left.addLayout(row)
        self.listw = QListWidget()
        self.listw.currentItemChanged.connect(self._on_select)
        left.addWidget(self.listw, 1)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        left.addWidget(self.stats)
        b_pose = QPushButton("自己位置から仮ラベル")
        b_pose.setToolTip("確認していない候補に、車両の自己位置を画像へ投影した仮ラベルを付ける")
        b_pose.clicked.connect(self.prelabel_pose)
        b_model = QPushButton("モデルで仮ラベル…")
        b_model.setToolTip("確認していない候補に、学習済みモデルの推論結果を仮ラベルとして付ける")
        b_model.clicked.connect(self.prelabel_model)
        b_ok = QPushButton("確認済みにして次へ（Enter）")
        b_ok.clicked.connect(self.confirm_next)
        b_none = QPushButton("車なしで確認済み（N）")
        b_none.clicked.connect(self.confirm_empty)
        b_unconf = QPushButton("確認を取り消す")
        b_unconf.clicked.connect(self.unconfirm)
        for b in (b_pose, b_model, b_ok, b_none, b_unconf):
            left.addWidget(b)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(300)
        lay.addWidget(lw)

        center = QVBoxLayout()
        self.info = QLabel("")
        center.addWidget(self.info)
        self.img = ImageView()
        self.img.overlays.append(self._draw)
        self.img.pressed.connect(self._press)
        self.img.dragged.connect(self._drag)
        self.img.released.connect(self._release)
        center.addWidget(self.img, 1)
        lay.addLayout(center, 3)

        right = QVBoxLayout()
        right.addWidget(QLabel("車載カメラ（同時刻）"))
        self.carimg = ImageView()
        self.carimg.setMinimumSize(240, 180)
        right.addWidget(self.carimg, 1)
        self.poseinfo = QLabel("")
        self.poseinfo.setWordWrap(True)
        right.addWidget(self.poseinfo)
        rw = QWidget()
        rw.setLayout(right)
        rw.setMaximumWidth(340)
        lay.addWidget(rw, 1)

        self.runner = ProcessRunner(runner_log, on_done=lambda _c: self.reload_store())
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(800)
        self._autosave_timer.timeout.connect(self._commit)
        for key, fn in (("Return", self.confirm_next), ("N", self.confirm_empty), ("Delete", self.delete_sel),
                        ("V", self.toggle_vis), ("1", lambda: self._set_next(0)), ("2", lambda: self._set_next(1)),
                        ("3", lambda: self._set_next(2)), ("4", lambda: self._set_next(3)),
                        ("Right", lambda: self._step_list(1)), ("Left", lambda: self._step_list(-1))):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        ctx.listeners.append(self.refresh)

    # --- 候補とストア ------------------------------------------------------
    def refresh(self):
        self.reload_store()

    def reload_store(self):
        self._commit()
        c = self.ctx
        self.store = LabelStore(c.clip) if c.clip else None
        cur = self.cur if self.store and self.cur in self._candidates() else None
        self.cur = None
        self._fill_list()
        if cur is None and self.store:
            cur = self._resume_frame()
        if cur is not None:
            self._select_frame(cur)

    def _resume_frame(self):
        """開き直したときの位置: 前回最後に開いていたフレーム → 無ければ最初の未確認 → 先頭。"""
        cands = self._candidates()
        if not cands:
            return None
        if self.store.last_frame in cands:
            return self.store.last_frame
        for i in cands:
            if not self.store.is_confirmed(i):
                return i
        return cands[0]

    def _candidates(self):
        if not self.store:
            return []
        d = os.path.join(self.ctx.clip.ext_dir, "frames", self.ctx.clip.name)
        cached = set()
        if os.path.isdir(d):
            cached = {int(f[:-4]) for f in os.listdir(d) if f.endswith(".jpg") and f[:-4].isdigit()}
        return sorted(cached | set(self.store.labeled_indices()))

    def _fill_list(self):
        self.listw.blockSignals(True)
        self.listw.clear()
        n_conf = n_pre = 0
        for i in self._candidates():
            it = QListWidgetItem()
            it.setData(Qt.UserRole, i)
            self._style_item(it, i)
            self.listw.addItem(it)
        self.listw.blockSignals(False)
        self._update_stats()

    def _style_item(self, it, i):
        insts = self.store.get(i) if self.store else None
        if self.store and self.store.is_confirmed(i):
            mark, col = "✔", "#108010"
        elif insts:
            mark, col = "○", "#c07000"
        else:
            mark, col = "・", "#808080"
        it.setText(f"{mark} {i:6d}   {len(insts or [])} 台")
        it.setForeground(QColor(col))

    def _update_stats(self):
        n_conf = n_pre = 0
        for r in range(self.listw.count()):
            i = self.listw.item(r).data(Qt.UserRole)
            if self.store.is_confirmed(i):
                n_conf += 1
            elif self.store.get(i):
                n_pre += 1
        self.stats.setText(f"候補 {self.listw.count()} ／ 確認済み {n_conf} ／ 仮ラベル {n_pre}")

    def _update_item(self, i):
        """1 行だけ表示を更新する（保存のたびに一覧を作り直すと選択やスクロールが飛ぶため）。"""
        for r in range(self.listw.count()):
            it = self.listw.item(r)
            if it.data(Qt.UserRole) == i:
                self._style_item(it, i)
                break
        self._update_stats()

    def make_candidates(self):
        c = self.ctx
        if not c.clip:
            return
        idx = c.clip.sample_indices(step=self.step.value())
        dlg = QProgressDialog("フレームを書き出し中…", "中止", 0, len(idx), self)
        dlg.setWindowModality(Qt.WindowModal)

        def prog(k, n):
            dlg.setMaximum(n)
            dlg.setValue(k)
            QApplication.processEvents()
        c.clip.extract(idx, progress=prog)
        dlg.close()
        self._fill_list()

    def _uncofirmed_candidates(self):
        return [i for i in self._candidates() if not self.store.is_confirmed(i)]

    def prelabel_pose(self):
        c = self.ctx
        if c.calib is None:
            QMessageBox.warning(self, "仮ラベル", "先に較正して保存してください")
            return
        self._commit()
        c.save_setup()
        n = trk.prelabel_from_pose(c.run_dir, c.clip.num, self._uncofirmed_candidates(), overwrite=True)
        self.reload_store()
        QMessageBox.information(self, "仮ラベル", f"{n} フレームに自己位置から仮ラベルを付けました（要確認）")

    def prelabel_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "学習済みモデル（best.pt）", _TOOL_ROOT, "PyTorch (*.pt)")
        if not path:
            return
        self._commit()
        c = self.ctx
        step = self.step.value()
        self.runner.start("extcam.tracker", ["prelabel", c.run_dir, "--clip", c.clip.num, "--from", "model",
                                             "--model", path, "--step", step, "--overwrite"])

    # --- フレーム ----------------------------------------------------------
    def _on_select(self, item, _prev=None):
        if item is None:
            return
        self._commit()
        self._load(int(item.data(Qt.UserRole)))

    def _select_frame(self, i):
        for r in range(self.listw.count()):
            if self.listw.item(r).data(Qt.UserRole) == i:
                self.listw.setCurrentRow(r)
                return

    def _step_list(self, d):
        r = self.listw.currentRow() + d
        if 0 <= r < self.listw.count():
            self.listw.setCurrentRow(r)

    def _load(self, i):
        c = self.ctx
        self.cur = i
        self.insts = json.loads(json.dumps(self.store.get(i) or []))
        self.sel = 0 if self.insts else None
        self.next_kpt = 0
        self.edited = False
        self.store.set_last(i)
        self.img.set_image(c.clip.cached_frame(i))
        t = c.clip.time_of(i)
        conf = "確認済み" if self.store.is_confirmed(i) else "未確認"
        self.info.setText(f"フレーム {i}（{conf}）  t={t:.3f}" if t else f"フレーム {i}")
        self._show_car(t)

    def _show_car(self, t):
        rl = self.ctx.runlog
        if rl is None or t is None:
            self.carimg.set_image(None)
            self.poseinfo.setText("自己位置なし")
            return
        j = rl.nearest(t, max_dt=0.2)
        path = rl.image_path(j)
        img = cv2.imread(path) if path and os.path.isfile(path) else None
        self.carimg.set_image(img)
        p = rl.pose_at(t)
        self.poseinfo.setText(
            f"自己位置（{rl.source}）: x={p[0]:.2f} y={p[1]:.2f} 向き={math.degrees(p[2]):.0f}°" if p else "自己位置なし")

    def _commit(self):
        """編集中の内容をストアへ（編集したフレームは確認済みにする）。"""
        if self.store is None or self.cur is None:
            return
        self._autosave_timer.stop()
        changed = False
        if getattr(self, "edited", False):
            for x in self.insts:
                x["src"] = "manual"
            self.store.set(self.cur, json.loads(json.dumps(self.insts)), confirmed=True)
            self.edited = False
            changed = True
        if self.store.needs_save():
            self.store.save()
        if changed:
            self._update_item(self.cur)
            self.info.setText(self.info.text().replace("（未確認）", "（確認済み）"))

    def _mark_edited(self):
        """編集した。少し待って自動保存する（落ちても失うのは最後の 0.8 秒の操作だけ）。"""
        self.edited = True
        self._autosave_timer.start()

    def confirm_next(self):
        if self.cur is None:
            return
        self.edited = True
        self._commit()
        self._step_list(1)

    def confirm_empty(self):
        if self.cur is None:
            return
        self.insts = []
        self.edited = True
        self._commit()
        self._step_list(1)

    def unconfirm(self):
        if self.cur is None:
            return
        self._commit()
        self.store.confirm(self.cur, False)
        self.store.save()
        self._update_item(self.cur)
        self.info.setText(self.info.text().replace("（確認済み）", "（未確認）"))

    # --- 編集 --------------------------------------------------------------
    def _hit_kpt(self, x, y, r_px=10):
        s = self.img._scale()
        best = None
        for a, inst in enumerate(self.insts):
            for k, (u, v, vis) in enumerate(inst["kpts"]):
                if vis <= 0:
                    continue
                d = math.hypot(u - x, v - y) * s
                if d <= r_px and (best is None or d < best[0]):
                    best = (d, a, k)
        return best

    def _hit_box(self, x, y):
        for a, inst in enumerate(self.insts):
            x1, y1, x2, y2 = inst["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return a
        return None

    def _press(self, x, y, button, _mods):
        if button != int(Qt.LeftButton) or self.cur is None:
            return
        hit = self._hit_kpt(x, y)
        if hit:
            self.sel = hit[1]
            self.drag = ("kpt", hit[1], hit[2])
            return
        if self.sel is not None and self.sel < len(self.insts):
            inst = self.insts[self.sel]
            missing = [k for k, kp in enumerate(inst["kpts"]) if kp[2] <= 0]
            k = self.next_kpt if self.next_kpt in missing or not missing else missing[0]
            if missing and self._hit_box(x, y) == self.sel:
                inst["kpts"][k] = [round(x, 1), round(y, 1), 2]
                rest = [m for m in missing if m != k]
                self.next_kpt = rest[0] if rest else 0
                self._mark_edited()
                self.img.update()
                return
        a = self._hit_box(x, y)
        if a is not None:
            self.sel = a
            self.img.update()
            return
        self.drag = ("box", x, y)

    def _drag(self, x, y):
        if not self.drag:
            return
        if self.drag[0] == "kpt":
            _, a, k = self.drag
            self.insts[a]["kpts"][k][0], self.insts[a]["kpts"][k][1] = round(x, 1), round(y, 1)
            if self.insts[a]["kpts"][k][2] <= 0:
                self.insts[a]["kpts"][k][2] = 2
            self._mark_edited()
        else:
            self._rubber = (self.drag[1], self.drag[2], x, y)
        self.img.update()

    def _release(self, x, y):
        if self.drag and self.drag[0] == "box":
            x0, y0 = self.drag[1], self.drag[2]
            if abs(x - x0) > 4 and abs(y - y0) > 4:
                self.insts.append({"bbox": [round(min(x0, x), 1), round(min(y0, y), 1),
                                            round(max(x0, x), 1), round(max(y0, y), 1)],
                                   "kpts": [[0, 0, 0] for _ in KPT_NAMES], "src": "manual"})
                self.sel = len(self.insts) - 1
                self.next_kpt = 0
                self._mark_edited()
        self._rubber = None
        self.drag = None
        self.img.update()

    def _set_next(self, k):
        self.next_kpt = k
        self.img.update()

    def toggle_vis(self):
        if self.sel is None or self.sel >= len(self.insts):
            return
        kp = self.insts[self.sel]["kpts"][self.next_kpt]
        kp[2] = {2: 1, 1: 0, 0: 2}[int(kp[2])]
        self._mark_edited()
        self.img.update()

    def delete_sel(self):
        if self.sel is None or self.sel >= len(self.insts):
            return
        del self.insts[self.sel]
        self.sel = 0 if self.insts else None
        self._mark_edited()
        self.img.update()

    def _draw(self, p, tw):
        for a, inst in enumerate(self.insts):
            x1, y1, x2, y2 = inst["bbox"]
            selected = a == self.sel
            col = "#ffffff" if selected else ("#ffd000" if inst.get("src") != "manual" else "#40ff60")
            p.setPen(pen(col, 2 if selected else 1.3, Qt.SolidLine if inst.get("src") == "manual" else Qt.DashLine))
            p.drawRect(QRectF(tw(x1, y1), tw(x2, y2)))
            k = inst["kpts"]
            # 前（0-1）と後（2-3）を結んで向きが分かるように
            for i0, i1, c in ((0, 1, "#ff5050"), (2, 3, "#5090ff"), (0, 2, "#c0c0c0"), (1, 3, "#c0c0c0")):
                if k[i0][2] > 0 and k[i1][2] > 0:
                    p.setPen(pen(c, 1.5))
                    p.drawLine(tw(k[i0][0], k[i0][1]), tw(k[i1][0], k[i1][1]))
            for j, (u, v, vis) in enumerate(k):
                if vis <= 0:
                    continue
                q = tw(u, v)
                p.setPen(pen(KPT_COLORS[j], 2.5 if vis == 2 else 1.2, Qt.SolidLine if vis == 2 else Qt.DotLine))
                p.drawEllipse(q, 5, 5)
                p.drawText(QPointF(q.x() + 6, q.y() - 6), KPT_SHORT[j])
        rb = getattr(self, "_rubber", None)
        if rb:
            p.setPen(pen("#ffffff", 1, Qt.DashLine))
            a, b = tw(rb[0], rb[1]), tw(rb[2], rb[3])
            p.drawRect(QRectF(a, b).normalized())
        p.setPen(pen("#ffffff", 1))
        p.drawText(QPointF(10, 20), f"次に置く点: {KPT_SHORT[self.next_kpt]}（1〜4 で変更、V で可視性）")


# ===========================================================================
# 3. 学習
# ===========================================================================

class TrainTab(QWidget):
    def __init__(self, ctx: Context, on_model):
        super().__init__()
        self.ctx = ctx
        self.on_model = on_model
        lay = QVBoxLayout(self)
        runs_box = QGroupBox("学習に使う走行フォルダ（確認済みのラベルを集める）")
        rl = QVBoxLayout(runs_box)
        self.runs = QListWidget()
        rl.addWidget(self.runs)
        rb = QHBoxLayout()
        b_add = QPushButton("追加…")
        b_add.clicked.connect(self._add_run)
        b_rm = QPushButton("外す")
        b_rm.clicked.connect(lambda: [self.runs.takeItem(self.runs.row(i)) for i in self.runs.selectedItems()])
        rb.addWidget(b_add)
        rb.addWidget(b_rm)
        rb.addStretch(1)
        rl.addLayout(rb)
        lay.addWidget(runs_box)

        form_box = QGroupBox("設定")
        form = QFormLayout(form_box)
        self.base = QLineEdit("yolo11n-pose.pt")
        b_base = QPushButton("…")
        b_base.clicked.connect(self._pick_base)
        hb = QHBoxLayout()
        hb.addWidget(self.base)
        hb.addWidget(b_base)
        form.addRow("初期重み（続きなら前回の best.pt）", hb)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 2000)
        self.epochs.setValue(100)
        self.imgsz = QComboBox()
        self.imgsz.addItems(["1280", "960", "640"])
        self.batch = QSpinBox()
        self.batch.setRange(1, 128)
        self.batch.setValue(8)
        self.device = QLineEdit("")
        self.device.setPlaceholderText("空=自動（0 / cpu など）")
        self.unconf = QCheckBox("未確認の仮ラベルも使う（自己位置が正確なときの近道）")
        self.out = QLineEdit(os.path.join(_TOOL_ROOT, "models", "extcam", time.strftime("%Y%m%d_%H%M%S")))
        form.addRow("エポック", self.epochs)
        form.addRow("画像サイズ", self.imgsz)
        form.addRow("バッチ", self.batch)
        form.addRow("デバイス", self.device)
        form.addRow("出力先", self.out)
        form.addRow("", self.unconf)
        lay.addWidget(form_box)
        row = QHBoxLayout()
        self.b_run = QPushButton("学習を開始")
        self.b_run.clicked.connect(self.start)
        b_stop = QPushButton("中止")
        b_stop.clicked.connect(lambda: self.runner.stop())
        row.addWidget(self.b_run)
        row.addWidget(b_stop)
        row.addStretch(1)
        lay.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("Monospace", 9))
        lay.addWidget(self.log, 1)
        self.best = None
        self.runner = ProcessRunner(self.log, on_line=self._line, on_done=self._done)
        ctx.listeners.append(self.refresh)

    def refresh(self):
        if self.ctx.run_dir and not any(self.runs.item(i).text() == self.ctx.run_dir for i in range(self.runs.count())):
            self.runs.addItem(self.ctx.run_dir)

    def _add_run(self):
        d = QFileDialog.getExistingDirectory(self, "走行フォルダ（data_<TS>）", self.ctx.run_dir or "")
        if d:
            self.runs.addItem(os.path.abspath(d))

    def _pick_base(self):
        p, _ = QFileDialog.getOpenFileName(self, "初期重み", _TOOL_ROOT, "PyTorch (*.pt)")
        if p:
            self.base.setText(p)

    def start(self):
        runs = [self.runs.item(i).text() for i in range(self.runs.count())]
        if not runs:
            return
        self.out.setText(os.path.join(_TOOL_ROOT, "models", "extcam", time.strftime("%Y%m%d_%H%M%S"))
                         if os.path.isdir(self.out.text()) else self.out.text())
        args = ["--runs", *runs, "--out", self.out.text(), "--model", self.base.text(),
                "--epochs", self.epochs.value(), "--imgsz", self.imgsz.currentText(), "--batch", self.batch.value()]
        if self.device.text().strip():
            args += ["--device", self.device.text().strip()]
        if self.unconf.isChecked():
            args.append("--include-unconfirmed")
        self.best = None
        self.runner.start("extcam.train", args)

    def _line(self, line):
        if line.startswith("BEST "):
            self.best = line[5:].strip()

    def _done(self, code):
        if self.best and self.best != "None" and os.path.isfile(self.best):
            self.on_model(self.best)
            QMessageBox.information(self, "学習", f"学習が終わりました。\n{self.best}\n（追跡タブに設定しました）")


# ===========================================================================
# 4. 追跡
# ===========================================================================

class TrackTab(QWidget):
    def __init__(self, ctx: Context):
        super().__init__()
        self.ctx = ctx
        self.raw = []
        self.track = []
        self.raw_by_frame = {}
        self.track_by_frame = {}
        self.summary = {}
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.model = QLineEdit("")
        self.model.setPlaceholderText("学習済みモデル（best.pt）")
        b_m = QPushButton("…")
        b_m.clicked.connect(self._pick_model)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.imgsz = QComboBox()
        self.imgsz.addItems(["1280", "960", "640"])
        b_run = QPushButton("追跡を実行")
        b_run.clicked.connect(self.run)
        b_stop = QPushButton("中止")
        b_stop.clicked.connect(lambda: self.runner.stop())
        b_load = QPushButton("結果を読み込む")
        b_load.clicked.connect(self.load_results)
        for w in (QLabel("モデル"), self.model, b_m, QLabel("conf"), self.conf, QLabel("imgsz"), self.imgsz,
                  b_run, b_stop, b_load):
            top.addWidget(w)
        top.setStretch(1, 1)
        lay.addLayout(top)
        self.slider, self.spin, self.tlabel = _frame_slider_row(lay, self.show_frame)
        play_row = QHBoxLayout()
        self.b_play = QPushButton("▶ 再生")
        self.b_play.setCheckable(True)
        self.b_play.toggled.connect(self._toggle_play)
        self.show_raw = QCheckBox("生の検出（箱・点）")
        self.show_raw.setChecked(True)
        self.show_raw.toggled.connect(lambda: self.img.update())
        play_row.addWidget(self.b_play)
        play_row.addWidget(self.show_raw)
        play_row.addStretch(1)
        lay.addLayout(play_row)

        split = QSplitter(Qt.Horizontal)
        self.img = ImageView()
        self.img.overlays.append(self._draw_img)
        self.map = MapView()
        self.map.draw_extra = self._draw_map
        split.addWidget(self.img)
        split.addWidget(self.map)
        split.setSizes([700, 500])
        lay.addWidget(split, 1)

        bottom = QHBoxLayout()
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["使う", "ID", "統合先", "件数", "外れ値", "位置 RMSE [m]",
                                              "前後の偏り [m]", "横の偏り [m]", "向き誤差 p50 [°]"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setMaximumHeight(170)
        bottom.addWidget(self.table, 3)
        side = QVBoxLayout()
        b_apply = QPushButton("手直しを反映して平滑化し直す")
        b_apply.clicked.connect(self.apply_edits)
        b_excl = QPushButton("このフレームの選択 ID を除外")
        b_excl.clicked.connect(self.exclude_frame)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setFont(QFont("Monospace", 9))
        side.addWidget(b_apply)
        side.addWidget(b_excl)
        side.addWidget(self.log, 1)
        bottom.addLayout(side, 2)
        lay.addLayout(bottom)

        self.runner = ProcessRunner(self.log, on_done=lambda c: self.load_results() if c == 0 else None)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        ctx.listeners.append(self.refresh)

    def set_model(self, path):
        self.model.setText(path)

    def _pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "学習済みモデル", _TOOL_ROOT, "PyTorch (*.pt)")
        if p:
            self.model.setText(p)

    def refresh(self):
        c = self.ctx
        n = len(c.clip) if c.clip else 0
        self.slider.setRange(0, max(0, n - 1))
        self.spin.setRange(0, max(0, n - 1))
        self.map.set_course(c.course)
        self.load_results(quiet=True)

    def run(self):
        c = self.ctx
        if not c.clip or not self.model.text():
            QMessageBox.warning(self, "追跡", "クリップとモデルを選んでください")
            return
        if c.calib is None:
            QMessageBox.warning(self, "追跡", "先に較正を保存してください")
            return
        c.save_setup()
        self.runner.start("extcam.tracker", ["run", c.run_dir, "--clip", c.clip.num, "--model", self.model.text(),
                                             "--conf", self.conf.value(), "--imgsz", self.imgsz.currentText()])

    def load_results(self, quiet=False):
        c = self.ctx
        self.raw, self.track, self.summary = [], [], {}
        self.raw_by_frame, self.track_by_frame = {}, {}
        if not c.clip:
            return
        p = trk._paths(c.clip)
        if os.path.isfile(p["raw"]):
            self.raw = trk.load_raw(p["raw"])
            for r in self.raw:
                self.raw_by_frame.setdefault(int(r["frame"]), []).append(r)
        if os.path.isfile(p["track"]):
            self.track = trk.load_track(p["track"])
            for r in self.track:
                self.track_by_frame.setdefault(r["frame"], []).append(r)
        if os.path.isfile(p["summary"]):
            with open(p["summary"], encoding="utf-8") as f:
                self.summary = json.load(f)
        self._fill_table()
        self.map.redraw()
        self.show_frame(self.spin.value())
        if not quiet and not self.raw:
            QMessageBox.information(self, "追跡", "結果がありません")

    def _fill_table(self):
        edits = trk.load_edits(trk._paths(self.ctx.clip)["edits"]) if self.ctx.clip else {}
        excl = set(int(x) for x in edits.get("exclude_ids") or [])
        merge = {int(k): int(v) for k, v in (edits.get("merge") or {}).items()}
        raw_ids = {}
        for r in self.raw:
            raw_ids[int(r["id"])] = raw_ids.get(int(r["id"]), 0) + 1
        ids = self.summary.get("ids", {})
        self.table.setRowCount(len(raw_ids))
        for row, (tid, n) in enumerate(sorted(raw_ids.items(), key=lambda kv: -kv[1])):
            e = ids.get(str(merge.get(tid, tid)), {})
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked if tid in excl else Qt.Checked)
            self.table.setItem(row, 0, chk)
            idit = QTableWidgetItem(str(tid))
            idit.setForeground(QColor(_id_color(tid)))
            idit.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 1, idit)
            self.table.setItem(row, 2, QTableWidgetItem(str(merge[tid]) if tid in merge else ""))
            vals = [n, e.get("outliers"), e.get("pos_rmse_m"), e.get("lon_mean_m"), e.get("lat_mean_m"),
                    e.get("yaw_err_p50_deg")]
            for col, v in enumerate(vals, start=3):
                it = QTableWidgetItem("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(row, col, it)

    def _read_edits_from_table(self) -> dict:
        p = trk._paths(self.ctx.clip)["edits"]
        edits = trk.load_edits(p)
        excl, merge = [], {}
        for r in range(self.table.rowCount()):
            tid = int(self.table.item(r, 1).text())
            if self.table.item(r, 0).checkState() != Qt.Checked:
                excl.append(tid)
            m = (self.table.item(r, 2).text() if self.table.item(r, 2) else "").strip()
            if m.lstrip("-").isdigit() and int(m) != tid:
                merge[str(tid)] = int(m)
        edits["exclude_ids"] = excl
        edits["merge"] = merge
        return edits

    def apply_edits(self):
        if not self.ctx.clip or not self.raw:
            return
        p = trk._paths(self.ctx.clip)
        trk.save_edits(p["edits"], self._read_edits_from_table())
        summ = trk.smooth_clip(self.ctx.run_dir, self.ctx.clip.num)
        self.log.appendPlainText("平滑化し直しました: " + json.dumps(summ.get("ids", {}), ensure_ascii=False)[:400])
        self.load_results(quiet=True)

    def exclude_frame(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows or not self.ctx.clip:
            return
        p = trk._paths(self.ctx.clip)
        edits = self._read_edits_from_table()
        fr = self.spin.value()
        for r in rows:
            edits.setdefault("exclude_frames", []).append([int(self.table.item(r, 1).text()), fr])
        trk.save_edits(p["edits"], edits)
        self.log.appendPlainText(f"フレーム {fr} を除外に追加（反映は「平滑化し直す」）")

    def _toggle_play(self, on):
        self.b_play.setText("■ 停止" if on else "▶ 再生")
        if on:
            self.timer.start(33)
        else:
            self.timer.stop()

    def _tick(self):
        v = self.spin.value() + 2
        if v > self.spin.maximum():
            self.b_play.setChecked(False)
            return
        self.spin.setValue(v)

    def show_frame(self, i):
        c = self.ctx
        if not c.clip:
            return
        self.img.set_image(c.clip.read(int(i)))
        t = c.clip.time_of(int(i))
        self.tlabel.setText(f"t={t:.3f}" if t else "")
        self.map.redraw()

    def _draw_img(self, p, tw):
        c = self.ctx
        i = self.spin.value()
        if self.show_raw.isChecked():
            for r in self.raw_by_frame.get(i, []):
                col = _id_color(int(r["id"]))
                p.setPen(pen(col, 1.3, Qt.DashLine))
                x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
                p.drawRect(QRectF(tw(x1, y1), tw(x2, y2)))
                p.drawText(tw(x1, y1) + QPointF(0, -4), f"ID {r['id']}  {float(r['conf']):.2f}")
                for j in range(len(KPT_NAMES)):
                    if float(r[f"k{j}c"]) >= 0.5:
                        p.setPen(pen(KPT_COLORS[j], 2))
                        p.drawEllipse(tw(float(r[f"k{j}u"]), float(r[f"k{j}v"])), 4, 4)
        if c.calib is None:
            return
        for r in self.track_by_frame.get(i, []):
            # 平滑化した姿勢をルーフの高さで描き戻す（検出とずれていれば平滑化か較正が怪しい）
            kb = c.veh.body_points()
            km = body_to_map(np.hstack([kb, np.full((4, 1), c.veh.height)]), r["x"], r["y"], r["yaw"])
            uv = c.to_frame(c.calib.project(km), i)
            p.setPen(pen("#ffffff", 2))
            for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
                p.drawLine(tw(*uv[a]), tw(*uv[b]))
            fm, rm_ = (uv[0] + uv[1]) / 2, (uv[2] + uv[3]) / 2
            p.setPen(pen("#ff3030", 2.5))
            p.drawLine(tw(*rm_), tw(*fm))

    def _draw_map(self, ax):
        c = self.ctx
        if c.runlog is not None and len(c.runlog) and c.clip and c.clip.t_jetson:
            t0, t1 = c.clip.t_jetson[0], c.clip.t_jetson[-1]
            pts = [p for t, p in zip(c.runlog.t, c.runlog.xyyaw) if p and t0 <= t <= t1]
            if pts:
                a = np.array(pts)
                ax.plot(a[:, 0], a[:, 1], "-", color="#888888", lw=1, label=f"自己位置（{c.runlog.source}）")
        by_id = {}
        for r in self.track:
            by_id.setdefault(r["id"], []).append(r)
        for tid, rs in by_id.items():
            a = np.array([[r["x"], r["y"]] for r in rs])
            ax.plot(a[:, 0], a[:, 1], "-", color=_id_color(tid), lw=1.4, label=f"ID {tid}")
        i = self.spin.value()
        for r in self.track_by_frame.get(i, []):
            ax.arrow(r["x"], r["y"], 0.3 * math.cos(r["yaw"]), 0.3 * math.sin(r["yaw"]), width=0.03,
                     color=_id_color(r["id"]), zorder=8)
        if c.runlog is not None and c.clip and c.clip.time_of(i):
            p = c.runlog.pose_at(c.clip.time_of(i))
            if p:
                ax.arrow(p[0], p[1], 0.3 * math.cos(p[2]), 0.3 * math.sin(p[2]), width=0.03, color="#444444", zorder=7)
        if by_id or (c.runlog is not None and len(c.runlog)):
            ax.legend(loc="upper right", fontsize=7)


# ===========================================================================

class ExtcamWindow(QMainWindow):
    def __init__(self, run_dir: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("外部カメラ: 取り込み・較正・ラベル・学習・追跡")
        self.resize(1500, 950)
        self.ctx = Context()
        central = QWidget()
        lay = QVBoxLayout(central)
        top = QHBoxLayout()
        self.run_edit = QLineEdit("")
        self.run_edit.setReadOnly(True)
        b_open = QPushButton("走行フォルダを開く…")
        b_open.clicked.connect(self._open_dialog)
        self.clip_combo = QComboBox()
        self.clip_combo.currentIndexChanged.connect(self._clip_changed)
        b_map = QPushButton("地図フォルダを変える…")
        b_map.clicked.connect(self._change_map)
        self.status = QLabel("")
        top.addWidget(QLabel("走行"))
        top.addWidget(self.run_edit, 1)
        top.addWidget(b_open)
        top.addWidget(QLabel("クリップ"))
        top.addWidget(self.clip_combo)
        top.addWidget(b_map)
        lay.addLayout(top)
        lay.addWidget(self.status)
        self.tabs = QTabWidget()
        self.calib_tab = CalibTab(self.ctx)
        self.track_tab = TrackTab(self.ctx)
        self.train_tab = TrainTab(self.ctx, on_model=self.track_tab.set_model)
        self.label_tab = LabelTab(self.ctx, self.train_tab.log)
        from extcam.import_tab import ImportTab
        self.import_tab = ImportTab(self.ctx, self.open_run)
        default_data = os.path.join(os.path.dirname(_TOOL_ROOT), "data")
        if os.path.isdir(default_data):
            self.import_tab.data_root.setText(default_data)
        self.tabs.addTab(self.import_tab, "0. 取り込み")
        self.tabs.addTab(self.calib_tab, "1. 較正")
        self.tabs.addTab(self.label_tab, "2. ラベル")
        self.tabs.addTab(self.train_tab, "3. 学習")
        self.tabs.addTab(self.track_tab, "4. 追跡")
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.ctx.listeners.append(self._update_status)
        if run_dir:
            self.open_run(run_dir)

    def _flush(self):
        """保存待ち（ラベルの自動保存・較正の寸法変更）をすぐ書く。"""
        try:
            self.label_tab._commit()
        except Exception as e:   # noqa: BLE001
            print("ラベルの保存に失敗:", e)
        if self.calib_tab._save_timer.isActive():
            self.calib_tab._autosave()

    def open_run(self, run_dir, select_last_clip=False):
        if not run_dir or not os.path.isdir(run_dir):
            return
        if self.ctx.run_dir:
            self._flush()
        self.ctx.load_run(run_dir)
        self.run_edit.setText(self.ctx.run_dir)
        self.clip_combo.blockSignals(True)
        self.clip_combo.clear()
        for num, path in list_clips(self.ctx.run_dir):
            self.clip_combo.addItem(os.path.basename(path), num)
        self.clip_combo.blockSignals(False)
        if not self.clip_combo.count():
            QMessageBox.information(self, "外部カメラ", "この走行フォルダには extcam/clip_NN.mkv がありません")
        elif select_last_clip:                       # 取り込み直後: 新しいクリップを開いて較正タブへ
            self.clip_combo.setCurrentIndex(self.clip_combo.count() - 1)
            self.tabs.setCurrentWidget(self.calib_tab)

    def _open_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "走行フォルダ（data_<TS>）", self.ctx.run_dir or "")
        if d:
            self.open_run(d)

    def _clip_changed(self, _i):
        num = self.clip_combo.currentData()
        if num is not None and self.ctx.run_dir:
            self.label_tab._commit()
            self.ctx.set_clip(int(num))

    def _change_map(self):
        d = QFileDialog.getExistingDirectory(self, "地図フォルダ（course_editor の地図: *.yaml と map_bounds.json）",
                                             self.ctx.course.map_dir if self.ctx.course and self.ctx.course.map_dir else "")
        if d:
            self.ctx.course = CourseMap(d)
            self.ctx.notify()

    def _update_status(self):
        c = self.ctx
        parts = []
        if c.clip:
            parts.append(f"{c.clip.name}: {len(c.clip)} フレーム")
        parts.append(f"地図: {c.course.map_dir if c.course and c.course.map_dir else 'なし'}"
                     f"（外形 {len(c.course.outer) if c.course else 0} 点）")
        parts.append(f"自己位置: {c.runlog.source if c.runlog and c.runlog.source else 'なし'}")
        parts.append("較正: " + ("未" if not c.calib else f"{c.calib.reproj_rms_px:.2f} px" if c.calib.reproj_rms_px is not None
                                else c.calib.intrinsics_source) + (f"（{os.path.basename(c.setup_path())}、手持ち）" if c.stab is not None else ""))
        self.status.setText("　|　".join(parts))

    def closeEvent(self, e):
        self._flush()
        for r in (self.train_tab.runner, self.track_tab.runner, self.label_tab.runner):
            if r.running():
                if QMessageBox.question(self, "終了", "学習・追跡が動いています。止めて閉じますか？") != QMessageBox.Yes:
                    e.ignore()
                    return
                r.stop()
        if self.ctx.clip:
            self.ctx.clip.close()
        super().closeEvent(e)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication(sys.argv)
    w = ExtcamWindow(argv[0] if argv else None)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
