# map_view.py
"""
走行軌跡マップビュー（鳥瞰）

PoseSourceManagerが読み取ったpose/slam/vslam/arucoの軌跡を2Dプロットし、
ソース切替・色分け・現在フレームのハイライト・軌跡クリックによるフレームジャンプを提供する。
背景にはROS map_server形式（.yaml + .pgm/.png）のSLAM地図を重畳できる。
"""
import math
import os
import yaml
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
                              QFileDialog, QDialog, QSpinBox, QDoubleSpinBox, QMessageBox, QFrame,
                              QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import RegularPolygon

from translations import get_text

# 日本語フォントの設定（data_analysis.pyと同じ設定。"MS Gothic"はこの環境の
# matplotlib(FreeType)でテキストが完全に不可視になる既知の不具合があるため使用しない）
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SOURCE_COLORS = {
    "aruco": "tab:red",
    "slam": "tab:blue",
    "vslam": "tab:green",
    "pose": "tab:gray",
}

STATUS_COLORS = {
    "ok": "tab:green",
    "init": "tab:orange",
    "lost": "tab:red",
    "unknown": "tab:gray",
    "interp": "tab:purple",
}


def load_ros_map(yaml_path: str):
    """ROS map_server形式の地図(.yaml)を読み込み、(画像パス, extent[m]) を返す

    extent は matplotlib imshow(..., origin='upper') とそのまま組み合わせられる
    (xmin, xmax, ymin, ymax) 形式。
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        meta = yaml.safe_load(f)

    image_name = meta["image"]
    resolution = float(meta["resolution"])
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    origin_x, origin_y = float(origin[0]), float(origin[1])

    image_path = os.path.join(os.path.dirname(yaml_path), image_name)

    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size

    extent = (origin_x, origin_x + width * resolution,
              origin_y, origin_y + height * resolution)
    return image_path, extent


class MapViewWidget(QWidget):
    """走行軌跡マップビュー。on_frame_selected(index:int) で親にフレームジャンプを通知する"""

    def __init__(self, parent=None, on_frame_selected=None):
        super().__init__(parent)
        self.on_frame_selected = on_frame_selected
        self.pose_manager = None

        self._bg_image_path = None
        self._bg_extent = None
        self._plotted_indexes = []
        self._current_marker = None
        # ジャンプ検出のマーカー表示に使う閾値（品質フィルタのスピンボックスと同期）
        self.jump_threshold = 1.0

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()

        controls.addWidget(QLabel(get_text('map_view_source_label')))
        self.source_combo = QComboBox()
        self.source_combo.addItem(get_text('map_view_source_auto'), None)
        self.source_combo.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.source_combo)

        controls.addWidget(QLabel(get_text('map_view_colorby_label')))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItem(get_text('map_view_colorby_time'), 'time')
        self.color_by_combo.addItem(get_text('map_view_colorby_speed'), 'speed')
        self.color_by_combo.addItem(get_text('map_view_colorby_source'), 'source')
        self.color_by_combo.addItem(get_text('map_view_colorby_status'), 'status')
        self.color_by_combo.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.color_by_combo)

        self.load_map_button = QPushButton(get_text('map_view_load_background'))
        self.load_map_button.clicked.connect(self._on_load_background_clicked)
        controls.addWidget(self.load_map_button)

        self.clear_map_button = QPushButton(get_text('map_view_clear_background'))
        self.clear_map_button.clicked.connect(self._on_clear_background_clicked)
        controls.addWidget(self.clear_map_button)

        # ジャンプ・スリップ・悪路マーカーの表示切替
        self.show_jumps_checkbox = QCheckBox(get_text('map_view_legend_jump'))
        self.show_jumps_checkbox.setChecked(True)
        self.show_jumps_checkbox.stateChanged.connect(self.refresh)
        controls.addWidget(self.show_jumps_checkbox)

        self.show_slip_checkbox = QCheckBox(get_text('map_view_legend_slip'))
        self.show_slip_checkbox.setChecked(True)
        self.show_slip_checkbox.stateChanged.connect(self.refresh)
        controls.addWidget(self.show_slip_checkbox)

        self.show_rough_checkbox = QCheckBox(get_text('map_view_legend_rough'))
        self.show_rough_checkbox.setChecked(True)
        self.show_rough_checkbox.stateChanged.connect(self.refresh)
        controls.addWidget(self.show_rough_checkbox)

        controls.addStretch()
        layout.addLayout(controls)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.status_label)

        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # matplotlib 標準のナビゲーションツールバー（ホーム/平行移動(pan)/
        # ズーム矩形/保存）。pan・zoom はボタンで切替、Home で全体表示に戻る。
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect('pick_event', self._on_pick)
        # マウスホイールでカーソル位置を中心にズーム（pan/zoomモード不要）
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    # --- 外部インターフェース -------------------------------------------------

    def set_pose_manager(self, pose_manager) -> None:
        self.pose_manager = pose_manager
        self._current_marker = None
        self._populate_source_combo()
        self.refresh()

    def _populate_source_combo(self) -> None:
        """有効な（縮退していない）ソースのみをプルダウンに表示する"""
        available = self.pose_manager.available_sources() if self.pose_manager else []
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem(get_text('map_view_source_auto'), None)
        for src in available:
            self.source_combo.addItem(src, src)
        self.source_combo.blockSignals(False)

    def set_background_map(self, yaml_path: str) -> None:
        try:
            image_path, extent = load_ros_map(yaml_path)
        except Exception as e:
            self.status_label.setText(get_text('map_view_background_load_error', str(e)))
            return
        self._bg_image_path = image_path
        self._bg_extent = extent
        self.refresh()

    def auto_load_background(self, data_dir: str) -> None:
        """データフォルダに紐づく地図（無ければ直近に保存された地図）を自動で
        背景に読み込む。解決順: ①<data_dir>/map/ 同梱スナップショット
        ②map_ref.json ③同タイムスタンプの地図 ④maps配下の最新地図。
        手動で背景を設定済みの場合や見つからない場合は何もしない。
        """
        if self._bg_image_path or not data_dir:
            return
        try:
            from utils.map_utils import resolve_background_map
            hit = resolve_background_map(data_dir)
        except Exception as e:
            self.status_label.setText(get_text('map_view_background_load_error', str(e)))
            return
        if not hit:
            return
        self.set_background_map(hit["map_yaml"])
        if self._bg_image_path:   # 読み込み成功時のみ由来を表示
            self.status_label.setText(get_text(
                'map_view_auto_loaded',
                os.path.basename(os.path.dirname(hit["map_yaml"])),
                hit["source"]))

    def clear_background_map(self) -> None:
        self._bg_image_path = None
        self._bg_extent = None
        self.refresh()

    def highlight_frame(self, index: int) -> None:
        if self.pose_manager is None:
            return
        pose = self.pose_manager.get_pose(index, prefer=self.source_combo.currentData())
        if self._current_marker is not None:
            try:
                self._current_marker.remove()
            except (ValueError, AttributeError):
                pass
            self._current_marker = None
        if pose is not None:
            # 現在位置は進行方向を向いた赤い三角で表示する。三角の大きさは
            # 現在の表示範囲に対する一定割合にして、ズームしても見やすく保つ。
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), 1e-6)
            radius = span * 0.02
            # RegularPolygon の orientation は +Y を 0 とするため theta-π/2 で
            # マップ方位（+X 基準）に合わせる。theta が無ければ上向き。
            theta = getattr(pose, 'theta', 0.0) or 0.0
            self._current_marker = RegularPolygon(
                (pose.x, pose.y), numVertices=3, radius=radius,
                orientation=theta - math.pi / 2.0,
                facecolor='red', edgecolor='darkred', linewidth=1.0, zorder=5)
            self.ax.add_patch(self._current_marker)
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        """マウスホイールでカーソル位置を中心にズームする。"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == 'up' else 1.25   # up=拡大 / down=縮小
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        # カーソル位置を固定点にして各軸を scale 倍に伸縮
        new_xlim = (cx + (xlim[0] - cx) * scale, cx + (xlim[1] - cx) * scale)
        new_ylim = (cy + (ylim[0] - cy) * scale, cy + (ylim[1] - cy) * scale)
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def refresh(self) -> None:
        self.ax.clear()
        self._current_marker = None
        # アノテーション画面のグリッド表示と同じ薄いグレーのグリッド
        self.ax.grid(True, color='gray', alpha=0.3, linewidth=0.5)
        self.ax.set_axisbelow(True)

        if self._bg_image_path and self._bg_extent:
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(self._bg_image_path)
                self.ax.imshow(img, extent=self._bg_extent, cmap='gray', origin='upper', zorder=0)
            except Exception as e:
                self.status_label.setText(get_text('map_view_background_load_error', str(e)))

        if self.pose_manager is None or not self.pose_manager.has_any_pose():
            self._plotted_indexes = []
            self.status_label.setText(get_text('map_view_no_pose_data'))
            self.canvas.draw()
            return

        source = self.source_combo.currentData()
        poses = self.pose_manager.get_trajectory(source=source)
        if not poses:
            self._plotted_indexes = []
            self.status_label.setText(get_text('map_view_no_pose_data'))
            self.canvas.draw()
            return

        xs = [p.x for p in poses]
        ys = [p.y for p in poses]
        self._plotted_indexes = [p.index for p in poses]

        color_mode = self.color_by_combo.currentData()
        colors, cmap = self._compute_colors(poses, color_mode)

        # 保存済みの学習用軌道ラベルがあるフレームは外枠（エッジ）付きで描画
        labeled = self.pose_manager.future_traj_indexes()
        if labeled:
            edgecolors = ['black' if p.index in labeled else 'none' for p in poses]
            linewidths = [0.7 if p.index in labeled else 0.0 for p in poses]
        else:
            edgecolors = 'none'
            linewidths = 0.0

        self.ax.scatter(xs, ys, c=colors, cmap=cmap, s=8, picker=5, zorder=2,
                        edgecolors=edgecolors, linewidths=linewidths)

        # ジャンプ（テレポート）が発生した位置は赤い×マーカーで形状を変えて可視化
        legend_handles = []
        jump_indexes = self.pose_manager.flag_jumps(poses, max_jump_m=self.jump_threshold)
        if self.show_jumps_checkbox.isChecked():
            if jump_indexes:
                jump_poses = [p for p in poses if p.index in jump_indexes]
                self.ax.scatter([p.x for p in jump_poses], [p.y for p in jump_poses],
                                marker='x', c='red', s=70, linewidths=1.8, zorder=4)
            legend_handles.append(
                Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=7,
                       label=f"{get_text('map_view_legend_jump')} ({len(jump_indexes)})"))

        # スリップ検知（pose/slip >= 1）はマゼンタの四角で可視化
        slip_indexes = self.pose_manager.slip_indexes()
        if self.show_slip_checkbox.isChecked():
            if slip_indexes:
                slip_poses = [p for p in poses if p.index in slip_indexes]
                self.ax.scatter([p.x for p in slip_poses], [p.y for p in slip_poses],
                                marker='s', facecolors='none', edgecolors='magenta',
                                s=55, linewidths=1.4, zorder=4)
            legend_handles.append(
                Line2D([0], [0], marker='s', markerfacecolor='none', markeredgecolor='magenta',
                       color='none', linestyle='None', markersize=7,
                       label=f"{get_text('map_view_legend_slip')} ({len(slip_indexes)})"))

        # 悪路検知（pose/road_condition == 1）はオレンジの三角で可視化
        rough_indexes = self.pose_manager.rough_road_indexes()
        if self.show_rough_checkbox.isChecked():
            if rough_indexes:
                rough_poses = [p for p in poses if p.index in rough_indexes]
                self.ax.scatter([p.x for p in rough_poses], [p.y for p in rough_poses],
                                marker='^', facecolors='none', edgecolors='darkorange',
                                s=60, linewidths=1.4, zorder=4)
            legend_handles.append(
                Line2D([0], [0], marker='^', markerfacecolor='none', markeredgecolor='darkorange',
                       color='none', linestyle='None', markersize=7,
                       label=f"{get_text('map_view_legend_rough')} ({len(rough_indexes)})"))

        # 凡例（表示中のマーカーのみ、件数付き）
        if legend_handles:
            self.ax.legend(handles=legend_handles, fontsize=8, loc='upper right', framealpha=0.8)

        available = self.pose_manager.available_sources()
        status_text = get_text('map_view_available_sources',
                               ", ".join(available) if available else "-")
        status_text += "　" + get_text('map_view_status_extras',
                                       len(labeled), len(jump_indexes))
        self.status_label.setText(status_text)

        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.canvas.draw()

    # --- 内部処理 ---------------------------------------------------------

    def _compute_colors(self, poses, mode: str):
        if mode == 'time':
            return list(range(len(poses))), 'viridis'
        if mode == 'speed':
            values = [p.extra.get('speed', p.extra.get('v_imu', 0.0)) for p in poses]
            return values, 'viridis'
        if mode == 'source':
            return [SOURCE_COLORS.get(p.source, 'black') for p in poses], None
        if mode == 'status':
            return [STATUS_COLORS.get(p.status, 'black') for p in poses], None
        return ['tab:blue' for _ in poses], None

    def _on_pick(self, event):
        if not self._plotted_indexes or not event.ind.size:
            return
        picked = int(event.ind[0])
        if picked >= len(self._plotted_indexes):
            return
        index = self._plotted_indexes[picked]
        if self.on_frame_selected:
            self.on_frame_selected(index)

    def _on_load_background_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, get_text('map_view_load_background'), "", "Map YAML (*.yaml *.yml)"
        )
        if path:
            self.set_background_map(path)

    def _on_clear_background_clicked(self):
        self.clear_background_map()


class MapViewDialog(QDialog):
    """走行軌跡マップビューを独立ウィンドウとして表示するダイアログ

    メインウィンドウのレイアウトを圧迫しないよう、data_analysis.py の
    DataAnalysisDialog と同じ非モーダル別ウィンドウ方式を採る。

    Phase 2: 品質フィルタ・区間ソース上書き・欠損補間・togivad/future_traj
    書き戻しの操作パネルを提供する。削除マーク/catalog書き戻しの確認・実行・
    結果表示は main_window（ImageAnnotationTool）側の対応メソッドに委譲する
    （overwrite_manifest_deleted_indexes と同じ「確認→実行→結果表示を1メソッド
    にまとめる」既存の流儀に合わせるため）。
    """

    jump_to_image = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._last_quality_flags = set()

        self.setWindowTitle(get_text('map_view_dock_title'))
        self.setMinimumSize(700, 750)
        self.resize(900, 900)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.map_widget = MapViewWidget(on_frame_selected=self._on_frame_selected)
        layout.addWidget(self.map_widget)

        self._build_edit_panel(layout)

    def _build_edit_panel(self, parent_layout):
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        parent_layout.addWidget(separator)

        # --- 品質フィルタ ---
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel(get_text('map_view_quality_label')))
        quality_row.addWidget(QLabel(get_text('map_view_quality_threshold_label')))
        self.quality_threshold_spin = QDoubleSpinBox()
        self.quality_threshold_spin.setRange(0.05, 10.0)
        self.quality_threshold_spin.setSingleStep(0.05)
        self.quality_threshold_spin.setValue(1.0)
        # マップ上のジャンプ×マーカー表示と同じ閾値を共有する
        self.quality_threshold_spin.valueChanged.connect(self._on_jump_threshold_changed)
        quality_row.addWidget(self.quality_threshold_spin)
        self.quality_detect_button = QPushButton(get_text('map_view_quality_detect_btn'))
        self.quality_detect_button.clicked.connect(self._on_quality_detect_clicked)
        quality_row.addWidget(self.quality_detect_button)
        self.quality_mark_deleted_button = QPushButton(get_text('map_view_quality_mark_deleted_btn'))
        self.quality_mark_deleted_button.clicked.connect(self._on_quality_mark_deleted_clicked)
        quality_row.addWidget(self.quality_mark_deleted_button)
        quality_row.addStretch()
        parent_layout.addLayout(quality_row)

        self.quality_result_label = QLabel("")
        self.quality_result_label.setStyleSheet("color: #888888; font-size: 11px;")
        parent_layout.addWidget(self.quality_result_label)

        # --- 区間ソース上書き ---
        segment_row = QHBoxLayout()
        segment_row.addWidget(QLabel(get_text('map_view_segment_label')))
        segment_row.addWidget(QLabel(get_text('map_view_segment_start_label')))
        self.segment_start_spin = QSpinBox()
        self.segment_start_spin.setRange(0, 0)
        segment_row.addWidget(self.segment_start_spin)
        segment_row.addWidget(QLabel(get_text('map_view_segment_end_label')))
        self.segment_end_spin = QSpinBox()
        self.segment_end_spin.setRange(0, 0)
        segment_row.addWidget(self.segment_end_spin)
        segment_row.addWidget(QLabel(get_text('map_view_segment_source_label')))
        self.segment_source_combo = QComboBox()
        segment_row.addWidget(self.segment_source_combo)
        self.segment_apply_button = QPushButton(get_text('map_view_segment_apply_btn'))
        self.segment_apply_button.clicked.connect(self._on_segment_apply_clicked)
        segment_row.addWidget(self.segment_apply_button)
        self.segment_clear_button = QPushButton(get_text('map_view_segment_clear_btn'))
        self.segment_clear_button.clicked.connect(self._on_segment_clear_clicked)
        segment_row.addWidget(self.segment_clear_button)
        segment_row.addStretch()
        parent_layout.addLayout(segment_row)

        # --- 欠損補間 ---
        interp_row = QHBoxLayout()
        interp_row.addWidget(QLabel(get_text('map_view_interp_label')))
        interp_row.addWidget(QLabel(get_text('map_view_interp_maxgap_label')))
        self.interp_maxgap_spin = QSpinBox()
        self.interp_maxgap_spin.setRange(1, 200)
        self.interp_maxgap_spin.setValue(10)
        interp_row.addWidget(self.interp_maxgap_spin)
        self.interp_run_button = QPushButton(get_text('map_view_interp_run_btn'))
        self.interp_run_button.clicked.connect(self._on_interp_run_clicked)
        interp_row.addWidget(self.interp_run_button)
        self.interp_clear_button = QPushButton(get_text('map_view_interp_clear_btn'))
        self.interp_clear_button.clicked.connect(self._on_interp_clear_clicked)
        interp_row.addWidget(self.interp_clear_button)
        interp_row.addStretch()
        parent_layout.addLayout(interp_row)

        self.interp_result_label = QLabel("")
        self.interp_result_label.setStyleSheet("color: #888888; font-size: 11px;")
        parent_layout.addWidget(self.interp_result_label)

        # --- 学習用軌道ラベル（togivad/future_traj）の計算・保存 ---
        writeback_row = QHBoxLayout()
        writeback_label = QLabel(get_text('map_view_writeback_label'))
        writeback_label.setToolTip(get_text('map_view_writeback_tooltip'))
        writeback_row.addWidget(writeback_label)
        writeback_row.addWidget(QLabel(get_text('map_view_writeback_horizon_label')))
        self.writeback_horizon_spin = QSpinBox()
        self.writeback_horizon_spin.setRange(1, 200)
        self.writeback_horizon_spin.setValue(20)
        self.writeback_horizon_spin.setToolTip(get_text('map_view_writeback_horizon_tooltip'))
        writeback_row.addWidget(self.writeback_horizon_spin)
        writeback_row.addWidget(QLabel(get_text('map_view_writeback_dt_label')))
        self.writeback_dt_spin = QDoubleSpinBox()
        self.writeback_dt_spin.setRange(0.01, 2.0)
        self.writeback_dt_spin.setSingleStep(0.01)
        self.writeback_dt_spin.setDecimals(2)
        self.writeback_dt_spin.setValue(0.05)
        self.writeback_dt_spin.setToolTip(get_text('map_view_writeback_dt_tooltip'))
        writeback_row.addWidget(self.writeback_dt_spin)
        self.writeback_button = QPushButton(get_text('map_view_writeback_btn'))
        self.writeback_button.setToolTip(get_text('map_view_writeback_tooltip'))
        self.writeback_button.clicked.connect(self._on_writeback_clicked)
        writeback_row.addWidget(self.writeback_button)
        # ②: 相手車矩形（opponent）→ togivad/agents 書き戻し
        self.agent_writeback_button = QPushButton(
            get_text('map_view_agent_writeback_btn'))
        self.agent_writeback_button.setToolTip(
            get_text('map_view_agent_writeback_tooltip'))
        self.agent_writeback_button.clicked.connect(
            self._on_agent_writeback_clicked)
        writeback_row.addWidget(self.agent_writeback_button)
        writeback_row.addStretch()
        parent_layout.addLayout(writeback_row)

        self.writeback_hint_label = QLabel(get_text('map_view_writeback_hint'))
        self.writeback_hint_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.writeback_hint_label.setWordWrap(True)
        parent_layout.addWidget(self.writeback_hint_label)

    def _on_frame_selected(self, index):
        self.jump_to_image.emit(index)

    def set_pose_manager(self, pose_manager) -> None:
        self.map_widget.jump_threshold = self.quality_threshold_spin.value()
        self.map_widget.set_pose_manager(pose_manager)
        self._last_quality_flags = set()
        self.quality_result_label.setText("")
        self.interp_result_label.setText("")
        self._refresh_segment_controls(pose_manager)

    def auto_load_background(self, data_dir: str) -> None:
        """走行データフォルダに紐づく地図を背景へ自動読み込み（widget へ委譲）"""
        self.map_widget.auto_load_background(data_dir)

    def _refresh_segment_controls(self, pose_manager) -> None:
        known = pose_manager.known_indexes() if pose_manager else []
        lo, hi = (known[0], known[-1]) if known else (0, 0)
        self.segment_start_spin.setRange(lo, hi)
        self.segment_end_spin.setRange(lo, hi)
        self.segment_start_spin.setValue(lo)
        self.segment_end_spin.setValue(hi)

        available = pose_manager.available_sources() if pose_manager else []
        self.segment_source_combo.clear()
        for src in available:
            self.segment_source_combo.addItem(src, src)

    def highlight_frame(self, index: int) -> None:
        self.map_widget.highlight_frame(index)

    # --- 品質フィルタ ------------------------------------------------------

    def _on_jump_threshold_changed(self, value):
        """ジャンプ閾値の変更をマップの×マーカー表示に即時反映する"""
        self.map_widget.jump_threshold = value
        if self.map_widget.pose_manager is not None:
            self.map_widget.refresh()

    def _on_quality_detect_clicked(self):
        pose_manager = self.map_widget.pose_manager
        if pose_manager is None:
            return
        threshold = self.quality_threshold_spin.value()
        self._last_quality_flags = pose_manager.flag_quality_issues(max_jump_m=threshold)
        if self._last_quality_flags:
            self.quality_result_label.setText(
                get_text('map_view_quality_result', len(self._last_quality_flags)))
        else:
            self.quality_result_label.setText(get_text('map_view_quality_none'))

    def _on_quality_mark_deleted_clicked(self):
        if not self._last_quality_flags:
            QMessageBox.information(
                self, get_text('map_view_quality_confirm_title'), get_text('map_view_quality_no_data'))
            return
        if self.main_window is not None:
            # parent_widget=self: このダイアログは常に最前面のため、メインウィンドウ親の
            # モーダルが裏に隠れて操作不能に見える問題を避ける
            self.main_window.mark_indices_as_deleted(
                sorted(self._last_quality_flags), parent_widget=self)
        self._last_quality_flags = set()
        self.quality_result_label.setText("")

    # --- 区間ソース上書き ----------------------------------------------------

    def _on_segment_apply_clicked(self):
        pose_manager = self.map_widget.pose_manager
        if pose_manager is None:
            return
        source = self.segment_source_combo.currentData()
        if source is None:
            return
        pose_manager.set_range_override(
            self.segment_start_spin.value(), self.segment_end_spin.value(), source)
        self.map_widget.refresh()

    def _on_segment_clear_clicked(self):
        pose_manager = self.map_widget.pose_manager
        if pose_manager is None:
            return
        pose_manager.clear_range_overrides()
        self.map_widget.refresh()

    # --- 欠損補間 -----------------------------------------------------------

    def _on_interp_run_clicked(self):
        pose_manager = self.map_widget.pose_manager
        if pose_manager is None:
            return
        filled = pose_manager.interpolate_gaps(max_gap=self.interp_maxgap_spin.value())
        self.interp_result_label.setText(get_text('map_view_interp_result', len(filled)))
        self.map_widget.refresh()

    def _on_interp_clear_clicked(self):
        pose_manager = self.map_widget.pose_manager
        if pose_manager is None:
            return
        pose_manager.clear_interpolation()
        self.interp_result_label.setText("")
        self.map_widget.refresh()

    # --- 学習用軌道ラベル（togivad/future_traj）の計算・保存 -----------------------

    def _on_writeback_clicked(self):
        if self.main_window is None:
            return
        # 実行中の二重起動を防ぐためボタンを無効化（処理中であることも視覚的に分かる）
        self.writeback_button.setEnabled(False)
        try:
            # parent_widget=self: このダイアログは常に最前面のため、メインウィンドウ親の
            # モーダル（確認・進捗・結果）が裏に隠れて操作不能に見える問題を避ける
            self.main_window.write_back_future_trajectories(
                self.writeback_horizon_spin.value(), self.writeback_dt_spin.value(),
                parent_widget=self)
        finally:
            self.writeback_button.setEnabled(True)

    def _on_agent_writeback_clicked(self):
        """② 相手車矩形 → togivad/agents（他車の ego 位置＋将来軌道）書き戻し。"""
        if self.main_window is None:
            return
        self.agent_writeback_button.setEnabled(False)
        try:
            self.main_window.write_back_agent_tracks(
                horizon=self.writeback_horizon_spin.value(), parent_widget=self)
        finally:
            self.agent_writeback_button.setEnabled(True)
