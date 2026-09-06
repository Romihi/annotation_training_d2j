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
                              QCheckBox, QSplitter, QTableWidget, QTableWidgetItem,
                              QAbstractItemView, QHeaderView, QStyledItemDelegate,
                              QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPen, QColor, QBrush, QCursor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import RegularPolygon, Rectangle, Polygon as MplPolygon
from matplotlib.path import Path as MplPath

from translations import get_text
from styles import get_location_color   # 位置クラス色（位置ボタン・バッジと共通）

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


class _CurrentLapDelegate(QStyledItemDelegate):
    """ラップ一覧テーブルで**現在フレームが属するラップの列**に赤枠を描く。

    列=ラップの転置レイアウト前提。current_col の列の各セルに左右の縦線、
    先頭行に上線・最終行に下線を足して、列全体を囲む赤枠として見せる。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_col = -1

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if index.column() != self.current_col:
            return
        painter.save()
        painter.setPen(QPen(QColor(220, 0, 0), 2))
        r = option.rect.adjusted(1, 1, -1, -1)
        painter.drawLine(r.topLeft(), r.bottomLeft())        # 左辺
        painter.drawLine(r.topRight(), r.bottomRight())      # 右辺
        if index.row() == 0:
            painter.drawLine(r.topLeft(), r.topRight())      # 上辺
        if index.row() == index.model().rowCount() - 1:
            painter.drawLine(r.bottomLeft(), r.bottomRight())  # 下辺
        painter.restore()


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
        # 初回描画後 True。以後の refresh はユーザーのズーム・パンを保持する
        self._view_initialized = False
        # 全軌跡（ラップフィルタ前）の index→ラップ番号（再生追従に使用）
        self._lap_by_index = {}
        # ラップタイム表示用（refresh で全軌跡から計算）
        self._lap_start_ts = []            # 各ラップ先頭の _timestamp_ms
        self._lap_times = []               # 各ラップ所要 [s]（末尾は走行中=部分）
        self._lap_first_index = []         # 各ラップ先頭のフレーム index
        self._session_start_ts = None
        self._current_index = None         # highlight_frame で更新

        # 位置領域（閉ポリゴン）: [{"loc": int, "polygon": [(x,y), ...]}, ...]
        # polygon は map 座標系 [m] の頂点列（3点以上・閉路は暗黙）。
        # 地図フォルダの location_regions.json へ永続化。
        # 編集の操作感はセグメンテーションアノテーションモード（main.py の
        # ImageLabel）に合わせる: クリックで頂点追加→始点クリック/右クリックで
        # 閉じる、領域クリックで選択＋ドラッグ移動、頂点は選択領域のみ表示して
        # ドラッグで変形、右クリックメニューで点追加/クラス変更、Delete で削除、
        # Shift+クリックで選択解除。
        self.location_regions = []
        self.region_edit_mode = False      # True の間はマップクリックが領域編集になる
        self.region_edit_class = 0         # 新規作成時に割り当てる位置クラス
        self._region_draft = []            # 作成中ポリゴンの頂点列 [(x,y), ...]
        self.selected_region_index = None  # 選択中の領域
        self.selected_vertex_index = None  # 選択中の頂点（選択領域内）
        self.hovering_region_index = None  # ホバー中の領域（本体）
        self.hovering_vertex_index = None  # ホバー中の頂点（(領域,頂点)の頂点側）
        self.hovering_vertex_region = None # ホバー中の頂点が属する領域
        self.is_moving_vertex = False      # 頂点ドラッグ中
        self.is_moving_region = False      # 領域全体ドラッグ中
        self._region_move_last = None      # 領域移動の直前マウス位置（増分方式）
        self._region_artists = []          # 領域ごとの描画アーティスト（部分更新用）
        self._map_dir = None               # 背景地図のフォルダ（領域定義の保存先）
        self._data_dir = None              # 走行データフォルダ（地図なし時の保存先）
        # 色分け「位置」用: index -> loc（メインウィンドウのアノテーションを参照）
        self.loc_provider = None
        # 位置推論結果: index -> {'pred_class', 'pose': {'x','y','theta'}, ...}（メイン
        # ウィンドウの location_inference_results を参照）。推定座標のマーカー描画と
        # 色分け「推論クラス」に使う
        self.inference_provider = None
        # 位置推論の表示設定 {'top_n', 'grid_mode': 'top1'|'weighted', 'grid_config'} を返す
        # コールバック（格子分類モデルの Top-N セル表示と推定位置の決め方に使う）
        self.inference_settings_provider = None
        # 現在フレームの推定位置マーカー（三角＋実測との誤差線、格子分類の Top-N セル）。blit 用に animated
        self._pred_artists = []
        # 推論済みフレームの推定座標（中空丸）。再生中に推論結果が増えても全体再描画
        # なしで追記できるよう animated な scatter として保持し、blit で描く
        self._pred_scatter = None
        self._pred_scatter_indexes = set()   # scatter に載せ済みのフレーム index
        self._pred_scatter_points = []       # scatter の座標列 [(x, y), ...]
        # 領域の追加・削除・読込をダイアログ側へ通知（引数: ヒント文字列 or None）
        self.on_regions_changed = None

        self._build_ui()

    # 推論位置の色: メイン画面の「位置推論結果」表示（紫）と揃える
    PRED_COLOR = 'purple'
    PRED_EDGE_COLOR = '#4B0082'

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)

        # 1段目: プルダウン類＋地図ボタン（コンパクト表示でも潰れないよう、
        # チェックボックス類は2段目へ分離。コンボは内容幅に自動調整）
        controls = QHBoxLayout()

        controls.addWidget(QLabel(get_text('map_view_source_label')))
        self.source_combo = QComboBox()
        self.source_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.source_combo.addItem(get_text('map_view_source_auto'), None)
        self.source_combo.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.source_combo)

        controls.addWidget(QLabel(get_text('map_view_colorby_label')))
        self.color_by_combo = QComboBox()
        self.color_by_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.color_by_combo.addItem(get_text('map_view_colorby_time'), 'time')
        self.color_by_combo.addItem(get_text('map_view_colorby_lap'), 'lap')
        self.color_by_combo.addItem(get_text('map_view_colorby_speed'), 'speed')
        self.color_by_combo.addItem(get_text('map_view_colorby_source'), 'source')
        self.color_by_combo.addItem(get_text('map_view_colorby_status'), 'status')
        self.color_by_combo.addItem(get_text('map_view_colorby_loc'), 'loc')
        self.color_by_combo.addItem(get_text('map_view_colorby_pred_loc'), 'pred_loc')
        self.color_by_combo.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.color_by_combo)

        # ラップ切替（全 / 1..N）。ラップは開始点への再接近で自動分割。
        # 項目は数字のみの簡素表示（凡例側は「ラップN」のまま）
        controls.addWidget(QLabel(get_text('map_view_lap_label')))
        self.lap_combo = QComboBox()
        self.lap_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.lap_combo.addItem(get_text('map_view_lap_all_short'), None)
        self.lap_combo.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.lap_combo)

        self.load_map_button = QPushButton(get_text('map_view_load_background'))
        self.load_map_button.clicked.connect(self._on_load_background_clicked)
        controls.addWidget(self.load_map_button)

        self.clear_map_button = QPushButton(get_text('map_view_clear_background'))
        self.clear_map_button.clicked.connect(self._on_clear_background_clicked)
        controls.addWidget(self.clear_map_button)

        controls.addStretch()
        layout.addLayout(controls)

        # 2段目: 表示トグル（線表示・ジャンプ・スリップ・悪路）
        toggles = QHBoxLayout()

        # 軌跡を点だけでなく連続線でも描く（ラップ・テレポートで分割）
        self.show_line_checkbox = QCheckBox(get_text('map_view_show_line'))
        self.show_line_checkbox.setChecked(True)
        self.show_line_checkbox.stateChanged.connect(self.refresh)
        toggles.addWidget(self.show_line_checkbox)

        self.show_jumps_checkbox = QCheckBox(get_text('map_view_legend_jump'))
        self.show_jumps_checkbox.setChecked(True)
        self.show_jumps_checkbox.stateChanged.connect(self.refresh)
        toggles.addWidget(self.show_jumps_checkbox)

        self.show_slip_checkbox = QCheckBox(get_text('map_view_legend_slip'))
        self.show_slip_checkbox.setChecked(True)
        self.show_slip_checkbox.stateChanged.connect(self.refresh)
        toggles.addWidget(self.show_slip_checkbox)

        self.show_rough_checkbox = QCheckBox(get_text('map_view_legend_rough'))
        self.show_rough_checkbox.setChecked(True)
        self.show_rough_checkbox.stateChanged.connect(self.refresh)
        toggles.addWidget(self.show_rough_checkbox)

        # 位置推論モデルの推定座標（座標・姿勢回帰）を青で重ねて表示する
        self.show_inference_checkbox = QCheckBox(get_text('map_view_show_inference'))
        self.show_inference_checkbox.setChecked(True)
        self.show_inference_checkbox.setToolTip(get_text('map_view_show_inference_tip'))
        self.show_inference_checkbox.stateChanged.connect(self._on_inference_toggle)
        toggles.addWidget(self.show_inference_checkbox)

        # ラップタイム一覧（折り畳み式。行クリックでそのラップ先頭へジャンプ）
        self.lap_table_button = QPushButton(get_text('map_view_lap_table_btn'))
        self.lap_table_button.setCheckable(True)
        self.lap_table_button.toggled.connect(self._toggle_lap_table)
        toggles.addWidget(self.lap_table_button)

        # ラップタイム表示（選択中ラップの所要＋ラップ内経過 / 全=ベスト＋総経過）
        # 他のボタン・チェックボックスと同じ既定フォントサイズで表示する
        self.lap_time_label = QLabel("")
        toggles.addWidget(self.lap_time_label)

        toggles.addStretch()
        layout.addLayout(toggles)

        # 折り畳み式ラップ一覧テーブル（トグル行の直下・既定は畳んだ状態）。
        # **転置レイアウト**: 行見出し=項目（タイム/開始/備考）・列=ラップ番号。
        # 高さがラップ数に依存せず一定でコンパクト（多周回は横スクロール）。
        self.lap_table = QTableWidget(3, 0)
        self.lap_table.setVerticalHeaderLabels([
            get_text('map_view_lap_table_col_time'),
            get_text('map_view_lap_table_col_start'),
            get_text('map_view_lap_table_col_note')])
        self.lap_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.lap_table.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.lap_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lap_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.lap_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.lap_table.setStyleSheet("font-size: 11px;")
        self.lap_table.setMaximumHeight(110)
        self.lap_table.setVisible(False)
        # 列クリック（セル・列ヘッダのどちらでも）でそのラップ先頭へジャンプ
        self.lap_table.cellClicked.connect(
            lambda _row, col: self._jump_to_lap_start(col))
        self.lap_table.horizontalHeader().sectionClicked.connect(
            self._jump_to_lap_start)
        # 現在フレームが属するラップの列に赤枠（全/各ラップ表示の両方で連動）
        self._lap_delegate = _CurrentLapDelegate(self.lap_table)
        self.lap_table.setItemDelegate(self._lap_delegate)
        layout.addWidget(self.lap_table)

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
        # 領域編集モードのポリゴン編集（軌跡点以外の任意位置をクリックできる
        # よう、pick_event ではなく button_press/motion/release イベントを使う。
        # Delete キーで頂点/領域を削除できるようキャンバスにフォーカスを持たせる）
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.canvas.mpl_connect('axes_leave_event', self._on_axes_leave)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        # blitting: 全描画（refresh/ズーム/リサイズ）のたびに背景をキャッシュし、
        # 再生中の現在位置▲更新は背景復元＋マーカーのみ描画にする
        # （毎フレームの図全体再描画 ~100ms超 → 数ms）
        self._blit_bg = None
        self.canvas.mpl_connect('draw_event', self._on_canvas_draw)

    # --- 外部インターフェース -------------------------------------------------

    def set_pose_manager(self, pose_manager) -> None:
        self.pose_manager = pose_manager
        self._current_marker = None
        self._view_initialized = False     # 新データ読込時は全体表示に戻す
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
        # 領域定義は地図に紐づけて保存するため、地図フォルダを保存先として記憶し、
        # 既存の location_regions.json があれば自動で読み込む
        self._map_dir = os.path.dirname(os.path.abspath(yaml_path))
        self._auto_load_regions()
        self.refresh()

    def auto_load_background(self, data_dir: str) -> None:
        """データフォルダに紐づく地図（無ければ直近に保存された地図）を自動で
        背景に読み込む。解決順: ①<data_dir>/map/ 同梱スナップショット
        ②map_ref.json ③同タイムスタンプの地図 ④maps配下の最新地図。
        手動で背景を設定済みの場合や見つからない場合は何もしない。
        """
        if data_dir:
            self._data_dir = os.path.abspath(data_dir)
        if self._bg_image_path or not data_dir:
            return
        try:
            from utils.map_utils import resolve_background_map
            hit = resolve_background_map(data_dir)
        except Exception as e:
            self.status_label.setText(get_text('map_view_background_load_error', str(e)))
            return
        if not hit:
            # 地図が無くても、データフォルダ直下の領域定義があれば読み込む
            self._auto_load_regions()
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

    # --- 位置領域（閉ポリゴン） ---------------------------------------------

    # 頂点スナップ半径 [画面px]: 既存領域の頂点をこの範囲でクリックすると
    # その頂点座標を再利用する（隣接領域と境界頂点を共有できる）。
    # 作成中ポリゴンの最初の頂点への同判定は「閉じて確定」になる。
    REGION_SNAP_PX = 12

    def regions_save_dir(self):
        """領域定義の保存先フォルダ（地図フォルダ優先、無ければデータフォルダ）"""
        return self._map_dir or self._data_dir

    def _auto_load_regions(self) -> None:
        """保存済みの location_regions.json があれば読み込む（未読込時のみ）。"""
        if self.location_regions:
            return
        from utils.map_utils import load_location_regions
        for d in (self._map_dir, self._data_dir):
            if not d:
                continue
            regions = load_location_regions(d)
            if regions:
                self.location_regions = regions
                if self.on_regions_changed:
                    self.on_regions_changed(get_text(
                        'map_view_region_loaded', len(regions),
                        os.path.basename(d)))
                return

    def save_regions(self) -> str:
        """領域定義を保存してパスを返す（保存先が無い場合は ValueError）。"""
        save_dir = self.regions_save_dir()
        if not save_dir:
            raise ValueError(get_text('map_view_region_no_save_dir'))
        from utils.map_utils import save_location_regions
        return save_location_regions(save_dir, self.location_regions)

    def set_region_edit_mode(self, enabled: bool, loc_class: int = None) -> None:
        self.region_edit_mode = bool(enabled)
        if loc_class is not None:
            self.region_edit_class = int(loc_class)
        self._region_draft = []
        self._clear_region_selection()
        self._clear_region_hover()
        self.canvas.setCursor(QCursor(Qt.ArrowCursor))
        self.refresh()
        self._notify_draft_hint()

    def undo_last_region(self) -> None:
        """作成中なら最後の頂点を、そうでなければ最後の領域を取り消す。"""
        if self._region_draft:
            self._region_draft.pop()
        elif self.location_regions:
            self.location_regions.pop()
        self._clear_region_selection()
        self._clear_region_hover()
        self.refresh()
        self._notify_draft_hint()

    def clear_regions(self) -> None:
        self.location_regions = []
        self._region_draft = []
        self._clear_region_selection()
        self._clear_region_hover()
        self.refresh()
        self._notify_draft_hint()

    def _clear_region_selection(self) -> None:
        self.selected_region_index = None
        self.selected_vertex_index = None
        self.is_moving_vertex = False
        self.is_moving_region = False
        self._region_move_last = None

    def _clear_region_hover(self) -> None:
        self.hovering_region_index = None
        self.hovering_vertex_index = None
        self.hovering_vertex_region = None

    def _notify_draft_hint(self) -> None:
        """編集状態に応じた操作ヒントをダイアログへ通知する。"""
        if not self.on_regions_changed:
            return
        if not self.region_edit_mode:
            self.on_regions_changed(None)
        elif self._region_draft:
            if len(self._region_draft) < 3:
                self.on_regions_changed(get_text(
                    'map_view_region_hint_progress', len(self._region_draft),
                    self.region_edit_class))
            else:
                self.on_regions_changed(get_text(
                    'map_view_region_hint_close', len(self._region_draft),
                    self.region_edit_class))
        elif self.selected_region_index is not None:
            region = self.location_regions[self.selected_region_index]
            self.on_regions_changed(get_text(
                'map_view_region_hint_selected',
                self.selected_region_index + 1, region["loc"]))
        else:
            self.on_regions_changed(get_text(
                'map_view_region_hint_start', self.region_edit_class))

    def _snap_to_region_vertex(self, event):
        """クリックが既存領域の頂点の近く（画面上 REGION_SNAP_PX 以内）なら
        その頂点座標を返す。一度閉じたポリゴンの頂点を新しい領域の頂点として
        そのまま再利用でき、隣接領域が境界を共有できる。該当なしは None。"""
        hit = self._find_vertex_at(event)
        if hit is None:
            return None
        ri, vi = hit
        vx, vy = self.location_regions[ri]["polygon"][vi]
        return (float(vx), float(vy))

    def _find_vertex_at(self, event):
        """画面上 REGION_SNAP_PX 以内の既存領域頂点 (region_i, vertex_i) を返す。

        セグメンテーションモードの check_vertex_hover と同様、選択の有無に
        関わらず全領域の頂点を対象にする。該当なしは None。"""
        best = None
        best_d2 = float(self.REGION_SNAP_PX) ** 2
        for ri, region in enumerate(self.location_regions):
            for vi, (vx, vy) in enumerate(region.get("polygon") or []):
                px, py = self.ax.transData.transform((vx, vy))
                d2 = (px - event.x) ** 2 + (py - event.y) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = (ri, vi)
        return best

    def _find_region_at(self, event):
        """クリック位置を内包する領域のインデックスを返す（該当なしは None）。

        重なりは後から定義した領域を優先（自動付与の判定と同じ規則）。"""
        if event.xdata is None or event.ydata is None:
            return None
        for ri in range(len(self.location_regions) - 1, -1, -1):
            poly = self.location_regions[ri].get("polygon") or []
            if len(poly) >= 3 and MplPath(poly).contains_point(
                    (event.xdata, event.ydata)):
                return ri
        return None

    def _is_near_draft_start(self, event) -> bool:
        """クリックが作成中ポリゴンの最初の頂点の近くか（=閉じる操作か）。"""
        if len(self._region_draft) < 3:
            return False
        fx, fy = self._region_draft[0]
        px, py = self.ax.transData.transform((fx, fy))
        return ((px - event.x) ** 2 + (py - event.y) ** 2
                <= float(self.REGION_SNAP_PX) ** 2)

    @staticmethod
    def _event_has_modifier(event, name: str) -> bool:
        """matplotlib イベントの修飾キー判定（'shift' / 'control'）。"""
        key = (event.key or "").lower()
        if name == 'control':
            return 'control' in key or 'ctrl' in key
        return name in key

    def _on_button_press(self, event):
        """領域編集モード中のマップクリック（セグメンテーションモードと同じ流儀）。

        作成中: 左クリックで頂点追加（既存頂点にスナップ）、最初の頂点クリック
        または右クリック（3点以上）で閉じて確定。
        非作成中: 頂点クリックで頂点ドラッグ開始、領域内クリックで選択＋移動
        ドラッグ開始（Shift+クリックで選択解除）、右クリックでメニュー
        （点を追加 / クラスを変更）、空き地クリックで新規ポリゴン開始
        （Ctrl+クリックは領域内からでも強制的に新規開始）。
        """
        if not self.region_edit_mode or event.inaxes != self.ax:
            return
        # pan/zoom ツール使用中はクリックを領域編集として扱わない
        if getattr(self.nav_toolbar, 'mode', None):
            return

        # --- 右クリック ---
        if event.button == 3:
            if self._region_draft:
                # セグメンテーションモードと同じく右クリックで閉じる（3点以上）
                if len(self._region_draft) >= 3:
                    self._close_region_draft()
                return
            ri = self._find_region_at(event)
            if ri is None:
                return
            if self._event_has_modifier(event, 'shift'):
                # Shift+右クリック: 位置クラスの変更ダイアログ
                self._change_region_class(ri)
            elif event.xdata is not None and event.ydata is not None:
                # 右クリック: クリック位置に最も近い辺へ点を挿入
                self._insert_point_to_region(
                    ri, float(event.xdata), float(event.ydata))
            return

        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        # --- 作成中: 頂点追加 / 閉じる ---
        if self._region_draft:
            if self._is_near_draft_start(event):
                self._close_region_draft()
                return
            snapped = self._snap_to_region_vertex(event)
            vertex = snapped if snapped is not None else (
                float(event.xdata), float(event.ydata))
            self._region_draft.append(vertex)
            self.refresh()
            self._notify_draft_hint()
            return

        # --- Ctrl+クリック: 既存領域の上からでも新規ポリゴンを開始 ---
        # （重ね塗り修正用。既存頂点近くならスナップして境界を共有）
        if self._event_has_modifier(event, 'control'):
            self._start_region_draft(event)
            return

        # --- 頂点クリック: 頂点ドラッグ開始（選択領域以外の頂点も掴める） ---
        hit = self._find_vertex_at(event)
        if hit is not None:
            self.selected_region_index, self.selected_vertex_index = hit
            self.is_moving_vertex = True
            self._refresh_region_styles()
            self._notify_draft_hint()
            return

        # --- 領域内クリック: 選択＋移動ドラッグ開始 / Shift+クリックで解除 ---
        ri = self._find_region_at(event)
        if ri is not None:
            if (self._event_has_modifier(event, 'shift')
                    and ri == self.selected_region_index):
                self._clear_region_selection()
                self._refresh_region_styles()
                self._notify_draft_hint()
                return
            self.selected_region_index = ri
            self.selected_vertex_index = None
            self.is_moving_region = True
            self._region_move_last = (float(event.xdata), float(event.ydata))
            self._refresh_region_styles()
            self._notify_draft_hint()
            return

        # --- 空き地クリック: 選択を外して新規ポリゴン開始 ---
        self._start_region_draft(event)

    def _start_region_draft(self, event) -> None:
        """新規ポリゴンの作成を開始する（最初の頂点を置く）。"""
        self._clear_region_selection()
        self._clear_region_hover()
        snapped = self._snap_to_region_vertex(event)
        vertex = snapped if snapped is not None else (
            float(event.xdata), float(event.ydata))
        self._region_draft = [vertex]
        self.refresh()
        self._notify_draft_hint()

    def _on_motion(self, event):
        """領域編集モード中のマウス移動: ドラッグ更新とホバー強調。"""
        if not self.region_edit_mode:
            return
        if getattr(self.nav_toolbar, 'mode', None):
            return

        # --- 頂点ドラッグ: リアルタイムに座標を書き換え ---
        if self.is_moving_vertex:
            if (event.inaxes != self.ax or event.xdata is None
                    or self.selected_region_index is None
                    or self.selected_vertex_index is None):
                return
            poly = self.location_regions[self.selected_region_index]["polygon"]
            poly[self.selected_vertex_index] = (
                float(event.xdata), float(event.ydata))
            self._sync_region_artist(self.selected_region_index)
            return

        # --- 領域全体ドラッグ: 増分デルタを全頂点へ適用 ---
        if self.is_moving_region:
            if (event.inaxes != self.ax or event.xdata is None
                    or self.selected_region_index is None
                    or self._region_move_last is None):
                return
            dx = float(event.xdata) - self._region_move_last[0]
            dy = float(event.ydata) - self._region_move_last[1]
            region = self.location_regions[self.selected_region_index]
            region["polygon"] = [(x + dx, y + dy)
                                 for x, y in region["polygon"]]
            self._region_move_last = (float(event.xdata), float(event.ydata))
            self._sync_region_artist(self.selected_region_index)
            return

        # --- ホバー強調（作成中は無効。セグメンテーションモードと同じ） ---
        if self._region_draft:
            return
        if event.inaxes != self.ax:
            self._update_region_hover(None, None, None)
            return
        hit = self._find_vertex_at(event)
        if hit is not None:
            self._update_region_hover(None, hit[0], hit[1])
            self.canvas.setCursor(QCursor(Qt.PointingHandCursor))
            return
        ri = self._find_region_at(event)
        if ri is not None:
            self._update_region_hover(ri, None, None)
            self.canvas.setCursor(QCursor(Qt.OpenHandCursor))
            return
        self._update_region_hover(None, None, None)

    def _update_region_hover(self, region_i, vertex_region_i, vertex_i) -> None:
        """ホバー状態を更新し、変化があったときだけ再スタイル・再描画する。"""
        if (self.hovering_region_index == region_i
                and self.hovering_vertex_region == vertex_region_i
                and self.hovering_vertex_index == vertex_i):
            return
        self.hovering_region_index = region_i
        self.hovering_vertex_region = vertex_region_i
        self.hovering_vertex_index = vertex_i
        if region_i is None and vertex_i is None:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))
        self._refresh_region_styles()

    def _on_button_release(self, event):
        """ドラッグ終了。選択状態は維持する（セグメンテーションモードと同じ）。"""
        if not self.region_edit_mode:
            return
        if self.is_moving_vertex or self.is_moving_region:
            self.is_moving_vertex = False
            self.is_moving_region = False
            self._region_move_last = None
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))
            self._notify_draft_hint()

    def _on_axes_leave(self, _event):
        """マウスがプロット外へ出たらホバー強調とカーソルを戻す。"""
        if not self.region_edit_mode:
            return
        if not (self.is_moving_vertex or self.is_moving_region):
            self._update_region_hover(None, None, None)

    def _on_key_press(self, event):
        if event.key in ('delete', 'backspace'):
            self.delete_by_key()

    def delete_by_key(self) -> None:
        """Delete/Backspace による削除。

        セグメンテーションモードの eventFilter と同じ優先順位:
        選択中の頂点 → 選択中の領域 → **ホバー中の領域**。
        メインウィンドウの eventFilter からも委譲される（マップビュー上に
        マウスがある間のDelete。後ろの画面のアノテーションは削除させない）。
        """
        if not self.region_edit_mode:
            return
        # 頂点削除（3点未満になる削除は不可。セグメンテーションモードと同じ）
        if (self.selected_region_index is not None
                and self.selected_vertex_index is not None
                and self.selected_region_index < len(self.location_regions)):
            region = self.location_regions[self.selected_region_index]
            if len(region["polygon"]) <= 3:
                QMessageBox.warning(self, get_text('dlg_warning'),
                                    get_text('msg_polygon_min_vertices'))
                return
            region["polygon"].pop(self.selected_vertex_index)
            self._clear_region_selection()
            self._clear_region_hover()
            self.refresh()
            self._notify_draft_hint()
            return
        # 領域削除: 選択中を優先し、無ければホバー中の領域
        target = self.selected_region_index
        if target is None:
            target = self.hovering_region_index
        if target is None or not (0 <= target < len(self.location_regions)):
            return
        removed = self.location_regions.pop(target)
        self._clear_region_selection()
        self._clear_region_hover()
        self.canvas.setCursor(QCursor(Qt.ArrowCursor))
        self.refresh()
        if self.on_regions_changed:
            self.on_regions_changed(get_text(
                'map_view_region_deleted', removed["loc"]))
        else:
            self._notify_draft_hint()

    def _change_region_class(self, region_i: int) -> None:
        """領域の位置クラスを変更する（セグメンテーションのクラス変更に相当）。"""
        region = self.location_regions[region_i]
        value, ok = QInputDialog.getInt(
            self, get_text('map_view_region_menu_change_class'),
            get_text('map_view_region_class_label'),
            int(region["loc"]), 0, 99)
        if not ok:
            return
        region["loc"] = int(value)
        self.refresh()
        if self.on_regions_changed:
            self.on_regions_changed(None)

    def _insert_point_to_region(self, region_i: int, x: float, y: float) -> None:
        """クリック位置に最も近い辺へ新しい頂点を挿入する
        （セグメンテーションモードの「点を追加」と同じ最近辺方式）。"""
        poly = self.location_regions[region_i]["polygon"]
        best_index = len(poly)
        best_dist = float('inf')
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            d = self._point_to_segment_distance(x, y, p1, p2)
            if d < best_dist:
                best_dist = d
                best_index = i + 1
        poly.insert(best_index, (x, y))
        self.selected_region_index = region_i
        self.selected_vertex_index = best_index
        self.refresh()
        if self.on_regions_changed:
            self.on_regions_changed(get_text(
                'map_view_region_point_added', best_index + 1))

    @staticmethod
    def _point_to_segment_distance(x, y, p1, p2) -> float:
        """点 (x,y) と線分 p1-p2 の距離（tを[0,1]にクランプ）。"""
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 0.0:
            return math.hypot(x - x1, y - y1)
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len2))
        return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))

    def _close_region_draft(self) -> None:
        """作成中ポリゴンを閉じて領域として確定する。"""
        polygon = list(self._region_draft)
        self._region_draft = []
        self.location_regions.append(
            {"loc": int(self.region_edit_class), "polygon": polygon})
        self.refresh()
        if self.on_regions_changed:
            self.on_regions_changed(
                get_text('map_view_region_added', self.region_edit_class,
                         len(polygon))
                + "　" + get_text('map_view_region_hint_start',
                                  self.region_edit_class))

    # 領域スタイル定数（セグメンテーションモードの選択/ホバー/通常の見た目に対応。
    # 塗りアルファは Qt の 150/100/120（/255）、線幅は 4/3/2 px を pt へ換算）
    REGION_FACE_ALPHA = {'selected': 0.59, 'hovered': 0.39, 'normal': 0.47}
    REGION_EDGE_WIDTH = {'selected': 2.5, 'hovered': 2.0, 'normal': 1.5}
    # 頂点マーカー（選択領域のみ表示。選択頂点=黄/黒縁、ホバー頂点=橙、通常=白）
    REGION_VERTEX_STYLE = {
        'selected': {'face': '#ffff00', 'edge': 'black', 'size': 110, 'lw': 1.5},
        'hovered': {'face': '#ffa500', 'edge': None, 'size': 80, 'lw': 1.0},
        'normal': {'face': 'white', 'edge': None, 'size': 45, 'lw': 1.0},
    }
    DRAFT_COLOR = '#ffff00'   # 作成中は黄色（セグメンテーションの作成中表示と同じ）

    def _region_state(self, region_i: int) -> str:
        if region_i == self.selected_region_index:
            return 'selected'
        if region_i == self.hovering_region_index:
            return 'hovered'
        return 'normal'

    def _draw_regions(self) -> None:
        """定義済みの位置領域（閉ポリゴン）と作成中ポリゴンを描く。

        領域ごとにアーティスト（本体ポリゴン・頂点・ラベル）への参照を保持し、
        ドラッグ・ホバー中はそれらの部分更新＋draw_idle だけで済ませる
        （全 refresh は重いので毎モーションでは行わない）。
        """
        self._region_artists = []
        # インデックスの妥当性を検証（undo等で領域数が減った直後の安全策）
        if (self.selected_region_index is not None
                and self.selected_region_index >= len(self.location_regions)):
            self._clear_region_selection()
        for ri, region in enumerate(self.location_regions):
            poly = region.get("polygon") or []
            if len(poly) < 3:
                self._region_artists.append(None)
                continue
            base = get_location_color(region["loc"])
            edge_color = base.darker().name()
            patch = MplPolygon(poly, closed=True, zorder=1.2,
                               facecolor=base.name(), edgecolor=edge_color)
            self.ax.add_patch(patch)
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            verts = self.ax.scatter(xs, ys, zorder=1.35, visible=False)
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            label = self.ax.text(
                cx, cy, str(region["loc"]), color='white', fontsize=9,
                fontweight='bold', ha='center', va='center', zorder=1.3,
                bbox=dict(facecolor=edge_color, edgecolor='none',
                          boxstyle='round,pad=0.25', alpha=0.9))
            self._region_artists.append(
                {'patch': patch, 'verts': verts, 'label': label,
                 'base': base, 'edge_color': edge_color})
            self._apply_region_style(ri)
        # 作成中ポリゴン: 黄の実線＋頂点。3点以上で始点→終点の破線と、
        # 「ここをクリックで閉じる」目印として始点を白塗り黄縁で強調する
        if self._region_draft:
            xs = [p[0] for p in self._region_draft]
            ys = [p[1] for p in self._region_draft]
            if len(xs) >= 2:
                self.ax.plot(xs, ys, '-', color=self.DRAFT_COLOR,
                             linewidth=2.0, zorder=3)
            if len(xs) >= 3:
                self.ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]], '--',
                             color=self.DRAFT_COLOR, linewidth=1.2, zorder=3)
            self.ax.scatter(xs, ys, c=self.DRAFT_COLOR, s=45, zorder=3.1,
                            edgecolors='black', linewidths=0.5)
            if len(xs) >= 3:
                self.ax.scatter([xs[0]], [ys[0]], facecolors='white',
                                edgecolors=self.DRAFT_COLOR, s=110,
                                linewidths=2.0, zorder=3.2)

    def _apply_region_style(self, region_i: int) -> None:
        """選択/ホバー状態に応じた本体・頂点のスタイルを領域へ適用する。"""
        art = (self._region_artists[region_i]
               if region_i < len(self._region_artists) else None)
        if art is None:
            return
        state = self._region_state(region_i)
        base = art['base']
        r, g, b = base.red() / 255.0, base.green() / 255.0, base.blue() / 255.0
        art['patch'].set_facecolor((r, g, b, self.REGION_FACE_ALPHA[state]))
        art['patch'].set_linewidth(self.REGION_EDGE_WIDTH[state])
        # 頂点は選択領域のみ表示（セグメンテーションモードと同じ）
        selected = (region_i == self.selected_region_index)
        art['verts'].set_visible(selected)
        if selected:
            poly = self.location_regions[region_i]["polygon"]
            faces, edges, sizes, lws = [], [], [], []
            for vi in range(len(poly)):
                if vi == self.selected_vertex_index:
                    style = self.REGION_VERTEX_STYLE['selected']
                elif (region_i == self.hovering_vertex_region
                        and vi == self.hovering_vertex_index):
                    style = self.REGION_VERTEX_STYLE['hovered']
                else:
                    style = self.REGION_VERTEX_STYLE['normal']
                faces.append(style['face'])
                edges.append(style['edge'] or art['edge_color'])
                sizes.append(style['size'])
                lws.append(style['lw'])
            art['verts'].set_facecolors(faces)
            art['verts'].set_edgecolors(edges)
            art['verts'].set_sizes(sizes)
            art['verts'].set_linewidths(lws)

    def _refresh_region_styles(self) -> None:
        """全領域のスタイルを再適用して軽量再描画する（ホバー・選択変更用）。"""
        for ri in range(len(self._region_artists)):
            self._apply_region_style(ri)
        self.canvas.draw_idle()

    def _sync_region_artist(self, region_i: int) -> None:
        """ドラッグ中の領域の形状をアーティストへ反映して軽量再描画する。"""
        art = (self._region_artists[region_i]
               if region_i < len(self._region_artists) else None)
        if art is None:
            return
        poly = self.location_regions[region_i]["polygon"]
        art['patch'].set_xy(poly)
        art['verts'].set_offsets(poly)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        art['label'].set_position((sum(xs) / len(xs), sum(ys) / len(ys)))
        self.canvas.draw_idle()

    def highlight_frame(self, index: int) -> None:
        if self.pose_manager is None:
            return
        self._current_index = index
        self._update_lap_time_label()
        self._update_lap_table_current()
        # ラップ絞り込み中に現在フレームが別ラップへ移った場合は、そのラップへ
        # 表示を自動で切り替える（再生追従。setCurrentIndex が refresh を起動し、
        # ズーム・パンは refresh 側で保持される）
        sel = self.lap_combo.currentData()
        if sel is not None:
            lap = self._lap_by_index.get(index)
            if lap is not None and lap != sel:
                pos = self.lap_combo.findData(lap)
                if pos >= 0:
                    self.lap_combo.setCurrentIndex(pos)
        self._build_current_markers(index)
        # blitting: 背景キャッシュがあれば「背景復元＋マーカーだけ描画」で済ませ、
        # 図全体の再描画（数十〜百ms超）を避ける。キャッシュが無い初回や
        # リサイズ直後は通常描画にフォールバック（draw_event で再キャッシュ）。
        if self._blit_bg is not None:
            self.canvas.restore_region(self._blit_bg)
            self._draw_animated_artists()
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def _build_current_markers(self, index: int) -> None:
        """現在フレームのマーカー（実測=赤三角、推定=紫三角＋誤差線）を作り直す（描画はしない）

        highlight_frame（blit 更新）と refresh（全体再描画）の両方から使う。refresh でも
        ここでマーカーを作っておくことで、トグル切替・色分け変更・ラップ切替の直後に
        現在フレームの三角が消えず、次のフレーム移動を待たずに表示される。
        """
        pose = self.pose_manager.get_pose(index, prefer=self.source_combo.currentData())
        if self._current_marker is not None:
            try:
                self._current_marker.remove()
            except (ValueError, AttributeError):
                pass
            self._current_marker = None
        # 位置推論の推定座標（三角＋実測との誤差線）。再生中に推論されたフレームの
        # 丸も同時に追記する（全体再描画なしで軌跡の丸が増えていく）
        self._append_pred_point(index)
        self._draw_pred_marker(index, pose)
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
            # animated=True: 通常の全描画から除外し blit でのみ描く
            self._current_marker.set_animated(True)
            self.ax.add_patch(self._current_marker)

    def _draw_animated_artists(self):
        """blit で描く animated アーティスト（推定座標の丸・三角・誤差線・現在位置）を描画"""
        if self._pred_scatter is not None:
            self.ax.draw_artist(self._pred_scatter)
        for artist in self._pred_artists:
            self.ax.draw_artist(artist)
        if self._current_marker is not None:
            self.ax.draw_artist(self._current_marker)

    def _on_canvas_draw(self, _event=None):
        """全描画の完了時に背景（マーカー以外の全要素）をキャッシュする。

        refresh・ズーム・パン・リサイズ等の通常描画は animated なマーカーを
        含まないため、その直後のキャンバスがそのまま blit 用の背景になる。
        マーカーがある場合はキャッシュ後に上へ blit で描き直す。
        """
        try:
            self._blit_bg = self.canvas.copy_from_bbox(self.ax.bbox)
        except Exception:
            self._blit_bg = None
            return
        if self._current_marker is not None or self._pred_artists or self._pred_scatter is not None:
            self._draw_animated_artists()
            self.canvas.blit(self.ax.bbox)

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
        # 初回以降はユーザーのズーム・パンを保持する（タブ切替・トグル変更等の
        # 再描画で表示範囲が初期化されないように、clear 前の範囲を復元する）
        keep_view = self._view_initialized
        saved_xlim = self.ax.get_xlim() if keep_view else None
        saved_ylim = self.ax.get_ylim() if keep_view else None

        def _finish_draw():
            if keep_view:
                self.ax.set_xlim(saved_xlim)
                self.ax.set_ylim(saved_ylim)
            else:
                self._view_initialized = True
            self.canvas.draw()

        self.ax.clear()
        self._current_marker = None
        self._pred_artists = []   # ax.clear() で消えているため参照だけ捨てる
        self._pred_scatter = None
        self._pred_scatter_indexes = set()
        self._pred_scatter_points = []
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

        # 位置領域は軌跡の有無に関わらず描く（領域だけ先に確認できるように）
        self._draw_regions()
        # 格子分類モデルの格子線（読み込み中のモデルが格子出力を持つときのみ）
        self._draw_grid_lines()

        if self.pose_manager is None or not self.pose_manager.has_any_pose():
            self._plotted_indexes = []
            self.status_label.setText(get_text('map_view_no_pose_data'))
            _finish_draw()
            return

        source = self.source_combo.currentData()
        poses = self.pose_manager.get_trajectory(source=source)
        if not poses:
            self._plotted_indexes = []
            self.status_label.setText(get_text('map_view_no_pose_data'))
            _finish_draw()
            return

        # ラップ分割（開始点への再接近で境界検出）と切替フィルタ
        laps = self._compute_laps(poses)
        n_laps = (max(laps) + 1) if laps else 0
        # フィルタ前の対応を保存（再生時のラップ自動追従が参照）
        self._lap_by_index = {p.index: l for p, l in zip(poses, laps)}
        self._compute_lap_times(poses, laps, n_laps)
        self._update_lap_combo(n_laps)
        self._update_lap_time_label()
        if self.lap_table.isVisible():     # 展開中はラップ再計算に追従して更新
            self._populate_lap_table()
        lap_sel = self.lap_combo.currentData()
        if lap_sel is not None:
            keep = [i for i, l in enumerate(laps) if l == lap_sel]
            poses = [poses[i] for i in keep]
            laps = [laps[i] for i in keep]
            if not poses:
                self._plotted_indexes = []
                self.status_label.setText(get_text('map_view_no_pose_data'))
                _finish_draw()
                return

        xs = [p.x for p in poses]
        ys = [p.y for p in poses]
        self._plotted_indexes = [p.index for p in poses]

        color_mode = self.color_by_combo.currentData()
        if color_mode == 'lap':
            colors = [self._lap_color(l) for l in laps]
            cmap = None
        else:
            colors, cmap = self._compute_colors(poses, color_mode)

        # 連続線表示: 点列をラップごとに結ぶ（テレポート級のギャップは繋がない）。
        # ラップ色分け時はラップ色、他モードは薄い青で点の下（zorder 1.5）に敷く
        if self.show_line_checkbox.isChecked() and len(xs) >= 2:
            gap_r = max(self.jump_threshold, 0.5)

            def _line_color(lap):
                return (self._lap_color(lap) if color_mode == 'lap'
                        else '#1f77b4')

            seg_x, seg_y = [xs[0]], [ys[0]]
            seg_lap = laps[0] if laps else 0
            for i in range(1, len(xs)):
                l = laps[i] if laps else 0
                gap = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) > gap_r
                if l != seg_lap or gap:
                    if len(seg_x) >= 2:
                        self.ax.plot(seg_x, seg_y, '-', color=_line_color(seg_lap),
                                     linewidth=1.3, alpha=0.85, zorder=1.5)
                    seg_x, seg_y = [], []
                    seg_lap = l
                seg_x.append(xs[i])
                seg_y.append(ys[i])
            if len(seg_x) >= 2:
                self.ax.plot(seg_x, seg_y, '-', color=_line_color(seg_lap),
                             linewidth=1.3, alpha=0.85, zorder=1.5)

        # 学習用軌道ラベル保存済みフレーム数（ステータス表示用のみ）。
        # 旧: ラベル済み点に黒枠エッジを描いていたが、書き戻し後は全点に付いて
        # 情報量ゼロのまま再描画だけ重くなる（実測1.6倍）ため表示機能は廃止。
        labeled = self.pose_manager.future_traj_indexes()

        self.ax.scatter(xs, ys, c=colors, cmap=cmap, s=8, picker=5, zorder=2,
                        edgecolors='none', linewidths=0.0)

        # ラップ色分け時はラップごとの凡例（点数付き・最大12項目）
        legend_handles = []
        if color_mode == 'lap' and laps:
            for l in sorted(set(laps))[:12]:
                legend_handles.append(Line2D(
                    [0], [0], marker='o', color=self._lap_color(l),
                    linestyle='None', markersize=7,
                    label=get_text('map_view_lap_item', l + 1)
                          + f" ({laps.count(l)})"))

        # ジャンプ（テレポート）が発生した位置は赤い×マーカーで形状を変えて可視化
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

        # 位置推論モデルの推定座標（座標・姿勢回帰の結果があるフレームのみ）
        self._draw_pred_trajectory(poses, legend_handles)

        # 凡例（表示中のマーカーのみ、件数付き）
        if legend_handles:
            self.ax.legend(handles=legend_handles, fontsize=8, loc='upper right', framealpha=0.8)

        available = self.pose_manager.available_sources()
        status_text = get_text('map_view_available_sources',
                               ", ".join(available) if available else "-")
        status_text += "　" + get_text('map_view_status_extras',
                                       len(labeled), len(jump_indexes))
        status_text += "　" + get_text('map_view_lap_status', n_laps)
        self.status_label.setText(status_text)

        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')

        # 現在フレームのマーカー（赤三角・推定位置の紫三角）を全体再描画に含める。
        # 表示範囲の復元後にサイズを決めるため、_finish_draw と同じ順で範囲を先に戻す
        if keep_view:
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)
        if self._current_index is not None:
            self._build_current_markers(self._current_index)
        _finish_draw()

    # --- 内部処理 ---------------------------------------------------------

    # ラップ分割パラメータ: 開始点に close_r まで再接近したらラップ境界。
    # 誤検出防止に、いったん rearm_r より離れる＆最低 min_arc_m 走ってから
    # のみ判定する（ヒステリシス。スタート付近のうろつきで刻まれない）。
    LAP_CLOSE_R_M = 1.0
    LAP_REARM_R_M = 2.0
    LAP_MIN_ARC_M = 5.0

    def _compute_laps(self, poses):
        """各 pose のラップ番号（0始まり）を返す（周回コース前提）。

        開始点 = 軌跡の先頭 pose。開始点への再接近（ヒステリシス付き）で
        ラップ境界を検出する。周回しないセッションでは全点ラップ0のまま。
        """
        laps = []
        if not poses:
            return laps
        sx, sy = poses[0].x, poses[0].y
        px, py = sx, sy
        lap, armed, arc = 0, False, 0.0
        for p in poses:
            arc += math.hypot(p.x - px, p.y - py)
            px, py = p.x, p.y
            d = math.hypot(p.x - sx, p.y - sy)
            if not armed:
                if d > self.LAP_REARM_R_M:
                    armed = True
            elif d < self.LAP_CLOSE_R_M and arc >= self.LAP_MIN_ARC_M:
                lap += 1
                armed = False
                arc = 0.0
            laps.append(lap)
        # 最終境界の直後で記録が終わると、走行距離が最短周回長に満たない
        # 「尻尾」が独立ラップとして残る。ノイズなので直前のラップへ併合する。
        if lap > 0 and arc < self.LAP_MIN_ARC_M:
            laps = [min(l, lap - 1) for l in laps]
        return laps

    def _timestamps(self) -> dict:
        """index→_timestamp_ms（pose_manager 保持。無ければ空）。"""
        return getattr(self.pose_manager, '_timestamps_ms', {}) or {}

    def _compute_lap_times(self, poses, laps, n_laps: int) -> None:
        """全軌跡から各ラップの開始時刻と所要時間[s]を計算して保持する。

        ラップ l の所要 = 次ラップ先頭時刻 − 当ラップ先頭時刻。最終ラップは
        記録末尾までの部分時間（走行中扱い）。タイムスタンプ欠損時は空のまま。
        """
        ts = self._timestamps()
        self._lap_start_ts = []
        self._lap_times = []
        self._lap_first_index = []
        self._session_start_ts = None
        if not poses or not ts:
            return
        first_ts = [None] * n_laps
        first_idx = [None] * n_laps
        last_ts = None
        for p, l in zip(poses, laps):
            if first_idx[l] is None:
                first_idx[l] = p.index
            t = ts.get(p.index)
            if t is None:
                continue
            if first_ts[l] is None:
                first_ts[l] = t
            last_ts = t
        if any(t is None for t in first_ts) or last_ts is None:
            return
        self._session_start_ts = first_ts[0]
        self._lap_start_ts = first_ts
        self._lap_first_index = first_idx
        for l in range(n_laps):
            end = first_ts[l + 1] if l + 1 < n_laps else last_ts
            self._lap_times.append(max(0.0, (end - first_ts[l]) / 1000.0))

    def _update_lap_time_label(self) -> None:
        """チェックボックス右のラップタイム表示を更新する。

        - ラップ選択時: 「ラップN: 所要s｜ラップ内 経過s」
        - 全選択時   : 「ベスト: ラップk 所要s｜経過 総経過s」
          （ベストは完了ラップ=最終ラップ以外から。完了ラップが無ければ経過のみ）
        """
        label = getattr(self, 'lap_time_label', None)
        if label is None:
            return
        ts = self._timestamps()
        if not self._lap_times or not ts:
            label.setText("")
            return
        cur_ts = ts.get(self._current_index) if self._current_index is not None \
            else None
        sel = self.lap_combo.currentData()
        if sel is not None and sel < len(self._lap_times):
            in_lap = ""
            cur_lap = self._lap_by_index.get(self._current_index)
            if cur_ts is not None and cur_lap is not None \
                    and cur_lap < len(self._lap_start_ts):
                in_lap = f"{(cur_ts - self._lap_start_ts[cur_lap]) / 1000.0:.2f}"
            label.setText(get_text('map_view_laptime_current',
                                   sel + 1, f"{self._lap_times[sel]:.2f}",
                                   in_lap or "-"))
            return
        # 全ラップ表示: 完了ラップ（最終ラップ以外）からベストを選ぶ
        total = ""
        if cur_ts is not None and self._session_start_ts is not None:
            total = f"{(cur_ts - self._session_start_ts) / 1000.0:.2f}"
        complete = self._lap_times[:-1]
        if complete:
            k = int(min(range(len(complete)), key=lambda i: complete[i]))
            label.setText(get_text('map_view_laptime_best',
                                   k + 1, f"{complete[k]:.2f}", total or "-"))
        else:
            label.setText(get_text('map_view_laptime_total', total or "-"))

    def _toggle_lap_table(self, checked: bool) -> None:
        """折り畳み式ラップ一覧の展開/収納。展開時に内容を再構築する。"""
        self.lap_table.setVisible(checked)
        if checked:
            self._populate_lap_table()

    def _populate_lap_table(self) -> None:
        """ラップ一覧テーブルを現在のラップタイムで再構築する。

        行クリックでそのラップの先頭フレームへジャンプする（既存の
        on_frame_selected 経由。ラップ絞り込み中なら自動追従で表示も切替わる）。
        """
        n = len(self._lap_times)
        self.lap_table.setColumnCount(n)
        self.lap_table.setHorizontalHeaderLabels([str(l + 1) for l in range(n)])
        complete = self._lap_times[:-1]
        best = (int(min(range(len(complete)), key=lambda i: complete[i]))
                if complete else None)
        for l in range(n):
            note = ""
            if l == best:
                note = get_text('map_view_lap_table_best')
            elif l == n - 1 and len(complete) == n - 1:
                note = get_text('map_view_lap_table_running')
            start = (self._lap_first_index[l]
                     if l < len(self._lap_first_index) else "")
            for row, text in enumerate([f"{self._lap_times[l]:.2f}",
                                        str(start), note]):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.lap_table.setItem(row, l, item)
        self._lap_delegate.current_col = -1        # 再構築後に赤枠を再適用
        self._update_lap_table_current()

    def _update_lap_table_current(self) -> None:
        """現在フレームが属するラップの列へ赤枠＋ヘッダー赤字を同期する。

        テーブルは常に全ラップを表示するため、ラップ絞り込み（各ラップ表示）
        中でも現在ラップの位置とタイムが表から分かる。再生・シークで
        highlight_frame 経由で毎フレーム呼ばれる（変化時のみ再描画）。
        """
        if not self.lap_table.isVisible():
            return
        cur = self._lap_by_index.get(self._current_index)
        col = cur if cur is not None and cur < self.lap_table.columnCount() else -1
        if self._lap_delegate.current_col == col:
            return
        self._lap_delegate.current_col = col
        red, black = QBrush(QColor(220, 0, 0)), QBrush(QColor(0, 0, 0))
        for c in range(self.lap_table.columnCount()):
            it = self.lap_table.horizontalHeaderItem(c)
            if it is not None:
                it.setForeground(red if c == col else black)
        if col >= 0:
            self.lap_table.scrollToItem(
                self.lap_table.item(0, col), QAbstractItemView.EnsureVisible)
        self.lap_table.viewport().update()

    def _jump_to_lap_start(self, lap: int) -> None:
        """ラップ一覧の列クリック: そのラップだけの表示に切替＋先頭へジャンプ。

        プルダウン選択と同じ絞り込み（lap_combo 切替 → refresh。ズームは保持）
        を行ってから先頭フレームへ飛ぶ。全ラップ表示へ戻すにはプルダウンで
        「全」を選ぶ。
        """
        if not (0 <= lap < len(self._lap_first_index)):
            return
        pos = self.lap_combo.findData(lap)
        if pos >= 0 and self.lap_combo.currentIndex() != pos:
            self.lap_combo.setCurrentIndex(pos)      # → refresh（絞り込み表示）
        idx = self._lap_first_index[lap]
        if idx is not None and self.on_frame_selected:
            self.on_frame_selected(int(idx))

    def _update_lap_combo(self, n_laps: int) -> None:
        """ラップ切替コンボを件数に合わせて再構築する（選択は維持）。

        コンパクト表示でも潰れないよう項目は簡素表示（「全」と数字のみ）。
        """
        current = self.lap_combo.currentData()
        self.lap_combo.blockSignals(True)
        self.lap_combo.clear()
        self.lap_combo.addItem(get_text('map_view_lap_all_short'), None)
        for l in range(n_laps):
            self.lap_combo.addItem(str(l + 1), l)
        if current is not None and 0 <= current < n_laps:
            self.lap_combo.setCurrentIndex(1 + current)
        self.lap_combo.blockSignals(False)

    @staticmethod
    def _lap_color(lap: int):
        """ラップ番号 → 判別しやすい色（tab20 を循環）。"""
        import matplotlib.pyplot as plt
        return plt.get_cmap('tab20')(lap % 20)

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
        if mode == 'loc':
            # 位置アノテーション（user/loc・自動付与を含む）で色分け。未付与は薄灰
            colors = []
            for p in poses:
                loc = self.loc_provider(p.index) if self.loc_provider else None
                colors.append(get_location_color(loc).name()
                              if loc is not None else '#d0d0d0')
            return colors, None
        if mode == 'pred_loc':
            # 位置推論モデルの予測クラスで色分け（「位置」と切り替えて比較する）。未推論は薄灰
            colors = []
            for p in poses:
                res = self._inference_result(p.index)
                pred = res.get('pred_class') if res else None
                colors.append(get_location_color(pred).name()
                              if pred is not None else '#d0d0d0')
            return colors, None
        return ['tab:blue' for _ in poses], None

    # --- 位置推論結果の重ね描き ----------------------------------------------

    def _inference_result(self, index):
        """フレーム index の位置推論結果（無ければ None）"""
        if self.inference_provider is None:
            return None
        try:
            return self.inference_provider(index)
        except Exception:
            return None

    def _inference_settings(self):
        """位置推論の表示設定（無ければ既定値）"""
        if self.inference_settings_provider is not None:
            try:
                s = self.inference_settings_provider() or {}
                return {'top_n': int(s.get('top_n', 3) or 3),
                        'grid_mode': s.get('grid_mode', 'weighted') or 'weighted',
                        'grid_config': s.get('grid_config')}
            except Exception:
                pass
        return {'top_n': 3, 'grid_mode': 'weighted', 'grid_config': None}

    def _predicted_pose(self, index):
        """フレーム index の推定座標 (x, y, theta|None)

        座標・姿勢回帰の出力があればそれを、無ければ格子分類の Top1 セル中心または
        Top1〜N の重み付き平均（表示設定に従う）を返す。どちらも無ければ None。
        """
        res = self._inference_result(index)
        if not res:
            return None
        pose = res.get('pose')
        if pose and pose.get('x') is not None and pose.get('y') is not None:
            return float(pose['x']), float(pose['y']), pose.get('theta')
        grid = res.get('grid')
        if grid and grid.get('top'):
            s = self._inference_settings()
            if s['grid_mode'] == 'top1':
                t1 = grid['top'][0]
                return float(t1['x']), float(t1['y']), None
            from model_catalog import grid_weighted_position
            wxy = grid_weighted_position(grid['top'], n=s['top_n'])
            if wxy is not None:
                return float(wxy[0]), float(wxy[1]), None
        return None

    def _draw_grid_cells(self, index):
        """格子分類の Top-N セルを現在フレームに重ねる（確率が高いほど濃い紫。Top1 は太枠）"""
        res = self._inference_result(index)
        grid = res.get('grid') if res else None
        if not grid or not grid.get('top'):
            return
        s = self._inference_settings()
        cfg = s['grid_config'] or {}
        cell = float(cfg.get('cell_size') or 0)
        if cell <= 0:
            return
        top = grid['top'][:s['top_n']]
        p_max = max(it['prob'] for it in top) or 1.0
        for rank, it in enumerate(top):
            x0, y0 = it['x'] - cell / 2.0, it['y'] - cell / 2.0
            alpha = 0.12 + 0.45 * (it['prob'] / p_max)
            rect = Rectangle((x0, y0), cell, cell, facecolor=self.PRED_COLOR, alpha=alpha,
                             edgecolor=self.PRED_EDGE_COLOR if rank == 0 else 'none',
                             linewidth=1.4 if rank == 0 else 0.0, zorder=3.5)
            rect.set_animated(True)
            self.ax.add_patch(rect)
            self._pred_artists.append(rect)

    def _draw_grid_lines(self):
        """格子分類モデルの格子線を薄く描く（全体再描画時）"""
        if not self.show_inference_checkbox.isChecked():
            return
        cfg = self._inference_settings()['grid_config'] or {}
        cell = float(cfg.get('cell_size') or 0)
        nx, ny = int(cfg.get('nx') or 0), int(cfg.get('ny') or 0)
        if cell <= 0 or nx <= 0 or ny <= 0 or nx * ny > 40000:
            return
        x0, y0 = float(cfg['x_min']), float(cfg['y_min'])
        x1, y1 = x0 + nx * cell, y0 + ny * cell
        for i in range(nx + 1):
            self.ax.plot([x0 + i * cell] * 2, [y0, y1], color=self.PRED_COLOR,
                         linewidth=0.4, alpha=0.18, zorder=1.2)
        for j in range(ny + 1):
            self.ax.plot([x0, x1], [y0 + j * cell] * 2, color=self.PRED_COLOR,
                         linewidth=0.4, alpha=0.18, zorder=1.2)

    def _on_inference_toggle(self, _state=None):
        # refresh が現在フレームのマーカー（赤三角・推定位置の紫三角）も作り直す
        self.refresh()

    def _clear_pred_artists(self):
        for artist in self._pred_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._pred_artists = []

    def _draw_pred_marker(self, index, actual_pose):
        """現在フレームの推定位置を青い三角で描き、実測位置との誤差を点線で結ぶ"""
        self._clear_pred_artists()
        if not self.show_inference_checkbox.isChecked():
            return
        # 格子分類なら Top-N セルを先に敷く（三角はその上）
        self._draw_grid_cells(index)
        pred = self._predicted_pose(index)
        if pred is None:
            return
        px, py, ptheta = pred
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), 1e-6)
        radius = span * 0.02
        theta = ptheta if ptheta is not None else (getattr(actual_pose, 'theta', 0.0) or 0.0)
        marker = RegularPolygon(
            (px, py), numVertices=3, radius=radius,
            orientation=theta - math.pi / 2.0,
            facecolor=self.PRED_COLOR, edgecolor=self.PRED_EDGE_COLOR,
            linewidth=1.0, alpha=0.9, zorder=5)
        marker.set_animated(True)
        self.ax.add_patch(marker)
        self._pred_artists.append(marker)
        if actual_pose is not None:
            link = Line2D([actual_pose.x, px], [actual_pose.y, py], color=self.PRED_COLOR,
                          linestyle='--', linewidth=1.0, alpha=0.8, zorder=4.5)
            link.set_animated(True)
            self.ax.add_line(link)
            self._pred_artists.append(link)

    def _ensure_pred_scatter(self):
        """推定座標の丸（animated scatter）を必要なら作成して返す"""
        if self._pred_scatter is None:
            self._pred_scatter = self.ax.scatter(
                [], [], marker='o', facecolors='none', edgecolors=self.PRED_COLOR,
                s=22, linewidths=1.0, alpha=0.9, zorder=3)
            self._pred_scatter.set_animated(True)
        return self._pred_scatter

    def _append_pred_point(self, index):
        """フレーム index の推定座標を丸の scatter へ追記する（表示中ラップ内のみ、未追記なら）"""
        if not self.show_inference_checkbox.isChecked():
            return
        if index in self._pred_scatter_indexes or index not in self._plotted_indexes:
            return
        pred = self._predicted_pose(index)
        if pred is None:
            return
        self._pred_scatter_indexes.add(index)
        self._pred_scatter_points.append((pred[0], pred[1]))
        self._ensure_pred_scatter().set_offsets(self._pred_scatter_points)

    def _draw_pred_trajectory(self, poses, legend_handles):
        """推論済みフレームの推定座標を中空丸で重ね描きし、凡例に件数と平均誤差を出す

        丸は animated な scatter に載せ、以後 highlight_frame で推論済みフレームが
        増えるたびに追記する（再生中も全体再描画なしで丸が増える）。
        """
        if not self.show_inference_checkbox.isChecked() or self.inference_provider is None:
            return
        errs = []
        for p in poses:
            pred = self._predicted_pose(p.index)
            if pred is None:
                continue
            self._pred_scatter_indexes.add(p.index)
            self._pred_scatter_points.append((pred[0], pred[1]))
            errs.append(math.hypot(pred[0] - p.x, pred[1] - p.y))
        if not errs:
            return
        self._ensure_pred_scatter().set_offsets(self._pred_scatter_points)
        mean_err = sum(errs) / len(errs)
        legend_handles.append(
            Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor=self.PRED_COLOR,
                   color='none', linestyle='None', markersize=7,
                   label=get_text('map_view_legend_inference', len(errs), f"{mean_err:.2f}")))

    def _on_pick(self, event):
        if not self._plotted_indexes or not event.ind.size:
            return
        picked = int(event.ind[0])
        if picked >= len(self._plotted_indexes):
            return
        # 領域編集モード中はフレームジャンプしない（クリックは button_press_event
        # 側でポリゴン頂点の追加として処理される）
        if self.region_edit_mode:
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
        layout.setSpacing(2)
        # マップ⇔操作パネルは縦スプリッタ。余白は**マップ側だけ**が吸収する
        # ため、下側パネルの行間が間延びしない（従来は余った縦スペースが
        # 全行へ分配されて行間が開いていた）。境界ドラッグで高さ配分を調整可。
        splitter = QSplitter(Qt.Vertical)
        self.map_widget = MapViewWidget(on_frame_selected=self._on_frame_selected)
        splitter.addWidget(self.map_widget)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(2)         # 下側パネル各行の行間をコンパクトに
        self._build_edit_panel(panel_layout)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)    # 余白はマップが取る
        splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(0, False)
        layout.addWidget(splitter)

        # 領域の追加・削除・読込をステータス行へ反映する
        self.map_widget.on_regions_changed = self._update_region_status

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
        writeback_row.addStretch()
        parent_layout.addLayout(writeback_row)

        # ボタンは 2 行目へ（コンパクト幅でも文言が見切れないように改行）
        writeback_btn_row = QHBoxLayout()
        self.writeback_button = QPushButton(get_text('map_view_writeback_btn'))
        self.writeback_button.setToolTip(get_text('map_view_writeback_tooltip'))
        self.writeback_button.clicked.connect(self._on_writeback_clicked)
        writeback_btn_row.addWidget(self.writeback_button)
        # ②: 相手車矩形（opponent）→ togivad/agents 書き戻し
        self.agent_writeback_button = QPushButton(
            get_text('map_view_agent_writeback_btn'))
        self.agent_writeback_button.setToolTip(
            get_text('map_view_agent_writeback_tooltip'))
        self.agent_writeback_button.clicked.connect(
            self._on_agent_writeback_clicked)
        writeback_btn_row.addWidget(self.agent_writeback_button)
        writeback_btn_row.addStretch()
        parent_layout.addLayout(writeback_btn_row)

        self.writeback_hint_label = QLabel(get_text('map_view_writeback_hint'))
        self.writeback_hint_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.writeback_hint_label.setWordWrap(True)
        parent_layout.addWidget(self.writeback_hint_label)

        # --- 位置領域（軌跡区間指定）＋ 位置自動アノテーション（Phase 1） ---
        region_sep = QFrame()
        region_sep.setFrameShape(QFrame.HLine)
        parent_layout.addWidget(region_sep)

        region_row = QHBoxLayout()
        region_row.addWidget(QLabel(get_text('map_view_region_label')))
        self.region_edit_button = QPushButton(get_text('map_view_region_edit_btn'))
        self.region_edit_button.setCheckable(True)
        self.region_edit_button.setToolTip(get_text('map_view_region_edit_tooltip'))
        self.region_edit_button.toggled.connect(self._on_region_edit_toggled)
        region_row.addWidget(self.region_edit_button)
        region_row.addWidget(QLabel(get_text('map_view_region_class_label')))
        self.region_class_spin = QSpinBox()
        self.region_class_spin.setRange(0, 99)
        self.region_class_spin.valueChanged.connect(self._on_region_class_changed)
        region_row.addWidget(self.region_class_spin)
        self.region_undo_button = QPushButton(get_text('map_view_region_undo_btn'))
        self.region_undo_button.clicked.connect(self._on_region_undo_clicked)
        region_row.addWidget(self.region_undo_button)
        self.region_clear_button = QPushButton(get_text('map_view_region_clear_btn'))
        self.region_clear_button.clicked.connect(self._on_region_clear_clicked)
        region_row.addWidget(self.region_clear_button)
        self.region_save_button = QPushButton(get_text('map_view_region_save_btn'))
        self.region_save_button.setToolTip(get_text('map_view_region_save_tooltip'))
        self.region_save_button.clicked.connect(self._on_region_save_clicked)
        region_row.addWidget(self.region_save_button)
        region_row.addStretch()
        parent_layout.addLayout(region_row)

        autoloc_row = QHBoxLayout()
        self.autoloc_button = QPushButton(get_text('map_view_autoloc_btn'))
        self.autoloc_button.setToolTip(get_text('map_view_autoloc_tooltip'))
        self.autoloc_button.clicked.connect(self._on_autoloc_clicked)
        autoloc_row.addWidget(self.autoloc_button)
        self.autoloc_keep_manual_checkbox = QCheckBox(
            get_text('map_view_autoloc_keep_manual'))
        self.autoloc_keep_manual_checkbox.setChecked(True)
        self.autoloc_keep_manual_checkbox.setToolTip(
            get_text('map_view_autoloc_keep_manual_tooltip'))
        autoloc_row.addWidget(self.autoloc_keep_manual_checkbox)
        autoloc_row.addStretch()
        parent_layout.addLayout(autoloc_row)

        self.region_status_label = QLabel(get_text('map_view_region_hint'))
        self.region_status_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.region_status_label.setWordWrap(True)
        parent_layout.addWidget(self.region_status_label)

    def _on_frame_selected(self, index):
        self.jump_to_image.emit(index)

    def set_pose_manager(self, pose_manager) -> None:
        self.map_widget.jump_threshold = self.quality_threshold_spin.value()
        # 色分け「位置」用: メインウィンドウの位置アノテーションを参照する
        # （dict自体は読み直し時に差し替わるため、呼び出し時に属性を引き直す）
        if self.main_window is not None:
            self.map_widget.loc_provider = (
                lambda idx: getattr(self.main_window, 'location_annotations',
                                    {}).get(idx))
            # 位置推論結果（推定座標・予測クラス）の参照。メイン側で更新される dict を
            # 呼び出し時に引き直す
            self.map_widget.inference_provider = (
                lambda idx: getattr(self.main_window, 'location_inference_results',
                                    {}).get(idx))
            # 表示設定（Top-N / Top1・重み付き / 格子定義）。メイン側のメソッドがあれば参照
            settings_fn = getattr(self.main_window, 'location_inference_settings', None)
            self.map_widget.inference_settings_provider = settings_fn if callable(settings_fn) else None
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

    def handle_delete_key(self) -> None:
        """メインウィンドウの eventFilter から委譲される Delete/Backspace 処理。

        マップビュー上にマウスがある間の削除キーは、メイン画面のアノテーション
        （運転・bbox・セグ等）ではなく、こちらの位置領域の削除として扱う。
        """
        self.map_widget.delete_by_key()

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

    # --- 位置領域（軌跡区間指定）＋ 位置自動アノテーション -----------------------

    def _update_region_status(self, hint=None):
        """領域数＋操作ヒントをステータス行へ表示する（widget からのコールバック）"""
        parts = [get_text('map_view_region_count',
                          len(self.map_widget.location_regions))]
        if hint:
            parts.append(hint)
        self.region_status_label.setText("　".join(parts))

    def _on_region_edit_toggled(self, checked):
        self.map_widget.set_region_edit_mode(
            checked, self.region_class_spin.value())

    def _on_region_class_changed(self, value):
        self.map_widget.region_edit_class = int(value)
        if self.map_widget.region_edit_mode:
            # 作成中ポリゴンの色・確定時のクラスが変わるため再描画してヒント更新
            self.map_widget.refresh()
            self.map_widget._notify_draft_hint()

    def _on_region_undo_clicked(self):
        self.map_widget.undo_last_region()

    def _on_region_clear_clicked(self):
        self.map_widget.clear_regions()

    def _on_region_save_clicked(self):
        if not self.map_widget.location_regions:
            QMessageBox.information(
                self, get_text('map_view_autoloc_confirm_title'),
                get_text('map_view_region_none'))
            return
        try:
            path = self.map_widget.save_regions()
        except Exception as e:
            QMessageBox.warning(
                self, get_text('map_view_autoloc_confirm_title'),
                get_text('map_view_region_save_error', str(e)))
            return
        self._update_region_status(get_text('map_view_region_saved', path))

    def _on_autoloc_clicked(self):
        """位置領域（ポリゴン）× pose の点内包判定で位置ラベルを自動付与（main側へ委譲）。"""
        if self.main_window is None:
            return
        regions = self.map_widget.location_regions
        if not regions:
            QMessageBox.information(
                self, get_text('map_view_autoloc_confirm_title'),
                get_text('map_view_region_none'))
            return
        self.autoloc_button.setEnabled(False)
        try:
            # parent_widget=self: このダイアログは常に最前面のため、メインウィンドウ親の
            # モーダルが裏に隠れて操作不能に見える問題を避ける
            self.main_window.auto_annotate_locations_from_regions(
                regions,
                keep_manual=self.autoloc_keep_manual_checkbox.isChecked(),
                prefer_source=self.map_widget.source_combo.currentData(),
                parent_widget=self)
        finally:
            self.autoloc_button.setEnabled(True)
        self.map_widget.refresh()   # 色分け「位置」表示を最新のラベルで更新
