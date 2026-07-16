# data_analysis.py
"""データ分析ダイアログ - アノテーションデータの統計分析と可視化"""

import numpy as np
from translations import get_text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QGroupBox, QGridLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea, QCheckBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QSpinBox,
    QButtonGroup, QRadioButton, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor


class CollapsibleSection(QWidget):
    """折り畳み・展開できるセクションウィジェット"""

    def __init__(self, title, collapsed=False, parent=None):
        super().__init__(parent)
        self._title = title
        self._collapsed = collapsed

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(0)

        # ─── ヘッダーボタン ───
        self.toggle_btn = QPushButton()
        self.toggle_btn.setCheckable(False)
        self.toggle_btn.clicked.connect(self._toggle)
        self.toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px 10px;
                font-weight: bold;
                font-size: 11px;
                background-color: #3a3a3a;
                color: #d0d0d0;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QPushButton:hover  { background-color: #484848; }
            QPushButton:pressed { background-color: #282828; }
        """)
        outer.addWidget(self.toggle_btn)

        # ─── コンテンツエリア ───
        self.content_widget = QWidget()
        self.content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer.addWidget(self.content_widget)

        self._update_label()
        if collapsed:
            self.content_widget.setVisible(False)

    def _update_label(self):
        arrow = '▶' if self._collapsed else '▼'
        self.toggle_btn.setText(f'  {arrow}  {self._title}')

    def _toggle(self):
        self._collapsed = not self._collapsed
        self.content_widget.setVisible(not self._collapsed)
        self._update_label()

    def set_content_layout(self, layout):
        self.content_widget.setLayout(layout)

# 日本語フォントの設定
# 注意: "MS Gothic" はこの環境のmatplotlib(FreeType)でテキストが完全に不可視になる
# 既知の不具合があるため使用しない（Yu Gothic/Meiryoは正常に描画される）。
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 色のパレット
COLORS = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]


class DataAnalysisDialog(QDialog):
    """データ分析ダイアログ"""

    # 画像ジャンプシグナル（インデックスを送信）
    jump_to_image = pyqtSignal(int)

    def __init__(self, parent=None, annotations=None, images=None, deleted_indexes=None,
                 downsampled_indexes=None, available_sensor_keys=None):
        super().__init__(parent)
        self.parent_window = parent
        self.annotations = annotations or {}
        self.images = images or []
        self.deleted_indexes = deleted_indexes or []
        self.downsampled_indexes = downsampled_indexes or []
        self.available_sensor_keys = available_sensor_keys or set()
        self.current_index = parent.current_index if parent else 0

        # 基本キー + センサーキー
        self.base_keys = ['angle', 'throttle', 'speed']
        self.all_keys = self.base_keys + sorted(list(self.available_sensor_keys))

        self.setWindowTitle(get_text('dlg_data_analysis'))
        self.setMinimumSize(600, 300)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.init_ui()
        self.update_analysis()

    def init_ui(self):
        """UIの初期化"""
        main_layout = QVBoxLayout(self)

        # スクロールエリア
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)

        # === 統計・分布セクション（横並び） ===
        self._stats_section = CollapsibleSection(get_text('section_stats_distribution'))
        stats_dist_layout = QHBoxLayout()
        stats_dist_layout.setContentsMargins(4, 4, 4, 4)

        # 左側: 統計量テーブル
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(6)
        self.stats_table.setHorizontalHeaderLabels([
            get_text('label_stats_item'), get_text('label_stats_mean'),
            get_text('label_stats_std'), get_text('label_stats_min'),
            get_text('label_stats_max'), get_text('label_stats_median')
        ])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setStyleSheet("""
            QTableWidget {
                font-size: 10px;
                gridline-color: #555;
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QTableWidget::item {
                padding: 1px 2px;
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #3a6ea5;
                color: white;
                font-weight: bold;
                font-size: 9px;
                padding: 2px;
                border: none;
                min-height: 18px;
                max-height: 18px;
            }
            QTableWidget::item:alternate {
                background-color: #353535;
            }
        """)
        self.stats_table.verticalHeader().setVisible(False)
        # 行の高さを狭める
        self.stats_table.verticalHeader().setDefaultSectionSize(20)
        self.stats_table.horizontalHeader().setFixedHeight(22)
        self.stats_table.setMinimumHeight(60)
        self.stats_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stats_dist_layout.addWidget(self.stats_table, 1)

        # 右側: 分布グラフ（AngleとThrottleを1つのグラフに統合）
        self.dist_figure = Figure(figsize=(3, 1.8), dpi=80)
        self.dist_canvas = FigureCanvas(self.dist_figure)
        self.dist_canvas.setMinimumHeight(80)
        self.dist_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stats_dist_layout.addWidget(self.dist_canvas, 1)

        self._stats_section.set_content_layout(stats_dist_layout)
        layout.addWidget(self._stats_section)

        # === 時系列セクション ===
        self._timeseries_section = CollapsibleSection(get_text('section_timeseries'))
        timeseries_layout = QVBoxLayout()
        timeseries_layout.setContentsMargins(4, 4, 4, 4)

        # ── 処理パイプライン（チェックボックス、上から順番に適用）──
        pipeline_label = QLabel("処理ステップ（上から順に適用）:")
        pipeline_label.setStyleSheet("font-size: 10px; color: #aaa;")
        timeseries_layout.addWidget(pipeline_label)

        # Step 1: 移動平均
        ma_row = QHBoxLayout()
        ma_row.setSpacing(4)
        self.ma_check = QCheckBox(get_text('label_moving_avg'))
        self.ma_check.setChecked(False)
        self.ma_check.toggled.connect(self._on_pipeline_changed)
        ma_row.addWidget(self.ma_check)
        ma_row.addWidget(QLabel(get_text('label_window')))
        self.moving_avg_spin = QSpinBox()
        self.moving_avg_spin.setRange(2, 300)
        self.moving_avg_spin.setValue(50)
        self.moving_avg_spin.setSingleStep(1)
        self.moving_avg_spin.setEnabled(False)
        self.moving_avg_spin.valueChanged.connect(self.update_timeseries_graph)
        ma_row.addWidget(self.moving_avg_spin)
        ma_row.addStretch()
        timeseries_layout.addLayout(ma_row)

        # Step 2: 正規化
        norm_row = QHBoxLayout()
        norm_row.setSpacing(4)
        self.norm_check = QCheckBox(get_text('label_normalized'))
        self.norm_check.setChecked(False)
        self.norm_check.toggled.connect(self._on_pipeline_changed)
        norm_row.addWidget(self.norm_check)
        norm_row.addStretch()
        timeseries_layout.addLayout(norm_row)

        # Step 3: 区間平均（バー表示）
        bin_row = QHBoxLayout()
        bin_row.setSpacing(4)
        self.bin_check = QCheckBox(get_text('label_bin_avg'))
        self.bin_check.setChecked(False)
        self.bin_check.toggled.connect(self._on_pipeline_changed)
        bin_row.addWidget(self.bin_check)
        bin_row.addWidget(QLabel(get_text('label_bin')))
        self.bin_size_spin = QSpinBox()
        self.bin_size_spin.setRange(10, 1000)
        self.bin_size_spin.setValue(200)
        self.bin_size_spin.setSingleStep(10)
        self.bin_size_spin.setSuffix(" idx")
        self.bin_size_spin.setEnabled(False)
        self.bin_size_spin.valueChanged.connect(self.update_timeseries_graph)
        bin_row.addWidget(self.bin_size_spin)
        bin_row.addStretch()
        timeseries_layout.addLayout(bin_row)

        # 2行目: 表示範囲設定
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel(get_text('label_display_range')))

        self.idx_min_spin = QSpinBox()
        self.idx_min_spin.setRange(0, 100000)
        self.idx_min_spin.setValue(0)
        self.idx_min_spin.setSingleStep(100)
        self.idx_min_spin.valueChanged.connect(self._on_range_changed)
        range_layout.addWidget(self.idx_min_spin)

        range_layout.addWidget(QLabel("〜"))

        self.idx_max_spin = QSpinBox()
        self.idx_max_spin.setRange(0, 100000)
        self.idx_max_spin.setValue(0)
        self.idx_max_spin.setSingleStep(100)
        self.idx_max_spin.setSpecialValueText(get_text('label_auto'))
        self.idx_max_spin.valueChanged.connect(self._on_range_changed)
        range_layout.addWidget(self.idx_max_spin)

        range_layout.addStretch()
        timeseries_layout.addLayout(range_layout)

        # 中央: グラフと表示項目リスト（横並び）
        graph_layout = QHBoxLayout()

        # 時系列グラフ（左側）
        self.timeseries_figure = Figure(figsize=(5, 2.5), dpi=80)
        self.timeseries_canvas = FigureCanvas(self.timeseries_figure)
        self.timeseries_canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.timeseries_canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.timeseries_canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.timeseries_canvas.mpl_connect('scroll_event', self._on_scroll)
        self.timeseries_canvas.setMinimumHeight(100)
        self.timeseries_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._drag_start = None  # ドラッグ開始位置 (x_data, y_data)
        self._is_dragging = False
        self._zoom_xlim = None  # ズーム後のX範囲（Noneで自動）
        self._zoom_ylim = None  # ズーム後のY範囲（Noneで自動）
        graph_layout.addWidget(self.timeseries_canvas, 4)

        # キー選択リスト（右側）
        key_select_layout = QVBoxLayout()
        key_select_layout.addWidget(QLabel(get_text('label_display_items')))
        self.key_list = QListWidget()
        self.key_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.key_list.setMaximumWidth(180)
        self.key_list.setMinimumWidth(140)
        self.key_list.setMinimumHeight(80)

        # 基本キーを追加
        for key in self.all_keys:
            item = QListWidgetItem(key)
            self.key_list.addItem(item)
            # デフォルトでangle, throttleを選択
            if key in ['angle', 'throttle']:
                item.setSelected(True)

        self.key_list.itemSelectionChanged.connect(self.update_timeseries_graph)
        key_select_layout.addWidget(self.key_list)
        graph_layout.addLayout(key_select_layout, 1)

        timeseries_layout.addLayout(graph_layout)

        # 説明ラベルとズームリセットボタン
        info_bar = QHBoxLayout()
        info_label = QLabel(get_text('label_click_to_jump') + "  |  " + get_text('label_zoom_hint'))
        info_label.setStyleSheet("color: gray;")
        info_bar.addWidget(info_label)
        info_bar.addStretch()
        self.zoom_reset_btn = QPushButton(get_text('btn_zoom_reset'))
        self.zoom_reset_btn.setMinimumWidth(120)
        self.zoom_reset_btn.setEnabled(False)
        self.zoom_reset_btn.clicked.connect(self._reset_zoom)
        info_bar.addWidget(self.zoom_reset_btn)
        timeseries_layout.addLayout(info_bar)

        self._timeseries_section.set_content_layout(timeseries_layout)
        layout.addWidget(self._timeseries_section)

        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        # 下部ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.close_button = QPushButton(get_text('btn_close'))
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

    def get_selected_keys(self):
        """選択されたキーのリストを取得"""
        return [item.text() for item in self.key_list.selectedItems()]

    def update_analysis(self):
        """全分析を更新"""
        # 親ウィンドウから最新データを取得
        if self.parent_window:
            self.annotations = self.parent_window.annotations
            self.images = self.parent_window.images
            self.deleted_indexes = getattr(self.parent_window, 'deleted_indexes', [])
            self.downsampled_indexes = getattr(self.parent_window, 'downsampled_indexes', [])
            self.current_index = self.parent_window.current_index

            # センサーキーも更新
            new_keys = getattr(self.parent_window, 'available_sensor_keys', set())
            if new_keys != self.available_sensor_keys:
                self.available_sensor_keys = new_keys
                self.all_keys = self.base_keys + sorted(list(self.available_sensor_keys))
                # キーリストを更新
                self.key_list.clear()
                for key in self.all_keys:
                    item = QListWidgetItem(key)
                    self.key_list.addItem(item)
                    if key in ['angle', 'throttle']:
                        item.setSelected(True)

        self.update_stats()
        self.update_distribution_graph()
        self.update_timeseries_graph()

    def update_stats(self):
        """統計情報を更新"""
        # 全キーのデータを収集
        key_data = {key: [] for key in self.all_keys}

        for idx, ann in self.annotations.items():
            if idx in self.deleted_indexes:
                continue
            if idx in self.downsampled_indexes:
                continue
            for key in self.all_keys:
                if key in ann and ann[key] is not None:
                    key_data[key].append(ann[key])

        # テーブル更新
        self.stats_table.setRowCount(len(self.all_keys))

        # 重要なキー（ハイライト対象）
        important_keys = ['angle', 'throttle']

        for row, key in enumerate(self.all_keys):
            data = key_data[key]
            is_important = key in important_keys

            # 項目名
            key_item = QTableWidgetItem(key)
            key_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            if is_important:
                font = key_item.font()
                font.setBold(True)
                key_item.setFont(font)
                key_item.setBackground(QColor('#e8f4fd'))
            self.stats_table.setItem(row, 0, key_item)

            if data:
                arr = np.array(data)
                values = [
                    f"{np.mean(arr):.2f}",
                    f"{np.std(arr):.2f}",
                    f"{np.min(arr):.2f}",
                    f"{np.max(arr):.2f}",
                    f"{np.median(arr):.2f}"
                ]
                for col, val in enumerate(values, 1):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if is_important:
                        item.setBackground(QColor('#e8f4fd'))
                    self.stats_table.setItem(row, col, item)
            else:
                for col in range(1, 6):
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setForeground(QColor('#999999'))
                    self.stats_table.setItem(row, col, item)

    def update_distribution_graph(self):
        """分布グラフを更新（元データとダウンサンプリング後を重ねて表示）"""
        self.dist_figure.clear()

        # 元データ収集（削除済みのみ除外）
        angles_orig = []
        throttles_orig = []
        # ダウンサンプリング後のデータ収集（削除済み+ダウンサンプリング除外）
        angles_ds = []
        throttles_ds = []

        for idx, ann in self.annotations.items():
            if idx in self.deleted_indexes:
                continue
            # 元データに追加
            if 'angle' in ann:
                angles_orig.append(ann['angle'])
            if 'throttle' in ann:
                throttles_orig.append(ann['throttle'])
            # ダウンサンプリング後のデータに追加
            if idx not in self.downsampled_indexes:
                if 'angle' in ann:
                    angles_ds.append(ann['angle'])
                if 'throttle' in ann:
                    throttles_ds.append(ann['throttle'])

        ax = self.dist_figure.add_subplot(111)

        if angles_orig or throttles_orig:
            # 共通のビン範囲を計算
            all_values = angles_orig + throttles_orig
            bin_min = min(all_values) if all_values else -1
            bin_max = max(all_values) if all_values else 1
            bins = np.linspace(bin_min, bin_max, 51)

            # 元データを薄い色で表示（背景）
            if angles_orig:
                ax.hist(angles_orig, bins=bins, color='steelblue', edgecolor='none',
                        alpha=0.25, label=get_text('label_angle_original', len(angles_orig)))
            if throttles_orig:
                ax.hist(throttles_orig, bins=bins, color='forestgreen', edgecolor='none',
                        alpha=0.25, label=get_text('label_throttle_original', len(throttles_orig)))

            # ダウンサンプリング後のデータを濃い色で表示（前景）
            if angles_ds:
                ax.hist(angles_ds, bins=bins, color='steelblue', edgecolor='white',
                        alpha=0.8, label=get_text('label_angle_ds', len(angles_ds)))
            if throttles_ds:
                ax.hist(throttles_ds, bins=bins, color='forestgreen', edgecolor='white',
                        alpha=0.8, label=get_text('label_throttle_ds', len(throttles_ds)))

            ds_count = len(self.downsampled_indexes)
            data_count = max(len(angles_ds), len(throttles_ds))
            if ds_count > 0:
                ax.set_title(get_text('label_dist_title_with_ds', f'{data_count:,}', ds_count))
            else:
                ax.set_title(get_text('label_dist_title', f'{data_count:,}'))
            ax.set_xlabel(get_text('label_value'))
            ax.set_ylabel(get_text('label_frequency'))
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, get_text('label_no_data'), ha='center', va='center', fontsize=12)
            ax.set_title(get_text('label_dist_title', '0'))

        self.dist_figure.tight_layout()
        self.dist_canvas.draw()

    def update_timeseries_graph(self):
        """時系列グラフを更新"""
        self.timeseries_figure.clear()

        selected_keys = self.get_selected_keys()
        if not selected_keys:
            ax = self.timeseries_figure.add_subplot(111)
            ax.text(0.5, 0.5, get_text('label_select_display_item'), ha='center', va='center', fontsize=14)
            self.timeseries_canvas.draw()
            return

        # データ収集（インデックス順にソート）
        data_by_key = {key: {} for key in selected_keys}
        indices_set = set()

        for idx, ann in self.annotations.items():
            if idx in self.deleted_indexes:
                continue
            if idx in self.downsampled_indexes:
                continue
            indices_set.add(idx)
            for key in selected_keys:
                if key in ann and ann[key] is not None:
                    data_by_key[key][idx] = ann[key]

        if not indices_set:
            ax = self.timeseries_figure.add_subplot(111)
            ax.text(0.5, 0.5, get_text('label_no_data_available'), ha='center', va='center', fontsize=14)
            self.timeseries_canvas.draw()
            return

        indices = sorted(list(indices_set))
        bin_size = self.bin_size_spin.value()
        window_size = self.moving_avg_spin.value()

        ax = self.timeseries_figure.add_subplot(111)

        use_ma   = self.ma_check.isChecked()
        use_norm = self.norm_check.isChecked()
        use_bin  = self.bin_check.isChecked()

        # 処理パイプラインを適用してグラフ描画
        if use_bin:
            # ── 区間平均（バー表示）──
            # Step1: 移動平均 → Step2: 正規化 → Step3: 区間平均
            min_idx = min(indices)
            max_idx = max(indices)
            bin_starts = list(range(min_idx, max_idx + 1, bin_size))
            num_keys = len(selected_keys)
            bar_width = bin_size * 0.8 / num_keys

            title_steps = []
            if use_ma:   title_steps.append(f"MA{window_size}")
            if use_norm: title_steps.append("正規化")
            title_steps.append(f"区間平均(bin={bin_size})")

            for key_idx, key in enumerate(selected_keys):
                key_indices = sorted(data_by_key[key].keys())
                key_values  = np.array([data_by_key[key][i] for i in key_indices], dtype=float)

                if len(key_values) == 0:
                    continue

                # Step1: 移動平均
                if use_ma and len(key_values) >= window_size:
                    kernel = np.ones(window_size) / window_size
                    ma = np.convolve(key_values, kernel, mode='valid')
                    offset = window_size // 2
                    key_indices = key_indices[offset: offset + len(ma)]
                    key_values  = ma

                # Step2: 正規化
                if use_norm:
                    abs_max = np.max(np.abs(key_values))
                    if abs_max > 0:
                        key_values = key_values / abs_max

                # Step3: 区間平均（バー）
                ki_arr = np.array(key_indices)
                means, bin_centers = [], []
                for bin_start in bin_starts:
                    bin_end    = bin_start + bin_size
                    bin_center = bin_start + bin_size / 2
                    bin_centers.append(bin_center)
                    mask = (ki_arr >= bin_start) & (ki_arr < bin_end)
                    vals = key_values[mask]
                    means.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)

                bar_offset = (key_idx - num_keys / 2 + 0.5) * bar_width
                color = COLORS[key_idx % len(COLORS)]
                ax.bar([c + bar_offset for c in bin_centers], means,
                       width=bar_width, color=color, alpha=0.7,
                       label=key, edgecolor='white')

            ax.set_title(" → ".join(title_steps))
            ax.axvline(x=self.current_index, color='purple', linestyle='--', alpha=0.8, linewidth=2)

        else:
            # ── 折れ線表示（Step1: MA → Step2: 正規化）──
            title_steps = []
            if use_ma:   title_steps.append(f"MA{window_size}")
            if use_norm: title_steps.append("正規化")
            if not title_steps: title_steps.append(get_text('label_raw_data'))

            for key_idx, key in enumerate(selected_keys):
                key_indices = sorted(data_by_key[key].keys())
                key_values  = np.array([data_by_key[key][i] for i in key_indices], dtype=float)

                if len(key_values) == 0:
                    continue

                color = COLORS[key_idx % len(COLORS)]
                raw_indices = key_indices[:]
                raw_values  = key_values.copy()  # MA前の元データ（背景表示用）
                applied_ma  = False

                # Step1: 移動平均
                if use_ma and len(key_values) >= window_size:
                    kernel = np.ones(window_size) / window_size
                    ma = np.convolve(key_values, kernel, mode='valid')
                    offset = window_size // 2
                    key_indices = key_indices[offset: offset + len(ma)]
                    key_values  = ma
                    applied_ma  = True

                # Step2: 正規化 — abs_max は MA 後の値から算出
                abs_max_label = ""
                norm_factor   = None
                if use_norm:
                    abs_max = np.max(np.abs(key_values))
                    if abs_max > 0:
                        norm_factor    = abs_max
                        key_values     = key_values / abs_max
                        abs_max_label  = f" (÷{abs_max:.3f})"

                # 元データの背景線: MA 有効時のみ、正規化も同係数で揃える
                if applied_ma:
                    bg = raw_values.copy()
                    if norm_factor is not None:
                        bg = bg / norm_factor  # 正規化と同じスケールに合わせる
                    ax.plot(raw_indices, bg, '-', color=color, alpha=0.15, linewidth=0.5)

                label = key + (f"(MA{window_size})" if applied_ma else "") + abs_max_label
                ax.plot(key_indices, key_values, '-', color=color, alpha=0.85,
                        label=label, linewidth=1.0)

            ax.set_title(" → ".join(title_steps))
            ax.axvline(x=self.current_index, color='purple', linestyle='--', alpha=0.8, linewidth=2)
            if self.current_index in indices and selected_keys:
                if self.current_index in data_by_key[selected_keys[0]]:
                    raw_y = data_by_key[selected_keys[0]][self.current_index]
                    # 正規化有効時は現在位置マーカーも正規化スケールで表示
                    if use_norm:
                        key0_vals = list(data_by_key[selected_keys[0]].values())
                        amax = max(abs(v) for v in key0_vals) if key0_vals else 1.0
                        raw_y = raw_y / amax if amax > 0 else raw_y
                    ax.scatter([self.current_index], [raw_y], color='purple', s=80, zorder=5, marker='o')

        ax.set_xlabel(get_text('label_index'))
        ax.set_ylabel(get_text('label_value'))
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

        # X軸の範囲を設定
        if self._zoom_xlim is not None:
            ax.set_xlim(self._zoom_xlim)
        elif indices:
            idx_min_setting = self.idx_min_spin.value()
            idx_max_setting = self.idx_max_spin.value()
            if idx_max_setting == 0:
                x_min = min(indices) - 10
                x_max = max(indices) + 10
            else:
                x_min = idx_min_setting
                x_max = idx_max_setting
            ax.set_xlim(x_min, x_max)

        # Y軸の範囲を設定（優先順位: ズーム > 正規化固定 > 自動）
        if self._zoom_ylim is not None:
            ax.set_ylim(self._zoom_ylim)
        elif use_norm and not use_bin:
            # 正規化時は [-1, 1] 基準に固定（自動スケールを上書き）
            ax.set_ylim(-1.1, 1.1)
            ax.autoscale(False, axis='y')

        self.timeseries_figure.tight_layout()
        self.timeseries_canvas.draw()

    def _on_pipeline_changed(self):
        """チェックボックスの状態に合わせてスピナー有効/無効を切り替え、ズームをリセットしてグラフ更新"""
        self.moving_avg_spin.setEnabled(self.ma_check.isChecked())
        self.bin_size_spin.setEnabled(self.bin_check.isChecked())
        # パイプライン変更時はズーム状態をリセット（前のスケールが残らないよう）
        self._zoom_xlim = None
        self._zoom_ylim = None
        self.zoom_reset_btn.setEnabled(False)
        self.update_timeseries_graph()

    def _on_mouse_press(self, event):
        """マウスボタン押下"""
        if event.inaxes is None or event.xdata is None:
            return
        if event.button == 3:
            # 右クリック: パン開始
            self._drag_start = (event.xdata, event.ydata)
            self._is_dragging = False

    def _on_mouse_move(self, event):
        """マウス移動（右ドラッグでパン）"""
        if self._drag_start is None or event.inaxes is None or event.xdata is None:
            return
        self._is_dragging = True
        ax = event.inaxes
        dx = self._drag_start[0] - event.xdata
        dy = self._drag_start[1] - event.ydata
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        ax.set_xlim(x_lo + dx, x_hi + dx)
        ax.set_ylim(y_lo + dy, y_hi + dy)
        self._zoom_xlim = ax.get_xlim()
        self._zoom_ylim = ax.get_ylim()
        self.zoom_reset_btn.setEnabled(True)
        self.timeseries_canvas.draw_idle()

    def _on_mouse_release(self, event):
        """マウスボタンリリース"""
        was_dragging = self._is_dragging
        self._drag_start = None
        self._is_dragging = False

        if was_dragging or event.button == 3:
            return  # ドラッグ終了 or 右クリックはジャンプしない

        # 左クリック: 画像ジャンプ
        if event.button == 1 and event.inaxes is not None and event.xdata is not None:
            click_x = event.xdata
            min_dist = float('inf')
            closest_idx = None
            for idx in self.annotations.keys():
                if idx in self.deleted_indexes:
                    continue
                dist = abs(idx - click_x)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx
            if closest_idx is not None:
                self.jump_to_image.emit(closest_idx)
                self.current_index = closest_idx
                self.update_timeseries_graph()

    def _on_scroll(self, event):
        """スクロールホイールでズーム"""
        if event.inaxes is None or event.xdata is None:
            return
        ax = event.inaxes
        # ズーム倍率
        scale = 0.8 if event.button == 'up' else 1.25

        # X軸ズーム（カーソル位置を中心）
        x_lo, x_hi = ax.get_xlim()
        x_center = event.xdata
        new_x_half = (x_hi - x_lo) * scale / 2
        ax.set_xlim(x_center - new_x_half, x_center + new_x_half)

        # Y軸ズーム（カーソル位置を中心）
        y_lo, y_hi = ax.get_ylim()
        y_center = event.ydata
        new_y_half = (y_hi - y_lo) * scale / 2
        ax.set_ylim(y_center - new_y_half, y_center + new_y_half)

        self._zoom_xlim = ax.get_xlim()
        self._zoom_ylim = ax.get_ylim()
        self.zoom_reset_btn.setEnabled(True)
        self.timeseries_canvas.draw_idle()

    def _on_range_changed(self):
        """表示範囲が変更されたらズームをリセットしてグラフを更新"""
        self._zoom_xlim = None
        self._zoom_ylim = None
        self.zoom_reset_btn.setEnabled(False)
        self.update_timeseries_graph()

    def _reset_zoom(self):
        """ズームをリセットして全体表示に戻す"""
        self._zoom_xlim = None
        self._zoom_ylim = None
        self.zoom_reset_btn.setEnabled(False)
        self.update_timeseries_graph()

    def update_current_position(self, index):
        """現在位置を更新（外部から呼び出し用）"""
        self.current_index = index
        self.update_timeseries_graph()
