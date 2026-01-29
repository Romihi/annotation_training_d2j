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
    QButtonGroup, QRadioButton, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

# 日本語フォントの設定
plt.rcParams['font.family'] = ['MS Gothic', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']
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
        self.setMinimumSize(900, 900)
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
        stats_dist_group = QGroupBox(get_text('section_stats_distribution'))
        stats_dist_layout = QHBoxLayout()

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
        stats_dist_layout.addWidget(self.stats_table, 1)

        # 右側: 分布グラフ（AngleとThrottleを1つのグラフに統合）
        self.dist_figure = Figure(figsize=(4, 3), dpi=100)
        self.dist_canvas = FigureCanvas(self.dist_figure)
        self.dist_canvas.setMinimumHeight(200)
        stats_dist_layout.addWidget(self.dist_canvas, 1)

        stats_dist_group.setLayout(stats_dist_layout)
        layout.addWidget(stats_dist_group)

        # === 時系列セクション ===
        timeseries_group = QGroupBox(get_text('section_timeseries'))
        timeseries_layout = QVBoxLayout()

        # 上部: 表示形式設定
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel(get_text('label_display')))

        self.display_mode_group = QButtonGroup(self)
        self.raw_data_radio = QRadioButton(get_text('label_raw_data'))
        self.raw_data_radio.setChecked(True)
        self.display_mode_group.addButton(self.raw_data_radio, 0)
        self.raw_data_radio.toggled.connect(self.update_timeseries_graph)
        settings_layout.addWidget(self.raw_data_radio)

        self.moving_avg_radio = QRadioButton(get_text('label_moving_avg'))
        self.display_mode_group.addButton(self.moving_avg_radio, 1)
        self.moving_avg_radio.toggled.connect(self.update_timeseries_graph)
        settings_layout.addWidget(self.moving_avg_radio)

        self.mean_hist_radio = QRadioButton(get_text('label_bin_avg'))
        self.display_mode_group.addButton(self.mean_hist_radio, 2)
        self.mean_hist_radio.toggled.connect(self.update_timeseries_graph)
        settings_layout.addWidget(self.mean_hist_radio)

        settings_layout.addWidget(QLabel("|"))

        # 移動平均の窓サイズ
        settings_layout.addWidget(QLabel(get_text('label_window')))
        self.moving_avg_spin = QSpinBox()
        self.moving_avg_spin.setRange(2, 300)
        self.moving_avg_spin.setValue(50)
        self.moving_avg_spin.setSingleStep(1)
        self.moving_avg_spin.valueChanged.connect(self.update_timeseries_graph)
        settings_layout.addWidget(self.moving_avg_spin)

        # 区間平均の区間サイズ
        settings_layout.addWidget(QLabel(get_text('label_bin')))
        self.bin_size_spin = QSpinBox()
        self.bin_size_spin.setRange(10, 1000)
        self.bin_size_spin.setValue(200)
        self.bin_size_spin.setSingleStep(10)
        self.bin_size_spin.setSuffix(" idx")
        self.bin_size_spin.valueChanged.connect(self.update_timeseries_graph)
        settings_layout.addWidget(self.bin_size_spin)

        settings_layout.addStretch()
        timeseries_layout.addLayout(settings_layout)

        # 2行目: 表示範囲設定
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel(get_text('label_display_range')))

        self.idx_min_spin = QSpinBox()
        self.idx_min_spin.setRange(0, 100000)
        self.idx_min_spin.setValue(0)
        self.idx_min_spin.setSingleStep(100)
        self.idx_min_spin.valueChanged.connect(self.update_timeseries_graph)
        range_layout.addWidget(self.idx_min_spin)

        range_layout.addWidget(QLabel("〜"))

        self.idx_max_spin = QSpinBox()
        self.idx_max_spin.setRange(0, 100000)
        self.idx_max_spin.setValue(0)
        self.idx_max_spin.setSingleStep(100)
        self.idx_max_spin.setSpecialValueText(get_text('label_auto'))
        self.idx_max_spin.valueChanged.connect(self.update_timeseries_graph)
        range_layout.addWidget(self.idx_max_spin)

        range_layout.addStretch()
        timeseries_layout.addLayout(range_layout)

        # 中央: グラフと表示項目リスト（横並び）
        graph_layout = QHBoxLayout()

        # 時系列グラフ（左側）
        self.timeseries_figure = Figure(figsize=(8, 4), dpi=100)
        self.timeseries_canvas = FigureCanvas(self.timeseries_figure)
        self.timeseries_canvas.mpl_connect('button_press_event', self.on_timeseries_click)
        self.timeseries_canvas.setMinimumHeight(300)
        graph_layout.addWidget(self.timeseries_canvas, 4)

        # キー選択リスト（右側）
        key_select_layout = QVBoxLayout()
        key_select_layout.addWidget(QLabel(get_text('label_display_items')))
        self.key_list = QListWidget()
        self.key_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.key_list.setMaximumWidth(120)

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

        # 説明ラベル
        info_label = QLabel(get_text('label_click_to_jump'))
        info_label.setStyleSheet("color: gray;")
        timeseries_layout.addWidget(info_label)

        timeseries_group.setLayout(timeseries_layout)
        layout.addWidget(timeseries_group)

        layout.addStretch()

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

        if self.mean_hist_radio.isChecked():
            # 平均値ヒストグラム表示
            min_idx = min(indices)
            max_idx = max(indices)
            bin_starts = list(range(min_idx, max_idx + 1, bin_size))
            num_keys = len(selected_keys)
            bar_width = bin_size * 0.8 / num_keys

            for key_idx, key in enumerate(selected_keys):
                means = []
                bin_centers = []

                for bin_start in bin_starts:
                    bin_end = bin_start + bin_size
                    bin_center = bin_start + bin_size / 2
                    bin_centers.append(bin_center)

                    # このビン内のデータを収集
                    bin_values = []
                    for idx in indices:
                        if bin_start <= idx < bin_end and idx in data_by_key[key]:
                            bin_values.append(data_by_key[key][idx])

                    # 平均値を計算
                    if bin_values:
                        means.append(np.mean(bin_values))
                    else:
                        means.append(np.nan)

                # バーをオフセットして描画
                offset = (key_idx - num_keys / 2 + 0.5) * bar_width
                color = COLORS[key_idx % len(COLORS)]
                ax.bar([c + offset for c in bin_centers], means,
                       width=bar_width, color=color, alpha=0.7,
                       label=key, edgecolor='white')

            ax.set_title(get_text('label_bin_avg_title', bin_size))

            # 現在位置マーカー
            ax.axvline(x=self.current_index, color='purple', linestyle='--', alpha=0.8, linewidth=2)

        elif self.moving_avg_radio.isChecked():
            # 移動平均表示
            for key_idx, key in enumerate(selected_keys):
                key_indices = sorted(data_by_key[key].keys())
                key_values = [data_by_key[key][i] for i in key_indices]

                if key_values:
                    color = COLORS[key_idx % len(COLORS)]

                    if len(key_values) >= window_size:
                        # 移動平均を計算
                        values_arr = np.array(key_values)
                        moving_avg = np.convolve(values_arr, np.ones(window_size)/window_size, mode='valid')
                        # 移動平均のインデックスを調整（中央揃え）
                        offset = window_size // 2
                        avg_indices = key_indices[offset:offset + len(moving_avg)]
                        ax.plot(avg_indices, moving_avg, '-', color=color, alpha=0.9, label=f'{key}(MA{window_size})', linewidth=1.2)
                        # 元データを薄く表示
                        ax.plot(key_indices, key_values, '-', color=color, alpha=0.2, linewidth=0.5)
                    else:
                        # データが少ない場合は生データのみ
                        ax.plot(key_indices, key_values, '-', color=color, alpha=0.7, label=key, linewidth=0.8)

            ax.set_title(get_text('label_data_trend_ma', window_size))

            # 現在位置マーカー
            ax.axvline(x=self.current_index, color='purple', linestyle='--', alpha=0.8, linewidth=2)
            if self.current_index in indices:
                if selected_keys and self.current_index in data_by_key[selected_keys[0]]:
                    y_val = data_by_key[selected_keys[0]][self.current_index]
                    ax.scatter([self.current_index], [y_val], color='purple', s=100, zorder=5, marker='o')

        else:
            # 生データ表示（線グラフ）
            for key_idx, key in enumerate(selected_keys):
                key_indices = sorted(data_by_key[key].keys())
                key_values = [data_by_key[key][i] for i in key_indices]

                if key_values:
                    color = COLORS[key_idx % len(COLORS)]
                    ax.plot(key_indices, key_values, '-', color=color, alpha=0.7, label=key, linewidth=0.8)

            ax.set_title(get_text('label_data_trend'))

            # 現在位置マーカー
            ax.axvline(x=self.current_index, color='purple', linestyle='--', alpha=0.8, linewidth=2)
            if self.current_index in indices:
                # 最初の選択キーの値をマーカーで表示
                if selected_keys and self.current_index in data_by_key[selected_keys[0]]:
                    y_val = data_by_key[selected_keys[0]][self.current_index]
                    ax.scatter([self.current_index], [y_val], color='purple', s=100, zorder=5, marker='o')

        ax.set_xlabel(get_text('label_index'))
        ax.set_ylabel(get_text('label_value'))
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

        # X軸の範囲を設定
        if indices:
            idx_min_setting = self.idx_min_spin.value()
            idx_max_setting = self.idx_max_spin.value()

            # 自動範囲（max=0の場合）またはカスタム範囲
            if idx_max_setting == 0:
                x_min = min(indices) - 10
                x_max = max(indices) + 10
            else:
                x_min = idx_min_setting
                x_max = idx_max_setting

            ax.set_xlim(x_min, x_max)

        self.timeseries_figure.tight_layout()
        self.timeseries_canvas.draw()

    def on_timeseries_click(self, event):
        """時系列グラフクリック時の処理"""
        if event.inaxes is None:
            return

        # クリック位置に最も近いインデックスを取得
        click_x = event.xdata
        if click_x is None:
            return

        # 最も近いアノテーション済みインデックスを探す
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

    def update_current_position(self, index):
        """現在位置を更新（外部から呼び出し用）"""
        self.current_index = index
        self.update_timeseries_graph()
