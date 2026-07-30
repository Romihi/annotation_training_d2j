# yolo_utils.py
"""YOLOv8を使用した物体検知のユーティリティ関数"""
import os
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (QLabel, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QPlainTextEdit, QProgressDialog, QMessageBox, QApplication,
                             QGroupBox, QProgressBar, QTextEdit)
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal

# Ultralytics YOLOv8 インポート
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics がインストールされていません。pip install ultralytics でインストールしてください。")

import sys
import re
from io import StringIO

# 物体カテゴリ定義
DEFAULT_CLASSES = ["traffic_cone", "person", "car", "bicycle", "motorcycle", "truck", "bus", "stop_sign", "parking_meter"]

# 学習済みYOLOで「他車(opponent)」とみなす既定の元クラス（COCO想定）。
# 自動アノテーションでこれらを target_class（既定 'opponent'）へ写像する。
DEFAULT_VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# 自車除外領域（正規化 x0,y0,x1,y1）。魚眼前方カメラでは画像下部中央に自車の
# 車体/ハンドルが常に写り YOLO が car/bicycle 等と誤検知するため、中心がこの
# 矩形に入る検出は捨てる（前方の他車は上部にあり除外されない）。
DEFAULT_EGO_REGION = (0.12, 0.66, 0.88, 1.0)


def _class_name(names, cid):
    """ultralytics の names（dict or list）からクラス名を安全に引く。"""
    cid = int(cid)
    if isinstance(names, dict):
        return names.get(cid, f"class_{cid}")
    if names is not None and 0 <= cid < len(names):
        return names[cid]
    return f"class_{cid}"


def dets_to_bboxes(dets_np, names, img_wh, class_map=None, conf_min=0.0,
                   ego_region=None):
    """YOLO検出 (N,6)=[x1,y1,x2,y2,conf,cls] → bbox_annotation 辞書のリスト。

    アノテーションツールの矩形形式 {x1,y1,x2,y2(0-1正規化), class, confidence} に
    変換する。run_single_yolo_inference と自動アノテーションで**同一コード**を共有。

    class_map: {元クラス名: 変換後クラス名}。指定時は map にあるクラスのみ採用し、
      名前を写像する（例 {'car':'opponent'} で車→他車ラベル）。None なら全クラス素通し。
    conf_min: この信頼度未満は捨てる。
    ego_region: (x0,y0,x1,y1) 正規化。中心がこの矩形内の検出は**自車**として捨てる。
    """
    W, H = img_wh
    out = []
    for det in np.asarray(dets_np):
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cid = det[:6]
        if float(conf) < conf_min:
            continue
        if ego_region is not None:                # 自車領域の検出は除外
            cx = (float(x1) + float(x2)) * 0.5 / W
            cy = (float(y1) + float(y2)) * 0.5 / H
            ex0, ey0, ex1, ey1 = ego_region
            if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                continue
        cname = _class_name(names, cid)
        if class_map is not None:
            if cname not in class_map:
                continue
            cname = class_map[cname]
        out.append({
            'x1': float(x1) / W, 'y1': float(y1) / H,
            'x2': float(x2) / W, 'y2': float(y2) / H,
            'class': cname, 'confidence': float(conf),
        })
    return out


def masks_to_segments(masks_np, classes, confs, class_map=None, conf_min=0.0,
                      ego_region=None):
    """YOLOセグメンテーションのマスク → segmentation_annotation のリスト。

    各マスクを輪郭抽出→ポリライン簡略化し {class, points(0-1正規化), confidence} に。
    masks_np: (N, mh, mw)。classes/confs: 長さ N のクラス名・信頼度。
    class_map: dets_to_bboxes と同じ意味。cv2 は呼び出し時に import。
    ego_region: (x0,y0,x1,y1) 正規化。重心がこの矩形内のマスクは自車として捨てる。
    """
    import cv2
    segs = []
    for i, mask in enumerate(np.asarray(masks_np)):
        conf = float(confs[i]) if i < len(confs) else 0.0
        if conf < conf_min:
            continue
        cname = classes[i] if i < len(classes) else "unknown"
        if class_map is not None:
            if cname not in class_map:
                continue
            cname = class_map[cname]
        m = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
        mh, mw = m.shape[:2]
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        pts = [[float(min(max(p[0][0] / mw, 0.0), 1.0)),
                float(min(max(p[0][1] / mh, 0.0), 1.0))] for p in approx]
        if len(pts) < 3:
            continue
        if ego_region is not None:                # 自車領域のマスクは除外
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            ex0, ey0, ex1, ey1 = ego_region
            if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                continue
        segs.append({'class': cname, 'points': pts, 'confidence': conf})
    return segs

# YOLO学習ワーカークラス
class YOLOTrainingWorker(QThread):
    """YOLO学習をバックグラウンドで実行し、出力をリアルタイムで通知"""
    output_received = pyqtSignal(str)  # 出力テキストを通知
    progress_updated = pyqtSignal(int, str)  # 進捗(%)とメッセージを通知
    epoch_updated = pyqtSignal(int, int)  # エポック進捗を通知 (current, total)
    metrics_updated = pyqtSignal(dict)  # メトリクス（loss等）を通知
    training_completed = pyqtSignal(object)  # 学習完了を通知
    error_occurred = pyqtSignal(str)  # エラーを通知

    def __init__(self, model, training_params):
        super().__init__()
        self.model = model
        self.training_params = training_params
        self.process = None
        self._stop_requested = False
        self.results = None  # 学習結果を保存
        self.total_epochs = training_params.get('epochs', 100)
        self.current_epoch = 0

    def run(self):
        """学習を実行し、出力をキャプチャ"""
        try:
            # 出力キャプチャクラス
            class SimpleOutputCapture:
                def __init__(self, worker, original_stream):
                    self.worker = worker
                    self.original_stream = original_stream
                    self.buffer = ""

                def write(self, text):
                    # 元のストリームにも出力（ターミナルに表示）
                    self.original_stream.write(text)
                    self.original_stream.flush()

                    # UIに出力を送信
                    if text:
                        # テキストをそのまま送信
                        self.worker.output_received.emit(text)

                        # バッファに蓄積
                        self.buffer += text

                        # 改行またはキャリッジリターンがあれば処理
                        if '\n' in self.buffer or '\r' in self.buffer:
                            # 改行とキャリッジリターンの両方で分割
                            lines = re.split(r'[\r\n]+', self.buffer)
                            # 完全な行のみを処理
                            for line in lines[:-1]:
                                if line.strip():
                                    # ANSI エスケープシーケンスを除去
                                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                                    self._parse_line(clean_line.strip())

                            # 最後の不完全な行を保持
                            self.buffer = lines[-1] if lines else ""

                def _parse_line(self, line):
                    # エポック進捗の解析（複数パターン対応）
                    if 'Starting training for' in line:
                        match = re.search(r'Starting training for (\d+) epochs', line)
                        if match:
                            total_epochs = int(match.group(1))
                            self.worker.progress_updated.emit(0, f"学習開始: {total_epochs}エポック")
                            return

                    # エポック番号の解析（複数パターン）
                    # Pattern 1: "Epoch 1/100" or "epoch: 1/100"
                    epoch_patterns = [
                        r'Epoch[:\s]+(\d+)/(\d+)',
                        r'(\d+)/(\d+)\s*epochs?',
                        r'Epoch\((\d+)/(\d+)\)',
                    ]

                    for pattern in epoch_patterns:
                        epoch_match = re.search(pattern, line, re.IGNORECASE)
                        if epoch_match:
                            current = int(epoch_match.group(1))
                            total = int(epoch_match.group(2))
                            progress = int((current / total) * 100)
                            self.worker.progress_updated.emit(progress, f"エポック {current}/{total}")
                            break

                    # Loss値の解析（YOLOv8/v11の数値行）
                    # パターン例: "      1/100      3.52G     0.9834     2.312     1.234        128        640"
                    # または: "1/100  0.9834  2.312  1.234"
                    if re.search(r'\d+/\d+', line) and re.search(r'\d+\.\d+', line):
                        # GPU_memを含む行かチェック
                        has_gpu_mem = bool(re.search(r'\d+\.\d+G', line))

                        # 浮動小数点数を全て抽出（GPU memは除外）
                        numbers = re.findall(r'(\d+\.\d+)(?!G)', line)

                        if has_gpu_mem and len(numbers) >= 3:
                            # GPU memがある場合、最初の数値をスキップして次の3つを使用
                            try:
                                metrics = {
                                    'box_loss': float(numbers[0]),
                                    'cls_loss': float(numbers[1]),
                                    'dfl_loss': float(numbers[2])
                                }
                                # セグメンテーションの場合はseg_lossも追加
                                if len(numbers) >= 4:
                                    metrics['seg_loss'] = float(numbers[3])
                                self.worker.metrics_updated.emit(metrics)
                            except:
                                pass
                        elif not has_gpu_mem and len(numbers) >= 3:
                            # GPU memがない場合、最初の3つを使用
                            try:
                                metrics = {
                                    'box_loss': float(numbers[0]),
                                    'cls_loss': float(numbers[1]),
                                    'dfl_loss': float(numbers[2])
                                }
                                if len(numbers) >= 4:
                                    metrics['seg_loss'] = float(numbers[3])
                                self.worker.metrics_updated.emit(metrics)
                            except:
                                pass

                    # mAP値の解析（複数パターン対応）
                    # パターン1: "all  100  100  0.95  0.85" (COCO形式)
                    if line.strip().startswith('all'):
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                # mAP50は3番目または4番目の数値
                                idx = 3 if len(parts) > 5 else 2
                                metrics = {
                                    'mAP50': float(parts[idx]),
                                    'mAP50-95': float(parts[idx + 1])
                                }
                                self.worker.metrics_updated.emit(metrics)
                            except:
                                pass

                    # パターン2: "metrics/mAP50(B): 0.95" 形式
                    map_match = re.search(r'metrics/mAP_?0?\.?5[^:]*:\s*([\d.]+)', line)
                    if map_match:
                        try:
                            metrics = {'mAP50': float(map_match.group(1))}
                            self.worker.metrics_updated.emit(metrics)
                        except:
                            pass

                    map5095_match = re.search(r'metrics/mAP_?0?\.?5:0?\.?95[^:]*:\s*([\d.]+)', line)
                    if map5095_match:
                        try:
                            metrics = {'mAP50-95': float(map5095_match.group(1))}
                            self.worker.metrics_updated.emit(metrics)
                        except:
                            pass

                def flush(self):
                    self.original_stream.flush()

                def fileno(self):
                    return self.original_stream.fileno()

            # 出力をキャプチャ
            stdout_capture = SimpleOutputCapture(self, sys.stdout)
            stderr_capture = SimpleOutputCapture(self, sys.stderr)

            # 標準出力を一時的に置き換え
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            try:
                # YOLO学習を実行
                self.progress_updated.emit(5, "学習を開始しています...")

                # コールバック関数を設定してエポック毎の進捗を取得
                worker_ref = self  # selfへの参照を保持

                def on_train_epoch_end(trainer):
                    """各トレーニングエポック終了時に呼ばれるコールバック"""
                    current_epoch = trainer.epoch + 1  # 0-indexed -> 1-indexed
                    total_epochs = trainer.epochs
                    worker_ref.current_epoch = current_epoch

                    # エポック進捗を通知
                    worker_ref.epoch_updated.emit(current_epoch, total_epochs)

                    # パーセント進捗も更新
                    progress = int((current_epoch / total_epochs) * 100)
                    worker_ref.progress_updated.emit(progress, f"エポック {current_epoch}/{total_epochs}")

                def on_fit_epoch_end(trainer):
                    """各エポック終了時（train + val後）に呼ばれるコールバック"""
                    # trainer.metricsからmAP値を含むメトリクスを取得
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        metrics = {}
                        try:
                            # trainer.metricsは辞書形式
                            trainer_metrics = trainer.metrics

                            # mAP値の取得（複数のキー形式に対応）
                            # YOLOv8/v11: 'metrics/mAP50(B)', 'metrics/mAP50-95(B)' など
                            for key, value in trainer_metrics.items():
                                if 'mAP50-95' in key or 'mAP_0.5:0.95' in key:
                                    metrics['mAP50-95'] = float(value)
                                elif 'mAP50' in key or 'mAP_0.5' in key:
                                    metrics['mAP50'] = float(value)
                                elif 'precision' in key.lower():
                                    metrics['precision'] = float(value)
                                elif 'recall' in key.lower():
                                    metrics['recall'] = float(value)

                            if metrics:
                                worker_ref.metrics_updated.emit(metrics)
                        except Exception as e:
                            print(f"メトリクス取得エラー: {e}")

                def on_train_batch_end(trainer):
                    """各バッチ終了時に呼ばれるコールバック（Loss更新用）"""
                    if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                        try:
                            metrics = {}
                            loss_tensor = trainer.tloss
                            if hasattr(loss_tensor, 'cpu'):
                                loss_values = loss_tensor.cpu().numpy()
                            else:
                                loss_values = loss_tensor

                            loss_names = ['box_loss', 'cls_loss', 'dfl_loss']
                            for i, name in enumerate(loss_names):
                                if i < len(loss_values):
                                    metrics[name] = float(loss_values[i])

                            if metrics:
                                worker_ref.metrics_updated.emit(metrics)
                        except:
                            pass

                # コールバックを登録
                self.model.add_callback('on_train_epoch_end', on_train_epoch_end)
                self.model.add_callback('on_fit_epoch_end', on_fit_epoch_end)  # mAP取得用
                self.model.add_callback('on_train_batch_end', on_train_batch_end)

                self.results = self.model.train(**self.training_params)
                self.training_completed.emit(self.results)

            finally:
                # 標準出力を復元
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            self.error_occurred.emit(str(e))
            import traceback
            traceback.print_exc()

    def stop(self):
        """学習を停止"""
        self._stop_requested = True
        if self.process:
            self.process.terminate()


class TrainingOutputDialog(QDialog):
    """YOLO学習の進捗を表示するダイアログ"""

    def __init__(self, parent=None, model_name="YOLO"):
        super().__init__(parent)
        self.setWindowTitle(f"{model_name} 学習進行状況")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)

        # レイアウト設定
        layout = QVBoxLayout(self)

        # 進捗グループ
        progress_group = QGroupBox("学習進捗")
        progress_layout = QVBoxLayout(progress_group)

        # エポック進捗
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("エポック:"))
        self.epoch_label = QLabel("0 / 0")
        epoch_layout.addWidget(self.epoch_label)
        epoch_layout.addStretch()
        progress_layout.addLayout(epoch_layout)

        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(progress_group)

        # メトリクスグループ
        metrics_group = QGroupBox("学習メトリクス")
        metrics_layout = QVBoxLayout(metrics_group)

        # Loss表示（物体検知用）
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Box Loss:"))
        self.box_loss_label = QLabel("-.----")
        loss_layout.addWidget(self.box_loss_label)
        loss_layout.addSpacing(20)

        loss_layout.addWidget(QLabel("Cls Loss:"))
        self.cls_loss_label = QLabel("-.----")
        loss_layout.addWidget(self.cls_loss_label)
        loss_layout.addSpacing(20)

        loss_layout.addWidget(QLabel("DFL Loss:"))
        self.dfl_loss_label = QLabel("-.----")
        loss_layout.addWidget(self.dfl_loss_label)
        loss_layout.addStretch()
        metrics_layout.addLayout(loss_layout)

        # セグメンテーション用Loss（オプション）
        seg_layout = QHBoxLayout()
        seg_layout.addWidget(QLabel("Seg Loss:"))
        self.seg_loss_label = QLabel("-.----")
        seg_layout.addWidget(self.seg_loss_label)
        seg_layout.addStretch()
        metrics_layout.addLayout(seg_layout)

        # mAP表示
        map_layout = QHBoxLayout()
        map_layout.addWidget(QLabel("mAP50:"))
        self.map50_label = QLabel("-.----")
        map_layout.addWidget(self.map50_label)
        map_layout.addSpacing(20)

        map_layout.addWidget(QLabel("mAP50-95:"))
        self.map5095_label = QLabel("-.----")
        map_layout.addWidget(self.map5095_label)
        map_layout.addStretch()
        metrics_layout.addLayout(map_layout)

        layout.addWidget(metrics_group)

        # ログ表示エリア
        log_group = QGroupBox("学習ログ")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)  # 最小高さのみ設定
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group, 1)  # stretch factor 1で残りスペースを埋める

        # ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.stop_button = QPushButton("学習を停止")
        self.stop_button.clicked.connect(self.stop_training)
        button_layout.addWidget(self.stop_button)

        self.close_button = QPushButton("閉じる")
        self.close_button.clicked.connect(self.close)
        self.close_button.setEnabled(False)  # 学習中は無効
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

        self.worker = None
        self.training_completed = False
        self.total_epochs = 0
        self.current_epoch = 0

    def start_training(self, model, training_params):
        """学習を開始"""
        self.total_epochs = training_params.get('epochs', 100)
        self.current_epoch = 0

        # 初期エポック表示を更新
        self.epoch_label.setText(f"0 / {self.total_epochs}")
        self.progress_bar.setValue(0)

        self.worker = YOLOTrainingWorker(model, training_params)
        self.worker.output_received.connect(self.append_log)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.epoch_updated.connect(self.update_epoch)  # エポック更新シグナルを接続
        self.worker.metrics_updated.connect(self.update_metrics)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.error_occurred.connect(self.on_error)

        self.worker.start()

    def add_preparation_message(self, message):
        """学習前の準備メッセージを追加"""
        self.append_log(message + "\n")

    def append_log(self, text):
        """ログテキストを追加"""
        # ANSI エスケープシーケンスを除去（端末制御文字・カラーコード）
        import re as _re
        text = _re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)   # CSI シーケンス
        text = _re.sub(r'\x1b[()][0-9A-Za-z]', '', text)      # その他エスケープ
        text = text.replace('\r', '')                          # キャリッジリターン
        # すべてのテキストをログに追加（フィルタリングを緩める）
        if text.strip():
            # 改行がない場合は現在の行に追加
            if not text.endswith('\n'):
                # カーソルを最後に移動
                cursor = self.log_text.textCursor()
                cursor.movePosition(cursor.End)
                cursor.insertText(text)
                self.log_text.setTextCursor(cursor)
            else:
                self.log_text.append(text.rstrip())

            # 自動スクロール
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, progress, message):
        """進捗を更新"""
        self.progress_bar.setValue(progress)
        # エポック情報を抽出
        if "エポック" in message:
            import re
            match = re.search(r'(\d+)/(\d+)', message)
            if match:
                self.current_epoch = int(match.group(1))
                total = int(match.group(2))
                self.epoch_label.setText(f"{self.current_epoch} / {total}")

    def update_epoch(self, current, total):
        """エポック進捗を直接更新"""
        self.current_epoch = current
        self.total_epochs = total
        self.epoch_label.setText(f"{current} / {total}")
        # プログレスバーも更新
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)

    def update_metrics(self, metrics):
        """メトリクスを更新"""
        if 'box_loss' in metrics:
            self.box_loss_label.setText(f"{metrics['box_loss']:.4f}")
        if 'cls_loss' in metrics:
            self.cls_loss_label.setText(f"{metrics['cls_loss']:.4f}")
        if 'dfl_loss' in metrics:
            self.dfl_loss_label.setText(f"{metrics['dfl_loss']:.4f}")
        if 'seg_loss' in metrics:
            self.seg_loss_label.setText(f"{metrics['seg_loss']:.4f}")
        if 'mAP50' in metrics:
            self.map50_label.setText(f"{metrics['mAP50']:.4f}")
        if 'mAP50-95' in metrics:
            self.map5095_label.setText(f"{metrics['mAP50-95']:.4f}")

    def stop_training(self):
        """学習を停止"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "確認",
                "学習を停止しますか？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.append_log("\n=== 学習を停止しました ===")
                self.stop_button.setEnabled(False)
                self.close_button.setEnabled(True)

    def on_training_completed(self, results):
        """学習完了時の処理"""
        self.training_completed = True
        self.append_log("\n=== 学習が完了しました ===")
        self.progress_bar.setValue(100)
        self.epoch_label.setText(f"{self.total_epochs} / {self.total_epochs} (完了)")
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)

        # 結果を親ウィンドウに通知
        if self.parent():
            if hasattr(self.parent(), 'on_yolo_training_completed'):
                self.parent().on_yolo_training_completed(results)

    def on_error(self, error_msg):
        """エラー発生時の処理"""
        self.append_log(f"\n=== エラーが発生しました ===\n{error_msg}")
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
        QMessageBox.critical(self, "エラー", f"学習中にエラーが発生しました:\n{error_msg}")

    def closeEvent(self, event):
        """ダイアログを閉じる際の処理"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "確認",
                "学習がまだ実行中です。本当に閉じますか？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self.worker.stop()
        event.accept()


def train_yolo_with_ui(model, training_params, parent=None, model_name="YOLO"):
    """UIダイアログ付きでYOLO学習を実行"""
    dialog = TrainingOutputDialog(parent, model_name)
    dialog.start_training(model, training_params)
    dialog.exec_()
    return dialog.training_completed

def get_yolo_model(model_path=None, pretrained=True):
    """
    YOLOモデルを読み込む
    Args:
        model_path: カスタムモデルのパス
        pretrained: 事前学習済みモデルを使用するかどうか
    Returns:
        YOLOモデル
    """
    if model_path and os.path.exists(model_path):
        try:
            model = YOLO(model_path)
            print(f"カスタムYOLOモデルを読み込みました: {model_path}")
            return model
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            
    if pretrained:
        # 事前学習済みのyolov8nを使用
        model = YOLO("yolov8n.pt")
        print("事前学習済みYOLOv8nモデルを読み込みました")
        return model
    
    return None

def detect_objects(image_path, model=None, conf_threshold=0.25):
    """
    画像内の物体を検出する
    Args:
        image_path: 画像のパス
        model: YOLOモデル
        conf_threshold: 信頼度のしきい値
    Returns:
        検出結果のリスト [{'class': クラス名, 'bbox': [x1, y1, x2, y2], 'confidence': 信頼度}, ...]
    """
    if model is None:
        model = get_yolo_model()
    
    if not os.path.exists(image_path):
        print(f"画像が見つかりません: {image_path}")
        return []
    
    try:
        # 画像を読み込み
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 推論実行
        results = model(image_path, conf=conf_threshold)[0]
        
        # 結果を整形
        detections = []
        for i, det in enumerate(results.boxes.data):
            x1, y1, x2, y2, conf, cls = det.tolist()
            
            # クラスIDからクラス名を取得
            class_name = results.names[int(cls)]
            
            # 結果を追加
            detections.append({
                'class': class_name,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf)
            })
        
        return detections
    
    except Exception as e:
        print(f"検出エラー: {e}")
        return []

def detect_objects_and_segments(image_path, model=None, conf_threshold=0.25):
    """
    画像内の物体検出とセグメンテーションを実行する
    Args:
        image_path: 画像のパス
        model: YOLOモデル (セグメンテーション対応)
        conf_threshold: 信頼度のしきい値
    Returns:
        検出結果の辞書 {'detections': [...], 'segments': [...]}
    """
    if model is None:
        model = get_yolo_model()
    
    if not os.path.exists(image_path):
        print(f"画像が見つかりません: {image_path}")
        return {'detections': [], 'segments': []}
    
    try:
        # 画像を読み込み
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 推論実行
        results = model(image_path, conf=conf_threshold)[0]
        
        detections = []
        segments = []
        
        # バウンディングボックス処理
        if hasattr(results, 'boxes') and results.boxes is not None:
            for i, det in enumerate(results.boxes.data):
                x1, y1, x2, y2, conf, cls = det.tolist()
                class_name = results.names[int(cls)]
                
                # ピクセル座標を正規化座標（0-1）に変換し、float型で統一
                norm_x1 = float(x1) / float(img_width)
                norm_y1 = float(y1) / float(img_height)
                norm_x2 = float(x2) / float(img_width)
                norm_y2 = float(y2) / float(img_height)
                
                # 座標の範囲を0-1に制限
                norm_x1 = max(0.0, min(1.0, norm_x1))
                norm_y1 = max(0.0, min(1.0, norm_y1))
                norm_x2 = max(0.0, min(1.0, norm_x2))
                norm_y2 = max(0.0, min(1.0, norm_y2))
                
                detections.append({
                    'class': class_name,
                    'bbox': [norm_x1, norm_y1, norm_x2, norm_y2],
                    'confidence': float(conf)
                })
        
        # セグメンテーションマスク処理
        if hasattr(results, 'masks') and results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                # マスクを numpy 配列に変換
                mask_array = mask.cpu().numpy()
                
                # マスクから輪郭ポイントを抽出
                import cv2
                mask_uint8 = (mask_array * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 最大の輪郭を取得
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # 輪郭を簡略化
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    # ポイントリストに変換（正規化座標に変換）
                    points = []
                    mask_height, mask_width = mask_array.shape
                    for point in approx:
                        x, y = point[0]
                        # マスク座標を0-1に正規化し、float型で統一
                        normalized_x = float(x) / float(mask_width)
                        normalized_y = float(y) / float(mask_height)
                        
                        # 座標を0-1範囲内に制限
                        normalized_x = max(0.0, min(1.0, normalized_x))
                        normalized_y = max(0.0, min(1.0, normalized_y))
                        
                        points.append([float(normalized_x), float(normalized_y)])
                    
                    if len(points) >= 3:  # 最低3点必要
                        # 対応するバウンディングボックスのクラス情報を取得
                        class_name = "unknown"
                        confidence = 0.0
                        if i < len(detections):
                            class_name = detections[i]['class']
                            confidence = detections[i]['confidence']
                        
                        segments.append({
                            'class': class_name,
                            'points': points,
                            'confidence': confidence
                        })
        
        return {
            'detections': detections,
            'segments': segments
        }
    
    except Exception as e:
        print(f"検出・セグメンテーションエラー: {e}")
        return {'detections': [], 'segments': []}

def train_yolo_model(dataset_dir, epochs=50, batch_size=16, img_size=640, save_dir=None, pretrained=True):
    """
    YOLOv8モデルを学習する
    Args:
        dataset_dir: データセットディレクトリ (YOLO形式)
        epochs: エポック数
        batch_size: バッチサイズ
        img_size: 画像サイズ
        save_dir: モデル保存ディレクトリ
        pretrained: 事前学習済みモデルを使用するかどうか
    Returns:
        学習済みモデルのパス
    """
    try:
        # 設定ファイルパスを確認
        yaml_path = os.path.join(dataset_dir, 'data.yaml')
        if not os.path.exists(yaml_path):
            print(f"設定ファイルが見つかりません: {yaml_path}")
            return None
        
        # モデルを初期化
        if pretrained:
            model = YOLO('yolov8n.pt')
            print("事前学習済みYOLOv8nモデルから学習を開始します")
        else:
            model = YOLO('yolov8n.yaml')
            print("YOLOv8nモデルをスクラッチから学習します")
        
        # トレーニングパラメータ
        params = {
            'data': yaml_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'patience': 10,  # Early stopping patience
            'device': 0 if torch.cuda.is_available() else 'cpu',
        }
        
        # 保存先が指定されている場合
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            params['project'] = save_dir
            params['name'] = f'yolov8n_custom_{time.strftime("%Y%m%d_%H%M%S")}'
        
        # トレーニング実行
        results = model.train(**params)
        
        # 最良のモデルのパスを取得
        best_model_path = results.best
        
        print(f"トレーニング完了! モデル保存先: {best_model_path}")
        return best_model_path
    
    except Exception as e:
        print(f"トレーニングエラー: {e}")
        return None

def convert_to_yolo_format(annotations, image_dir, output_dir, classes=None):
    """
    アノテーションをYOLO形式に変換する
    Args:
        annotations: アノテーション辞書 {画像パス: {クラス名: [[x1,y1,x2,y2], ...], ...}}
        image_dir: 入力画像ディレクトリ
        output_dir: 出力ディレクトリ
        classes: クラスリスト
    Returns:
        変換されたデータセットのディレクトリパス
    """
    if classes is None:
        classes = DEFAULT_CLASSES
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 画像とラベルを分割するディレクトリ
    train_images_dir = os.path.join(images_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    train_labels_dir = os.path.join(labels_dir, 'train')
    val_labels_dir = os.path.join(labels_dir, 'val')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # アノテーションを変換
    image_files = list(annotations.keys())
    np.random.shuffle(image_files)
    
    # 訓練/検証分割 (80/20)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # データ変換
    for img_path in train_files:
        _convert_single_annotation(img_path, annotations[img_path], train_images_dir, train_labels_dir, classes)
    
    for img_path in val_files:
        _convert_single_annotation(img_path, annotations[img_path], val_images_dir, val_labels_dir, classes)
    
    # YAML設定ファイル作成
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'nc': len(classes),
        'names': classes
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml_content_str = ""
        for key, value in yaml_content.items():
            if key == 'names':
                yaml_content_str += f"{key}: {value}\n"
            else:
                yaml_content_str += f"{key}: {value}\n"
        f.write(yaml_content_str)
    
    print(f"YOLOデータセットの変換が完了しました: {output_dir}")
    print(f"トレーニングデータ: {len(train_files)}個, 検証データ: {len(val_files)}個")
    
    return output_dir

def _convert_single_annotation(img_path, annotations, output_images_dir, output_labels_dir, classes):
    """
    1枚の画像のアノテーションをYOLO形式に変換する
    Args:
        img_path: 画像パス
        annotations: アノテーション {クラス名: [[x1,y1,x2,y2], ...], ...}
        output_images_dir: 出力画像ディレクトリ
        output_labels_dir: 出力ラベルディレクトリ
        classes: クラスリスト
    """
    try:
        # 画像ファイル名取得
        img_filename = os.path.basename(img_path)
        img_name = os.path.splitext(img_filename)[0]
        
        # 画像をコピー
        import shutil
        output_img_path = os.path.join(output_images_dir, img_filename)
        shutil.copy2(img_path, output_img_path)
        
        # 画像サイズ取得
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # ラベルファイル作成
        label_file = os.path.join(output_labels_dir, f"{img_name}.txt")
        
        with open(label_file, 'w') as f:
            for class_name, boxes in annotations.items():
                if class_name not in classes:
                    continue
                
                class_id = classes.index(class_name)
                
                for box in boxes:
                    x1, y1, x2, y2 = box
                    
                    # YOLO形式に変換 (中心x, 中心y, 幅, 高さ) - 正規化
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # ラベル行を書き込み: <class_id> <center_x> <center_y> <width> <height>
                    f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
    
    except Exception as e:
        print(f"変換エラー ({img_path}): {e}")

def batch_detect_objects(image_paths, model=None, conf_threshold=0.25, progress_callback=None, use_index_keys=False):
    """
    複数の画像で物体検出を実行する
    Args:
        image_paths: 画像パスのリスト
        model: YOLOモデル
        conf_threshold: 信頼度のしきい値
        progress_callback: 進捗コールバック関数
        use_index_keys: Trueの場合、インデックスをキーにして結果を返す
    Returns:
        検出結果の辞書 {画像パス or インデックス: 検出結果, ...}
    """
    if model is None:
        model = get_yolo_model()
    
    results = {}
    total = len(image_paths)
    
    for i, img_path in enumerate(image_paths):
        if progress_callback:
            if not progress_callback(i, total, f"画像 {i+1}/{total} を処理中: {os.path.basename(img_path)}"):
                break
                
        detections = detect_objects(img_path, model, conf_threshold)
        key = i if use_index_keys else img_path
        results[key] = detections
    
    return results

def batch_detect_objects_and_segments(image_paths, model=None, conf_threshold=0.25, progress_callback=None, indices=None):
    """
    複数の画像で物体検出とセグメンテーションを実行する
    Args:
        image_paths: 画像パスのリスト
        model: YOLOモデル (セグメンテーション対応)
        conf_threshold: 信頼度のしきい値
        progress_callback: 進捗コールバック関数
        indices: 画像に対応するインデックスのリスト（指定された場合、このインデックスをキーとして使用）
    Returns:
        検出結果の辞書 {インデックス: {'detections': [...], 'segments': [...]}, ...}
    """
    if model is None:
        model = get_yolo_model()

    results = {}
    total = len(image_paths)

    for i, img_path in enumerate(image_paths):
        if progress_callback:
            if not progress_callback(i, total, f"画像 {i+1}/{total} を処理中: {os.path.basename(img_path)}"):
                break

        result = detect_objects_and_segments(img_path, model, conf_threshold)
        # indicesが指定されている場合は対応するインデックスを使用
        key = indices[i] if indices is not None else i
        results[key] = result

    return results

def draw_detection_preview(image_path, detections, output_path=None):
    """
    検出結果のプレビュー画像を生成する
    Args:
        image_path: 元画像のパス
        detections: 検出結果のリスト
        output_path: 出力画像のパス
    Returns:
        描画された画像 (PIL.Image)
    """
    try:
        # 画像を開く
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # 各検出結果を描画
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # 色を決定（クラスによって変える）
            from hashlib import md5
            color_hash = int(md5(class_name.encode()).hexdigest(), 16) % 0xFFFFFF
            r = (color_hash & 0xFF0000) >> 16
            g = (color_hash & 0x00FF00) >> 8
            b = color_hash & 0x0000FF
            color = (r, g, b)
            
            # バウンディングボックスを描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # ラベルを描画
            label = f"{class_name} {confidence:.2f}"
            label_size = draw.textlength(label, font=None)
            draw.rectangle([x1, y1, x1 + label_size, y1 + 15], fill=color)
            draw.text([x1, y1], label, fill=(255, 255, 255))
        
        # 出力パスが指定されている場合は保存
        if output_path:
            img.save(output_path)
        
        return img
    
    except Exception as e:
        print(f"描画エラー: {e}")
        return None

# --- ここから追加するイメージラベルクラス ---

class ObjectDetectionImageLabel(QLabel):
    """物体検知用の画像ラベルクラス"""
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1000, 800)
        
        self.current_class = "traffic_cone"  # デフォルトクラス
        self.zoom_factor = 2.5  # 拡大率
        
        # 物体検知用の変数
        self.start_point = None
        self.current_point = None
        self.drawing = False
        self.boxes = []  # 描画したバウンディングボックスを保存
        self.selected_box = None  # 選択中のボックス
        self.dragging = False
        self.drag_start = None
        self.drag_corner = None  # リサイズ中のコーナー
        
        # 検出結果表示用
        self.detections = []
        self.show_detections = True
    
    def set_detections(self, detections):
        """検出結果を設定する"""
        self.detections = detections
        self.update()
    
    def toggle_detection_display(self):
        """検出結果の表示をトグル"""
        self.show_detections = not self.show_detections
        self.update()
    
    def set_current_class(self, class_name):
        """現在のクラスを設定する"""
        self.current_class = class_name
    
    def get_boxes(self):
        """描画したバウンディングボックスを取得する"""
        return self.boxes
    
    def clear_boxes(self):
        """バウンディングボックスをクリアする"""
        self.boxes = []
        self.selected_box = None
        self.update()
    
    def set_boxes(self, boxes):
        """バウンディングボックスを設定する"""
        self.boxes = boxes
        self.selected_box = None
        self.update()
        
    def mousePressEvent(self, event):
        if not self.pixmap() or not self.main_window:
            return
        
        # クリック位置を取得
        pos = event.pos()
        
        # 元の画像サイズ
        pix_width = self.pixmap().width()
        pix_height = self.pixmap().height()
        
        # ズーム係数を使用して拡大後のサイズを計算
        scaled_width = int(pix_width * self.zoom_factor)
        scaled_height = int(pix_height * self.zoom_factor)
        
        # 表示領域の計算
        x = (self.width() - scaled_width) // 2
        y = (self.height() - scaled_height) // 2
        target_rect = QRect(x, y, scaled_width, scaled_height)
        
        # クリック位置が画像内かチェック
        if not target_rect.contains(pos):
            return
        
        # 画像内の相対位置を計算
        rel_x = (pos.x() - target_rect.x()) / target_rect.width()
        rel_y = (pos.y() - target_rect.y()) / target_rect.height()
        
        # 元の画像の座標に変換
        orig_x = int(rel_x * pix_width)
        orig_y = int(rel_y * pix_height)
        
        # 既存のボックスの選択/操作をチェック
        for i, box in enumerate(self.boxes):
            box_class, (x1, y1, x2, y2) = box
            
            # ボックスの内部または境界上か判定
            is_inside = x1 <= orig_x <= x2 and y1 <= orig_y <= y2
            
            # コーナーの近くかチェック (リサイズ用)
            corner_size = 10
            corner_points = [
                ("tl", x1, y1), ("tr", x2, y1),
                ("bl", x1, y2), ("br", x2, y2)
            ]
            
            near_corner = None
            for corner_id, cx, cy in corner_points:
                if abs(orig_x - cx) <= corner_size and abs(orig_y - cy) <= corner_size:
                    near_corner = corner_id
                    break
            
            if near_corner:
                # コーナーをドラッグ開始（リサイズ）
                self.selected_box = i
                self.dragging = True
                self.drag_corner = near_corner
                self.drag_start = (orig_x, orig_y)
                self.update()
                return
            elif is_inside:
                # ボックス内部をクリック（選択または移動）
                if event.button() == Qt.LeftButton:
                    self.selected_box = i
                    self.dragging = True
                    self.drag_corner = None
                    self.drag_start = (orig_x, orig_y)
                    self.update()
                    return
                elif event.button() == Qt.RightButton and self.selected_box == i:
                    # 右クリックで選択中のボックスを削除
                    self.boxes.pop(i)
                    self.selected_box = None
                    self.update()
                    return
        
        # 新しいボックスの描画開始
        if event.button() == Qt.LeftButton:
            self.start_point = (orig_x, orig_y)
            self.current_point = (orig_x, orig_y)
            self.drawing = True
            self.selected_box = None
            self.update()
    
    def mouseMoveEvent(self, event):
        if not self.pixmap() or not self.main_window:
            return
        
        # 元の画像サイズ
        pix_width = self.pixmap().width()
        pix_height = self.pixmap().height()
        
        # ズーム係数を使用して拡大後のサイズを計算
        scaled_width = int(pix_width * self.zoom_factor)
        scaled_height = int(pix_height * self.zoom_factor)
        
        # 表示領域の計算
        x = (self.width() - scaled_width) // 2
        y = (self.height() - scaled_height) // 2
        target_rect = QRect(x, y, scaled_width, scaled_height)
        
        # クリック位置が画像内かチェック
        pos = event.pos()
        if not target_rect.contains(pos):
            return
        
        # 画像内の相対位置を計算
        rel_x = (pos.x() - target_rect.x()) / target_rect.width()
        rel_y = (pos.y() - target_rect.y()) / target_rect.height()
        
        # 元の画像の座標に変換
        orig_x = int(rel_x * pix_width)
        orig_y = int(rel_y * pix_height)
        
        # 範囲内に制限
        orig_x = max(0, min(orig_x, pix_width))
        orig_y = max(0, min(orig_y, pix_height))
        
        if self.drawing:
            # 描画中は現在位置を更新
            self.current_point = (orig_x, orig_y)
            self.update()
        elif self.dragging and self.selected_box is not None:
            # ドラッグ中はボックスを移動/リサイズ
            if self.drag_corner:
                # コーナーをドラッグしてリサイズ
                class_name, (x1, y1, x2, y2) = self.boxes[self.selected_box]
                
                if self.drag_corner == "tl":
                    x1, y1 = orig_x, orig_y
                elif self.drag_corner == "tr":
                    x2, y1 = orig_x, orig_y
                elif self.drag_corner == "bl":
                    x1, y2 = orig_x, orig_y
                elif self.drag_corner == "br":
                    x2, y2 = orig_x, orig_y
                
                # x1 < x2, y1 < y2 を保証
                if x1 > x2:
                    x1, x2 = x2, x1
                    if self.drag_corner in ["tl", "bl"]:
                        self.drag_corner = "tr" if self.drag_corner == "tl" else "br"
                    else:
                        self.drag_corner = "tl" if self.drag_corner == "tr" else "bl"
                
                if y1 > y2:
                    y1, y2 = y2, y1
                    if self.drag_corner in ["tl", "tr"]:
                        self.drag_corner = "bl" if self.drag_corner == "tl" else "br"
                    else:
                        self.drag_corner = "tl" if self.drag_corner == "bl" else "tr"
                
                self.boxes[self.selected_box] = (class_name, (x1, y1, x2, y2))
            else:
                # ボックス全体を移動
                class_name, (x1, y1, x2, y2) = self.boxes[self.selected_box]
                
                # 移動量を計算
                delta_x = orig_x - self.drag_start[0]
                delta_y = orig_y - self.drag_start[1]
                
                # 新しい座標を計算
                new_x1 = x1 + delta_x
                new_y1 = y1 + delta_y
                new_x2 = x2 + delta_x
                new_y2 = y2 + delta_y
                
                # 画像内に収まるように調整
                if new_x1 < 0:
                    new_x2 -= new_x1
                    new_x1 = 0
                elif new_x2 > pix_width:
                    new_x1 -= (new_x2 - pix_width)
                    new_x2 = pix_width
                
                if new_y1 < 0:
                    new_y2 -= new_y1
                    new_y1 = 0
                elif new_y2 > pix_height:
                    new_y1 -= (new_y2 - pix_height)
                    new_y2 = pix_height
                
                self.boxes[self.selected_box] = (class_name, (new_x1, new_y1, new_x2, new_y2))
                self.drag_start = (orig_x, orig_y)
            
            self.update()
    
    def mouseReleaseEvent(self, event):
        if not self.pixmap() or not self.main_window:
            return
        
        if self.drawing:
            # 描画終了
            self.drawing = False
            
            # 新しいボックスを追加（最小サイズチェック）
            if self.start_point and self.current_point:
                x1 = min(self.start_point[0], self.current_point[0])
                y1 = min(self.start_point[1], self.current_point[1])
                x2 = max(self.start_point[0], self.current_point[0])
                y2 = max(self.start_point[1], self.current_point[1])
                
                # 最小サイズチェック (10x10ピクセル以上)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.boxes.append((self.current_class, (x1, y1, x2, y2)))
                    
                    # コールバック呼び出し（アノテーション更新通知）
                    if self.main_window and hasattr(self.main_window, 'handle_object_annotation'):
                        self.main_window.handle_object_annotation()
            
            self.start_point = None
            self.current_point = None
        
        if self.dragging:
            # ドラッグ終了
            self.dragging = False
            self.drag_start = None
            self.drag_corner = None
            
            # コールバック呼び出し（アノテーション更新通知）
            if self.main_window and hasattr(self.main_window, 'handle_object_annotation'):
                self.main_window.handle_object_annotation()
        
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if not self.pixmap():
            painter = QPainter(self)
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "フォルダを選択し、読込ボタンを押してください")
            painter.end()
            return
        
        painter = QPainter(self)
        
        # 元の画像のサイズ
        pix_width = self.pixmap().width()
        pix_height = self.pixmap().height()
        
        # ズーム係数を使用して拡大後のサイズを計算
        scaled_width = int(pix_width * self.zoom_factor)
        scaled_height = int(pix_height * self.zoom_factor)
        
        # 中央に配置するための座標計算
        x = (self.width() - scaled_width) // 2
        y = (self.height() - scaled_height) // 2
        
        # 画像を拡大して描画
        target_rect = QRect(x, y, scaled_width, scaled_height)
        painter.drawPixmap(target_rect, self.pixmap())
        
        # 検出結果を描画
        if self.show_detections and self.detections:
            for det in self.detections:
                box = det['bbox']
                x1, y1, x2, y2 = box
                class_name = det['class']
                confidence = det['confidence']
                
                # 表示座標に変換
                x1_scaled = x + int(x1 * scaled_width / pix_width)
                y1_scaled = y + int(y1 * scaled_height / pix_height)
                x2_scaled = x + int(x2 * scaled_width / pix_width)
                y2_scaled = y + int(y2 * scaled_height / pix_height)
                
                # クラス別の色を決定
                from hashlib import md5
                color_hash = int(md5(class_name.encode()).hexdigest(), 16)
                r = (color_hash & 0xFF0000) >> 16
                g = (color_hash & 0x00FF00) >> 8
                b = color_hash & 0x0000FF
                color = QColor(r, g, b)
                
                # バウンディングボックスを描画
                painter.setPen(QPen(color, 2))
                painter.drawRect(x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled)
                
                # ラベルを描画
                label = f"{class_name} {confidence:.2f}"
                painter.setFont(QFont("Arial", 8))
                text_width = painter.fontMetrics().horizontalAdvance(label)
                
                # ラベル背景
                painter.fillRect(x1_scaled, y1_scaled - 18, text_width + 4, 18, color)
                
                # ラベルテキスト
                painter.setPen(QPen(Qt.white))
                painter.drawText(x1_scaled + 2, y1_scaled - 5, label)
        
        # 現在の描画中のボックスを描画
        if self.drawing and self.start_point and self.current_point:
            x1 = min(self.start_point[0], self.current_point[0])
            y1 = min(self.start_point[1], self.current_point[1])
            x2 = max(self.start_point[0], self.current_point[0])
            y2 = max(self.start_point[1], self.current_point[1])
            
            # 表示座標に変換
            x1_scaled = x + int(x1 * scaled_width / pix_width)
            y1_scaled = y + int(y1 * scaled_height / pix_height)
            x2_scaled = x + int(x2 * scaled_width / pix_width)
            y2_scaled = y + int(y2 * scaled_height / pix_height)
            
            # 描画中のボックスは点線で表示
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            painter.drawRect(x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled)
            
            # クラス名を表示
            painter.setFont(QFont("Arial", 8))
            painter.setPen(QPen(QColor(255, 255, 0)))
            painter.drawText(x1_scaled + 2, y1_scaled - 5, self.current_class)
        
        # 既存のバウンディングボックスを描画
        for i, box in enumerate(self.boxes):
            class_name, (x1, y1, x2, y2) = box
            
            # 表示座標に変換
            x1_scaled = x + int(x1 * scaled_width / pix_width)
            y1_scaled = y + int(y1 * scaled_height / pix_height)
            x2_scaled = x + int(x2 * scaled_width / pix_width)
            y2_scaled = y + int(y2 * scaled_height / pix_height)
            
            # クラス別の色を決定
            from hashlib import md5
            color_hash = int(md5(class_name.encode()).hexdigest(), 16)
            r = (color_hash & 0xFF0000) >> 16
            g = (color_hash & 0x00FF00) >> 8
            b = color_hash & 0x0000FF
            color = QColor(r, g, b)
            
            # 選択中のボックスは太線で表示
            if i == self.selected_box:
                painter.setPen(QPen(color, 3))
            else:
                painter.setPen(QPen(color, 2))
            
            painter.drawRect(x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled)
            
            # ラベルを描画
            painter.setFont(QFont("Arial", 8))
            
            # ラベル背景
            painter.fillRect(x1_scaled, y1_scaled - 18, 
                            painter.fontMetrics().horizontalAdvance(class_name) + 4, 18, color)
            
            # ラベルテキスト
            painter.setPen(QPen(Qt.white))
            painter.drawText(x1_scaled + 2, y1_scaled - 5, class_name)
            
            # 選択中のボックスにはコーナーポイントを表示
            if i == self.selected_box:
                corner_size = 6
                corner_points = [
                    (x1_scaled, y1_scaled),  # 左上
                    (x2_scaled, y1_scaled),  # 右上
                    (x1_scaled, y2_scaled),  # 左下
                    (x2_scaled, y2_scaled)   # 右下
                ]
                
                painter.setBrush(QBrush(Qt.white))
                for cx, cy in corner_points:
                    painter.drawRect(cx - corner_size // 2, cy - corner_size // 2, 
                                    corner_size, corner_size)
        
        painter.end()
