# enhanced_annotations.py
"""物体検知アノテーションの表示強化モジュール"""

import os
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QFrame, QGraphicsOpacityEffect)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QPoint

# 1. アノテーション情報ラベルを更新する関数
def update_annotation_info_label(self):
    """物体検知アノテーション情報を表示する"""
    if not self.images:
        return ""
        
    current_img_path = self.images[self.current_index]
    is_deleted = hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes
    
    # 物体検知アノテーション情報
    bbox_info = ""
    if not is_deleted and current_img_path in self.bbox_annotations and self.bbox_annotations[current_img_path]:
        bboxes = self.bbox_annotations[current_img_path]
        bbox_info = f"<b>物体検知アノテーション:</b><br>"
        
        # クラスごとのカウント辞書
        class_counts = {}
        for bbox in bboxes:
            class_name = bbox.get('class', 'unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # クラスカウント情報のフォーマット
        bbox_info += "検出オブジェクト:<br>"
        for class_name, count in class_counts.items():
            # このクラスの色を取得
            class_colors = {
                'car': "#FF0000",     # 赤
                'person': "#00FF00",  # 緑
                'sign': "#0000FF",    # 青
                'cone': "#FFFF00",    # 黄
                'unknown': "#808080"  # グレー
            }
            color = class_colors.get(class_name, "#FF0000")
            
            bbox_info += f"<span style='color: {color}; font-weight: bold;'>● {class_name}</span>: {count}個<br>"
        
        bbox_info += f"合計: {len(bboxes)}個のオブジェクト<br>"
    
    return bbox_info

# 2. display_current_imageメソッドを強化する関数
def enhanced_display_current_image(self):
    """物体検知アノテーション情報を含むよう画像表示を強化"""
    if not self.images:
        return
    
    current_img_path = self.images[self.current_index]
    is_deleted = hasattr(self, 'deleted_indexes') and self.current_index in self.deleted_indexes

    # スライダーの表示を更新
    self.slider_value_label.setText(f"{self.current_index + 1}/{len(self.images)}")

    # 画像情報表示の更新
    filename = os.path.basename(current_img_path)
    status_text = " [削除済み]" if is_deleted else ""
    self.current_image_info.setText(
        f"画像 {self.current_index + 1} of {len(self.images)}:{status_text}\n{filename}"
    )
    
    # 削除済みの場合は赤字で表示
    if is_deleted:
        self.current_image_info.setStyleSheet("color: #FF5555; font-weight: bold;")
    else:
        self.current_image_info.setStyleSheet("color: #333333; font-weight: bold;")
    
    # アノテーション情報の表示
    if self.current_index in self.annotations and not is_deleted:
        anno = self.annotations[self.current_index]
        
        # 基本的なアノテーション情報
        annotation_text = f"<b>運転アノテーション情報:</b><br>"
        annotation_text += f"angle = <span style='color: #FF6666;'>{anno['angle']:.4f}</span><br>"
        annotation_text += f"throttle = <span style='color: #FF6666;'>{anno['throttle']:.4f}</span>"
        
        # 位置情報があれば追加して強調表示
        if 'loc' in anno:
            loc_value = anno['loc']
            loc_color = get_location_color(loc_value)
            
            # 位置情報を色付きのバッジとして表示
            annotation_text += f"<br><div style='margin-top: 10px;'>"
            annotation_text += f"<div style='display: inline-block; background-color: {loc_color.name()}; color: white; font-weight: bold; padding: 5px; border-radius: 5px;'>"
            annotation_text += f"位置 {loc_value}</div></div>"
        
        # 物体検知アノテーション情報を追加
        bbox_info = update_annotation_info_label(self)
        if bbox_info:
            annotation_text += f"<br><br>{bbox_info}"
        
        # リッチテキストとして設定
        self.annotation_info_label.setText(annotation_text)
        self.annotation_info_label.setTextFormat(Qt.RichText)
    elif not is_deleted and current_img_path in self.bbox_annotations and self.bbox_annotations[current_img_path]:
        # 自動運転アノテーションはないが、物体検知アノテーションはある場合
        bbox_info = update_annotation_info_label(self)
        self.annotation_info_label.setText(bbox_info)
        self.annotation_info_label.setTextFormat(Qt.RichText)
    elif is_deleted:
        # 削除済みの場合のメッセージ
        self.annotation_info_label.setText(
            "<span style='color: #FF5555;'>この画像は削除済みです。<br>"
            "画像をクリックするか「削除状態を復元」ボタンを押して<br>"
            "再度アノテーションを行えます。</span>"
        )
        self.annotation_info_label.setTextFormat(Qt.RichText)
    else:
        self.annotation_info_label.setText("")
    
    # 位置情報ラベルの更新
    location_value = None
    if not is_deleted:
        # アノテーションの位置情報を確認
        if self.current_index in self.annotations and 'loc' in self.annotations[self.current_index]:
            location_value = self.annotations[self.current_index]['loc']
        # 位置情報専用の辞書を確認
        elif self.current_index in self.location_annotations:
            location_value = self.location_annotations[self.current_index]
    
    # 位置情報ラベルの更新
    if location_value is not None and not is_deleted:
        # 位置情報ラベルの更新（self.current_locationは更新しない）
        self.current_location_label.setText(f"現在の位置情報: {location_value}")
        
        # 位置情報に基づいた色を取得
        loc_color = get_location_color(location_value)
        self.current_location_label.setStyleSheet(f"color: {loc_color.name()}; font-weight: bold;")
        
        # ボタンの選択状態を更新
        for button in self.location_buttons:
            button_value = button.property("location_value")
            button.setChecked(button_value == location_value)
    else:
        # 位置情報がない場合
        self.current_location_label.setText("現在の位置情報: なし")
        self.current_location_label.setStyleSheet("")
        
        # すべてのボタンの選択を解除
        for button in self.location_buttons:
            button.setChecked(False)
    
    # 推論結果の表示も更新（現在の画像に推論結果がある場合）
    if self.inference_checkbox.isChecked() and not is_deleted:
        self.update_inference_display()
    else:
        self.inference_info_label.setText("")
    
    # 画像を読み込んで表示
    pixmap = QPixmap(current_img_path)
    if not pixmap.isNull():
        self.main_image_view.setPixmap(pixmap)
        
        # アノテーションポイントの設定
        if not is_deleted and self.current_index in self.annotations :
            anno = self.annotations[self.current_index]
            self.main_image_view.annotation_point = QPoint(anno['x'], anno['y'])
        else:
            self.main_image_view.annotation_point = None
        
        # 推論ポイントの設定
        if not is_deleted and self.inference_checkbox.isChecked() and self.current_index in self.inference_results:
            inference = self.inference_results[self.current_index]
            self.main_image_view.inference_point = QPoint(inference['x'], inference['y'])
        else:
            self.main_image_view.inference_point = None
        
        # 削除済みの場合
        if is_deleted:
            # 削除済みフラグを設定
            self.main_image_view.is_deleted = True
        else:
            self.main_image_view.is_deleted = False
            
        # UIを更新
        self.main_image_view.update()

# 3. 強化されたサムネイルウィジェットクラス
class EnhancedThumbnailWidget(QWidget):
    def __init__(self, parent=None, img_path="", index=0, is_selected=False, 
                 annotation=None, on_click=None, location_value=None, is_deleted=False,
                 bbox_annotations=None):  # bbox_annotations parameter added
        super().__init__(parent)
        self.img_path = img_path
        self.index = index
        self.on_click = on_click
        self.is_selected = is_selected
        self.annotation = annotation  # アノテーション情報
        self.location_value = location_value  # 変更: 辞書ではなく直接位置情報の値を受け取る
        self.is_deleted = is_deleted  # 削除済みフラグ
        self.bbox_annotations = bbox_annotations  # 物体検知アノテーション情報を追加
        
        # サムネイル全体のサイズも調整
        self.setMinimumWidth(210)
        self.setMinimumHeight(170)  # 高さを少し小さく
        
        # メインレイアウト（水平レイアウト）
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # マージンをなくす
        self.layout.setSpacing(1)  # 最小限のスペーシング
        
        # 左側の情報パネル
        info_panel = QWidget()
        info_panel.setFixedWidth(70)  # 情報パネル幅
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(1)
        
        # インデックス番号
        self.idx_label = QLabel(f"{index + 1}")
        self.idx_label.setAlignment(Qt.AlignCenter)
        self.idx_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.idx_label)
        
        # 削除済みバッジ（削除されている場合）
        if is_deleted:
            deleted_badge = QLabel("削除済")
            deleted_badge.setAlignment(Qt.AlignCenter)
            deleted_badge.setStyleSheet("""
                background-color: #FF5555;
                color: white;
                font-weight: bold;
                border-radius: 10px;
                min-width: 20px;
                min-height: 20px;
                padding: 1px;
            """)
            info_layout.addWidget(deleted_badge)
        # アノテーション情報
        elif annotation and not is_deleted:
            angle_label = QLabel(f"A: {annotation.get('angle', 0):.2f}")
            angle_label.setStyleSheet("color: #FF6666; font-size: 12px;font-weight: bold;")
            info_layout.addWidget(angle_label)
            
            throttle_label = QLabel(f"T: {annotation.get('throttle', 0):.2f}")
            throttle_label.setStyleSheet("color: #FF6666; font-size: 12px;font-weight: bold;")
            info_layout.addWidget(throttle_label)

            # 位置情報バッジ（位置情報がある場合）
            # 変更: 辞書からの参照ではなく、直接location_valueを使用
            if location_value is not None:
                loc_color = get_location_color(location_value)
                
                loc_badge = QLabel(str(location_value))
                loc_badge.setAlignment(Qt.AlignCenter)
                loc_badge.setStyleSheet(f"""
                    background-color: {loc_color.name()};
                    color: white;
                    font-weight: bold;
                    border-radius: 10px;
                    min-width: 20px;
                    min-height: 20px;
                    padding: 1px;
                """)
                info_layout.addWidget(loc_badge)
        
        # 物体検知アノテーション情報を追加 (新規追加)
        if bbox_annotations and not is_deleted:
            # オブジェクト数を表示するバッジ
            obj_count = len(bbox_annotations)
            bbox_badge = QLabel(f"物体: {obj_count}")
            bbox_badge.setAlignment(Qt.AlignCenter)
            bbox_badge.setStyleSheet("""
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 10px;
                min-width: 20px;
                min-height: 20px;
                padding: 1px;
                font-size: 10px;
            """)
            info_layout.addWidget(bbox_badge)
            
            # クラスごとのカウントを集計
            class_counts = {}
            for bbox in bbox_annotations:
                class_name = bbox.get('class', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # 主要なクラスを最大2つまで表示
            for i, (class_name, count) in enumerate(class_counts.items()):
                if i >= 2:  # 最大2クラスまで表示
                    break
                    
                class_label = QLabel(f"{class_name}: {count}")
                class_label.setStyleSheet("font-size: 10px; color: #333;")
                info_layout.addWidget(class_label)
        
        # 残りのスペースを埋めるスペーサー
        info_layout.addStretch()
        
        # 左側の情報パネルをメインレイアウトに追加
        self.layout.addWidget(info_panel)
        
        # 右側の画像パネル
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)  # スペーシングをなくす
        
        # ファイル名ラベルを画像の上部に配置
        filename = os.path.basename(img_path)
        if len(filename) > 20:  # ファイル名が長い場合は切り詰める
            filename = filename[:18] + "..."
        
        name_label = QLabel(filename)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-size: 12px; color: #444444; background-color: #f8f8f8;font-weight: bold;")
        name_label.setFixedHeight(10)  # 高さを最小限に
        image_layout.addWidget(name_label)
        
        # 画像コンテナ（枠を付けるための外側のコンテナ）
        image_container = QFrame()
        
        # ボーダーのスタイル設定
        border_style = ""
        if is_selected:
            border_style = "border: 2px solid red;"
        elif is_deleted:
            border_style = "border: 2px solid #FF5555;"  # 削除済みは赤い枠線
        elif location_value is not None:
            loc_color = get_location_color(location_value)
            border_style = f"border: 2px solid {loc_color.name()};"
        elif annotation:
            border_style = "border: 2px solid #FF9966;"  # アノテーションのみはオレンジ系
        elif bbox_annotations:  # 物体検知アノテーションがある場合は青い枠線
            border_style = "border: 2px solid #2196F3;"
        else:
            border_style = "border: 1px solid #dddddd;"
        
        # 画像コンテナのレイアウト - マージンを完全に削除
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)  # 余白なし
        image_container_layout.setSpacing(0)  # スペーシングなし
        
        # 画像ラベル
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(150, 140)  # 幅を少し広げる
        
        # 削除済みの場合は半透明になるスタイルを追加
        if is_deleted:
            self.img_label.setStyleSheet(f"{border_style} padding: 0px; opacity: 0.5;")
        else:
            self.img_label.setStyleSheet(f"{border_style} padding: 0px;")
            
        image_container_layout.addWidget(self.img_label)
        
        # 画像コンテナをイメージパネルに追加
        image_layout.addWidget(image_container)
        
        # 画像パネルをメインレイアウトに追加
        self.layout.addWidget(image_panel)
        
        # 画像を読み込む
        self.load_image(img_path)
        
        # ウィジェット全体の枠線はなし
        self.setStyleSheet("border: none;")

    def mousePressEvent(self, event):
        # クリック時にon_click関数を呼び出す
        if self.on_click and event.button() == Qt.LeftButton:
            self.on_click(self.index)

    # def load_image(self, img_path):
    #     if not os.path.exists(img_path):
    #         return
        
    #     try:
    #         # PILで画像を開く
    #         pil_img = Image.open(img_path)
            
    #         # 画像のコピーを作成して描画する
    #         draw_img = pil_img.copy()
    #         draw = ImageDraw.Draw(draw_img)
            
    #         # 基本的なアノテーションを描画（以前のコード部分）
    #         if self.annotation and not self.is_deleted:
    #             # アノテーションの座標を取得
    #             x, y = self.annotation["x"], self.annotation["y"]
                
    #             # 丸を描画
    #             circle_size = 15  # サムネイル用の丸のサイズを大きく
    #             draw.ellipse((x-circle_size, y-circle_size, x+circle_size, y+circle_size), 
    #                          outline='red', width=4)
            
    #         # 物体検知アノテーションがある場合は矩形を描画 (新規追加)
    #         if self.bbox_annotations and not self.is_deleted:
    #             img_width, img_height = pil_img.size
                
    #             for bbox in self.bbox_annotations:
    #                 # クラスに応じた色を定義
    #                 class_colors = {
    #                     'car': (255, 0, 0),      # 赤
    #                     'person': (0, 255, 0),   # 緑
    #                     'sign': (0, 0, 255),     # 青
    #                     'cone': (255, 255, 0),   # 黄
    #                     'unknown': (128, 128, 128)  # グレー
    #                 }
                    
    #                 class_name = bbox.get('class', 'unknown')
    #                 color = class_colors.get(class_name, (255, 0, 0))
                    
    #                 # 正規化された座標を実際の座標に変換
    #                 x1 = int(bbox['x1'] * img_width)
    #                 y1 = int(bbox['y1'] * img_height)
    #                 x2 = int(bbox['x2'] * img_width)
    #                 y2 = int(bbox['y2'] * img_height)
                    
    #                 # 矩形を描画
    #                 draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
    #                 # ラベルを表示 (サムネイルでは小さいのでクラス名の1文字目だけ表示)
    #                 label_text = class_name[0].upper()  # 頭文字のみ
                    
    #                 # ラベル背景
    #                 text_size = 10  # 大まかなテキストサイズ（PILのfont.getsize()の代替）
    #                 label_bg = (x1, y1-text_size, x1+text_size, y1)
    #                 draw.rectangle(label_bg, fill=color)
                    
    #                 # テキスト描画
    #                 draw.text((x1+2, y1-text_size), label_text, fill=(255, 255, 255))
            
    #         # 画像をQImageに変換
    #         draw_img = draw_img.convert("RGBA")
    #         data = draw_img.tobytes("raw", "RGBA")
    #         qimg = QImage(data, draw_img.width, draw_img.height, QImage.Format_RGBA8888)
            
    #         # QImageをQPixmapに変換してサムネイルに設定
    #         pixmap = QPixmap.fromImage(qimg)
            
    #         if not pixmap.isNull():
    #             pixmap = pixmap.scaled(170, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #             self.img_label.setPixmap(pixmap)
                
    #             # 削除済みの場合は半透明にする追加の処理
    #             if self.is_deleted:
    #                 self.setGraphicsEffect(QGraphicsOpacityEffect(opacity=0.5))
            
    #     except Exception as e:
    #         print(f"Error loading image {img_path}: {e}")

    def load_image(self, img_path):
        if not os.path.exists(img_path):
            return
        
        try:
            # PILで画像を開く
            pil_img = Image.open(img_path)
            
            # 画像のコピーを作成して描画する
            draw_img = pil_img.copy()
            draw = ImageDraw.Draw(draw_img)
            
            # 基本的なアノテーションを描画
            if self.annotation and not self.is_deleted:
                # アノテーションの座標を取得
                x, y = self.annotation["x"], self.annotation["y"]
                
                # 丸を描画
                circle_size = 15  # サムネイル用の丸のサイズ
                draw.ellipse((x-circle_size, y-circle_size, x+circle_size, y+circle_size), 
                            outline='red', width=4)
            
            # 物体検知アノテーションがある場合は矩形を描画
            if self.bbox_annotations and not self.is_deleted:
                img_width, img_height = pil_img.size
                
                for bbox in self.bbox_annotations:
                    # クラスに応じた色を定義
                    class_colors = {
                        'car': (255, 0, 0),      # 赤
                        'person': (0, 255, 0),   # 緑
                        'sign': (0, 0, 255),     # 青
                        'cone': (255, 255, 0),   # 黄
                        'unknown': (128, 128, 128)  # グレー
                    }
                    
                    class_name = bbox.get('class', 'unknown')
                    color = class_colors.get(class_name, (255, 0, 0))
                    
                    # 正規化された座標を実際の座標に変換
                    x1 = int(bbox['x1'] * img_width)
                    y1 = int(bbox['y1'] * img_height)
                    x2 = int(bbox['x2'] * img_width)
                    y2 = int(bbox['y2'] * img_height)
                    
                    # 矩形を描画
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # ラベルを表示 (サムネイルでは小さいのでクラス名の1文字目だけ表示)
                    label_text = class_name[0].upper()  # 頭文字のみ
                    
                    # ラベル背景
                    text_size = 10  # 大まかなテキストサイズ
                    label_bg = (x1, y1-text_size, x1+text_size, y1)
                    draw.rectangle(label_bg, fill=color)
                    
                    # テキスト描画
                    draw.text((x1+2, y1-text_size), label_text, fill=(255, 255, 255))
            
            # 画像をQImageに変換
            draw_img = draw_img.convert("RGBA")
            data = draw_img.tobytes("raw", "RGBA")
            qimg = QImage(data, draw_img.width, draw_img.height, QImage.Format_RGBA8888)
            
            # QImageをQPixmapに変換してサムネイルに設定
            pixmap = QPixmap.fromImage(qimg)
            
            if not pixmap.isNull():
                # 画像ラベルのサイズを取得
                label_width = self.img_label.width()
                label_height = self.img_label.height()
                
                # サイズが0の場合は固定サイズを使用
                if label_width == 0 or label_height == 0:
                    label_width = 150
                    label_height = 140
                
                # 重要な変更: ラベルサイズと同じサイズでスケーリング
                scaled_pixmap = pixmap.scaled(
                    label_width, 
                    label_height,
                    Qt.KeepAspectRatio,  # アスペクト比を維持
                    Qt.SmoothTransformation  # 滑らかな変換
                )
                
                self.img_label.setPixmap(scaled_pixmap)
                self.img_label.setAlignment(Qt.AlignCenter)  # 中央揃え
                
                # 削除済みの場合は半透明にする追加の処理
                if self.is_deleted:
                    self.setGraphicsEffect(QGraphicsOpacityEffect(opacity=0.5))
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# 4. ギャラリー更新メソッドを強化する関数
def enhanced_update_gallery(self):
    """物体検知アノテーション対応のギャラリー更新"""
    # 現在のギャラリーをクリア
    for i in reversed(range(self.gallery_layout.count())): 
        self.gallery_layout.itemAt(i).widget().deleteLater()
    
    if not self.images:
        return
    
    # スキップ枚数を取得
    skip_count = self.skip_count_spin.value()
    
    # 現在の位置とインデックス情報
    current_idx = self.current_index
    total_images = len(self.images)
    
    # 前に表示する2枚の画像インデックスを計算
    prev_indices = []
    for i in range(1, 3):
        idx = current_idx - i * skip_count
        if idx >= 0:
            prev_indices.append(idx)
    prev_indices.reverse()  # 近い順に並べる
    
    # 次に表示する2枚の画像インデックスを計算
    next_indices = []
    for i in range(1, 3):
        idx = current_idx + i * skip_count
        if idx < total_images:
            next_indices.append(idx)
    
    # 表示する画像インデックスを組み合わせる
    display_indices = prev_indices + [current_idx] + next_indices
    
    # ギャラリーのグリッドレイアウトを調整
    col_count = 5  # 一行あたりの列数
    
    # ギャラリーにサムネイルを追加
    for i, idx in enumerate(display_indices):
        if 0 <= idx < total_images:
            img_path = self.images[idx]
            
            # 削除されたインデックスの場合、削除済みフラグをセット
            is_deleted = hasattr(self, 'deleted_indexes') and idx in self.deleted_indexes
            
            # アノテーション情報を取得
            annotation = None
            location_value = None  # Initialize here to prevent the error
            
            if not is_deleted and idx in self.annotations:
                annotation = self.annotations[idx]
            
                # 位置情報を事前に特定
                if annotation and 'loc' in annotation:
                    location_value = annotation['loc']

            # 物体検知アノテーションを取得 (新規追加)
            bbox_annotations = None
            if not is_deleted and img_path in self.bbox_annotations:
                bbox_annotations = self.bbox_annotations[img_path]
            
            # 拡張サムネイルウィジェットを作成
            thumb = EnhancedThumbnailWidget(
                img_path=img_path,
                index=idx,
                is_selected=(idx == current_idx),
                annotation=annotation,
                on_click=self.select_image,
                location_value=location_value,
                is_deleted=is_deleted,
                bbox_annotations=bbox_annotations  # 物体検知アノテーションを追加
            )
            
            # col_count列のグリッドで配置
            row = i // col_count
            col = i % col_count
            
            self.gallery_layout.addWidget(thumb, row, col)

# 5. すべての機能強化を適用する関数
def apply_enhanced_annotations_display(self):
    """物体検知アノテーション表示強化を適用する"""
    # 1. display_current_imageメソッドを置き換え
    self._original_display_current_image = self.display_current_image
    self.display_current_image = lambda: enhanced_display_current_image(self)
    
    # 2. update_galleryメソッドを置き換え
    self._original_update_gallery = self.update_gallery
    self.update_gallery = lambda: enhanced_update_gallery(self)
    
    # 3. アノテーション情報更新関数を保存しておく
    self.update_annotation_info_label = lambda: update_annotation_info_label(self)
    
    # 4. 初期更新
    self.display_current_image()
    self.update_gallery()
    
    print("物体検知アノテーション表示機能が有効化されました。")

# get_location_color関数が必要なので、念のため定義しておく
def get_location_color(location_value):
    """位置情報の値から色を取得する"""
    # カラーリストを作成（8つの明確に異なる色）
    colors = [
        QColor(255, 0, 0),      # 赤
        QColor(0, 150, 0),      # 緑
        QColor(0, 0, 255),      # 青
        QColor(255, 165, 0),    # オレンジ
        QColor(128, 0, 128),    # 紫
        QColor(0, 128, 128),    # ティール
        QColor(255, 0, 255),    # マゼンタ
        QColor(128, 128, 0)     # オリーブ
    ]
    
    # 位置情報の値に基づいて色を選択（8で割った余りを使用）
    if location_value is None:
        return QColor(200, 200, 200)  # グレー（位置情報なし）
    
    # 色インデックスを取得（0〜7の範囲）
    color_index = location_value % 8
    return colors[color_index]
