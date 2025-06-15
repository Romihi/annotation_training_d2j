#未実装
class ImageAnnotationTool(QMainWindow):
    # UI関連
    def toggle_theme(self, theme_name):
        """テーマを切り替える"""
        # styles.pyのset_theme関数を呼び出す
        current_theme = set_theme(theme_name)
        
        # 1. まずテーマ切り替えボタンの状態とスタイルを更新
        self.light_theme_button.setChecked(current_theme == "light")
        self.dark_theme_button.setChecked(current_theme == "dark")
        self.update_theme_button_styles()
        
        # 2. テーマに応じた背景色と文字色の基本設定
        self.adjust_base_colors_for_theme()
        
        # 3. 各ウィジェットにスタイルを適用
        self.apply_theme_to_widgets()
        
        # テーマ変更の通知
        theme_display_name = "ライト" if theme_name == "light" else "ダーク"
        self.statusBar().showMessage(f"テーマを {theme_display_name} に変更しました", 3000)

    def adjust_base_colors_for_theme(self):
        """テーマに応じた基本的な背景色と文字色を設定する"""
        current_theme = get_current_theme()
        
        if current_theme == "dark":
            # ダークモードの背景色を設定
            self.setStyleSheet("background-color: #1F2937;")
        else:
            # ライトモードの場合はスタイルシートをクリア
            self.setStyleSheet("")


    def update_theme_button_styles(self):
        """テーマ切り替えボタンのスタイルを現在のテーマに合わせて更新"""
        current_theme = get_current_theme()
        
        # ライトモードボタンのスタイル
        light_button_style = """
            QPushButton {
                background-color: #F9FAFB; /* ライトテーマの背景色 */
                color: #111827; /* 暗いテキスト */
                font-weight: bold;
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:checked {
                background-color: #E5E7EB; /* 選択時は少し暗い */
                border: 2px solid #2563EB; /* 選択時は青いボーダー */
            }
        """
        
        # ダークモードボタンのスタイル
        dark_button_style = """
            QPushButton {
                background-color: #1F2937; /* ダークテーマの背景色 */
                color: white; /* 白いテキスト */
                font-weight: bold;
                border: 1px solid #4B5563;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:checked {
                background-color: #374151; /* 選択時は少し明るい */
                border: 2px solid #3B82F6; /* 選択時は青いボーダー */
            }
        """
        
        # 明示的にスタイルを設定（テーマに関係なく）
        self.light_theme_button.setStyleSheet(light_button_style)
        self.dark_theme_button.setStyleSheet(dark_button_style)
        
        # チェック状態を再設定（現在のテーマに合わせて）
        self.light_theme_button.setChecked(current_theme == "light")
        self.dark_theme_button.setChecked(current_theme == "dark")

    def apply_theme_to_widgets(self):
        """テーマ変更後に主要なウィジェットにスタイルを再適用する"""
        # 基本ボタン
        if hasattr(self, 'load_button'):
            apply_style(self.load_button, 'primary')
        if hasattr(self, 'load_annotation_button'):
            apply_style(self.load_annotation_button, 'primary')
        
        # モデル関連ボタン
        if hasattr(self, 'model_refresh_button'):
            apply_style(self.model_refresh_button, 'model')
        if hasattr(self, 'model_load_button'):
            apply_style(self.model_load_button, 'model')
            
        # 学習関連ボタン
        if hasattr(self, 'train_model_button'):
            apply_style(self.train_model_button, 'training')
        
        # オートアノテーション関連
        if hasattr(self, 'auto_annotate_button'):
            apply_style(self.auto_annotate_button, 'training')
        if hasattr(self, 'batch_inference_button'):
            apply_style(self.batch_inference_button, 'special')
        
        # YOLO関連ボタン
        if hasattr(self, 'yolo_refresh_button'):
            apply_style(self.yolo_refresh_button, 'model')
        if hasattr(self, 'yolo_load_button'):
            apply_style(self.yolo_load_button, 'model')
        if hasattr(self, 'train_yolo_button'):
            apply_style(self.train_yolo_button, 'training')
        if hasattr(self, 'load_yolo_btn'):
            apply_style(self.load_yolo_btn, 'primary')
        
        # 位置情報関連ボタン
        if hasattr(self, 'location_refresh_button'):
            apply_style(self.location_refresh_button, 'model')
        if hasattr(self, 'location_load_button'):
            apply_style(self.location_load_button, 'model')
        if hasattr(self, 'train_location_button'):
            apply_style(self.train_location_button, 'training')
        
        # エクスポート関連ボタン
        for button_name in ['donkey_btn', 'jetracer_btn', 'yolo_btn']:
            if hasattr(self, button_name):
                apply_style(getattr(self, button_name), 'export')
        
        # 動画作成ボタン
        if hasattr(self, 'create_video_button'):
            apply_style(self.create_video_button, 'export')
        
        # MLflow関連ボタン
        if hasattr(self, 'mlflow_compare_button'):
            apply_style(self.mlflow_compare_button, 'special')
        
        # 削除関連ボタン
        if hasattr(self, 'delete_current_button'):
            apply_style(self.delete_current_button, 'destructive')
        if hasattr(self, 'clip_button'):
            apply_style(self.clip_button, 'destructive')
        
        # 復元ボタン
        if hasattr(self, 'restore_button'):
            apply_style(self.restore_button, 'primary')
        if hasattr(self, 'restore_all_button'):
            apply_style(self.restore_all_button, 'primary')
        
        # ナビゲーションボタン
        for nav_button in ['prev_button', 'next_button', 'prev_multi_button', 'next_multi_button']:
            if hasattr(self, nav_button):
                apply_style(getattr(self, nav_button), 'nav')
        
        # 再生ボタン
        for play_button in ['play_button', 'reverse_play_button']:
            if hasattr(self, play_button):
                apply_style(getattr(self, play_button), 'nav')
        
        # アノテーションモードボタン
        if hasattr(self, 'auto_mode_button'):
            # 選択状態に応じてスタイルを変更しない（カスタムスタイルシートが適用されているため）
            pass
        if hasattr(self, 'detection_mode_button'):
            # 選択状態に応じてスタイルを変更しない（カスタムスタイルシートが適用されているため）
            pass
        
        # フォルダブラウズボタン
        if hasattr(self, 'browse_button'):
            apply_style(self.browse_button, 'primary')
        
        # グループボックス
        for group_box in self.findChildren(QGroupBox):
            apply_style(group_box, 'group_box')
        
        # コンボボックス
        for combo_box in self.findChildren(QComboBox):
            apply_style(combo_box, 'combo_box')
        
        # スピンボックス
        for spin_box in self.findChildren(QSpinBox):
            apply_style(spin_box, 'spin_box')
        for double_spin_box in self.findChildren(QDoubleSpinBox):
            apply_style(double_spin_box, 'spin_box')
        
        # チェックボックス
        for checkbox in self.findChildren(QCheckBox):
            apply_style(checkbox, 'checkbox')
        
        # ラジオボタン
        for radio_button in self.findChildren(QRadioButton):
            apply_style(radio_button, 'radio')
        
        # スライダー
        for slider in self.findChildren(QSlider):
            apply_style(slider, 'slider')
        
        # テキスト入力
        for line_edit in self.findChildren(QLineEdit):
            apply_style(line_edit, 'text_input')
        
        # スクロールエリア
        for scroll_area in self.findChildren(QScrollArea):
            apply_style(scroll_area, 'scroll')
        
        # テーマ切り替えボタンのスタイルを更新
        self.update_theme_button_styles()
        
        # テーマに応じた色調整
        # self.adjust_text_colors_for_theme()

    def adjust_text_colors_for_theme(self):
        """テーマに応じて文字色と背景色を調整する"""
        current_theme = get_current_theme()
        
        if current_theme == "dark":
            # ダークモードの場合
            text_color = "white"
            secondary_text_color = "#D1D5DB"  # 薄い白/グレー
            
            # ダークモードの時だけ背景色を設定
            self.setStyleSheet("background-color: #1F2937;")
            
            # すべてのQWidgetとその子クラスを対象に文字色を設定
            # QLabel
            for label in self.findChildren(QLabel):
                # すでに特別なスタイルが適用されている可能性があるラベルをスキップ
                skip_labels = ["detection_inference_info_label", "inference_info_label", "location_inference_info_label"]
                if hasattr(label, 'objectName') and label.objectName() in skip_labels:
                    continue
                    
                # スタイルを取得し、文字色を更新
                current_style = label.styleSheet()
                new_style = current_style
                
                # 文字色指定を追加/更新
                if "color:" in current_style:
                    # すでに色指定がある場合は置換
                    new_style = re.sub(r'color:\s*[^;]+;', f"color: {text_color};", current_style)
                else:
                    # 色指定がない場合は追加
                    new_style = f"{current_style}; color: {text_color};"
                
                label.setStyleSheet(new_style)
            
            # QCheckBox
            for checkbox in self.findChildren(QCheckBox):
                checkbox.setStyleSheet(f"color: {text_color};")
            
            # QRadioButton
            for radio in self.findChildren(QRadioButton):
                radio.setStyleSheet(f"color: {text_color};")
            
            # QGroupBox
            for group in self.findChildren(QGroupBox):
                group.setStyleSheet(f"color: {text_color}; border: 1px solid #4B5563;")
            
            # QComboBox
            for combo in self.findChildren(QComboBox):
                combo.setStyleSheet(f"color: {text_color}; background-color: #374151; selection-background-color: #4B5563;")
            
            # QLineEdit
            for edit in self.findChildren(QLineEdit):
                edit.setStyleSheet(f"color: {text_color}; background-color: #374151; border: 1px solid #4B5563;")
            
            # QSpinBox と QDoubleSpinBox
            for spin in self.findChildren(QSpinBox):
                spin.setStyleSheet(f"color: {text_color}; background-color: #374151; border: 1px solid #4B5563;")
            
            for dspin in self.findChildren(QDoubleSpinBox):
                dspin.setStyleSheet(f"color: {text_color}; background-color: #374151; border: 1px solid #4B5563;")
            
            # タブウィジェット（存在する場合）
            for tab in self.findChildren(QTabWidget):
                tab.setStyleSheet(f"color: {text_color}; background-color: #1F2937;")
            
            # スライダー（存在する場合）
            for slider in self.findChildren(QSlider):
                slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        border: 1px solid #4B5563;
                        height: 8px;
                        background: #374151;
                        margin: 2px 0;
                        border-radius: 4px;
                    }
                    QSlider::handle:horizontal {
                        background: #3B82F6;
                        border: 1px solid #3B82F6;
                        width: 18px;
                        height: 18px;
                        margin: -6px 0;
                        border-radius: 9px;
                    }
                """)
        else:
            # ライトモードの場合
            # スタイルシートをクリアして、デフォルトに戻す
            self.setStyleSheet("")
            
            # テーマ切り替えボタン以外のウィジェットのスタイルをデフォルトに戻す
            excluded_widgets = [self.light_theme_button, self.dark_theme_button]
            for widget in self.findChildren(QWidget):
                if widget not in excluded_widgets:
                    widget.setStyleSheet("")
            
        # テーマ切り替えボタンのスタイルを更新
        self.update_theme_button_styles()
