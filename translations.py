# translations.py
"""
多言語対応のための翻訳辞書モジュール

===========================================
【編集ガイド】
===========================================
1. 新しいキーを追加する場合は、必ず ja と en 両方に追加
2. 動的値は {0}, {1} で指定
3. 複数行のテキストは \n で改行

【キー命名規則 - プレフィックス一覧】
-----------------------------------------
| プレフィックス | 用途                    | 例                          |
|---------------|-------------------------|----------------------------|
| app_          | アプリ・ウィンドウ設定    | app_title, app_language    |
| section_      | セクションタイトル        | section_data_load          |
| btn_          | ボタンラベル             | btn_browse, btn_save       |
| label_        | ラベル・説明文           | label_annotated_count      |
| tip_          | ツールチップ             | tip_load_model             |
| placeholder_  | プレースホルダー         | placeholder_folder         |
| chk_          | チェックボックス         | chk_dark_mode              |
| dlg_          | ダイアログタイトル        | dlg_warning, dlg_error     |
| msg_          | メッセージ（ダイアログ等）| msg_no_images              |
| status_       | ステータスバー           | status_deleted             |
-----------------------------------------

使用方法:
    from translations import get_text, set_language, get_current_language

    # テキスト取得
    button.setText(get_text('btn_load_images'))

    # フォーマット付き
    label.setText(get_text('label_annotated_count', 10, 100))

    # 言語切替
    set_language('en')
"""

from config import LANGUAGE

# 現在の言語設定（モジュールレベル）
_current_language = LANGUAGE

def get_current_language():
    """現在の言語設定を取得"""
    return _current_language

def set_language(lang: str):
    """言語を設定（'ja' または 'en'）"""
    global _current_language
    if lang in TRANSLATIONS:
        _current_language = lang
    else:
        raise ValueError(f"Unsupported language: {lang}. Available: {list(TRANSLATIONS.keys())}")

def get_text(key: str, *args, **kwargs) -> str:
    """
    翻訳テキストを取得

    Args:
        key: 翻訳キー
        *args, **kwargs: フォーマット用の引数

    Returns:
        翻訳されたテキスト。キーが見つからない場合はキー自体を返す

    Example:
        get_text('label_annotated_count', 10, 100)  # "アノテーション済み: 10 / 100"
    """
    lang = _current_language

    # 現在の言語で取得を試みる
    if lang in TRANSLATIONS and key in TRANSLATIONS[lang]:
        text = TRANSLATIONS[lang][key]
    # フォールバック: 日本語
    elif key in TRANSLATIONS.get('ja', {}):
        text = TRANSLATIONS['ja'][key]
    # キーが見つからない場合はキー自体を返す
    else:
        return key

    # フォーマット引数がある場合は適用
    if args or kwargs:
        try:
            return text.format(*args, **kwargs)
        except (IndexError, KeyError):
            return text
    return text

def t(key: str, *args, **kwargs) -> str:
    """get_text のエイリアス（短縮形）"""
    return get_text(key, *args, **kwargs)


# =============================================================================
# 翻訳辞書
# =============================================================================

TRANSLATIONS = {
    # =========================================================================
    # 日本語 (Japanese)
    # =========================================================================
    'ja': {
        # =====================================================================
        # app_ : アプリケーション・ウィンドウ設定
        # =====================================================================
        'app_title': '画像アノテーションツール',
        'app_language': '言語',
        'app_language_ja': '日本語',
        'app_language_en': 'English',
        'app_language_switch': '言語切替',
        'app_language_changed': '言語設定を変更しました。再起動後に反映されます。',
        'app_restart_required': '再起動が必要です',

        # =====================================================================
        # section_ : セクションタイトル
        # =====================================================================
        'section_data_load': 'データ読込（imagesフォルダの親フォルダ:',
        'section_save_annotation': 'アノテーションデータ保存:',
        'section_pilot_model': '自動運転モデル:',
        'section_object_detection': '物体検知・セグメンテーションモデル:',
        'section_model_management': 'モデル管理やクラウド学習:',
        'section_display_settings': '表示設定:',

        # =====================================================================
        # btn_ : ボタンラベル
        # =====================================================================
        # --- データ読込 ---
        'btn_browse': '参照...',
        'btn_load_images': '画像読込',
        'btn_load_annotations': 'アノテーション読込',

        # --- エクスポート ---
        'btn_create_video': 'アノテーション動画作成',

        # --- 自動運転モデル ---
        'btn_train_save': '学習・保存',
        'btn_load_model': 'モデル読込',
        'btn_auto_annotate': 'オートアノテーション実行',
        'btn_batch_inference': '全画像を推論',

        # --- 物体検知 ---
        'btn_load_yolo_annotation': 'YOLOアノテーション読込',
        'btn_preset': 'プリセット',
        'btn_apply': '反映',
        'btn_train_yolo': '学習・保存',
        'btn_yolo_auto_annotate': 'YOLO オートアノテーション',

        # --- モデル管理 ---
        'btn_open_mlflow': 'MLflowを開く',
        'btn_open_databricks': 'Databricksを開く',
        'btn_sync': '同期',
        'btn_transfer': '転送',
        'btn_settings': '設定',
        'btn_open_colab': 'Colabを開く',
        'btn_download': '取得',

        # --- 表示設定 ---
        'btn_window_font_settings': 'ウィンドウ・フォントサイズ設定',

        # --- ナビゲーション ---
        'btn_reverse_play': '◀逆再生',
        'btn_forward_play': '▶再生',
        'btn_delete_current': '現在のアノテーション削除',
        'btn_restore_deleted': '削除状態を復元',
        'btn_restore_all_deleted': '全ての削除状態を復元',
        'btn_current_position': '現在位置',
        'btn_range_delete': '範囲削除',

        # --- ダウンサンプリング ---
        'btn_detect': '検出',
        'btn_redetect': '再検出',
        'btn_clear': '解除',

        # --- 情報パネル ---
        'btn_analysis': '分析',

        # =====================================================================
        # label_ : ラベル・説明文
        # =====================================================================
        # --- データ読込 ---
        'label_annotated_count': 'アノテーション済み: {0} / {1}',
        'label_image_count': '画像 {0} of {1}:{2}',
        'label_deleted_suffix': '[削除済み]',
        'label_image_source': '画像ソース',

        # --- 自動運転モデル ---
        'label_pilot_model_select': '走行モデル選択:',

        # --- 物体検知 ---
        'label_detection_classes': '検知クラス:',
        'label_classes_example': '例: car,red_sign,green_sign,dog',
        'label_yolo_model': 'YOLOモデル:',

        # --- モデル管理 ---
        'label_mlflow_local': 'MLflow（ローカル）:',
        'label_databricks_integration': 'Databricks連携',
        'label_colab_integration': 'Google Colab連携',

        # --- ナビゲーション ---
        'label_canvas_zoom_label': 'ズーム',
        'label_canvas_zoom_tooltip': 'キャンバスのズーム倍率を調整',
        'label_image_seek': '画像シーク:',
        'label_play': '再生:',
        'label_delete_restore': '削除/復元:',
        'label_delete_range': '削除範囲指定:',
        'label_from': 'から',

        # --- ダウンサンプリング ---
        'label_downsampling': 'ダウンサンプリング:',
        'label_angle_range': 'angle範囲:',
        'label_throttle_range': 'throttle範囲:',
        'label_consecutive': '連続:',
        'label_interval': '間隔:',
        'label_items': '{0}件',
        'label_items_added': '(+{0}件)',

        # --- CAM設定 ---
        'label_cam_method': '手法:',
        'label_cam_target': '対象:',
        'label_cam_direction': '方向:',

        # --- 情報パネル ---
        'label_image_info': '画像情報',
        'label_data_distribution': 'データ分布',
        'label_no_annotation': 'アノテーションがありません',
        'label_no_image_selected': '画像が選択されていません',
        'label_inference': '推論:{0}',

        # --- その他 ---
        'label_select_folder_prompt': 'フォルダを選択し、読込ボタンを押してください',

        # =====================================================================
        # chk_ : チェックボックス
        # =====================================================================
        'chk_show_future_annotation': '5,10個先のアノテーション表示（燈色）',
        'chk_show_inference': '推論結果表示（青丸）',
        'chk_show_diff_vector': '差分ベクトル表示（緑矢印）',
        'chk_show_detection_inference': '物体検知推論結果表示',
        'chk_show_segmentation_inference': 'セグメンテーション推論結果表示',
        'chk_dark_mode': 'ダークモード',

        # =====================================================================
        # placeholder_ : プレースホルダー
        # =====================================================================
        'placeholder_folder': 'フォルダパスを入力または参照ボタンで複数選択可能',
        'placeholder_classes': 'カンマ区切りでクラス名を入力',

        # =====================================================================
        # tip_ : ツールチップ
        # =====================================================================
        # --- 自動運転モデル ---
        'tip_load_model': 'modelsフォルダのモデルを読込む',
        'tip_show_future_annotation': '5フレーム先と10フレーム先のアノテーションを表示',

        # --- CAM設定 ---
        'tip_cam_method': 'CAM可視化手法を選択\nScoreCAM: 勾配を使わない高精度手法（計算時間長め）',
        'tip_cam_target': 'CAMで可視化する出力を選択',
        'tip_cam_direction': '可視化する勾配の方向\nboth: 正負両方を同時表示（赤=正/青=負）\npositive: 出力を増加させる根拠（右に切る/加速）\nnegative: 出力を減少させる根拠（左に切る/減速）',

        # --- 物体検知 ---
        'tip_train_yolo': '物体検知またはセグメンテーションを学習',

        # --- モデル管理 ---
        'tip_open_mlflow': 'ローカルMLflow UIを起動',
        'tip_open_databricks': 'Databricks MLflow UIを開く',
        'tip_sync': 'ローカルの学習記録をDatabricksにアップロード',
        'tip_transfer': '現在のアノテーションを転送',
        'tip_open_colab': 'Google Colabをブラウザで開く',
        'tip_colab_transfer': '現在のアノテーションをGoogle Driveに転送してColabで学習',
        'tip_colab_download': 'Colabで学習したモデルをGoogle Driveからダウンロード',

        # --- ナビゲーション ---
        'tip_set_start': '現在のインデックスを開始位置に設定',
        'tip_set_end': '現在のインデックスを終了位置に設定',

        # --- ダウンサンプリング ---
        'tip_consecutive': 'この数以上連続した場合にダウンサンプリング対象とする',
        'tip_interval': '何枚ごとに1枚残すか（例：3なら3枚中1枚を残す、0なら全て対象）',
        'tip_detect': '条件に該当するインデックスを検出してダウンサンプリング対象に設定',
        'tip_clear_downsampling': 'ダウンサンプリング対象をすべて解除',

        # --- 情報パネル ---
        'tip_analysis': 'アノテーションデータの統計分析と可視化',

        # --- モードツールチップ ---
        'tip_auto_driving_mode': '自動運転アノテーションモード\n・画像をクリックしてAngle/Throttleを設定\n・右クリック: ポイント削除\n・数字キー(0-7): 運転位置を設定（同じキー再押下で解除）\n・Deleteキー: 現在の画像のアノテーション（angle/throttle/位置）を削除',
        'tip_detection_mode': '物体検知アノテーションモード\n・ドラッグしてバウンディングボックスを作成\n・作成したボックスをクリックして選択/移動\n・ボックスの角をドラッグしてサイズ調整\n・右クリック: 選択したボックスを削除\n・Deleteキー: 選択したボックスを削除',
        'tip_segmentation_mode': 'セグメンテーションアノテーションモード\n・左クリック: ポリゴン頂点を追加\n・右クリック: ポリゴンを閉じる/完成させる\n・ポリゴン上で右クリック: 新しい頂点を追加\n・頂点をドラッグ: 頂点位置を調整\n・Deleteキー: 選択したポリゴンを削除\n・Escキー: 作成中のポリゴンをキャンセル',
        'tip_waypoint_mode': 'waypointアノテーションモード\n・左クリック: waypoint座標を追加\n・右クリック: 最後のwaypointを削除\n・緑色の丸でwaypointが表示されます\n・Deleteキー: 現在の画像のwaypointを全削除',

        # =====================================================================
        # status_ : ステータスバーメッセージ
        # =====================================================================
        'status_bbox_hint': 'Bキーを押しながらクリックすると、いつでもバウンディングボックスを作成できます。Deleteキーで選択したボックスを削除できます。',
        'status_deleted': '削除済み',
        'status_ds_target': 'DS対象',
        'status_click_to_reannotate': '削除済み\nクリックで再アノテーション',
        'status_model_not_loaded': '自動運転モデルが読み込まれていません',
        'status_detection_model_not_loaded': '物体検知モデルが読み込まれていません',
        'status_segmentation_model_not_loaded': 'セグメンテーションモデルが読み込まれていません',

        # =====================================================================
        # dlg_ : ダイアログタイトル
        # =====================================================================
        'dlg_warning': '警告',
        'dlg_error': 'エラー',
        'dlg_info': '情報',
        'dlg_complete': '完了',
        'dlg_confirm': '確認',
        'dlg_select_folder': 'フォルダを選択',
        'dlg_export_complete': 'エクスポート完了',
        'dlg_load_complete': '読み込み完了',
        'dlg_load_complete_warning': '読み込み完了（警告あり）',
        'dlg_load_error': '読み込みエラー',
        'dlg_sync_complete': '同期完了',
        'dlg_sync_cancel': '同期キャンセル',
        'dlg_transfer_complete': '転送完了',
        'dlg_transfer_cancel': '転送キャンセル',
        'dlg_add_complete': '追加完了',
        'dlg_copy_complete': 'コピー完了',

        # =====================================================================
        # msg_ : ダイアログ・エラーメッセージ
        # =====================================================================
        'msg_no_images': '画像が読み込まれていません。',
        'msg_no_images_loaded': '画像が読み込まれていません。先に画像を読み込んでください。',
        'msg_no_annotations': '学習用のアノテーションがありません。',
        'msg_no_task_selected': 'タスクが選択されていません。',
        'msg_no_classes': '検知クラスが設定されていません。\n先にクラス設定を行ってください。',
        'msg_no_save_folder': '保存先フォルダが指定されていません。',
        'msg_no_export_format': 'エクスポートする形式が選択されていません。',
        'msg_no_export_data': 'エクスポートするアノテーションがありません。',
        'msg_no_valid_model': '有効なYOLOモデルが選択されていません。',
        'msg_model_not_found': '選択されたモデルが見つかりません: {0}',
        'msg_load_images_first': '先に画像を読み込んでください。',
        'msg_package_not_installed': '{0}パッケージがインストールされていません。\npip install {0} でインストールしてください。',
        'msg_no_detection_annotations': '物体検知アノテーションがありません。',
        'msg_no_segmentation_annotations': 'セグメンテーションアノテーションがありません。',
        'msg_pretrained_model_failed': '事前学習済み {0} モデルの準備に失敗しました。',
        'msg_no_image_source': '画像ソースが選択されていません。',
        'msg_class_settings_applied': 'クラス設定が反映されました。',
        'msg_no_class_name': 'クラス名が入力されていません。',
        'msg_no_valid_class': '有効なクラス名がありません。',
        'msg_export_failed': 'エクスポートに失敗しました。',
        'msg_downsample_cleared': 'ダウンサンプリング対象を解除しました。',
        'msg_location_exists': '位置情報 {0} は既に存在します。',
        'msg_location_added': '位置情報 {0} を追加しました。',
        'label_location_button': '位置 {0}',
        'label_inference_location': '推論位置 {0}',
        'msg_env_copied': '環境変数設定テンプレートをクリップボードにコピーしました',
        'msg_readme_not_found': 'READMEが見つかりません:\n{0}',
        'msg_file_open_error': 'ファイルを開けませんでした:\n{0}',
        'msg_model_download_failed': 'モデルのダウンロードに失敗しました。',
        'msg_transfer_cancelled': '転送がキャンセルされました。',
        'msg_no_sync_option': '同期オプションが選択されていません。',
        'msg_load_data_first': '画像とアノテーションデータを読み込んでください。',
        'msg_angle_range_error': 'angle範囲の最小値は最大値より小さくしてください。',
        'msg_throttle_range_error': 'throttle範囲の最小値は最大値より小さくしてください。',
        'msg_no_downsample_targets': 'ダウンサンプリング対象はありません。',
        'msg_load_folder_first': '先に画像フォルダを選択して画像を読み込んでください。',
        'msg_no_annotation_data': 'アノテーションデータがありません。',
        'msg_inference_method_not_supported': 'サポートされていない推論方法です。',
        'msg_folder_not_exists': 'フォルダが存在しません: {0}',
        'msg_images_folder_not_found': 'フォルダの下にimagesフォルダが見つかりません: {0}',
        'msg_no_images_in_folder': '選択されたフォルダ内のimagesフォルダに画像ファイルがありません。',
        'msg_no_subfolders': '現在のフォルダ内にサブフォルダが見つかりません。',
        'msg_image_source_switch_failed': "画像ソース '{0}' への切り替えに失敗しました。",
        'msg_segmentation_model_not_loaded': '現在のセグメンテーションモデルが読み込まれていません。事前学習済みモデルを使用するか、モデルを読み込んでから再試行してください。',
        'msg_detection_model_not_loaded': '現在の物体検知モデルが読み込まれていません。事前学習済みモデルを使用するか、モデルを読み込んでから再試行してください。',

        # =====================================================================
        # アノテーションモード・制御パネル
        # =====================================================================
        'label_annotation_mode': 'アノテーションモード:',
        'label_waypoint_control': '打点制御:',
        'label_num_points': '打点数:',
        'label_point_position': '打点位置:',
        'label_steering_direction': '走行方向計算:',
        'label_class_id': 'クラスID:',
        'label_y_coordinate': 'Y座標:',
        'label_max_steering': '最大舵角:',
        'label_display_mode': '表示モード:',
        'label_mode_hint': '※Bキーを押すとモードが切り替わります',
        'label_location_info': 'コースの位置情報:',
        'label_current_location': '現在の位置情報: なし',
        'label_current_location_value': '現在の位置情報: {0}',
        'label_gallery': 'ギャラリー:',
        'label_deleted': '削除済',

        # =====================================================================
        # 学習ダイアログ
        # =====================================================================
        'dlg_model_training': 'モデル学習',
        'dlg_select_task': '学習するタスクを選択してください:',
        'label_bbox_none': '✗ バウンディングボックス: なし',
        'label_bbox_count': '✓ バウンディングボックス: {0}個',
        'label_seg_none': '✗ セグメンテーション: なし',
        'label_seg_count': '✓ セグメンテーション: {0}個',
        'label_current_model': '現在のモデル: なし',
        'label_current_model_value': '現在のモデル: {0}',
        'label_epochs': '学習エポック数:',
        'label_batch_size': 'バッチサイズ:',
        'label_input_image_size': '入力画像サイズ:',
        'label_size_note': '注: 640以外のサイズを選択すると精度や速度に影響します',
        'label_patience': '忍耐エポック数:',
        'label_learning_rate': '学習率:',
        'label_probability': '確率:',
        'label_hue': '色相 (H):',
        'label_saturation': '彩度 (S):',
        'label_brightness': '明度 (V):',
        'label_translation': '平行移動:',
        'label_scale': 'スケール:',
        'label_model_name': 'モデル名:',
        'label_comment': 'コメント:',

        # =====================================================================
        # エクスポートダイアログ
        # =====================================================================
        'label_save_folder': '保存先フォルダ:',
        'label_select_image_source': 'エクスポートする画像ソースを選択してください（複数選択可）：',
        'label_donkey_key_note': '※ Donkeycarのデフォルトキーは \'cam/image_array\' です。',
        'dlg_yolo_export': 'YOLOアノテーションエクスポート',

        # =====================================================================
        # プリセットダイアログ
        # =====================================================================
        'dlg_select_preset': 'よく使われるクラスセットを選択してください:',
        'dlg_class_preset_select': 'クラスプリセット選択',
        'label_custom': 'カスタム:',
        'placeholder_comma_separated_classes': 'カンマ区切りでクラス名を入力',

        # =====================================================================
        # 設定ダイアログ
        # =====================================================================
        'dlg_display_settings': '表示設定',
        'label_width': '幅:',
        'label_height': '高さ:',
        'label_preset': 'プリセット:',
        'label_font_size': 'サイズ:',
        'label_preview': 'プレビュー: アノテーションツール',

        # =====================================================================
        # 同期ダイアログ
        # =====================================================================
        'dlg_syncing': '同期中',
        'dlg_transferring_to_databricks': 'Databricksへ転送中',
        'dlg_transferring_to_colab': 'Google Colabへ転送中',
        'msg_syncing_to_databricks': 'Databricksに同期中...',
        'msg_preparing_transfer': '転送準備中...',
        'msg_authenticating_google_drive': 'Google Driveに認証中...',
        'msg_connecting_google_drive': 'Google Driveに接続中...',
        'msg_downloading_mlflow_data': 'MLflow実験データをダウンロード中...',
        'msg_downloading_model': 'モデルをダウンロード中...',
        'dlg_authenticating': '認証中',
        'dlg_fetching_model_list': 'モデル一覧を取得中',
        'dlg_download_model': 'モデルをダウンロード',
        'dlg_downloading': 'ダウンロード中',
        'dlg_model_download': 'モデルダウンロード',
        'msg_sync_delete_warning': '※ 削除オプションを有効にすると、Databricks上のRunが削除されます。\n   この操作は元に戻せません。',

        # =====================================================================
        # ダークモード
        # =====================================================================
        'btn_dark_mode': 'ダークモード',
        'btn_light_mode': 'ライトモード',

        # =====================================================================
        # 位置推論・ウェイポイント推論モデル
        # =====================================================================
        'label_location_model': '位置推論モデル:',
        'label_model_type': 'モデルタイプ:',
        'label_waypoint_model': 'ウェイポイント推論モデル:',
        'chk_location_inference': '位置推論結果表示',
        'chk_waypoint_inference': 'ウェイポイント推論結果表示',

        # =====================================================================
        # 情報パネル - 運転アノテーション
        # =====================================================================
        'label_driving_annotation_info': '運転アノテーション情報:',
        'label_location_inference_result': '位置推論結果:',
        'label_driving_inference_header': '自動運転推論結果:',
        'label_inference_result_header': '推論結果:',
        'label_detection_inference_header': '物体検知推論結果:',
        'label_detected_objects': '検出オブジェクト:',
        'label_object_count': '{0}個',
        'label_total_objects': '合計: {0}個のオブジェクト',
        'label_segmentation_inference_header': 'セグメンテーション推論結果:',
        'label_position_rank': '{0}. 位置 {1}: {2}',
        'label_pretrained': '事前学習済み',
        'dlg_inference_recalculate': '推論結果の再計算確認',
        'msg_inference_recalculate_body': "現在、{0}個の推論結果が保存されています。\n一括推論を実行すると、すべての推論結果が現在のモデル '{1}' を使って再計算されます。\n\n続行しますか？",
        'msg_inference_all_running': '全画像の推論を実行中...',
        'msg_inference_running': '推論実行中...',
        'dlg_inference_complete': '推論完了',
        'msg_inference_complete_body': '{0}枚の画像に対する推論を完了しました。\n{1}個の新しい結果が追加され、{2}個の結果が更新されました。\n\n使用モデル: {3}{4}',
        'msg_inference_processing_error': '推論中にエラーが発生しました: {0}',
        'msg_location_inference_auto_on': '\n\n位置推論結果表示が自動的にオンになりました。',

        # =====================================================================
        # アノテーションモード
        # =====================================================================
        'label_auto_driving_mode': '自動運転アノテーションモード',
        'label_auto_driving_mode_desc': '自動運転アノテーションモード\n画像をクリックして角度・スロットルを設定',
        'status_switched_to_auto_driving': '自動運転アノテーションモードに切り替えました。',
        'status_switched_to_detection': '物体検知アノテーションモードに切り替えました。',
        'status_switched_to_segmentation': 'セグメンテーションアノテーションモードに切り替えました。',
        'status_switched_to_waypoint': 'waypointアノテーションモードに切り替えました。',

        # =====================================================================
        # 追加チェックボックス
        # =====================================================================
        'chk_show_driving_direction': '走行方向を表示',
        'chk_apply_last_bbox': '前回のバウンディングボックスを適用',
        'chk_apply_last_segmentation': '前回のセグメンテーションを適用',
        'chk_auto_skip_on_click': 'クリック時自動スキップ枚数',
        'chk_apply_location': '前回の位置情報を適用',
        'chk_detection_inference': '物体検知推論結果表示',
        'chk_early_stopping': 'Early Stopping を有効にする',
        'chk_data_augmentation': 'データオーグメンテーションを有効にする',
        'chk_aug_mosaic': 'モザイク',
        'chk_aug_flip': '水平反転',
        'chk_aug_hsv': 'HSV調整',
        'chk_aug_geometry': '幾何変換',
        'chk_aug_erase': 'ランダムイレース',
        'chk_aug_color': '色調整',
        'chk_bbox_export': 'バウンディングボックス (物体検知用)',
        'chk_seg_export': 'セグメンテーション (インスタンスセグメンテーション用)',
        'chk_unified_available': '統合形式 (1つのデータセットに両方を含める)',
        'chk_unified_unavailable': '統合形式 (利用不可)',
        'chk_save_settings': '次回起動時にこの設定を適用する',
        'chk_upload_to_databricks': 'ローカル→Databricks（新規Runをアップロード）',
        'chk_delete_from_databricks': 'ローカルで削除したRunをDatabricksからも削除',
        'chk_generate_notebook': '学習用Notebookを生成',
        'chk_open_colab_after': '転送後にColabを開く',
        'chk_download_mlruns': 'MLflow実験データ(mlruns)もダウンロードしてマージする',
        'chk_show_inference_result': '推論結果を表示する（水色丸）',
        'chk_show_diff_vector': '差分ベクトルを表示（緑矢印）',
        'chk_add_speed_output': 'Speed（速度）を出力に追加',
        'chk_add_future_prediction': '将来フレームの予測を出力に追加',
        'chk_exclude_downsampled': 'ダウンサンプリング対象を除外',
        'chk_overwrite_annotation': '既存のアノテーションを上書きする',

        # =====================================================================
        # モードボタン
        # =====================================================================
        'btn_auto_driving': '自動運転',
        'btn_detection': '物体検知',
        'btn_segmentation': 'セグメンテーション',
        'btn_waypoint': 'ウェイポイント',

        # =====================================================================
        # ツールバーボタン
        # =====================================================================
        'toolbar_open': '開く',
        'toolbar_save': '保存',
        'toolbar_mlflow': 'MLflow',
        'toolbar_cloud': 'クラウド',

        # =====================================================================
        # 保存メニュー
        # =====================================================================
        'menu_driving_annotation': '自動運転アノテーション',
        'menu_detection_annotation': '物体検知/セグメンテーション',
        'menu_video_export': '動画作成',

        # =====================================================================
        # 共通ボタン
        # =====================================================================
        'btn_add_location': '位置情報を追加',
        'btn_apply_settings': '適用',
        'btn_cancel': 'キャンセル',
        'btn_close': '閉じる',
        'btn_copy_template': '設定テンプレートをコピー',
        'btn_open_readme': 'READMEを開く',
        'btn_connection_test': '接続テスト',
        'btn_copy_env_template': '環境変数テンプレートをコピー',
        'btn_aug_preview': 'オーグメンテーションプレビュー',

        # =====================================================================
        # 追加ラベル（学習・推論ダイアログ）
        # =====================================================================
        'label_framerate': 'フレームレート (fps):',
        'label_image_skip': '画像スキップ数:',
        'label_start': '開始:',
        'label_end': '終了:',
        'label_multi_source_note': '※複数ソース選択時は下記から複数の画像ソースを選択してください。\n選択した順に左から配置されます。',
        'label_total_frames': '合計フレーム数: {0}',
        'label_total_frames_calculating': '合計フレーム数: 計算中...',
        'label_model_architecture': 'モデルアーキテクチャ:',
        'label_base_model': 'ベースモデル:',
        'label_speed_normalize': '正規化値:',
        'label_speed_normalize_note': '※ Speed値はこの値で除算されます',
        'label_future_info': '※ 5フレーム先と10フレーム先のangle, throttle(, speed)を追加出力',
        'label_future_detail': '出力例（speed有）: [angle, throttle, speed, t+5_angle, t+5_throttle, t+5_speed, t+10_angle, t+10_throttle, t+10_speed]',
        'label_min_delta': '最小改善量:',
        'label_validation_ratio': '検証データ割合:',
        'label_skip_count': 'スキップ枚数:',
        'label_color_brightness': '明るさ:',
        'label_color_contrast': 'コントラスト:',
        'label_color_saturation': '彩度:',
        'label_rotation_angle': '回転角度 (±度):',
        'label_translation_ratio': '平行移動 (±比率):',
        'label_erase_min_ratio': '最小比率:',
        'label_erase_max_ratio': '最大比率:',
        'label_auto_annotation_range': 'オートアノテーションを実行する範囲を指定してください。',
        'label_auto_annotation_method': 'オートアノテーションの実行方法を選択してください',
        'label_skip_every': '枚ごと',
        'label_aug_preview_title': 'オーグメンテーションプレビュー',
        'label_original_image': 'オリジナル画像',
        'label_waypoint_count': 'ウェイポイント数:',

        # =====================================================================
        # ダイアログタイトル・キャンバステキスト
        # =====================================================================
        'dlg_select_yolo_folder': 'YOLOアノテーションフォルダを選択',
        'dlg_select_save_folder': '保存先フォルダを選択',
        'dlg_select_export_folder': 'エクスポート先フォルダを選択',
        'dlg_select_annotation_subfolder': 'アノテーションを読み込むサブフォルダを選択してください:',
        'dlg_select_subfolder_title': 'サブフォルダの選択',
        'dlg_select_image_folder': '画像フォルダを選択',
        'dlg_save_video': '動画の保存先を選択',
        'canvas_no_image': 'フォルダを選択し、読込ボタンを押してください',
        'status_no_images_loaded': '画像が読み込まれていません',
        'combo_model_not_found': 'モデルが見つかりません',
        'combo_select_folder': 'フォルダを選択してください',
        'msg_selected_model_not_found': '選択されたモデルが見つかりません: {0}',

        # =====================================================================
        # セッション復元・読み込みダイアログ
        # =====================================================================
        'dlg_restore_session': '前回のセッションを復元',
        'msg_restore_folders': '前回の作業フォルダ（{0}個）を読み込みますか？\n\n最初のフォルダ: {1}\n{2}',
        'msg_restore_folder': '前回の作業フォルダを読み込みますか？\n\nフォルダ: {0}',
        'msg_other_folders': '他 {0} フォルダ',
        'dlg_loading_progress': '読み込み進捗',
        'msg_loading_folder': 'フォルダを読み込み中...',
        'msg_searching_folders': '{0}個のフォルダを検索中...',
        'msg_loading_folder_progress': "フォルダ '{0}' を読み込み中... ({1}/{2})",
        'dlg_load_complete': '読み込み完了',
        'msg_images_loaded': '{0}個のフォルダから合計{1}枚の画像を読み込みました。\n画像データのキー: {2}\n現在のキー \'{3}\' の画像数: {4}\n元の画像サイズ: {5}x{6}\n\nアノテーションデータは読み込まれていません。',
        'dlg_load_annotation': 'アノテーションデータ読み込み',
        'msg_load_annotation_prompt': '画像読み込みが完了しました。\n続けてアノテーションデータを読み込みますか？',
        'dlg_load_success': '読み込み成功',
        'msg_annotations_loaded': '{0}個のフォルダから合計{1}個のアノテーションを読み込みました。{2}',
        'msg_no_annotations_found': '選択したフォルダからアノテーションデータが見つかりませんでした。',
        'msg_annotation_load_error': 'アノテーションの読み込み中にエラーが発生しました: {0}',
        'msg_model_load_complete': 'モデル読み込み完了',
        'msg_model_loaded': 'モデルを読み込みました: {0}',
        'msg_yolo_model_loaded': '{0}モデル「{1}」を読み込みました。\n信頼度閾値: {2}\n\n画像送りごとに自動的に{0}推論が実行されます。',
        'msg_pilot_model_loaded': 'モデル「{0}」を読み込みました。\n\n画像送りごとに自動的に推論が実行されます。',
        'msg_model_saved_loaded': 'モデルを保存しました:\n\nファイル: {0}\nモデル: {1}\n\n「モデル読込」ボタンから推論に使用できます。',
        'dlg_model_save_complete': 'モデル保存完了',
        'msg_model_saved': 'モデルファイルを保存しました:\n{0}\n\n「モデル読込」ボタンから読み込んでください。',
        'msg_same_level_annotations_loaded': '同階層から{0}個のアノテーションを読み込みました。',
        'msg_subfolder_annotations_loaded': 'サブフォルダ「{0}」から{1}個のアノテーションを読み込みました。',
        'msg_no_subfolder_annotations': '選択されたサブフォルダ「{0}」から読み込めるアノテーションデータがありませんでした。',

        # =====================================================================
        # アノテーション読み込み進捗
        # =====================================================================
        'dlg_annotation_loading': 'アノテーション読み込み',
        'btn_cancel': 'キャンセル',
        'msg_searching_annotations': '{0}個のフォルダからアノテーションを検索中...',
        'msg_processing_folder': 'フォルダ {0}/{1} を処理中...\n{2}',
        'msg_preparing_load': 'アノテーションデータ読み込み準備中...',
        'msg_checking_manifest': 'マニフェストファイルを確認中...',
        'msg_annotations_count_loaded': '{0}個のアノテーションを読み込みました',
        'label_details': '\n\n詳細:\n',
        'label_item_count': '• {0}: {1}個\n',
        'msg_catalog_files_found': '{0}個のカタログファイルを検出しました',
        'msg_searching_image_folder': '画像フォルダを検索中...',
        'msg_processing_catalog_file': 'カタログファイル処理中: {0} ({1}/{2})',
        'msg_processing_catalog_entry': 'カタログエントリ処理中: {0}/{1} エントリ',

        # YOLO アノテーション読み込み
        'msg_loading_yolo_annotations': 'YOLOアノテーションを読み込み中...',
        'dlg_loading': '読み込み中',
        'msg_yolo_loaded': 'YOLOアノテーションを読み込みました。\n\n処理画像数: {0}/{1}\nアノテーション付き画像: {2}\nバウンディングボックス: {3}\nセグメンテーション: {4}\nクラス: {5}\n\nクラス別アノテーション数:\n{6}{7}',
        'msg_yolo_load_error': 'YOLOアノテーションの読み込み中にエラーが発生しました。\n\n処理済み画像: {0}/{1}\n読み込み済みバウンディングボックス: {2}\n読み込み済みセグメンテーション: {3}\n\nエラー詳細:\n{4}',
        'msg_other_errors': '...他{0}件のエラー',
        'msg_warning_errors': '\n\n警告: {0}件のエラーが発生しました',
        'msg_and_more': '...他{0}件',

        # =====================================================================
        # エクスポート関連
        # =====================================================================
        'dlg_export_warning': 'エクスポート警告',
        'msg_no_exportable_entries': 'エクスポート可能なエントリがありませんでした。',
        'msg_donkey_export_complete': 'アノテーションをDonkeycar形式でエクスポートしました。\n選択画像ソース: {0}\n保存先: {1}\nエクスポート数: {2}個',
        'msg_donkey_export_error': 'Donkeycarエクスポート中にエラーが発生しました: {0}\n\n詳細: {1}',
        'msg_jetracer_export_complete': 'アノテーションをJetracer形式でエクスポートしました。\n保存先: {0}\nエクスポート数: {1}個',
        'msg_jetracer_export_error': 'Jetracerエクスポート中にエラーが発生しました: {0}\n\n詳細: {1}',
        'msg_yolo_export_error': 'YOLO統合エクスポート中にエラーが発生しました: {0}',

        # YOLO エクスポートダイアログ
        'dlg_yolo_export_settings': 'YOLO統合エクスポート設定',
        'label_annotation_status': 'アノテーション状況',
        'label_bbox_status': '✓ バウンディングボックス: {0}個 ({1}枚の画像)',
        'label_seg_status': '✓ セグメンテーション: {0}個 ({1}枚の画像)',
        'label_export_format': 'エクスポート形式選択',
        'tip_bbox_export_count': '{0}個のバウンディングボックスをエクスポートします',
        'tip_no_bbox': 'バウンディングボックスアノテーションがありません',
        'tip_seg_export_count': '{0}個のセグメンテーションをエクスポートします',
        'tip_no_seg': 'セグメンテーションアノテーションがありません',
        'tip_unified_export': 'バウンディングボックスとセグメンテーションを1つのYOLOデータセットに統合します',
        'tip_unified_requires_both': '両方のアノテーション形式が必要です',
        'label_class_settings': 'クラス設定',
        'label_save_settings': '保存先設定',
        'label_deleted_indexes_info': '削除済みインデックス数: {0}個（エクスポートから除外されます）',
        'msg_yolo_export_confirm': '以下の設定でYOLO形式でエクスポートします：\n\n保存先: {0}\nクラス: {1}\n\nエクスポート内容:\n{2}{3}{4}',
        'label_bbox_count_item': 'バウンディングボックス: {0}個',
        'label_seg_count_item': 'セグメンテーション: {0}個',
        'msg_unified_note': '\n※ 統合形式で1つのデータセットに保存されます',
        'msg_separate_note': '\n※ 各形式別々のデータセットとして保存されます',
        'msg_deleted_excluded': '\n\n削除済みインデックス: {0}個（除外）',
        'dlg_yolo_export_confirm': 'YOLO統合エクスポート確認',
        'msg_continue_question': '\n\n続行しますか？',
        'msg_yolo_export_preparing': 'YOLOエクスポート準備中...',
        'dlg_exporting': 'エクスポート実行中',
        'msg_exporting_unified': '統合YOLO形式でエクスポート中...',
        'msg_exporting_bbox': 'バウンディングボックス形式でエクスポート中...',
        'msg_exporting_seg': 'セグメンテーション形式でエクスポート中...',
        'label_unified_format': '統合形式: {0}',
        'label_detection_format': '物体検知: {0}',

        # YOLO学習ダイアログ
        'dlg_yolo_task_selection': 'YOLO学習タスク選択',
        'label_training_task': '学習タスク',
        'label_detection_task': '物体検知 (Detection)',
        'label_segmentation_task': 'セグメンテーション (Segmentation)',
        'tip_detection_task': 'バウンディングボックスを使用した物体検知モデルを学習',
        'tip_segmentation_task': 'ポリゴンを使用したセグメンテーションモデルを学習',
        'dlg_yolo_training_settings': 'YOLO{0}モデル学習設定',
        'label_training_stats': '学習データ統計:',
        'label_total_loaded_images': '総読み込み画像数: {0}枚',
        'label_annotated_images_count': '{0}アノテーション済み画像数: {1}枚',
        'label_actual_training_count': '実際の学習使用枚数: {0}枚',
        'label_excluded_calculation': '({0}枚 - 削除済み{1}枚)',
        'label_total_annotations_count': '総{0}アノテーション数: {1}個',
        'label_deleted_excluded_note': '※ 削除マークされた画像は学習対象から除外されます',
        'label_model_init_settings': 'モデル初期化設定',
        'label_use_pretrained_weights': '事前学習済みの重みを使用 (推奨)',
        'label_use_current_model_weights': '現在読み込まれているモデルの重みを使用',
        'label_current_model_name': '現在のモデル: {0}',
        'label_no_model_loaded': '現在のモデル: なし（先にモデルを読み込んでください）',
        'tab_basic_settings': '基本設定',
        'tab_data_augmentation': 'データオーグメンテーション',
        'label_model_name_settings': 'モデル名設定',
        'placeholder_custom_name': 'カスタム名を入力',
        'label_model_name_note': '※ モデルタイプ ({0}) のプレフィックスは変更できません。.ptは自動的に付与されます',
        'label_model_name_note_pth': '※ モデルタイプ ({0}) のプレフィックスは変更できません。.pthは自動的に付与されます',
        'label_training_comment': '学習コメント (MLflowに記録)',
        'placeholder_training_comment': 'この学習についてのメモやコメントを入力してください (任意)',
        'label_original_image_size': '元画像: {0}×{1}',

        # YOLOモデル読み込みダイアログ
        'dlg_yolo_confidence_settings': '{0}モデル信頼度設定',
        'label_confidence_threshold': '{0}の信頼度閾値 (0.0-1.0):',
        'msg_loading_yolo_model': "{0}モデル '{1}' を読み込み中...",
        'dlg_unified_model_loading': '統合モデル読み込み',
        'msg_loading_model_to_memory': '{0}モデルをメモリに読み込み中...',
        'msg_getting_class_info': 'クラス情報を取得中...',
        'tip_segmentation_model_loaded': 'セグメンテーションモデルが読み込まれています',
        'tip_detection_model_loaded': '物体検知モデルが読み込まれています',
        'msg_unified_model_loaded': "統合{0}モデル '{1}' を読み込みました。\n信頼度閾値: {2}\nクラス: {3}個\n\n画像送りごとに自動的に{0}推論が実行されます。",
        'msg_yolo_load_failed': 'YOLOモデルの読み込みに失敗しました: {0}',
        'label_segmentation_short': 'セグメンテーション',
        'label_detection_short': '物体検知',
        'label_seg_tag': '[セグ]',
        'label_det_tag': '[物検]',
        'label_det_tag_short': '物検',
        'label_seg_tag_short': 'セグ',
        'status_models_loaded_count': '{0}個の{1}モデルを読み込みました (物体検知: {2}, セグメンテーション: {3})',
        'msg_no_labels_directory': '選択されたフォルダ内にlabelsディレクトリが見つかりません。\nYOLOデータセットの構造:\n- dataset/\n  - images/\n  - labels/\n  - classes.txt',
        'msg_no_annotations_found_fallback': 'アノテーションが見つかりませんでした',
        'label_env_status': '環境変数の状態',
        'label_env_tab': '環境変数',
        'msg_config_databricks_not_found': 'databricks/config_databricks.py が見つかりません',
        'msg_config_colab_not_found': 'colab/config_colab.py が見つかりません',
        'label_status_message': '状態: {0}\n{1}',
        'dlg_no_models': 'モデルなし',
        'msg_google_drive_no_models': 'Google Driveにモデルファイルが見つかりませんでした。\n\nColabでモデルを学習し、Google Driveに保存してください。',
        'dlg_images_folder_not_found': 'imagesフォルダ未検出',
        'msg_images_folder_missing': '次のフォルダ内にimagesフォルダが見つかりませんでした：\n{0}\n\n有効なフォルダのみ処理を続行します。',
        'msg_no_segmentation_class_mismatch': '有効なセグメンテーションアノテーションが見つかりません。\n\n発見されたクラス名: {0}\n期待されるクラス名: {1}\n\nクラス名が一致していることを確認してください。',
        'section_window_size': 'ウィンドウサイズ',
        'section_font_size': 'フォントサイズ',
        'section_save_settings': '設定の保存',
        'section_current_status': '現在の状態',
        'section_sync_options': '同期オプション',
        'section_connection_status': '接続状態',
        'section_env_setup': '環境変数の設定方法',
        'section_transfer_content': '転送内容',
        'section_options': 'オプション',
        'section_save_location': '保存先',
        'section_range': '範囲',
        'label_location_value': '位置 {0}',
        'dlg_databricks_sync_settings': 'Databricks同期設定',
        'dlg_databricks_settings': 'Databricks設定',
        'dlg_colab_transfer_settings': 'Google Colab転送設定',
        'dlg_colab_settings': 'Google Colab設定',
        'msg_mlflow_tracking_uri': 'MLflowトラッキングURI: {0}',
        'msg_mlflow_init_success': 'MLflow初期化成功: {0}',
        'msg_setting_experiment': '実験を設定: {0}',
        'msg_yolo_training_preparing': 'YOLO{0}モデル \'{1}\' の学習準備中...',
        'label_dependency_package': '依存パッケージ',
        'msg_class_index_mapping': 'クラス-インデックスマッピング: {0}',
        'label_detection_result': '物体検知: {0}',
        'msg_exporting_bbox': 'バウンディングボックスエクスポート中: {0}',
        'msg_exporting_segmentation': 'セグメンテーションエクスポート中: {0}',
        'msg_processing_file': '処理中: {0}',
        'msg_downloading_size': 'ダウンロード中: {0} MB / {1} MB',
        'msg_preparing_with_input_size': '入力サイズ: {0} で学習準備中...',
        'msg_initializing_model_waypoints': 'モデル \'{0}\' を初期化中... ({1}ウェイポイント)',
        'msg_location_inference_all_running': '全画像の位置推論を実行中...',
        'msg_location_inference_running': '位置推論実行中...',
        'dlg_location_inference_complete': '位置推論完了',
        'msg_location_inference_result': '{0}枚の画像に対する位置推論を完了しました。\n{1}個の新しい結果が追加され、{2}個の結果が更新されました。\n\n使用モデル: {3} ({4}){5}',
        'msg_location_inference_error': '位置推論中にエラーが発生しました: {0}',
        'msg_location_model_load_error': '位置モデルの読み込み中にエラーが発生しました: {0}',
        'msg_location_model_training_error': '位置モデル学習中にエラーが発生しました: {0}',
        'tip_seg_disabled_detection': 'セグメンテーションモデルが読み込まれているため無効',
        'tip_detection_disabled_seg': '物体検知モデルが読み込まれているため無効',
        'msg_running_inference_test': '推論テストを実行中...',
        'msg_yolo_load_error': '{0}モデルの読み込み中にエラーが発生しました: {1}',
        'msg_yolo_model_type_changed': 'YOLOモデルタイプを「{0}」に変更しました。モデルリストを更新します...',

        # 動画作成ダイアログ
        'msg_no_annotations_for_video': 'アノテーションがありません。',
        'dlg_video_settings': '動画作成設定',
        'label_target_range': '対象範囲',
        'label_all_images': 'すべての画像',
        'label_specify_range': 'インデックス範囲を指定',
        'label_output_mode': '出力モード',
        'label_single_source': '単一ソース出力（通常モード）',
        'label_multi_source': '複数ソース出力（横に並べる）',
        'label_image_sources': '画像ソース',
        'label_images_count': '({0}枚)',
        'msg_no_source_selected': '合計フレーム数: 画像ソースが選択されていません',
        'msg_no_valid_source': '合計フレーム数: 有効な画像ソースがありません',
        'msg_start_must_be_less': '合計フレーム数: 開始は終了以下にしてください',
        'label_range_info': '\n範囲: {0} - {1}',
        'label_time_format_min_sec': '{0}分{1}秒',
        'label_time_format_sec': '{0}秒',
        'label_selected_sources': '\n選択ソース: {0} (各{1}枚)',
        'label_total_frames_info': '合計フレーム数: {0}フレーム (約{1}){2}{3}',
        'label_total_frames_simple': '合計フレーム数: {0}フレーム',
        'msg_start_index_error': '開始インデックスは終了インデックス以下である必要があります。',
        'msg_no_source_images': '画像ソースが選択されていません。',
        'msg_source_not_found': "ソース '{0}' の画像が見つかりません。",
        'msg_no_images_in_source': '選択したソースのいずれかに画像がありません。',
        'msg_no_images_in_selected': "選択したソース '{0}' に画像がありません。",
        'dlg_save_video': '動画の保存先を選択',
        'msg_creating_video': '動画作成中...',
        'dlg_processing': '処理中',
        'dlg_success': '成功',
        'msg_video_created': 'アノテーション動画を作成しました:\nファイル: {0}\nフレーム数: {1}フレーム\n{2}{3}\n設定: {4}fps, {5}枚ごと',
        'label_multi_sources': '複数ソース: {0}',
        'label_single_source_info': 'ソース: {0}',
        'msg_video_creation_failed': '動画の作成に失敗しました。処理可能なアノテーションデータがありませんでした。',
        'msg_video_creation_error': '動画作成中にエラーが発生しました: {0}',

        # Donkey/Jetracerエクスポートダイアログ
        'dlg_donkey_export_settings': 'Donkeycarエクスポート設定',
        'dlg_jetracer_export_settings': 'Jetracerエクスポート設定',
        'label_image_source_selection': '画像ソース選択',
        'label_variant_images_count': '{0} ({1}枚)',
        'label_catalog_key_settings': 'カタログキー設定',
        'label_key_name': '{0} キー名:',
        'label_deleted_indexes_export': '削除済みインデックス数: {0}個（削除情報も併せてエクスポートされます）',
        'msg_export_confirm': '以下の設定で{0}形式でエクスポートします：\n\n保存先: {1}\n',
        'label_image_source_item': '・画像ソース: {0} ({1}枚)\n',
        'label_key_name_item': '  キー名: {0}\n',
        'label_annotation_count': '\nアノテーション数: {0}個',
        'label_deleted_count': '\n削除済みインデックス数: {0}個',
        'dlg_export_confirm': '{0}エクスポート確認',

        # =====================================================================
        # 自動運転モデルセクション
        # =====================================================================
        # 学習設定ダイアログ
        'msg_need_annotations_to_train': 'モデルを学習するにはアノテーションが必要です。',
        'dlg_training_settings': '学習設定',
        'label_init_settings': '初期化設定',
        'label_use_pretrained': '事前学習済みの重みを使用（推奨）',
        'tip_pretrained': 'ImageNetで事前学習済みの重みを使用して学習します（転移学習）',
        'tip_no_pretrained': 'このモデルには事前学習済みの重みがありません',
        'label_random_init': 'ランダム初期化（スクラッチから学習）',
        'tip_random_init': '重みをランダムに初期化して最初から学習します（学習に時間がかかります）',
        'label_finetune': '既存モデルの重みを使用（ファインチューニング）',
        'tip_finetune': '選択したモデルの重みを使用してファインチューニングします',
        'tip_no_model_for_type': '選択したモデルタイプに対応するモデルがありません',
        'msg_no_model_for_type': '{0}のモデルがありません',
        'label_output_settings': '出力設定',
        'label_training_params': '学習パラメータ',
        'label_speed_data_info': '※ {0}個のアノテーションにspeedデータが含まれています',
        'tip_speed_normalize': 'Speed値を正規化する際の除数（デフォルト: 10.0）',
        'tip_future_prediction': '5, 10フレーム先のangle, throttle(, speed)を追加出力',
        'tip_validation_ratio': '学習データから検証用に分割する割合',
        'tip_weight_decay': 'L2正則化の強さ（過学習防止）',
        'tip_optimizer': 'Adam: 汎用的, AdamW: Weight Decay改良版, SGD: 古典的だが安定',
        'tip_scheduler': 'ReduceLROnPlateau: 損失停滞時に学習率低下, StepLR: 固定ステップで低下, CosineAnnealing: コサイン曲線で調整',
        'label_training_data_selection': '学習データ選択',
        'label_use_all_annotations': 'すべてのアノテーションデータを使用',
        'label_use_skip': 'スキップ設定でデータを間引く',
        'label_specify_index_range': 'インデックス範囲を指定',
        'label_range_separator': '〜',
        'tip_exclude_downsampled': '直進時などのダウンサンプリング対象データ（現在{0}件）を学習から除外します',
        'label_exclude_downsampled_zero': 'ダウンサンプリング対象を除外 (0件)',
        'label_exclude_downsampled_count': 'ダウンサンプリング対象を除外 ({0}件)',
        'label_data_count_all': '使用データ数: {0}枚',
        'label_data_count_all_detail': '<b>使用データ数: {0}枚</b> (全{1}枚 - {2})',
        'label_data_count_skip_detail': '<b>使用データ数: {0}枚</b> ({1}枚ごと、対象{2}枚 - {3})',
        'label_data_count_range_detail': '<b>使用データ数: {0}枚</b> (範囲{1}-{2}、対象{3}枚 - {4})',
        'label_excluded_deleted': '削除済み{0}枚',
        'label_excluded_ds': ' + DS{0}枚',
        'btn_start_training': '学習開始',
        'msg_training_starting': '{0}学習開始...',
        'dlg_yolo_model_training': 'YOLO{0}モデル学習',
        'dlg_training_result_logging': '学習結果記録',
        'dlg_training_complete': '学習完了',
        'btn_open_mlflow': 'MLflowを開く',
        'msg_downloading_pretrained': '事前学習済み {0} モデルをダウンロードしています...',
        'msg_pretrained_download_failed': '事前学習済み {0} モデルの準備に失敗しました。',
        'msg_package_not_installed': '{0}パッケージがインストールされていません。\npip install {0} でインストールしてください。',

        # 位置モデル関連
        'msg_need_location_annotations': '位置モデルを学習するには位置アノテーションが必要です。',
        'msg_need_at_least_2_locations': '位置モデルを学習するには少なくとも2つの異なる位置ラベルが必要です。現在: {0}種類',
        'msg_no_valid_location_annotations': '有効な位置アノテーションがありません。',
        'msg_preparing_location_training': "位置モデル '{0}' の学習データを準備中...",
        'dlg_location_model_training': '位置モデル学習',
        'msg_init_location_model': "モデル '{0}' を初期化中... (固定{1}クラス)",
        'dlg_location_training_settings': '位置モデル学習設定',
        'label_location_stats': '<b>学習データ統計:</b><br>総読み込み画像数: {0}枚<br>位置アノテーション済み画像数: {1}枚<br><b style="color: #2E7D32; font-size: 14px;">実際の学習使用枚数: {2}枚</b><br>({1}枚 - 削除済み{3}枚)<br><span style="color: #FF6600;">{4}</span>',
        'label_detected_locations': '検出された位置ラベル: {0}種類 ({1})',
        'label_fixed_class_note': '※ 位置モデルは常に{0}クラス出力で作成されます。',
        'msg_no_valid_location_model': '有効な位置モデルが選択されていません。',
        'msg_loading_location_model': "位置モデル '{0} ({1})' を読み込み中...",
        'msg_location_model_load_error': '位置モデルの読み込み中にエラーが発生しました: {0}',
        'msg_running_initial_inference': '初期推論を実行中...',
        'msg_updating_inference_display': '推論表示を更新中...',
        'tip_location_model_loaded': '位置モデルが読み込まれています',
        'tip_location_model_not_loaded': '位置モデルが読み込まれていません',

        # 推論表示チェックボックスツールチップ
        'tip_driving_model_loaded': '自動運転モデルが読み込まれています',
        'tip_driving_model_not_loaded': '自動運転モデルが読み込まれていません',
        'tip_detection_model_loaded': '物体検知モデルが読み込まれています',
        'tip_detection_model_not_loaded': '物体検知モデルが読み込まれていません',
        'tip_seg_disabled_by_detection': '物体検知モデルが読み込まれているため無効',
        'tip_segmentation_model_loaded': 'セグメンテーションモデルが読み込まれています',
        'tip_segmentation_model_not_loaded': 'セグメンテーションモデルが読み込まれていません',
        'tip_batch_inference': '全ての画像に対して推論を実行します',
        'tip_gradcam_heatmap': 'モデルの注目領域をヒートマップで表示',

        # 追加モデルスロット
        'btn_add_model': '＋ モデル追加',
        'tip_add_model': '追加の走行モデルスロットを追加します（最大3個）',
        'btn_remove_model': '－ モデル削除',
        'tip_remove_model': '最後に追加した走行モデルスロットを削除します',
        'label_driving_model_n': '走行モデル{0}',
        'chk_inference_result_n': '推論結果{0}表示',
        'tip_extra_model_not_loaded': 'モデル未読込',
        'label_driving_model_n_inference': '走行モデル{0}推論',

        # ウェイポイントモデル関連
        'dlg_waypoint_training_settings': 'ウェイポイントモデル学習設定',
        'label_waypoint_stats': '<b>学習データ統計:</b><br>総読み込み画像数: {0}枚<br>ウェイポイントアノテーション済み画像数: {1}枚<br><b style="color: #2E7D32; font-size: 14px;">実際の学習使用枚数: {2}枚</b><br>({1}枚 - 削除済み{3}枚)<br><span style="color: #FF6600;">{4}</span>',
        'msg_no_waypoint_model_selected': '読み込むウェイポイントモデルが選択されていません。',
        'msg_waypoint_model_not_found': 'ウェイポイントモデルファイルが見つかりません: {0}',
        'tip_waypoint_model_loaded': 'ウェイポイントモデル ({0}, {1}ポイント) が読み込まれています',
        'tip_waypoint_model_not_loaded': 'ウェイポイントモデルが読み込まれていません',
        'msg_waypoint_model_loaded': 'ウェイポイントモデルを読み込みました\nモデル: {0}\nウェイポイント数: {1}',
        'msg_need_waypoint_annotations': 'ウェイポイントモデルを学習するには少なくとも5枚のアノテーションが必要です。現在: {0}枚',
        'msg_no_valid_waypoint_annotations': '有効なウェイポイントアノテーションがありません。',
        'msg_preparing_waypoint_training': "ウェイポイントモデル '{0}' の学習データを準備中...",
        'dlg_waypoint_model_training': 'ウェイポイントモデル学習',
        'msg_skipped_images_header': '以下の{0}枚の画像はwaypoint数が一致しないためスキップされます:\n\n',
        'msg_skipped_images_item': '  {0}: {1} - {2}\n',
        'msg_skipped_images_more': '\n...他 {0}件',
        'msg_skipped_images_footer': '\n\n学習に使用される画像: {0}枚\n続行しますか？',
        'msg_need_waypoint_annotations_first': 'ウェイポイントモデルを学習するにはウェイポイントアノテーションが必要です。',

        # YOLOモデル読み込み
        'msg_loading_yolo_model_display': "YOLOモデル '{0}' を読み込み中...",
        'dlg_detection_threshold': '検出閾値',
        'label_detection_threshold': '検出信頼度閾値 (0.0-1.0):',
        'msg_loading_model_to_memory_display': "モデル '{0}' をメモリに読み込み中...",
        'msg_running_inference_on_current': '現在の画像に対して推論実行中...',
        'msg_yolo_model_load_error': 'YOLOモデルの読み込み中にエラーが発生しました: {0}',

        # Waypointモード選択
        'label_auto_advance': '自動遷移',
        'tip_auto_advance': '最後のwaypointが配置されたら自動で次の画像に遷移',
        'label_apply_last_waypoint': '前回のウエイポイントを適用',
        'tip_apply_last_waypoint': '前回の画像のwaypointを次の画像に自動適用',

        # セグメンテーション表示モード
        'tip_show_driving_direction': 'セグメンテーション推論結果から走行方向を計算して矢印で表示',
        'tip_seg_class_id': '走行可能エリアのセグメンテーションクラスID',
        'tip_seg_y_coordinate': '走行方向計算に使用するY座標（画像上からのピクセル）',
        'tip_seg_max_steering': '走行軌跡計算に使用する最大舵角（度）',
        'label_trajectory_mode': '軌跡',
        'tip_trajectory_mode': '走行軌跡を円弧で表示',
        'label_waypoint_mode': 'ウェイポイント',
        'tip_waypoint_mode': '目標Y座標までのウェイポイント（4点等間隔）を表示',

        # 位置アノテーション
        'tip_apply_last_bbox': '前回作成したバウンディングボックスを現在の画像にも適用します',
        'tip_apply_last_segmentation': '前回作成したセグメンテーションを現在の画像にも適用します',
        'tip_apply_location': '前回選択した位置情報を現在の画像にも適用します',

        # 画像バッジ
        'badge_objects': '物体: {0}',
        'badge_segments': 'セグ: {0}',
        'badge_deleted': '削除済み',
        'badge_ds_target': 'DS対象',
        'label_deleted_click_to_restore': '削除済み\nクリックで再アノテーション',
        'label_inference_prefix': '推論:',

        # ステータスバーメッセージ
        'status_polygon_point_added': 'ポリゴンに新しい点を追加しました (位置: {0})',
        'status_waypoint_complete_auto_advance': 'waypoint配置完了 ({0}個) - 次の画像に自動遷移',
        'status_speed_updated': 'Speed値を更新: {0:.2f}',
        'status_bbox_deselected': 'バウンディングボックスの選択を解除しました',
        'status_vertex_editing': '頂点を編集中... (頂点 {0})',
        'status_seg_deselected': 'セグメンテーションの選択を解除しました',
        'status_waypoint_count_reached': '設定された打点数({0})に達しています',
        'status_y_exceeds_image': 'Y座標が画像サイズ({0})を超えています',
        'status_waypoint_added': 'waypoint{0}追加: ({1}, {2}) - 総数: {3}/{4}',
        'status_freehand_start': '一筆書きモード開始 - ドラッグしてウェイポイントを配置',
        'status_freehand_placed': '一筆書きで{0}個のウェイポイントを配置しました',
        'status_waypoint_adjusted': 'ウェイポイントの位置を調整しました',
        'status_creating_bbox': '新規バウンディングボックス作成中... 幅: {0}px, 高さ: {1}px',
        'status_moving_bbox': "'{0}' バウンディングボックスを移動中...",
        'status_moving_seg': "'{0}' セグメンテーションを移動中...",
        'status_bbox_deleted': "'{0}' のバウンディングボックスを削除しました",
        'status_seg_deleted': "'{0}' のセグメンテーションを削除しました",
        'status_driving_annotation_deleted': '自動運転アノテーション ({0}) を削除しました',
        'status_no_annotation_to_delete': '削除するアノテーションがありませんでした',
        'status_waypoints_deleted': 'waypoint {0}個を削除しました',
        'status_no_waypoint_to_delete': '削除するwaypointがありませんでした',
        'status_start_y_set': '開始Y位置を{0}に設定しました',
        'status_end_y_set': '終了Y位置を{0}に設定しました',
        'status_mouse_not_in_image': 'マウスが画像内にありません',
        'status_image_view_not_initialized': '画像ビューが初期化されていません',
        'status_waypoint_auto_apply_switched': '前回waypoint自動適用モードに切り替え - {0}個を適用しました',
        'status_waypoint_auto_apply_mode': '前回waypoint自動適用モードに切り替えました',
        'status_waypoint_auto_apply_no_data': '前回waypoint自動適用モードに切り替えました（適用するwaypointがありません）',
        'status_auto_advance_mode': '配置完了時自動遷移モードに切り替えました',
        'status_location_inference_on': '位置推論結果表示をオンにしました',
        'status_location_inference_off': '位置推論結果表示をオフにしました',
        'status_updating_model_list': 'モデルリストを更新中...',
        'status_model_not_found': '{0}のモデルが見つかりません。他のアーキを選択するか、モデルを学習してください',
        'status_models_loaded': '{0}個の{1}モデルを読み込みました',
        'status_auto_play': '自動再生中 ({0}, {1}, {2}) - 停止するには再度ボタンをクリック',
        'status_direction_forward': '順方向',
        'status_direction_backward': '逆方向',
        'status_speed_slow': '低速',
        'status_speed_fast': '高速',
        'status_skip_count': '{0}枚スキップ',
        'status_no_skip': 'スキップなし',
        'status_location_auto_applied': '位置情報 {0} を自動適用しました',
        'status_bbox_auto_applied': "前回の '{0}' バウンディングボックスを適用しました",
        'status_detection_inference_on': '物体検知推論結果表示をオンにしました',
        'status_detection_inference_off': '物体検知推論結果表示をオフにしました',
        'status_switched_to_driving_training': '自動運転モデル学習モードに切り替えました。',
        'status_switched_to_detection_training': '物体検知モデル学習モードに切り替えました。',
        'status_segmentation_inference_on': 'セグメンテーション推論結果表示をオンにしました',
        'status_segmentation_inference_off': 'セグメンテーション推論結果表示をオフにしました',
        'status_updating_yolo_model_list': '統合YOLOモデルリストを更新中...',
        'status_updating_yolo_model_list_simple': 'YOLOモデルリストを更新中...',
        'status_yolo_model_not_found': '{0}のYOLOモデルが見つかりません',
        'status_future_annotation_on': '将来アノテーション表示をオンにしました',
        'status_future_annotation_off': '将来アノテーション表示をオフにしました',
        'status_cam_on': 'CAM表示をオンにしました',
        'status_cam_off': 'CAM表示をオフにしました',
        'status_cam_error': 'CAM生成エラー: {0}',
        'status_driving_inference_on': '自動運転推論結果表示をオンにしました',
        'status_driving_inference_off': '自動運転推論結果表示をオフにしました',
        'status_inference_processing': '推論処理中... モデル: {0} ({1})',
        'status_location_inference_processing': '位置推論処理中... モデル: {0} ({1})',
        'status_inference_for_source_switch': '画像ソース切り替えのため推論を実行中...',
        'status_jumped_to_index': 'インデックス {0} にジャンプしました',
        'status_waypoints_auto_applied': '前回のwaypoint {0}個を自動適用しました',
        'status_reached_first_image': '最初の画像に到達したため自動再生を停止しました',
        'status_reached_last_image': '最後の画像に到達したため自動再生を停止しました',
        'status_bboxes_auto_applied': '前回の {0}個のバウンディングボックスを適用しました',
        'status_segs_auto_applied': '前回の {0}個のセグメンテーションを適用しました',
        'status_seg_auto_applied': "前回の '{0}' セグメンテーションを適用しました",
        'status_training_cancelled': '学習がキャンセルされました',
        'status_updating_waypoint_model_list': 'ウェイポイントモデルリストを更新中...',
        'status_waypoint_model_not_found': '{0}のウェイポイントモデルが見つかりません。モデルを学習してください',
        'status_waypoint_models_found': '{0}個のウェイポイントモデルが見つかりました',
        'status_waypoint_inference_on': 'ウェイポイント推論表示を有効化しました',
        'status_waypoint_inference_off': 'ウェイポイント推論表示を無効化しました',
        'status_updating_location_model_list': '位置モデルリストを更新中...',
        'status_location_model_not_found': '{0}の位置モデルが見つかりません。モデルを学習してください',
        'status_location_models_loaded': '{0}個の{1}位置モデルを読み込みました',
        'status_location_model_loaded': "位置モデル '{0} ({1})' を読み込みました (クラス数: {2})",
        'status_location_model_type_changed': '位置モデルタイプを「{0}」に変更しました。モデルリストを更新します...',
        'status_google_auth': 'Google認証中... ブラウザで認証を完了してください（タイムアウト: 60秒）',
        'status_pretrained_model_saved': '事前学習済み {0} モデルをmodelsフォルダに保存しました: {1}',

        # QMessageBoxダイアログ
        'dlg_warning': '警告',
        'dlg_error': 'エラー',
        'dlg_info': '情報',
        'dlg_complete': '完了',
        'dlg_sync_cancelled': '同期キャンセル',
        'dlg_sync_complete': '同期完了',
        'dlg_sync_complete_with_errors': '同期完了（一部エラー）',
        'sync_progress': '同期中 {0}/{1}',
        'sync_already_running': '同期が実行中です',
        'btn_cancel_sync': '同期キャンセル',
        'dlg_connection_test': '接続テスト',
        'dlg_transfer_complete': '転送完了',
        'dlg_add_complete': '追加完了',
        'msg_location_already_exists': '位置情報 {0} は既に存在します。',
        'msg_location_added': '位置情報 {0} を追加しました。',
        'msg_image_source_switch_failed': "画像ソース '{0}' への切り替えに失敗しました。",
        'msg_readme_not_found': 'READMEが見つかりません:\n{0}',
        'msg_file_open_failed': 'ファイルを開けませんでした:\n{0}',
        'msg_folder_not_found': 'フォルダが存在しません: {0}',
        'msg_images_folder_not_found': 'フォルダの下にimagesフォルダが見つかりません: {0}',
        'msg_no_training_data': '学習データがありません。',
        'msg_insufficient_data': 'データ数が不足しています。最低2枚の画像が必要です。',
        'msg_start_index_error': '開始インデックスは終了インデックス以下である必要があります。',
        'msg_no_images_to_process': '処理対象の画像がありません。',
        'msg_yolo_model_not_loaded': 'YOLOモデルが読み込まれていません。',
        'msg_no_preview_images': 'プレビュー対象の画像がありません。',
        'msg_augmentation_disabled': 'データオーグメンテーションが無効になっています。',
        'msg_preview_error': 'プレビュー生成中にエラーが発生しました: {0}',
        'msg_waypoint_model_load_failed': 'ウェイポイントモデルの読み込みに失敗しました:\n{0}',
        'msg_waypoint_model_not_loaded': 'ウェイポイントモデルが読み込まれていません。',
        'msg_waypoint_training_error': 'ウェイポイントモデルの学習中にエラーが発生しました:\n{0}',
        'msg_no_valid_bbox_annotations': '有効な物体検知アノテーションがありません。\n（削除マーク済み: {0}件）',

        # アノテーションモードツールチップ
        'tip_auto_driving_mode': '自動運転アノテーションモード\n・画像上をクリックして角度(angle)とスロットル(throttle)を設定\n・左クリック: ポイント追加/移動\n・右クリック: ポイント削除\n・数字キー(0-7): 運転位置を設定（同じキー再押下で解除）\n・Deleteキー: 現在の画像のアノテーション（angle/throttle/位置）を削除',
        'tip_waypoint_count': '配置するwaypoint数',
        'tip_cam_target': 'CAMで可視化する出力を選択',
        'tip_waypoint_start_y': 'waypoint開始位置のY座標',
        'tip_waypoint_end_y': 'waypoint終了位置のY座標',
        'status_vertex_moving': "'{0}' 頂点 {1} を移動中... ({2}, {3})",
        'status_bbox_resizing': "'{0}' バウンディングボックスのサイズ変更中... [位置: ({1:.0f}, {2:.0f}), サイズ: {3:.0f}x{4:.0f}]",
        'label_graph_error': 'グラフ作成エラー: {0}',
        'msg_polygon_min_vertices': 'ポリゴンは最低3つの頂点が必要です。\n頂点を削除できません。',
        'btn_location_with_count': '{0} | 位置 {1}',
        'btn_play': '▶再生',
        'btn_stop': '■停止',
        'btn_reverse_play': '◀逆再生',
        'dlg_waypoint_shortage': 'Waypoint不足',
        'msg_waypoint_shortage': '現在の画像には{0}個のwaypointが配置されていますが、\n{1}個必要です。\n\n残り{2}個のwaypointを配置してから次の画像に進んでください。\n\n※配置を中止する場合は、Deleteキーで全てのwaypointを削除してください。',
        'msg_cannot_set_location_deleted': '削除済みの画像には位置情報を設定できません。\n先に「削除状態を復元」を実行してください。',

        # Databricks/Colab関連
        'tip_keep_current_input': '現在の入力内容を保持',
        'label_databricks_combined': '✓ Databricks+ローカル併用',
        'label_databricks_disconnected': '✗ Databricks: 未接続',
        'label_local_mlflow': 'ローカルMLflow使用中',
        'tip_upload_runs': 'ローカルにあってDatabricksにないRunをアップロードします',
        'tip_delete_runs': 'ローカルに存在しないRunをDatabricksから削除します（注意: 元に戻せません）',
        'chk_delete_runs': 'ローカルで削除したRunをDatabricksからも削除 ({0}件)',
        'label_colab_authenticated': '認証済み',
        'label_colab_not_authenticated': '未認証',
        'label_colab_disabled': '無効',
        'label_colab_no_config': '設定ファイルなし',
        'tip_generate_notebook': '転送後にGoogle Colabで使用できるNotebookを生成します',
        'tip_open_colab': '転送完了後にブラウザでColabを開きます',
        'tip_merge_mlruns': 'Colabで記録されたMLflow実験データをローカルにマージします',
        'label_processing_count': '処理予定: {0}枚',
        'label_processing_count_skip': '処理予定: 約{0}枚（{1}枚ごと）',

        'label_no_annotations': 'アノテーションがありません',
        'label_no_valid_annotations': '有効なアノテーションがありません',
        'msg_no_annotations_to_transfer': '転送するアノテーションがありません。\n\n先にアノテーションを作成してください。',
        'msg_no_deleted_annotations_to_restore': '復元する削除済みのアノテーションがありません。',
        'dlg_confirm': '確認',
        'msg_confirm_restore_all': '全ての削除状態をクリアします。よろしいですか？\n\n削除済みインデックス数: {0}個',
        'dlg_zip_filename': 'ZIPファイル名',
        'msg_enter_zip_filename_databricks': 'Databricksに転送するZIPファイル名を入力してください:\n（.zipは自動で付加されます）',
        'msg_enter_zip_filename_gdrive': 'Google Driveに転送するZIPファイル名を入力してください:\n（.zipは自動で付加されます）',

        # モデル読み込み
        'msg_no_valid_model_selected': '有効なモデルが選択されていません。',
        'msg_loading_model': "モデル '{0} ({1})' を読み込み中...",
        'dlg_model_loading': 'モデル読み込み',
        'dlg_clear_inference_confirm': '推論結果のクリア確認',
        'msg_clear_inference_prompt': '現在、{0}個の推論結果が保存されています。\nモデルを変更すると古い推論結果が新しいモデルと不整合を起こす可能性があります。\n\n既存の推論結果をクリアしますか？',
        'msg_existing_inference': '既存の推論結果: {0}個\n確認ダイアログを表示します...',
        'msg_clearing_inference': '既存の推論結果をクリア中...',
        'msg_cleared_old_inference': '{0}個の古い推論結果をクリアしました',
        'msg_init_model_arch': 'モデルアーキテクチャの初期化中...',
        'msg_loading_model_file': 'モデルファイルを読み込み中: {0}',
        'msg_init_model': 'モデルを初期化中...',
        'msg_transfer_to_device': 'モデルを{0}に転送中...',
        'msg_running_inference': '推論実行中: {0}',
        'msg_saving_inference': '推論結果を保存中...',
        'msg_updating_inference': '推論表示を更新中...',
        'msg_model_loaded_suffix': ' (古い推論結果はクリアされました)',
        'msg_model_loaded': "モデル '{0} ({1})' を読み込みました{2}",
        'msg_model_loaded_detail': "モデル '{0} ({1})' を読み込みました。",
        'msg_new_inference_available': '\n\n{0}個の新しい推論結果が利用可能です。',
        'msg_existing_kept': '\n\n既存の推論結果は保持されています。必要に応じて「一括推論実行」ボタンで更新してください。',
        'msg_inference_auto_on': '\n\n推論結果表示が自動的にオンになりました。',
        'msg_model_load_error': 'モデル読み込み中にエラーが発生しました: {0}',

        # オートアノテーション
        'dlg_auto_annotation_settings': 'オートアノテーション設定',
        'dlg_existing_annotation_handling': '既存アノテーションの処理',
        'btn_overwrite': '上書き',
        'btn_append': '追加',
        'dlg_augmentation_preview': 'オーグメンテーションプレビュー',
        'dlg_insufficient_data': 'データ不足',
        'msg_insufficient_segmentation_data': '有効なセグメンテーションデータが {0} 枚しかありません。\nセグメンテーション学習には最低4枚以上の画像が推奨されます。',
        'msg_yolo_training_error': 'YOLO{0}モデル学習中にエラーが発生しました: {1}',
        'dlg_no_segmentation_data': 'セグメンテーションデータなし',
        'msg_no_segmentation_generate_from_bbox': '有効なセグメンテーションアノテーションが見つかりません。\n\nバウンディングボックスから矩形セグメンテーションを自動生成しますか？\n（より高精度な結果を得るには、手動でポリゴンアノテーションを作成することを推奨します）',
        'msg_no_segmentation_manual_required': '有効なセグメンテーションアノテーションが見つかりません。\n\nセグメンテーション学習には最低3点以上のポリゴンアノテーションが必要です。\n手動でポリゴンアノテーションを作成してから再試行してください。',
        'label_pretrained_weights_downloaded': '事前学習済みの重み (ダウンロード済み: {0})',
        'label_current_model_weights': '現在のモデル重み: {0}',
        'msg_yolo_training_complete': 'YOLO{0}モデルの学習が完了しました。\n最終mAP: {1}\n使用デバイス: {2}\n初期化: {3}\n\nモデル保存先: {4}\n{5}',
        'msg_need_manual_annotation': 'オートアノテーションを実行するには、まず数枚の画像に手動でアノテーションを行ってください。',
        'dlg_auto_annotation_range': 'オートアノテーション範囲指定',
        'label_range': '範囲',
        'label_all_unannotated': 'すべてのアノテーションされていない画像',
        'label_specify_index_range': 'インデックス範囲を指定',
        'tip_overwrite': 'チェックすると、既にアノテーションされている画像も再推論で上書きします',
        'msg_no_unannotated_in_range': '指定範囲にアノテーションされていない画像がありません。',
        'msg_auto_annotation_preparing': 'オートアノテーション準備中... ({0}枚の画像)',
        'dlg_auto_annotation_running': 'オートアノテーション実行中',
        'msg_preparing_model': "モデル '{0}' を使用した処理を準備中...",
        'msg_init_model_type': "モデル '{0}' を初期化中...",
        'msg_loading_model_basename': "モデル '{0}' を読み込み中...",
        'msg_preparing_pretrained': "事前学習済みモデル '{0}' を準備中...",
        'msg_batch_processing': 'バッチ {0}/{1} 処理中...\n画像 {2}-{3}/{4}',
        'msg_batch_image_processing': 'バッチ {0}/{1} 処理中...\n画像 {2}/{3} を処理中',
        'msg_updating_location_buttons': '位置情報ボタンを更新中...',
        'msg_verifying_image_files': '画像ファイルを確認中...',
        'msg_organizing_image_data': '画像データを整理中...',
        'msg_updating_display': '画面を更新中...',
        'msg_parsing_manifest': 'マニフェストファイルを解析中...',
        'msg_parsing_image_index': '画像ファイルのインデックスを解析中...',
        'msg_updating_gallery': 'ギャラリー表示を更新中...',
        'msg_using_detection_model': '物体検知モデルを使用します...',
        'msg_using_segmentation_model': 'セグメンテーションモデルを使用します...',
        'msg_running_segmentation': 'セグメンテーションを実行中...',
        'msg_running_detection': '物体検知を実行中...',
        'msg_integrating_annotations': 'アノテーションデータを統合中...',
        'msg_saving_training_curve': '学習曲線を保存中...',
        'msg_recording_mlflow': 'MLflowに学習結果を記録中...',
        'msg_updating_ui': 'UI表示を更新中...',
        'msg_updating_graph': '分布グラフを更新中...',
        'msg_auto_annotation_complete': '完了: {0}枚の画像にオートアノテーションを適用しました',
        'msg_auto_annotation_success': '{0}枚の画像にオートアノテーションを適用しました。\n使用モデル: {1}{2}',
        'label_pretrained_suffix': ' (事前学習済み)',
        'dlg_cancelled': 'キャンセル',
        'msg_auto_annotation_cancelled': 'オートアノテーションがキャンセルされました。\n{0}枚の画像が処理されました。',
        'msg_auto_annotation_error': 'オートアノテーション中にエラーが発生しました: {0}',

        # 一括推論
        'dlg_batch_inference_confirm': '一括推論実行確認',
        'msg_batch_inference_prompt': '全{0}枚の画像に対して推論を実行します。\n現在のモデル: {1}{2}\n\n進行中は操作ができなくなります。続行しますか？',
        'dlg_existing_inference': '既存の推論結果',
        'msg_overwrite_inference_prompt': '現在、{0}個の推論結果が保存されています。これらを上書きしますか？\n\n「はい」: 全ての推論結果を新しいモデルで上書きします。\n「いいえ」: 推論結果がない画像のみ処理します。',
        'msg_preparing_inference': '推論処理の準備中...',
        'dlg_batch_inference_running': '一括推論実行中',
        'msg_batch_inference_cancelled': '一括推論がキャンセルされました。\n処理済み: {0}/{1}枚\n成功: {2}枚, スキップ: {3}枚',
        'msg_batch_inference_complete': '全画像の推論が完了しました。\n処理済み: {0}枚\n成功: {1}枚, スキップ: {2}枚\n\n推論結果表示がONになりました。',
        'msg_batch_inference_error': '一括推論実行中にエラーが発生しました: {0}',

        # データ分析ダイアログ
        'dlg_data_analysis': 'データ分析',
        'section_stats_distribution': '統計・分布',
        'section_timeseries': '時系列',
        'label_stats_item': '項目',
        'label_stats_mean': '平均',
        'label_stats_std': '標準偏差',
        'label_stats_min': '最小',
        'label_stats_max': '最大',
        'label_stats_median': '中央値',
        'label_display': '表示:',
        'label_raw_data': '生データ',
        'label_moving_avg': '移動平均',
        'label_bin_avg': '区間平均',
        'label_window': '窓:',
        'label_bin': '区間:',
        'label_display_range': '表示範囲:',
        'label_auto': '自動',
        'label_display_items': '表示項目:',
        'label_click_to_jump': 'グラフをクリックすると該当画像にジャンプします',
        'btn_close': '閉じる',
        'label_angle_original': 'Angle(元: {0})',
        'label_throttle_original': 'Throttle(元: {0})',
        'label_angle_ds': 'Angle(DS後: {0})',
        'label_throttle_ds': 'Throttle(DS後: {0})',
        'label_dist_title_with_ds': 'Angle / Throttle 分布 (n={0}, DS除外: {1})',
        'label_dist_title': 'Angle / Throttle 分布 (n={0})',
        'label_value': '値',
        'label_frequency': '頻度',
        'label_no_data': 'データなし',
        'label_select_display_item': '表示項目を選択してください',
        'label_no_data_available': 'データがありません',
        'label_bin_avg_title': '区間平均（{0}インデックスごと）',
        'label_data_trend_ma': 'データ推移（移動平均: 窓{0}）',
        'label_data_trend': 'データ推移',
        'label_index': 'インデックス',

        # Databricks/Colab設定ダイアログ
        'label_not_set': '(未設定)',
        'label_using_default': '(デフォルト使用)',
        'msg_env_setup_help': 'セキュリティのため、認証情報は環境変数で設定してください:\n\nWindows (PowerShell):\n  $env:DATABRICKS_ENABLED = "true"\n  $env:DATABRICKS_HOST = "https://..."\n  $env:DATABRICKS_TOKEN = "dapi..."\n\nLinux/Mac:\n  export DATABRICKS_ENABLED="true"\n  export DATABRICKS_HOST="https://..."\n  export DATABRICKS_TOKEN="dapi..."',
        'msg_oauth_setup_guide': 'Google Colab連携を有効にするには、以下の手順を実行してください:\n\n1. Google Cloud Consoleでプロジェクトを作成\n2. Google Drive APIを有効化\n3. OAuth 2.0クライアントIDを作成し、client_secrets.jsonをダウンロード\n4. 環境変数を設定:\n   COLAB_ENABLED=true\n   GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json',
        'tab_oauth_guide': 'OAuth設定ガイド',
        'dlg_auth_required': '認証が必要',
        'msg_auth_required': '初回接続のため、ブラウザでGoogleアカウント認証が必要です。\n\nブラウザが開いたら、Googleアカウントを選択して認証を完了してください。\n（タイムアウト: 60秒）\n\n続行しますか？',
        'dlg_import_error': 'インポートエラー',
        'msg_import_error_colab': '必要なライブラリがインストールされていません:\n\n{0}\n\npip install pydrive2 google-auth google-auth-oauthlib pyyaml でインストールしてください。',
        'dlg_auth_timeout': '認証タイムアウト',
        'msg_auth_timeout': '{0}\n\nブラウザを閉じた場合や、認証に時間がかかりすぎた場合に発生します。\n再度「接続テスト」ボタンをクリックしてお試しください。',
        'dlg_connection_test_error': '接続テストエラー',
        'msg_connection_test_error': '接続テスト中にエラーが発生しました:\n\n{0}',

        # 表示設定ダイアログ
        'label_current_size': '現在のサイズ: {0} x {1}',
        'label_current_font_size': '現在のフォントサイズ: {0}pt',
        'status_display_settings_applied': '表示設定を適用しました - ウィンドウ: {0}x{1}, フォント: {2}pt',

        # Databricks/Colabステータス
        'status_disabled': '無効',
        'status_config_error': '設定エラー',
        'status_configured': '設定済み',
        'msg_databricks_disabled': 'Databricks連携は無効です（ローカルMLflowを使用）\n\n有効にするには環境変数を設定してください:\n  DATABRICKS_ENABLED=true\n  DATABRICKS_HOST=https://...\n  DATABRICKS_TOKEN=dapi...',
        'msg_databricks_workspace': 'Databricksワークスペース: {0}',
        'msg_env_host_not_set': '環境変数 DATABRICKS_HOST が設定されていません',
        'msg_env_host_https_required': 'DATABRICKS_HOST は https:// で始まる必要があります',
        'msg_env_token_not_set': '環境変数 DATABRICKS_TOKEN が設定されていません',
        'msg_colab_disabled': 'Google Colab連携は無効です\n\n有効にするには環境変数を設定してください:\n  COLAB_ENABLED=true\n  GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json',
        'msg_colab_workspace': 'Google Drive フォルダ: {0}',
        'msg_env_client_secrets_not_set': '環境変数 GOOGLE_CLIENT_SECRETS が設定されていません',
        'msg_env_client_secrets_not_found': 'クライアントシークレットファイルが見つかりません: {0}',
        'status_authenticated': '認証済み',
        'status_not_authenticated': '未認証',
        'msg_colab_authenticated': '\n認証済み',
        'msg_colab_auth_required': '\n要認証（初回転送時にブラウザ認証）',
        'msg_databricks_env_template': '''# Databricks設定用環境変数

# Windows (PowerShell):
$env:DATABRICKS_ENABLED = "true"
$env:DATABRICKS_HOST = "https://your-workspace.cloud.databricks.com"
$env:DATABRICKS_TOKEN = "dapi..."
$env:DATABRICKS_EXPERIMENT_PREFIX = "/Users/your-email@example.com/experiments"

# Windows (コマンドプロンプト):
set DATABRICKS_ENABLED=true
set DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
set DATABRICKS_TOKEN=dapi...
set DATABRICKS_EXPERIMENT_PREFIX=/Users/your-email@example.com/experiments

# Linux/Mac:
export DATABRICKS_ENABLED="true"
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_EXPERIMENT_PREFIX="/Users/your-email@example.com/experiments"

# .env ファイル形式:
DATABRICKS_ENABLED=true
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
DATABRICKS_EXPERIMENT_PREFIX=/Users/your-email@example.com/experiments
''',
        'msg_colab_env_template': '''# Google Colab設定用環境変数

# Windows (PowerShell):
$env:COLAB_ENABLED = "true"
$env:GOOGLE_CLIENT_SECRETS = "C:\\path\\to\\client_secrets.json"
$env:COLAB_DRIVE_FOLDER_NAME = "annotation_data"

# Windows (コマンドプロンプト):
set COLAB_ENABLED=true
set GOOGLE_CLIENT_SECRETS=C:\\path\\to\\client_secrets.json
set COLAB_DRIVE_FOLDER_NAME=annotation_data

# Linux/Mac:
export COLAB_ENABLED="true"
export GOOGLE_CLIENT_SECRETS="/path/to/client_secrets.json"
export COLAB_DRIVE_FOLDER_NAME="annotation_data"

# .env ファイル形式:
COLAB_ENABLED=true
GOOGLE_CLIENT_SECRETS=/path/to/client_secrets.json
COLAB_DRIVE_FOLDER_NAME=annotation_data
''',
        'msg_oauth_setup_guide_full': '''================================================================================
Google Cloud Console での OAuth設定手順
================================================================================

1. Google Cloud Console にアクセス
   https://console.cloud.google.com/

2. プロジェクトを作成または選択
   - 画面上部のプロジェクト選択メニューから「新しいプロジェクト」
   - プロジェクト名を入力して作成

3. Google Drive API を有効化
   - 左メニューから「APIとサービス」→「ライブラリ」
   - 「Google Drive API」を検索
   - 「有効にする」をクリック

4. OAuth同意画面を設定
   - 「APIとサービス」→「OAuth同意画面」
   - ユーザータイプ: 「外部」を選択（個人使用の場合）
   - アプリ名、メールアドレスを入力
   - スコープの追加は不要（後で自動設定される）
   - テストユーザーに自分のメールアドレスを追加

5. OAuth クライアントIDを作成
   - 「APIとサービス」→「認証情報」
   - 「認証情報を作成」→「OAuthクライアントID」
   - アプリケーションの種類: 「デスクトップアプリ」
   - 名前を入力して「作成」

6. client_secrets.json をダウンロード
   - 作成したクライアントIDの右側にあるダウンロードアイコンをクリック
   - 「JSONをダウンロード」
   - ファイル名を「client_secrets.json」に変更して保存

7. 環境変数を設定
   COLAB_ENABLED=true
   GOOGLE_CLIENT_SECRETS=保存したclient_secrets.jsonのパス

================================================================================
注意事項
================================================================================

- client_secrets.json は秘密情報です。Gitにコミットしないでください
- .gitignore に client_secrets.json を追加することを推奨します
- 初回転送時にブラウザでGoogleアカウントの認証が求められます
- 認証情報は .google_credentials.json に保存され、2回目以降は自動的に使用されます

================================================================================''',

        # MLflow UI
        'dlg_mlflow_ui': 'MLflow UI',
        'msg_mlflow_ui_started': 'ローカルMLflow UIを起動しました。\n\nブラウザで http://localhost:5000 にアクセスして実験結果を確認できます。\n\nUIを終了するには、コマンドウィンドウを閉じてください。',
        'msg_mlflow_ui_failed': 'MLflow UIの起動に失敗しました:\n\n{0}\n\nMLflowがインストールされているか確認してください: pip install mlflow',
        'dlg_databricks_not_enabled': 'Databricks未有効',
        'msg_databricks_not_enabled': 'Databricks連携が有効になっていません。\n\n「Databricks連携」チェックボックスをONにしてください。',
        'msg_databricks_enable_confirm': 'Databricks連携が有効になっていません。\n\n有効にして接続しますか？',
        'dlg_databricks_connection_failed': 'Databricks接続失敗',
        'msg_databricks_connection_failed': 'Databricksへの接続に失敗しました。\n\n環境変数の設定を確認してください。',
        'dlg_databricks_connection_success': 'Databricks接続成功',
        'msg_databricks_connection_success': 'Databricksへの接続に成功しました。',
        'dlg_databricks_connection_error': 'Databricks接続エラー',
        'msg_databricks_connection_error_env': 'Databricksへの接続に失敗しました。\n\n環境変数の設定を確認してください：\n- DATABRICKS_HOST\n- DATABRICKS_TOKEN\n\nローカルMLflowモードにフォールバックします。',
        'status_disconnected': '未接続',
        'dlg_volumes_path_not_exist': 'Volumesパスが存在しません',
        'msg_volumes_path_not_exist': '転送先のVolumesパスが存在しません:\n\n{0}\n\n詳細: {1}\n\nDatabricksでこのパスを作成してから再度お試しください。\n\n環境変数 DATABRICKS_VOLUMES_PATH で\n別のパスを指定することもできます。\n\n例: /Volumes/workspace/default/test',
        'dlg_transfer_confirm': '転送確認',
        'msg_transfer_confirm_databricks': '以下の内容でDatabricksに転送します:\n\nアノテーション数: {0}\nファイル名: {1}\n転送先: {2}/{1}\n\n続行しますか？',
        'dlg_colab_not_enabled': 'Google Colab未有効',
        'msg_colab_not_enabled': 'Google Colab連携が有効になっていません。\n\n有効にするには環境変数を設定してください:\n  COLAB_ENABLED=true\n  GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json\n\n設定ボタンから詳細を確認できます。',

        # 同期ダイアログ
        'label_local_runs': 'ローカルのRun数: {0}',
        'label_databricks_runs': 'DatabricksのRun数: {0}',
        'label_unsynced_runs': '推定未同期Run数: {0}',
        'label_orphaned_runs': 'Databricksにのみ存在するRun数: {0}',
        'dlg_delete_confirm': '削除確認',
        'msg_delete_runs_confirm': 'Databricksから{0}件のRunを削除します。\n\nこの操作は元に戻せません。続行しますか？',

        # 転送完了/エラー
        'dlg_transfer_complete': '転送完了',
        'msg_transfer_complete_databricks': 'Databricksへの転送が完了しました。\n\nアノテーション数: {0}\nZIPサイズ: {1:.2f} MB\n転送先: {2}',
        'dlg_transfer_error': '転送エラー',
        'msg_transfer_error': '転送中にエラーが発生しました:\n\n{0}',
        'msg_unknown_error': '不明なエラー',
        'label_google_drive_models': 'Google Drive上のモデル: {0}件',
        'label_unknown_date': '不明',

        # 自動学習パイプライン
        'chk_auto_train_after_transfer': '転送後に自動学習を開始する',
        'tip_auto_train_cluster_required': '自動学習にはクラスターIDの設定が必要です（環境変数: DATABRICKS_CLUSTER_ID）',
        'msg_auto_train_started': '自動学習を開始しました (Run ID: {0})',
        'msg_auto_train_failed': '転送は成功しましたが学習の起動に失敗しました:\n{0}',
        'msg_transfer_and_train_complete': 'Databricksへの転送が完了しました。\n\nアノテーション数: {0}\nZIPサイズ: {1:.1f} MB\n転送先: {2}\n\n自動学習を開始しました (Run ID: {3})',
        'label_cluster_id': 'クラスターID',
        'label_notebook_path': 'ノートブックパス',
        'label_auto_train_settings': '自動学習パイプライン設定',
        'label_set_via_env': '環境変数で設定してください',
    },

    # =========================================================================
    # English
    # =========================================================================
    'en': {
        # =====================================================================
        # app_ : Application / Window Settings
        # =====================================================================
        'app_title': 'Image Annotation Tool',
        'app_language': 'Language',
        'app_language_ja': '日本語',
        'app_language_en': 'English',
        'app_language_switch': 'Language',
        'app_language_changed': 'Language setting changed. Restart required to apply.',
        'app_restart_required': 'Restart Required',

        # =====================================================================
        # section_ : Section Titles
        # =====================================================================
        'section_data_load': 'Data Load (parent folder of images folder):',
        'section_save_annotation': 'Save Annotation Data:',
        'section_pilot_model': 'Autonomous Driving Model:',
        'section_object_detection': 'Object Detection / Segmentation Model:',
        'section_model_management': 'Model Management / Cloud Training:',
        'section_display_settings': 'Display Settings:',

        # =====================================================================
        # btn_ : Button Labels
        # =====================================================================
        # --- Data Load ---
        'btn_browse': 'Browse...',
        'btn_load_images': 'Load Images',
        'btn_load_annotations': 'Load Annotations',

        # --- Export ---
        'btn_create_video': 'Create Video',

        # --- Pilot Model ---
        'btn_train_save': 'Train',
        'btn_load_model': 'Load Model',
        'btn_auto_annotate': 'Auto Annotate',
        'btn_batch_inference': 'Infer All',

        # --- Object Detection ---
        'btn_load_yolo_annotation': 'Load YOLO',
        'btn_preset': 'Preset',
        'btn_apply': 'Apply',
        'btn_train_yolo': 'Train',
        'btn_yolo_auto_annotate': 'YOLO Auto',

        # --- Model Management ---
        'btn_open_mlflow': 'Open MLflow',
        'btn_open_databricks': 'Databricks',
        'btn_sync': 'Sync',
        'btn_transfer': 'Send',
        'btn_settings': 'Settings',
        'btn_open_colab': 'Colab',
        'btn_download': 'Get',

        # --- Display Settings ---
        'btn_window_font_settings': 'Preference',

        # --- Navigation ---
        'btn_reverse_play': '◀ Reverse',
        'btn_forward_play': '▶ Play',
        'btn_delete_current': 'Delete Current',
        'btn_restore_deleted': 'Restore Deleted',
        'btn_restore_all_deleted': 'Restore All',
        'btn_current_position': 'Current',
        'btn_range_delete': 'Delete Range',

        # --- Downsampling ---
        'btn_detect': 'Detect',
        'btn_redetect': 'Re-detect',
        'btn_clear': 'Clear',

        # --- Info Panel ---
        'btn_analysis': 'Analysis',

        # =====================================================================
        # label_ : Labels / Descriptions
        # =====================================================================
        # --- Data Load ---
        'label_annotated_count': 'Annotated: {0} / {1}',
        'label_image_count': 'Image {0} of {1}:{2}',
        'label_deleted_suffix': '[Deleted]',
        'label_image_source': 'Image Source',

        # --- Pilot Model ---
        'label_pilot_model_select': 'Driving Model:',

        # --- Object Detection ---
        'label_detection_classes': 'Detection Classes:',
        'label_classes_example': 'e.g.: car,red_sign,green_sign,dog',
        'label_yolo_model': 'YOLO Model:',

        # --- Model Management ---
        'label_mlflow_local': 'MLflow (Local):',
        'label_databricks_integration': 'Databricks Integration',
        'label_colab_integration': 'Google Colab Integration',

        # --- Navigation ---
        'label_canvas_zoom_label': 'Zoom',
        'label_canvas_zoom_tooltip': 'Adjust canvas zoom level',
        'label_image_seek': 'Image Seek:',
        'label_play': 'Play:',
        'label_delete_restore': 'Delete/Restore:',
        'label_delete_range': 'Delete Range:',
        'label_from': 'to',

        # --- Downsampling ---
        'label_downsampling': 'Downsampling:',
        'label_angle_range': 'Angle Range:',
        'label_throttle_range': 'Throttle Range:',
        'label_consecutive': 'Consecutive:',
        'label_interval': 'Interval:',
        'label_items': '{0} items',
        'label_items_added': '(+{0} items)',

        # --- CAM Settings ---
        'label_cam_method': 'Method:',
        'label_cam_target': 'Target:',
        'label_cam_direction': 'Direction:',

        # --- Info Panel ---
        'label_image_info': 'Image Info',
        'label_data_distribution': 'Data Distribution',
        'label_no_annotation': 'No annotations',
        'label_no_image_selected': 'No image selected',
        'label_inference': 'Infer:{0}',

        # --- Other ---
        'label_select_folder_prompt': 'Select a folder and click Load button',

        # =====================================================================
        # chk_ : Checkboxes
        # =====================================================================
        'chk_show_future_annotation': '+5, +10 frames(orange)',
        'chk_show_inference': 'Inference result (blue)',
        'chk_show_diff_vector': 'Diff vector (green arrow)',
        'chk_show_detection_inference': 'Detection Inference',
        'chk_show_segmentation_inference': 'Segmentation Inference',
        'chk_dark_mode': 'Dark Mode',

        # =====================================================================
        # placeholder_ : Placeholders
        # =====================================================================
        'placeholder_folder': 'Enter folder path or use Browse for multiple selection',
        'placeholder_classes': 'Enter class names separated by commas',

        # =====================================================================
        # tip_ : Tooltips
        # =====================================================================
        # --- Pilot Model ---
        'tip_load_model': 'Load model from models folder',
        'tip_show_future_annotation': 'Show annotations for 5 and 10 frames ahead',

        # --- CAM Settings ---
        'tip_cam_method': 'Select CAM visualization method\nScoreCAM: High accuracy without gradients (slower)',
        'tip_cam_target': 'Select output to visualize with CAM',
        'tip_cam_direction': 'Gradient direction to visualize\nboth: Show positive/negative simultaneously (red=positive/blue=negative)\npositive: Evidence for increasing output (turn right/accelerate)\nnegative: Evidence for decreasing output (turn left/decelerate)',

        # --- Object Detection ---
        'tip_train_yolo': 'Train object detection or segmentation',

        # --- Model Management ---
        'tip_open_mlflow': 'Launch local MLflow UI',
        'tip_open_databricks': 'Open Databricks MLflow UI',
        'tip_sync': 'Upload local training records to Databricks',
        'tip_transfer': 'Transfer current annotations',
        'tip_open_colab': 'Open Google Colab in browser',
        'tip_colab_transfer': 'Transfer annotations to Google Drive for Colab training',
        'tip_colab_download': 'Download Colab-trained model from Google Drive',

        # --- Navigation ---
        'tip_set_start': 'Set current index as start position',
        'tip_set_end': 'Set current index as end position',

        # --- Downsampling ---
        'tip_consecutive': 'Mark as downsampling target when consecutive count exceeds this',
        'tip_interval': 'Keep every N frames (e.g., 3 means keep 1 of every 3, 0 targets all)',
        'tip_detect': 'Detect indexes matching criteria and mark as downsampling targets',
        'tip_clear_downsampling': 'Clear all downsampling targets',

        # --- Info Panel ---
        'tip_analysis': 'Statistical analysis and visualization of annotation data',

        # --- Mode Tooltips ---
        'tip_auto_driving_mode': 'Auto Driving Annotation Mode\n- Click image to set Angle/Throttle\n- Right-click: Delete point\n- Number keys (0-7): Set driving position (press again to unset)\n- Delete key: Delete annotation (angle/throttle/position) for current image',
        'tip_detection_mode': 'Object Detection Annotation Mode\n- Drag to create bounding box\n- Click created box to select/move\n- Drag box corners to resize\n- Right-click: Delete selected box\n- Delete key: Delete selected box',
        'tip_segmentation_mode': 'Segmentation Annotation Mode\n- Left-click: Add polygon vertex\n- Right-click: Close/complete polygon\n- Right-click on polygon: Add new vertex\n- Drag vertex: Adjust vertex position\n- Delete key: Delete selected polygon\n- Esc key: Cancel polygon in progress',
        'tip_waypoint_mode': 'Waypoint Annotation Mode\n- Left-click: Add waypoint coordinate\n- Right-click: Delete last waypoint\n- Waypoints are displayed as green circles\n- Delete key: Delete all waypoints for current image',

        # =====================================================================
        # status_ : Status Bar Messages
        # =====================================================================
        'status_bbox_hint': 'Hold B key and click to create bounding box. Press Delete to remove selected box.',
        'status_deleted': 'Deleted',
        'status_ds_target': 'DS Target',
        'status_click_to_reannotate': 'Deleted\nClick to re-annotate',
        'status_model_not_loaded': 'Autonomous model not loaded',
        'status_detection_model_not_loaded': 'Detection model not loaded',
        'status_segmentation_model_not_loaded': 'Segmentation model not loaded',

        # =====================================================================
        # dlg_ : Dialog Titles
        # =====================================================================
        'dlg_warning': 'Warning',
        'dlg_error': 'Error',
        'dlg_info': 'Information',
        'dlg_complete': 'Complete',
        'dlg_confirm': 'Confirm',
        'dlg_select_folder': 'Select Folder',
        'dlg_export_complete': 'Export Complete',
        'dlg_load_complete': 'Load Complete',
        'dlg_load_complete_warning': 'Load Complete (with warnings)',
        'dlg_load_error': 'Load Error',
        'dlg_sync_complete': 'Sync Complete',
        'dlg_sync_cancel': 'Sync Cancelled',
        'dlg_transfer_complete': 'Transfer Complete',
        'dlg_transfer_cancel': 'Transfer Cancelled',
        'dlg_add_complete': 'Add Complete',
        'dlg_copy_complete': 'Copy Complete',

        # =====================================================================
        # msg_ : Dialog / Error Messages
        # =====================================================================
        'msg_no_images': 'No images loaded.',
        'msg_no_images_loaded': 'No images loaded. Please load images first.',
        'msg_no_annotations': 'No annotations available for training.',
        'msg_no_task_selected': 'No task selected.',
        'msg_no_classes': 'Detection classes not configured.\nPlease configure classes first.',
        'msg_no_save_folder': 'Save folder not specified.',
        'msg_no_export_format': 'No export format selected.',
        'msg_no_export_data': 'No annotations to export.',
        'msg_no_valid_model': 'No valid YOLO model selected.',
        'msg_model_not_found': 'Selected model not found: {0}',
        'msg_load_images_first': 'Please load images first.',
        'msg_package_not_installed': '{0} package is not installed.\nPlease install with: pip install {0}',
        'msg_no_detection_annotations': 'No object detection annotations.',
        'msg_no_segmentation_annotations': 'No segmentation annotations.',
        'msg_pretrained_model_failed': 'Failed to prepare pre-trained {0} model.',
        'msg_no_image_source': 'No image source selected.',
        'msg_class_settings_applied': 'Class settings applied.',
        'msg_no_class_name': 'No class name entered.',
        'msg_no_valid_class': 'No valid class names.',
        'msg_export_failed': 'Export failed.',
        'msg_downsample_cleared': 'Downsampling targets cleared.',
        'msg_location_exists': 'Location {0} already exists.',
        'msg_location_added': 'Location {0} added.',
        'label_location_button': 'Location {0}',
        'label_inference_location': 'Inference Location {0}',
        'msg_env_copied': 'Environment variable template copied to clipboard',
        'msg_readme_not_found': 'README not found:\n{0}',
        'msg_file_open_error': 'Failed to open file:\n{0}',
        'msg_model_download_failed': 'Model download failed.',
        'msg_transfer_cancelled': 'Transfer cancelled.',
        'msg_no_sync_option': 'No sync option selected.',
        'msg_load_data_first': 'Please load images and annotation data first.',
        'msg_angle_range_error': 'Angle range minimum must be less than maximum.',
        'msg_throttle_range_error': 'Throttle range minimum must be less than maximum.',
        'msg_no_downsample_targets': 'No downsampling targets.',
        'msg_load_folder_first': 'Please select and load an image folder first.',
        'msg_no_annotation_data': 'No annotation data available.',
        'msg_inference_method_not_supported': 'Inference method not supported.',
        'msg_folder_not_exists': 'Folder does not exist: {0}',
        'msg_images_folder_not_found': 'Images folder not found in: {0}',
        'msg_no_images_in_folder': 'No image files found in the images folder.',
        'msg_no_subfolders': 'No subfolders found in current folder.',
        'msg_image_source_switch_failed': "Failed to switch to image source '{0}'.",
        'msg_segmentation_model_not_loaded': 'Current segmentation model not loaded. Please use a pre-trained model or load a model and try again.',
        'msg_detection_model_not_loaded': 'Current detection model not loaded. Please use a pre-trained model or load a model and try again.',

        # =====================================================================
        # Annotation Mode / Control Panel
        # =====================================================================
        'label_annotation_mode': 'Annotation Mode:',
        'label_waypoint_control': 'Waypoint Control:',
        'label_num_points': 'Num Points:',
        'label_point_position': 'Point Position:',
        'label_steering_direction': 'Steering Direction:',
        'label_class_id': 'Class ID:',
        'label_y_coordinate': 'Y Coordinate:',
        'label_max_steering': 'Max Steering:',
        'label_display_mode': 'Display Mode:',
        'label_mode_hint': '* Press B key to switch mode',
        'label_location_info': 'Course Location Info:',
        'label_current_location': 'Current Location: None',
        'label_current_location_value': 'Current Location: {0}',
        'label_gallery': 'Gallery:',
        'label_deleted': 'Deleted',

        # =====================================================================
        # Training Dialog
        # =====================================================================
        'dlg_model_training': 'Model Training',
        'dlg_select_task': 'Select task to train:',
        'label_bbox_none': '✗ Bounding Box: None',
        'label_bbox_count': '✓ Bounding Box: {0}',
        'label_seg_none': '✗ Segmentation: None',
        'label_seg_count': '✓ Segmentation: {0}',
        'label_current_model': 'Current Model: None',
        'label_current_model_value': 'Current Model: {0}',
        'label_epochs': 'Training Epochs:',
        'label_batch_size': 'Batch Size:',
        'label_input_image_size': 'Input Image Size:',
        'label_size_note': 'Note: Selecting sizes other than 640 may affect accuracy and speed',
        'label_patience': 'Patience Epochs:',
        'label_learning_rate': 'Learning Rate:',
        'label_probability': 'Probability:',
        'label_hue': 'Hue (H):',
        'label_saturation': 'Saturation (S):',
        'label_brightness': 'Brightness (V):',
        'label_translation': 'Translation:',
        'label_scale': 'Scale:',
        'label_model_name': 'Model Name:',
        'label_comment': 'Comment:',

        # =====================================================================
        # Export Dialog
        # =====================================================================
        'label_save_folder': 'Save Folder:',
        'label_select_image_source': 'Select image source(s) to export (multiple selection allowed):',
        'label_donkey_key_note': "* Donkeycar's default key is 'cam/image_array'.",
        'dlg_yolo_export': 'YOLO Annotation Export',

        # =====================================================================
        # Preset Dialog
        # =====================================================================
        'dlg_select_preset': 'Select a commonly used class set:',
        'dlg_class_preset_select': 'Class Preset Selection',
        'label_custom': 'Custom:',
        'placeholder_comma_separated_classes': 'Enter class names separated by commas',

        # =====================================================================
        # Settings Dialog
        # =====================================================================
        'dlg_display_settings': 'Display Settings',
        'label_width': 'Width:',
        'label_height': 'Height:',
        'label_preset': 'Preset:',
        'label_font_size': 'Size:',
        'label_preview': 'Preview: Annotation Tool',

        # =====================================================================
        # Sync Dialog
        # =====================================================================
        'dlg_syncing': 'Syncing',
        'dlg_transferring_to_databricks': 'Transferring to Databricks',
        'dlg_transferring_to_colab': 'Transferring to Google Colab',
        'msg_syncing_to_databricks': 'Syncing to Databricks...',
        'msg_preparing_transfer': 'Preparing transfer...',
        'msg_authenticating_google_drive': 'Authenticating with Google Drive...',
        'msg_connecting_google_drive': 'Connecting to Google Drive...',
        'msg_downloading_mlflow_data': 'Downloading MLflow experiment data...',
        'msg_downloading_model': 'Downloading model...',
        'dlg_authenticating': 'Authenticating',
        'dlg_fetching_model_list': 'Fetching model list',
        'dlg_download_model': 'Download Model',
        'dlg_downloading': 'Downloading',
        'dlg_model_download': 'Model Download',
        'msg_sync_delete_warning': '* Enabling delete option will remove Runs from Databricks.\n   This action cannot be undone.',

        # =====================================================================
        # Dark Mode
        # =====================================================================
        'btn_dark_mode': 'Dark Mode',
        'btn_light_mode': 'Light Mode',

        # =====================================================================
        # Location / Waypoint Inference Model
        # =====================================================================
        'label_location_model': 'Location Inference Model:',
        'label_model_type': 'Model Type:',
        'label_waypoint_model': 'Waypoint Inference Model:',
        'chk_location_inference': 'Show Location Inference',
        'chk_waypoint_inference': 'Show Waypoint Inference',

        # =====================================================================
        # Info Panel - Driving Annotation
        # =====================================================================
        'label_driving_annotation_info': 'Driving Annotation Info:',
        'label_location_inference_result': 'Location Inference Result:',
        'label_driving_inference_header': 'Driving Inference Result:',
        'label_inference_result_header': 'Inference Result:',
        'label_detection_inference_header': 'Detection Inference Result:',
        'label_detected_objects': 'Detected Objects:',
        'label_object_count': '{0}',
        'label_total_objects': 'Total: {0} objects',
        'label_segmentation_inference_header': 'Segmentation Inference Result:',
        'label_position_rank': '{0}. Position {1}: {2}',
        'label_pretrained': 'pre-trained',
        'dlg_inference_recalculate': 'Inference Recalculation',
        'msg_inference_recalculate_body': "Currently, {0} inference results are stored.\nBatch inference will recalculate all results using the current model '{1}'.\n\nContinue?",
        'msg_inference_all_running': 'Running inference on all images...',
        'msg_inference_running': 'Running inference...',
        'dlg_inference_complete': 'Inference Complete',
        'msg_inference_complete_body': 'Inference completed for {0} images.\n{1} new results added, {2} results updated.\n\nModel: {3}{4}',
        'msg_inference_processing_error': 'Error during inference: {0}',
        'msg_location_inference_auto_on': '\n\nLocation inference display has been automatically enabled.',

        # =====================================================================
        # Annotation Mode
        # =====================================================================
        'label_auto_driving_mode': 'Auto Driving Annotation Mode',
        'label_auto_driving_mode_desc': 'Auto Driving Annotation Mode\nClick image to set angle/throttle',
        'status_switched_to_auto_driving': 'Switched to auto driving annotation mode.',
        'status_switched_to_detection': 'Switched to object detection annotation mode.',
        'status_switched_to_segmentation': 'Switched to segmentation annotation mode.',
        'status_switched_to_waypoint': 'Switched to waypoint annotation mode.',

        # =====================================================================
        # Additional Checkboxes
        # =====================================================================
        'chk_show_driving_direction': 'Show Driving Direction',
        'chk_apply_last_bbox': 'Apply Previous Bounding Box',
        'chk_apply_last_segmentation': 'Apply Previous Segmentation',
        'chk_auto_skip_on_click': 'Auto Skip on Click',
        'chk_apply_location': 'Apply Previous Location',
        'chk_detection_inference': 'Show Detection Inference',
        'chk_early_stopping': 'Enable Early Stopping',
        'chk_data_augmentation': 'Enable Data Augmentation',
        'chk_aug_mosaic': 'Mosaic',
        'chk_aug_flip': 'Horizontal Flip',
        'chk_aug_hsv': 'HSV Adjustment',
        'chk_aug_geometry': 'Geometric Transform',
        'chk_aug_erase': 'Random Erase',
        'chk_aug_color': 'Color Adjustment',
        'chk_bbox_export': 'Bounding Box (for Object Detection)',
        'chk_seg_export': 'Segmentation (for Instance Segmentation)',
        'chk_unified_available': 'Unified Format (include both in one dataset)',
        'chk_unified_unavailable': 'Unified Format (unavailable)',
        'chk_save_settings': 'Apply these settings on next startup',
        'chk_upload_to_databricks': 'Local → Databricks (upload new Runs)',
        'chk_delete_from_databricks': 'Delete Runs from Databricks that were deleted locally',
        'chk_generate_notebook': 'Generate Training Notebook',
        'chk_open_colab_after': 'Open Colab after transfer',
        'chk_download_mlruns': 'Download and merge MLflow experiment data (mlruns)',
        'chk_show_inference_result': 'Show Inference Result (cyan circle)',
        'chk_show_diff_vector': 'Show Diff Vector (green arrow)',
        'chk_add_speed_output': 'Add Speed to Output',
        'chk_add_future_prediction': 'Add Future Frame Prediction to Output',
        'chk_exclude_downsampled': 'Exclude Downsampling Targets',
        'chk_overwrite_annotation': 'Overwrite Existing Annotations',

        # =====================================================================
        # Mode Buttons
        # =====================================================================
        'btn_auto_driving': 'Auto Driving',
        'btn_detection': 'Detection',
        'btn_segmentation': 'Segmentation',
        'btn_waypoint': 'Waypoint',

        # =====================================================================
        # Toolbar Buttons
        # =====================================================================
        'toolbar_open': 'Open',
        'toolbar_save': 'Save',
        'toolbar_mlflow': 'MLflow',
        'toolbar_cloud': 'Cloud',

        # =====================================================================
        # Save Menu
        # =====================================================================
        'menu_driving_annotation': 'Driving Annotation',
        'menu_detection_annotation': 'Detection/Segmentation',
        'menu_video_export': 'Video Export',

        # =====================================================================
        # Common Buttons
        # =====================================================================
        'btn_add_location': 'Add Location',
        'btn_apply_settings': 'Apply',
        'btn_cancel': 'Cancel',
        'btn_close': 'Close',
        'btn_copy_template': 'Copy Template',
        'btn_open_readme': 'Open README',
        'btn_connection_test': 'Connection Test',
        'btn_copy_env_template': 'Copy Env Template',
        'btn_aug_preview': 'Aug Preview',

        # =====================================================================
        # Additional Labels (Training / Inference Dialog)
        # =====================================================================
        'label_framerate': 'Frame Rate (fps):',
        'label_image_skip': 'Image Skip Count:',
        'label_start': 'Start:',
        'label_end': 'End:',
        'label_multi_source_note': '* For multiple sources, select from the list below.\nSources are placed from left to right in selected order.',
        'label_total_frames': 'Total Frames: {0}',
        'label_total_frames_calculating': 'Total Frames: Calculating...',
        'label_model_architecture': 'Model Architecture:',
        'label_base_model': 'Base Model:',
        'label_speed_normalize': 'Normalize Value:',
        'label_speed_normalize_note': '* Speed value is divided by this value',
        'label_future_info': '* Adds angle, throttle(, speed) for +5 and +10 frames',
        'label_future_detail': 'Output example (with speed): [angle, throttle, speed, t+5_angle, t+5_throttle, t+5_speed, t+10_angle, t+10_throttle, t+10_speed]',
        'label_min_delta': 'Min Delta:',
        'label_validation_ratio': 'Validation Ratio:',
        'label_skip_count': 'Skip Count:',
        'label_color_brightness': 'Brightness:',
        'label_color_contrast': 'Contrast:',
        'label_color_saturation': 'Saturation:',
        'label_rotation_angle': 'Rotation Angle (±deg):',
        'label_translation_ratio': 'Translation (±ratio):',
        'label_erase_min_ratio': 'Min Ratio:',
        'label_erase_max_ratio': 'Max Ratio:',
        'label_auto_annotation_range': 'Specify the range for auto annotation.',
        'label_auto_annotation_method': 'Select auto annotation method',
        'label_skip_every': 'every',
        'label_aug_preview_title': 'Augmentation Preview',
        'label_original_image': 'Original Image',
        'label_waypoint_count': 'Waypoint Count:',

        # =====================================================================
        # Dialog Titles / Canvas Text
        # =====================================================================
        'dlg_select_yolo_folder': 'Select YOLO Annotation Folder',
        'dlg_select_save_folder': 'Select Save Folder',
        'dlg_select_export_folder': 'Select Export Folder',
        'dlg_select_annotation_subfolder': 'Select subfolder to load annotations from:',
        'dlg_select_subfolder_title': 'Select Subfolder',
        'dlg_select_image_folder': 'Select Image Folder',
        'dlg_save_video': 'Select Video Save Location',
        'canvas_no_image': 'Select a folder and click Load button',
        'status_no_images_loaded': 'No images loaded',
        'combo_model_not_found': 'Model not found',
        'combo_select_folder': 'Please select a folder',
        'msg_selected_model_not_found': 'Selected model not found: {0}',

        # =====================================================================
        # Session Restore / Loading Dialogs
        # =====================================================================
        'dlg_restore_session': 'Restore Previous Session',
        'msg_restore_folders': 'Load previous work folders ({0})?\n\nFirst folder: {1}\n{2}',
        'msg_restore_folder': 'Load previous work folder?\n\nFolder: {0}',
        'msg_other_folders': 'and {0} more folders',
        'dlg_loading_progress': 'Loading Progress',
        'msg_loading_folder': 'Loading folder...',
        'msg_searching_folders': 'Searching {0} folders...',
        'msg_loading_folder_progress': "Loading folder '{0}'... ({1}/{2})",
        'dlg_load_complete': 'Load Complete',
        'msg_images_loaded': 'Loaded {1} images from {0} folders.\nImage data keys: {2}\nCurrent key \'{3}\' image count: {4}\nOriginal image size: {5}x{6}\n\nAnnotation data not loaded.',
        'dlg_load_annotation': 'Load Annotation Data',
        'msg_load_annotation_prompt': 'Image loading complete.\nWould you like to load annotation data?',
        'dlg_load_success': 'Load Successful',
        'msg_annotations_loaded': 'Loaded {1} annotations from {0} folders.{2}',
        'msg_no_annotations_found': 'No annotation data found in selected folders.',
        'msg_annotation_load_error': 'Error loading annotations: {0}',
        'msg_model_load_complete': 'Model Load Complete',
        'msg_model_loaded': 'Loaded model: {0}',
        'msg_yolo_model_loaded': 'Loaded {0} model "{1}".\nConfidence threshold: {2}\n\nInference will run automatically on each image.',
        'msg_pilot_model_loaded': 'Loaded model "{0}".\n\nInference will run automatically on each image.',
        'msg_model_saved_loaded': 'Model saved:\n\nFile: {0}\nModel: {1}\n\nUse "Load Model" button for inference.',
        'dlg_model_save_complete': 'Model Save Complete',
        'msg_model_saved': 'Model file saved:\n{0}\n\nUse "Load Model" button to load it.',
        'msg_same_level_annotations_loaded': 'Loaded {0} annotations from same level.',
        'msg_subfolder_annotations_loaded': 'Loaded {1} annotations from subfolder "{0}".',
        'msg_no_subfolder_annotations': 'No loadable annotation data found in selected subfolder "{0}".',

        # =====================================================================
        # Annotation Loading Progress
        # =====================================================================
        'dlg_annotation_loading': 'Loading Annotations',
        'btn_cancel': 'Cancel',
        'msg_searching_annotations': 'Searching for annotations in {0} folders...',
        'msg_processing_folder': 'Processing folder {0}/{1}...\n{2}',
        'msg_preparing_load': 'Preparing to load annotation data...',
        'msg_checking_manifest': 'Checking manifest file...',
        'msg_annotations_count_loaded': 'Loaded {0} annotations',
        'label_details': '\n\nDetails:\n',
        'label_item_count': '• {0}: {1}\n',
        'msg_catalog_files_found': 'Detected {0} catalog files',
        'msg_searching_image_folder': 'Searching for image folder...',
        'msg_processing_catalog_file': 'Processing catalog file: {0} ({1}/{2})',
        'msg_processing_catalog_entry': 'Processing catalog entries: {0}/{1} entries',

        # YOLO Annotation Loading
        'msg_loading_yolo_annotations': 'Loading YOLO annotations...',
        'dlg_loading': 'Loading',
        'msg_yolo_loaded': 'Loaded YOLO annotations.\n\nProcessed images: {0}/{1}\nImages with annotations: {2}\nBounding boxes: {3}\nSegmentation: {4}\nClasses: {5}\n\nAnnotations by class:\n{6}{7}',
        'msg_yolo_load_error': 'Error loading YOLO annotations.\n\nProcessed images: {0}/{1}\nLoaded bounding boxes: {2}\nLoaded segmentation: {3}\n\nError details:\n{4}',
        'msg_other_errors': '...and {0} more errors',
        'msg_warning_errors': '\n\nWarning: {0} errors occurred',
        'msg_and_more': '...and {0} more',

        # =====================================================================
        # Export Related
        # =====================================================================
        'dlg_export_warning': 'Export Warning',
        'msg_no_exportable_entries': 'No exportable entries found.',
        'msg_donkey_export_complete': 'Exported annotations in Donkeycar format.\nSelected image sources: {0}\nSave location: {1}\nExported: {2} entries',
        'msg_donkey_export_error': 'Error during Donkeycar export: {0}\n\nDetails: {1}',
        'msg_jetracer_export_complete': 'Exported annotations in Jetracer format.\nSave location: {0}\nExported: {1} entries',
        'msg_jetracer_export_error': 'Error during Jetracer export: {0}\n\nDetails: {1}',
        'msg_yolo_export_error': 'Error during YOLO unified export: {0}',

        # YOLO Export Dialog
        'dlg_yolo_export_settings': 'YOLO Unified Export Settings',
        'label_annotation_status': 'Annotation Status',
        'label_bbox_status': '✓ Bounding Boxes: {0} ({1} images)',
        'label_seg_status': '✓ Segmentation: {0} ({1} images)',
        'label_export_format': 'Export Format Selection',
        'tip_bbox_export_count': 'Export {0} bounding boxes',
        'tip_no_bbox': 'No bounding box annotations',
        'tip_seg_export_count': 'Export {0} segmentations',
        'tip_no_seg': 'No segmentation annotations',
        'tip_unified_export': 'Combine bounding boxes and segmentation into one YOLO dataset',
        'tip_unified_requires_both': 'Both annotation formats required',
        'label_class_settings': 'Class Settings',
        'label_save_settings': 'Save Settings',
        'label_deleted_indexes_info': 'Deleted indexes: {0} (excluded from export)',
        'msg_yolo_export_confirm': 'Export in YOLO format with these settings:\n\nSave location: {0}\nClasses: {1}\n\nExport contents:\n{2}{3}{4}',
        'label_bbox_count_item': 'Bounding Boxes: {0}',
        'label_seg_count_item': 'Segmentation: {0}',
        'msg_unified_note': '\n* Will be saved as a unified dataset',
        'msg_separate_note': '\n* Each format will be saved as separate datasets',
        'msg_deleted_excluded': '\n\nDeleted indexes: {0} (excluded)',
        'dlg_yolo_export_confirm': 'YOLO Unified Export Confirmation',
        'msg_continue_question': '\n\nContinue?',
        'msg_yolo_export_preparing': 'Preparing YOLO export...',
        'dlg_exporting': 'Exporting',
        'msg_exporting_unified': 'Exporting in unified YOLO format...',
        'msg_exporting_bbox': 'Exporting in bounding box format...',
        'msg_exporting_seg': 'Exporting in segmentation format...',
        'label_unified_format': 'Unified format: {0}',
        'label_detection_format': 'Object detection: {0}',

        # YOLO Training Dialog
        'dlg_yolo_task_selection': 'YOLO Training Task Selection',
        'label_training_task': 'Training Task',
        'label_detection_task': 'Object Detection',
        'label_segmentation_task': 'Segmentation',
        'tip_detection_task': 'Train object detection model using bounding boxes',
        'tip_segmentation_task': 'Train segmentation model using polygons',
        'dlg_yolo_training_settings': 'YOLO {0} Model Training Settings',
        'label_training_stats': 'Training Data Statistics:',
        'label_total_loaded_images': 'Total loaded images: {0}',
        'label_annotated_images_count': '{0} annotated images: {1}',
        'label_actual_training_count': 'Actual training count: {0} images',
        'label_excluded_calculation': '({0} - {1} deleted)',
        'label_total_annotations_count': 'Total {0} annotations: {1}',
        'label_deleted_excluded_note': '* Deleted-marked images are excluded from training',
        'label_model_init_settings': 'Model Initialization Settings',
        'label_use_pretrained_weights': 'Use pre-trained weights (recommended)',
        'label_use_current_model_weights': 'Use weights from currently loaded model',
        'label_current_model_name': 'Current model: {0}',
        'label_no_model_loaded': 'Current model: None (load model first)',
        'tab_basic_settings': 'Basic Settings',
        'tab_data_augmentation': 'Data Augmentation',
        'label_model_name_settings': 'Model Name Settings',
        'placeholder_custom_name': 'Enter custom name',
        'label_model_name_note': '* Model type ({0}) prefix cannot be changed. .pt is automatically appended',
        'label_model_name_note_pth': '* Model type ({0}) prefix cannot be changed. .pth is automatically appended',
        'label_training_comment': 'Training Comment (Logged to MLflow)',
        'placeholder_training_comment': 'Enter notes or comments about this training (optional)',
        'label_original_image_size': 'Original image: {0}×{1}',

        # YOLO Model Loading Dialog
        'dlg_yolo_confidence_settings': '{0} Model Confidence Settings',
        'label_confidence_threshold': '{0} confidence threshold (0.0-1.0):',
        'msg_loading_yolo_model': "Loading {0} model '{1}'...",
        'dlg_unified_model_loading': 'Loading Unified Model',
        'msg_loading_model_to_memory': 'Loading {0} model to memory...',
        'msg_getting_class_info': 'Getting class information...',
        'tip_segmentation_model_loaded': 'Segmentation model is loaded',
        'tip_detection_model_loaded': 'Object detection model is loaded',
        'msg_unified_model_loaded': "Loaded unified {0} model '{1}'.\nConfidence threshold: {2}\nClasses: {3}\n\n{0} inference will run automatically on each image.",
        'msg_yolo_load_failed': 'Failed to load YOLO model: {0}',
        'label_segmentation_short': 'Segmentation',
        'label_detection_short': 'Object Detection',
        'label_seg_tag': '[Seg]',
        'label_det_tag': '[Det]',
        'label_det_tag_short': 'Det',
        'label_seg_tag_short': 'Seg',
        'status_models_loaded_count': 'Loaded {0} {1} models (Detection: {2}, Segmentation: {3})',
        'msg_no_labels_directory': 'No labels directory found in the selected folder.\nYOLO dataset structure:\n- dataset/\n  - images/\n  - labels/\n  - classes.txt',
        'msg_no_annotations_found_fallback': 'No annotations found',
        'label_env_status': 'Environment Variable Status',
        'label_env_tab': 'Environment Variables',
        'msg_config_databricks_not_found': 'databricks/config_databricks.py not found',
        'msg_config_colab_not_found': 'colab/config_colab.py not found',
        'label_status_message': 'Status: {0}\n{1}',
        'dlg_no_models': 'No Models',
        'msg_google_drive_no_models': 'No model files found on Google Drive.\n\nPlease train a model on Colab and save it to Google Drive.',
        'dlg_images_folder_not_found': 'Images folder not found',
        'msg_images_folder_missing': 'Images folder not found in the following folders:\n{0}\n\nProceeding with valid folders only.',
        'msg_no_segmentation_class_mismatch': 'No valid segmentation annotations found.\n\nFound class names: {0}\nExpected class names: {1}\n\nPlease verify that class names match.',
        'section_window_size': 'Window Size',
        'section_font_size': 'Font Size',
        'section_save_settings': 'Save Settings',
        'section_current_status': 'Current Status',
        'section_sync_options': 'Sync Options',
        'section_connection_status': 'Connection Status',
        'section_env_setup': 'Environment Variable Setup',
        'section_transfer_content': 'Transfer Content',
        'section_options': 'Options',
        'section_save_location': 'Save Location',
        'section_range': 'Range',
        'label_location_value': 'Location {0}',
        'dlg_databricks_sync_settings': 'Databricks Sync Settings',
        'dlg_databricks_settings': 'Databricks Settings',
        'dlg_colab_transfer_settings': 'Google Colab Transfer Settings',
        'dlg_colab_settings': 'Google Colab Settings',
        'msg_mlflow_tracking_uri': 'MLflow Tracking URI: {0}',
        'msg_mlflow_init_success': 'MLflow initialization success: {0}',
        'msg_setting_experiment': 'Setting experiment: {0}',
        'msg_yolo_training_preparing': 'Preparing YOLO {0} model \'{1}\' for training...',
        'label_dependency_package': 'dependency package',
        'msg_class_index_mapping': 'Class-Index Mapping: {0}',
        'label_detection_result': 'Object Detection: {0}',
        'msg_exporting_bbox': 'Exporting bounding boxes: {0}',
        'msg_exporting_segmentation': 'Exporting segmentation: {0}',
        'msg_processing_file': 'Processing: {0}',
        'msg_downloading_size': 'Downloading: {0} MB / {1} MB',
        'msg_preparing_with_input_size': 'Preparing training with input size: {0}...',
        'msg_initializing_model_waypoints': 'Initializing model \'{0}\'... ({1} waypoints)',
        'msg_location_inference_all_running': 'Running location inference on all images...',
        'msg_location_inference_running': 'Running location inference...',
        'dlg_location_inference_complete': 'Location Inference Complete',
        'msg_location_inference_result': 'Location inference completed for {0} images.\n{1} new results added, {2} results updated.\n\nModel used: {3} ({4}){5}',
        'msg_location_inference_error': 'Error during location inference: {0}',
        'msg_location_model_load_error': 'Error loading location model: {0}',
        'msg_location_model_training_error': 'Error during location model training: {0}',
        'tip_seg_disabled_detection': 'Disabled because segmentation model is loaded',
        'tip_detection_disabled_seg': 'Disabled because object detection model is loaded',
        'msg_running_inference_test': 'Running inference test...',
        'msg_yolo_load_error': 'Error loading {0} model: {1}',
        'msg_yolo_model_type_changed': 'Changed YOLO model type to "{0}". Refreshing model list...',

        # Video Creation Dialog
        'msg_no_annotations_for_video': 'No annotations available.',
        'dlg_video_settings': 'Video Creation Settings',
        'label_target_range': 'Target Range',
        'label_all_images': 'All images',
        'label_specify_range': 'Specify index range',
        'label_output_mode': 'Output Mode',
        'label_single_source': 'Single source output (normal mode)',
        'label_multi_source': 'Multiple sources output (side by side)',
        'label_image_sources': 'Image Sources',
        'label_images_count': '({0} images)',
        'msg_no_source_selected': 'Total frames: No image source selected',
        'msg_no_valid_source': 'Total frames: No valid image source',
        'msg_start_must_be_less': 'Total frames: Start must be less than or equal to end',
        'label_range_info': '\nRange: {0} - {1}',
        'label_time_format_min_sec': '{0}m {1}s',
        'label_time_format_sec': '{0}s',
        'label_selected_sources': '\nSelected sources: {0} ({1} images each)',
        'label_total_frames_info': 'Total frames: {0} frames (approx. {1}){2}{3}',
        'label_total_frames_simple': 'Total frames: {0} frames',
        'msg_start_index_error': 'Start index must be less than or equal to end index.',
        'msg_no_source_images': 'No image source selected.',
        'msg_source_not_found': "Images for source '{0}' not found.",
        'msg_no_images_in_source': 'No images in one of the selected sources.',
        'msg_no_images_in_selected': "No images in selected source '{0}'.",
        'dlg_save_video': 'Select Video Save Location',
        'msg_creating_video': 'Creating video...',
        'dlg_processing': 'Processing',
        'dlg_success': 'Success',
        'msg_video_created': 'Annotation video created:\nFile: {0}\nFrames: {1} frames\n{2}{3}\nSettings: {4}fps, every {5} images',
        'label_multi_sources': 'Multiple sources: {0}',
        'label_single_source_info': 'Source: {0}',
        'msg_video_creation_failed': 'Video creation failed. No processable annotation data.',
        'msg_video_creation_error': 'Error during video creation: {0}',

        # Donkey/Jetracer Export Dialog
        'dlg_donkey_export_settings': 'Donkeycar Export Settings',
        'dlg_jetracer_export_settings': 'Jetracer Export Settings',
        'label_image_source_selection': 'Image Source Selection',
        'label_variant_images_count': '{0} ({1} images)',
        'label_catalog_key_settings': 'Catalog Key Settings',
        'label_key_name': '{0} Key Name:',
        'label_deleted_indexes_export': 'Deleted indexes: {0} (deletion info will also be exported)',
        'msg_export_confirm': 'Export in {0} format with these settings:\n\nSave location: {1}\n',
        'label_image_source_item': '・Image source: {0} ({1} images)\n',
        'label_key_name_item': '  Key name: {0}\n',
        'label_annotation_count': '\nAnnotation count: {0}',
        'label_deleted_count': '\nDeleted indexes: {0}',
        'dlg_export_confirm': '{0} Export Confirmation',

        # =====================================================================
        # Autonomous Driving Model Section
        # =====================================================================
        # Training Settings Dialog
        'msg_need_annotations_to_train': 'Annotations are required to train the model.',
        'dlg_training_settings': 'Training Settings',
        'label_init_settings': 'Initialization Settings',
        'label_use_pretrained': 'Use pre-trained weights (Recommended)',
        'tip_pretrained': 'Train with ImageNet pre-trained weights (transfer learning)',
        'tip_no_pretrained': 'This model does not have pre-trained weights',
        'label_random_init': 'Random initialization (train from scratch)',
        'tip_random_init': 'Randomly initialize weights and train from scratch (takes longer)',
        'label_finetune': 'Use existing model weights (fine-tuning)',
        'tip_finetune': 'Fine-tune using selected model weights',
        'tip_no_model_for_type': 'No model available for the selected model type',
        'msg_no_model_for_type': 'No models available for {0}',
        'label_output_settings': 'Output Settings',
        'label_training_params': 'Training Parameters',
        'label_speed_data_info': '* {0} annotations contain speed data',
        'tip_speed_normalize': 'Divisor for normalizing speed values (default: 10.0)',
        'tip_future_prediction': 'Add 5, 10 frame ahead angle, throttle(, speed) outputs',
        'tip_validation_ratio': 'Ratio of training data to split for validation',
        'tip_weight_decay': 'L2 regularization strength (prevents overfitting)',
        'tip_optimizer': 'Adam: versatile, AdamW: improved weight decay, SGD: classic but stable',
        'tip_scheduler': 'ReduceLROnPlateau: reduce on loss plateau, StepLR: fixed step reduction, CosineAnnealing: cosine curve adjustment',
        'label_training_data_selection': 'Training Data Selection',
        'label_use_all_annotations': 'Use all annotation data',
        'label_use_skip': 'Thin out data with skip setting',
        'label_specify_index_range': 'Specify index range',
        'label_range_separator': '-',
        'tip_exclude_downsampled': 'Exclude downsampled data (e.g. straight driving) from training (currently {0} items)',
        'label_exclude_downsampled_zero': 'Exclude downsampled items (0 items)',
        'label_exclude_downsampled_count': 'Exclude downsampled items ({0} items)',
        'label_data_count_all': 'Training data: {0} images',
        'label_data_count_all_detail': '<b>Training data: {0} images</b> (Total {1} - {2})',
        'label_data_count_skip_detail': '<b>Training data: {0} images</b> (Every {1}, Target {2} - {3})',
        'label_data_count_range_detail': '<b>Training data: {0} images</b> (Range {1}-{2}, Target {3} - {4})',
        'label_excluded_deleted': '{0} deleted',
        'label_excluded_ds': ' + DS {0}',
        'btn_start_training': 'Start Training',
        'msg_training_starting': 'Starting {0} training...',
        'dlg_yolo_model_training': 'YOLO {0} Model Training',
        'dlg_training_result_logging': 'Logging Training Results',
        'dlg_training_complete': 'Training Complete',
        'btn_open_mlflow': 'Open MLflow',
        'msg_downloading_pretrained': 'Downloading pre-trained {0} model...',
        'msg_pretrained_download_failed': 'Failed to prepare pre-trained {0} model.',
        'msg_package_not_installed': '{0} package is not installed.\nPlease install with: pip install {0}',

        # Location Model
        'msg_need_location_annotations': 'Location annotations are required to train a location model.',
        'msg_need_at_least_2_locations': 'At least 2 different location labels are required. Current: {0} types',
        'msg_no_valid_location_annotations': 'No valid location annotations.',
        'msg_preparing_location_training': "Preparing training data for location model '{0}'...",
        'dlg_location_model_training': 'Location Model Training',
        'msg_init_location_model': "Initializing model '{0}'... (Fixed {1} classes)",
        'dlg_location_training_settings': 'Location Model Training Settings',
        'label_location_stats': '<b>Training Data Statistics:</b><br>Total loaded images: {0}<br>Location annotated images: {1}<br><b style="color: #2E7D32; font-size: 14px;">Actual training count: {2}</b><br>({1} - {3} deleted)<br><span style="color: #FF6600;">{4}</span>',
        'label_detected_locations': 'Detected location labels: {0} types ({1})',
        'label_fixed_class_note': '* Location model always outputs {0} classes.',
        'msg_no_valid_location_model': 'No valid location model selected.',
        'msg_loading_location_model': "Loading location model '{0} ({1})'...",
        'msg_location_model_load_error': 'Error loading location model: {0}',
        'msg_running_initial_inference': 'Running initial inference...',
        'msg_updating_inference_display': 'Updating inference display...',
        'tip_location_model_loaded': 'Location model is loaded',
        'tip_location_model_not_loaded': 'Location model is not loaded',

        # Inference Checkbox Tooltips
        'tip_driving_model_loaded': 'Autonomous driving model is loaded',
        'tip_driving_model_not_loaded': 'Autonomous driving model is not loaded',
        'tip_detection_model_loaded': 'Object detection model is loaded',
        'tip_detection_model_not_loaded': 'Object detection model is not loaded',
        'tip_seg_disabled_by_detection': 'Disabled because object detection model is loaded',
        'tip_segmentation_model_loaded': 'Segmentation model is loaded',
        'tip_segmentation_model_not_loaded': 'Segmentation model is not loaded',
        'tip_batch_inference': 'Run inference on all images',
        'tip_gradcam_heatmap': 'Display model attention regions as heatmap',

        # Extra Model Slots
        'btn_add_model': '+ Add Model',
        'tip_add_model': 'Add an extra driving model slot (up to 3)',
        'btn_remove_model': '- Remove Model',
        'tip_remove_model': 'Remove the last added driving model slot',
        'label_driving_model_n': 'Driving Model {0}',
        'chk_inference_result_n': 'Inference Result {0}',
        'tip_extra_model_not_loaded': 'Model not loaded',
        'label_driving_model_n_inference': 'Driving Model {0} Inference',

        # Waypoint Model
        'dlg_waypoint_training_settings': 'Waypoint Model Training Settings',
        'label_waypoint_stats': '<b>Training Data Statistics:</b><br>Total loaded images: {0}<br>Waypoint annotated images: {1}<br><b style="color: #2E7D32; font-size: 14px;">Actual training count: {2}</b><br>({1} - {3} deleted)<br><span style="color: #FF6600;">{4}</span>',
        'msg_no_waypoint_model_selected': 'No waypoint model selected to load.',
        'msg_waypoint_model_not_found': 'Waypoint model file not found: {0}',
        'tip_waypoint_model_loaded': 'Waypoint model ({0}, {1} points) is loaded',
        'tip_waypoint_model_not_loaded': 'Waypoint model is not loaded',
        'msg_waypoint_model_loaded': 'Waypoint model loaded\nModel: {0}\nWaypoint count: {1}',
        'msg_need_waypoint_annotations': 'At least 5 annotations are required to train waypoint model. Current: {0}',
        'msg_no_valid_waypoint_annotations': 'No valid waypoint annotations.',
        'msg_preparing_waypoint_training': "Preparing training data for waypoint model '{0}'...",
        'dlg_waypoint_model_training': 'Waypoint Model Training',
        'msg_skipped_images_header': 'The following {0} images will be skipped due to waypoint count mismatch:\n\n',
        'msg_skipped_images_item': '  {0}: {1} - {2}\n',
        'msg_skipped_images_more': '\n...and {0} more',
        'msg_skipped_images_footer': '\n\nImages to be used for training: {0}\nContinue?',
        'msg_need_waypoint_annotations_first': 'Waypoint annotations are required to train a waypoint model.',

        # YOLO Model Loading
        'msg_loading_yolo_model_display': "Loading YOLO model '{0}'...",
        'dlg_detection_threshold': 'Detection Threshold',
        'label_detection_threshold': 'Detection confidence threshold (0.0-1.0):',
        'msg_loading_model_to_memory_display': "Loading model '{0}' to memory...",
        'msg_running_inference_on_current': 'Running inference on current image...',
        'msg_yolo_model_load_error': 'Error loading YOLO model: {0}',

        # Waypoint Mode Selection
        'label_auto_advance': 'Auto Advance',
        'tip_auto_advance': 'Automatically advance to next image when last waypoint is placed',
        'label_apply_last_waypoint': 'Apply Previous Waypoints',
        'tip_apply_last_waypoint': 'Automatically apply previous image waypoints to next image',

        # Segmentation Display Mode
        'tip_show_driving_direction': 'Calculate and display driving direction arrow from segmentation inference',
        'tip_seg_class_id': 'Segmentation class ID for drivable area',
        'tip_seg_y_coordinate': 'Y coordinate used for direction calculation (pixels from top)',
        'tip_seg_max_steering': 'Maximum steering angle used for trajectory calculation (degrees)',
        'label_trajectory_mode': 'Trajectory',
        'tip_trajectory_mode': 'Display driving trajectory as arc',
        'label_waypoint_mode': 'Waypoint',
        'tip_waypoint_mode': 'Display waypoints (4 equally spaced points) to target Y coordinate',

        # Location Annotation
        'tip_apply_last_bbox': 'Apply previously created bounding box to current image',
        'tip_apply_last_segmentation': 'Apply previously created segmentation to current image',
        'tip_apply_location': 'Apply previously selected location to current image',

        # Image Badges
        'badge_objects': 'Obj: {0}',
        'badge_segments': 'Seg: {0}',
        'badge_deleted': 'Deleted',
        'badge_ds_target': 'DS Target',
        'label_deleted_click_to_restore': 'Deleted\nClick to re-annotate',
        'label_inference_prefix': 'Inf:',

        # Status Bar Messages
        'status_polygon_point_added': 'Added new point to polygon (position: {0})',
        'status_waypoint_complete_auto_advance': 'Waypoint placement complete ({0} points) - Auto advancing to next image',
        'status_speed_updated': 'Speed value updated: {0:.2f}',
        'status_bbox_deselected': 'Bounding box deselected',
        'status_vertex_editing': 'Editing vertex... (vertex {0})',
        'status_seg_deselected': 'Segmentation deselected',
        'status_waypoint_count_reached': 'Waypoint count ({0}) reached',
        'status_y_exceeds_image': 'Y coordinate exceeds image size ({0})',
        'status_waypoint_added': 'Waypoint {0} added: ({1}, {2}) - Total: {3}/{4}',
        'status_freehand_start': 'Freehand mode started - Drag to place waypoints',
        'status_freehand_placed': 'Placed {0} waypoints via freehand',
        'status_waypoint_adjusted': 'Waypoint position adjusted',
        'status_creating_bbox': 'Creating new bounding box... Width: {0}px, Height: {1}px',
        'status_moving_bbox': "Moving '{0}' bounding box...",
        'status_moving_seg': "Moving '{0}' segmentation...",
        'status_bbox_deleted': "Deleted '{0}' bounding box",
        'status_seg_deleted': "Deleted '{0}' segmentation",
        'status_driving_annotation_deleted': 'Driving annotation ({0}) deleted',
        'status_no_annotation_to_delete': 'No annotation to delete',
        'status_waypoints_deleted': 'Deleted {0} waypoints',
        'status_no_waypoint_to_delete': 'No waypoints to delete',
        'status_start_y_set': 'Start Y position set to {0}',
        'status_end_y_set': 'End Y position set to {0}',
        'status_mouse_not_in_image': 'Mouse is not within image',
        'status_image_view_not_initialized': 'Image view not initialized',
        'status_waypoint_auto_apply_switched': 'Switched to auto-apply previous waypoints mode - Applied {0} waypoints',
        'status_waypoint_auto_apply_mode': 'Switched to auto-apply previous waypoints mode',
        'status_waypoint_auto_apply_no_data': 'Switched to auto-apply previous waypoints mode (no waypoints to apply)',
        'status_auto_advance_mode': 'Switched to auto-advance on completion mode',
        'status_location_inference_on': 'Location inference display enabled',
        'status_location_inference_off': 'Location inference display disabled',
        'status_updating_model_list': 'Updating model list...',
        'status_model_not_found': 'No {0} models found. Select another architecture or train a model',
        'status_models_loaded': 'Loaded {0} {1} models',
        'status_auto_play': 'Auto playing ({0}, {1}, {2}) - Click button again to stop',
        'status_direction_forward': 'Forward',
        'status_direction_backward': 'Backward',
        'status_speed_slow': 'Slow',
        'status_speed_fast': 'Fast',
        'status_skip_count': 'Skip {0}',
        'status_no_skip': 'No skip',
        'status_location_auto_applied': 'Location {0} auto-applied',
        'status_bbox_auto_applied': "Applied previous '{0}' bounding box",
        'status_detection_inference_on': 'Object detection inference display enabled',
        'status_detection_inference_off': 'Object detection inference display disabled',
        'status_switched_to_driving_training': 'Switched to autonomous driving model training mode.',
        'status_switched_to_detection_training': 'Switched to object detection model training mode.',
        'status_segmentation_inference_on': 'Segmentation inference display enabled',
        'status_segmentation_inference_off': 'Segmentation inference display disabled',
        'status_updating_yolo_model_list': 'Updating unified YOLO model list...',
        'status_updating_yolo_model_list_simple': 'Updating YOLO model list...',
        'status_yolo_model_not_found': 'No YOLO models found for {0}',
        'status_future_annotation_on': 'Future annotation display enabled',
        'status_future_annotation_off': 'Future annotation display disabled',
        'status_cam_on': 'CAM display enabled',
        'status_cam_off': 'CAM display disabled',
        'status_cam_error': 'CAM generation error: {0}',
        'status_driving_inference_on': 'Driving inference display enabled',
        'status_driving_inference_off': 'Driving inference display disabled',
        'status_inference_processing': 'Processing inference... Model: {0} ({1})',
        'status_location_inference_processing': 'Processing location inference... Model: {0} ({1})',
        'status_inference_for_source_switch': 'Running inference for image source switch...',
        'status_jumped_to_index': 'Jumped to index {0}',
        'status_waypoints_auto_applied': 'Auto-applied {0} previous waypoints',
        'status_reached_first_image': 'Reached first image, auto play stopped',
        'status_reached_last_image': 'Reached last image, auto play stopped',
        'status_bboxes_auto_applied': 'Applied {0} previous bounding boxes',
        'status_segs_auto_applied': 'Applied {0} previous segmentations',
        'status_seg_auto_applied': "Applied previous '{0}' segmentation",
        'status_training_cancelled': 'Training cancelled',
        'status_updating_waypoint_model_list': 'Updating waypoint model list...',
        'status_waypoint_model_not_found': 'No waypoint models found for {0}. Please train a model',
        'status_waypoint_models_found': 'Found {0} waypoint models',
        'status_waypoint_inference_on': 'Waypoint inference display enabled',
        'status_waypoint_inference_off': 'Waypoint inference display disabled',
        'status_updating_location_model_list': 'Updating location model list...',
        'status_location_model_not_found': 'No location models found for {0}. Please train a model',
        'status_location_models_loaded': 'Loaded {0} {1} location models',
        'status_location_model_loaded': "Loaded location model '{0} ({1})' (classes: {2})",
        'status_location_model_type_changed': 'Location model type changed to "{0}". Updating model list...',
        'status_google_auth': 'Google authentication... Please complete authentication in browser (timeout: 60 seconds)',
        'status_pretrained_model_saved': 'Saved pretrained {0} model to models folder: {1}',

        # QMessageBox Dialogs
        'dlg_warning': 'Warning',
        'dlg_error': 'Error',
        'dlg_info': 'Information',
        'dlg_complete': 'Complete',
        'dlg_sync_cancelled': 'Sync Cancelled',
        'dlg_sync_complete': 'Sync Complete',
        'dlg_sync_complete_with_errors': 'Sync Complete (Some Errors)',
        'sync_progress': 'Syncing {0}/{1}',
        'sync_already_running': 'Sync is already running',
        'btn_cancel_sync': 'Cancel Sync',
        'dlg_connection_test': 'Connection Test',
        'dlg_transfer_complete': 'Transfer Complete',
        'dlg_add_complete': 'Add Complete',
        'msg_location_already_exists': 'Location {0} already exists.',
        'msg_location_added': 'Location {0} has been added.',
        'msg_image_source_switch_failed': "Failed to switch to image source '{0}'.",
        'msg_readme_not_found': 'README not found:\n{0}',
        'msg_file_open_failed': 'Failed to open file:\n{0}',
        'msg_folder_not_found': 'Folder does not exist: {0}',
        'msg_images_folder_not_found': 'Images folder not found under: {0}',
        'msg_no_training_data': 'No training data available.',
        'msg_insufficient_data': 'Insufficient data. At least 2 images required.',
        'msg_start_index_error': 'Start index must be less than or equal to end index.',
        'msg_no_images_to_process': 'No images to process.',
        'msg_yolo_model_not_loaded': 'YOLO model is not loaded.',
        'msg_no_preview_images': 'No images available for preview.',
        'msg_augmentation_disabled': 'Data augmentation is disabled.',
        'msg_preview_error': 'Error during preview generation: {0}',
        'msg_waypoint_model_load_failed': 'Failed to load waypoint model:\n{0}',
        'msg_waypoint_model_not_loaded': 'Waypoint model is not loaded.',
        'msg_waypoint_training_error': 'Error during waypoint model training:\n{0}',
        'msg_no_valid_bbox_annotations': 'No valid object detection annotations.\n(Deleted: {0})',

        # Annotation Mode Tooltips
        'tip_auto_driving_mode': 'Autonomous Driving Annotation Mode\n・Click on image to set angle and throttle\n・Left click: Add/move point\n・Right click: Delete point\n・Number keys (0-7): Set driving position (press again to deselect)\n・Delete key: Delete current image annotations (angle/throttle/position)',
        'tip_waypoint_count': 'Number of waypoints to place',
        'tip_cam_target': 'Select output to visualize with CAM',
        'tip_waypoint_start_y': 'Y coordinate for waypoint start position',
        'tip_waypoint_end_y': 'Y coordinate for waypoint end position',
        'status_vertex_moving': "Moving '{0}' vertex {1}... ({2}, {3})",
        'status_bbox_resizing': "Resizing '{0}' bounding box... [Pos: ({1:.0f}, {2:.0f}), Size: {3:.0f}x{4:.0f}]",
        'label_graph_error': 'Graph creation error: {0}',
        'msg_polygon_min_vertices': 'Polygon requires at least 3 vertices.\nCannot delete vertex.',
        'btn_location_with_count': '{0} | Loc {1}',
        'btn_play': '▶Play',
        'btn_stop': '■Stop',
        'btn_reverse_play': '◀Reverse',
        'dlg_waypoint_shortage': 'Waypoint Shortage',
        'msg_waypoint_shortage': 'Current image has {0} waypoints placed,\nbut {1} are required.\n\nPlease place {2} more waypoints before moving to the next image.\n\n*To cancel placement, delete all waypoints with the Delete key.',
        'msg_cannot_set_location_deleted': 'Cannot set location on deleted image.\nPlease restore the deleted state first.',

        # Databricks/Colab
        'tip_keep_current_input': 'Keep current input',
        'label_databricks_combined': '✓ Databricks + Local',
        'label_databricks_disconnected': '✗ Databricks: Disconnected',
        'label_local_mlflow': 'Using Local MLflow',
        'tip_upload_runs': 'Upload runs that exist locally but not on Databricks',
        'tip_delete_runs': 'Delete runs from Databricks that do not exist locally (Warning: Cannot be undone)',
        'chk_delete_runs': 'Delete runs from Databricks that were deleted locally ({0})',
        'label_colab_authenticated': 'Authenticated',
        'label_colab_not_authenticated': 'Not Authenticated',
        'label_colab_disabled': 'Disabled',
        'label_colab_no_config': 'No Config File',
        'tip_generate_notebook': 'Generate a notebook that can be used in Google Colab after transfer',
        'tip_open_colab': 'Open Colab in browser after transfer is complete',
        'tip_merge_mlruns': 'Merge MLflow experiment data recorded in Colab to local',
        'label_processing_count': 'To process: {0} images',
        'label_processing_count_skip': 'To process: ~{0} images (every {1})',

        'label_no_annotations': 'No annotations',
        'label_no_valid_annotations': 'No valid annotations',
        'msg_no_annotations_to_transfer': 'No annotations to transfer.\n\nPlease create annotations first.',
        'msg_no_deleted_annotations_to_restore': 'No deleted annotations to restore.',
        'dlg_confirm': 'Confirm',
        'msg_confirm_restore_all': 'Clear all deleted states. Are you sure?\n\nDeleted index count: {0}',
        'dlg_zip_filename': 'ZIP Filename',
        'msg_enter_zip_filename_databricks': 'Enter ZIP filename to transfer to Databricks:\n(.zip will be added automatically)',
        'msg_enter_zip_filename_gdrive': 'Enter ZIP filename to transfer to Google Drive:\n(.zip will be added automatically)',

        'label_speed_data_info': '* {0} annotations contain speed data',

        # Model Loading
        'msg_no_valid_model_selected': 'No valid model selected.',
        'msg_loading_model': "Loading model '{0} ({1})'...",
        'dlg_model_loading': 'Loading Model',
        'dlg_clear_inference_confirm': 'Clear Inference Results',
        'msg_clear_inference_prompt': '{0} inference results are currently saved.\nChanging the model may cause inconsistencies with old inference results.\n\nDo you want to clear existing inference results?',
        'msg_existing_inference': 'Existing inference results: {0}\nShowing confirmation dialog...',
        'msg_clearing_inference': 'Clearing existing inference results...',
        'msg_cleared_old_inference': 'Cleared {0} old inference results',
        'msg_init_model_arch': 'Initializing model architecture...',
        'msg_loading_model_file': 'Loading model file: {0}',
        'msg_init_model': 'Initializing model...',
        'msg_transfer_to_device': 'Transferring model to {0}...',
        'msg_running_inference': 'Running inference: {0}',
        'msg_saving_inference': 'Saving inference results...',
        'msg_updating_inference': 'Updating inference display...',
        'msg_model_loaded_suffix': ' (old inference results cleared)',
        'msg_model_loaded': "Model '{0} ({1})' loaded{2}",
        'msg_model_loaded_detail': "Model '{0} ({1})' loaded.",
        'msg_new_inference_available': '\n\n{0} new inference results available.',
        'msg_existing_kept': '\n\nExisting inference results have been kept. Use "Infer All Images" button to update if needed.',
        'msg_inference_auto_on': '\n\nInference result display has been automatically enabled.',
        'msg_model_load_error': 'Error loading model: {0}',

        # Auto Annotation
        'dlg_auto_annotation_settings': 'Auto Annotation Settings',
        'dlg_existing_annotation_handling': 'Existing Annotation Handling',
        'btn_overwrite': 'Overwrite',
        'btn_append': 'Append',
        'dlg_augmentation_preview': 'Augmentation Preview',
        'dlg_insufficient_data': 'Insufficient Data',
        'msg_insufficient_segmentation_data': 'Only {0} images with valid segmentation data.\nAt least 4 images are recommended for segmentation training.',
        'msg_yolo_training_error': 'Error during YOLO {0} model training: {1}',
        'dlg_no_segmentation_data': 'No Segmentation Data',
        'msg_no_segmentation_generate_from_bbox': 'No valid segmentation annotations found.\n\nWould you like to automatically generate rectangular segmentation from bounding boxes?\n(For better accuracy, manual polygon annotation is recommended)',
        'msg_no_segmentation_manual_required': 'No valid segmentation annotations found.\n\nSegmentation training requires at least 3-point polygon annotations.\nPlease create manual polygon annotations and try again.',
        'label_pretrained_weights_downloaded': 'Pre-trained weights (downloaded: {0})',
        'label_current_model_weights': 'Current model weights: {0}',
        'msg_yolo_training_complete': 'YOLO {0} model training completed.\nFinal mAP: {1}\nDevice used: {2}\nInitialization: {3}\n\nModel saved at: {4}\n{5}',
        'msg_need_manual_annotation': 'To run auto annotation, please manually annotate a few images first.',
        'dlg_auto_annotation_range': 'Auto Annotation Range',
        'label_range': 'Range',
        'label_all_unannotated': 'All unannotated images',
        'label_specify_index_range': 'Specify index range',
        'tip_overwrite': 'If checked, already annotated images will be overwritten with new inference',
        'msg_no_unannotated_in_range': 'No unannotated images in the specified range.',
        'msg_auto_annotation_preparing': 'Preparing auto annotation... ({0} images)',
        'dlg_auto_annotation_running': 'Running Auto Annotation',
        'msg_preparing_model': "Preparing processing with model '{0}'...",
        'msg_init_model_type': "Initializing model '{0}'...",
        'msg_loading_model_basename': "Loading model '{0}'...",
        'msg_preparing_pretrained': "Preparing pre-trained model '{0}'...",
        'msg_batch_processing': 'Processing batch {0}/{1}...\nImages {2}-{3}/{4}',
        'msg_batch_image_processing': 'Processing batch {0}/{1}...\nProcessing image {2}/{3}',
        'msg_updating_location_buttons': 'Updating location buttons...',
        'msg_verifying_image_files': 'Verifying image files...',
        'msg_organizing_image_data': 'Organizing image data...',
        'msg_updating_display': 'Updating display...',
        'msg_parsing_manifest': 'Parsing manifest file...',
        'msg_parsing_image_index': 'Parsing image file index...',
        'msg_updating_gallery': 'Updating gallery display...',
        'msg_using_detection_model': 'Using object detection model...',
        'msg_using_segmentation_model': 'Using segmentation model...',
        'msg_running_segmentation': 'Running segmentation...',
        'msg_running_detection': 'Running object detection...',
        'msg_integrating_annotations': 'Integrating annotation data...',
        'msg_saving_training_curve': 'Saving training curve...',
        'msg_recording_mlflow': 'Recording training results to MLflow...',
        'msg_updating_ui': 'Updating UI...',
        'msg_updating_graph': 'Updating distribution graph...',
        'msg_auto_annotation_complete': 'Complete: Applied auto annotation to {0} images',
        'msg_auto_annotation_success': 'Applied auto annotation to {0} images.\nModel used: {1}{2}',
        'label_pretrained_suffix': ' (pre-trained)',
        'dlg_cancelled': 'Cancelled',
        'msg_auto_annotation_cancelled': 'Auto annotation was cancelled.\n{0} images were processed.',
        'msg_auto_annotation_error': 'Error during auto annotation: {0}',

        # Batch Inference
        'dlg_batch_inference_confirm': 'Batch Inference Confirmation',
        'msg_batch_inference_prompt': 'Run inference on all {0} images.\nCurrent model: {1}{2}\n\nOperations will be blocked during processing. Continue?',
        'dlg_existing_inference': 'Existing Inference Results',
        'msg_overwrite_inference_prompt': '{0} inference results are currently saved. Do you want to overwrite them?\n\n"Yes": Overwrite all inference results with new model.\n"No": Only process images without inference results.',
        'msg_preparing_inference': 'Preparing inference processing...',
        'dlg_batch_inference_running': 'Running Batch Inference',
        'msg_batch_inference_cancelled': 'Batch inference was cancelled.\nProcessed: {0}/{1} images\nSuccess: {2}, Skipped: {3}',
        'msg_batch_inference_complete': 'Inference completed for all images.\nProcessed: {0} images\nSuccess: {1}, Skipped: {2}\n\nInference result display is now ON.',
        'msg_batch_inference_error': 'Error during batch inference: {0}',

        # Data Analysis Dialog
        'dlg_data_analysis': 'Data Analysis',
        'section_stats_distribution': 'Statistics & Distribution',
        'section_timeseries': 'Time Series',
        'label_stats_item': 'Item',
        'label_stats_mean': 'Mean',
        'label_stats_std': 'Std Dev',
        'label_stats_min': 'Min',
        'label_stats_max': 'Max',
        'label_stats_median': 'Median',
        'label_display': 'Display:',
        'label_raw_data': 'Raw Data',
        'label_moving_avg': 'Moving Avg',
        'label_bin_avg': 'Bin Avg',
        'label_window': 'Window:',
        'label_bin': 'Bin:',
        'label_display_range': 'Range:',
        'label_auto': 'Auto',
        'label_display_items': 'Display Items:',
        'label_click_to_jump': 'Click graph to jump to image',
        'btn_close': 'Close',
        'label_angle_original': 'Angle(orig: {0})',
        'label_throttle_original': 'Throttle(orig: {0})',
        'label_angle_ds': 'Angle(DS: {0})',
        'label_throttle_ds': 'Throttle(DS: {0})',
        'label_dist_title_with_ds': 'Angle / Throttle Distribution (n={0}, DS excluded: {1})',
        'label_dist_title': 'Angle / Throttle Distribution (n={0})',
        'label_value': 'Value',
        'label_frequency': 'Frequency',
        'label_no_data': 'No Data',
        'label_select_display_item': 'Please select display items',
        'label_no_data_available': 'No data available',
        'label_bin_avg_title': 'Bin Average (every {0} indices)',
        'label_data_trend_ma': 'Data Trend (Moving Avg: window {0})',
        'label_data_trend': 'Data Trend',
        'label_index': 'Index',

        # Databricks/Colab Settings Dialog
        'label_not_set': '(not set)',
        'label_using_default': '(using default)',
        'msg_env_setup_help': 'For security, set credentials via environment variables:\n\nWindows (PowerShell):\n  $env:DATABRICKS_ENABLED = "true"\n  $env:DATABRICKS_HOST = "https://..."\n  $env:DATABRICKS_TOKEN = "dapi..."\n\nLinux/Mac:\n  export DATABRICKS_ENABLED="true"\n  export DATABRICKS_HOST="https://..."\n  export DATABRICKS_TOKEN="dapi..."',
        'msg_oauth_setup_guide': 'To enable Google Colab integration, follow these steps:\n\n1. Create a project in Google Cloud Console\n2. Enable Google Drive API\n3. Create OAuth 2.0 Client ID and download client_secrets.json\n4. Set environment variables:\n   COLAB_ENABLED=true\n   GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json',
        'tab_oauth_guide': 'OAuth Setup Guide',
        'dlg_auth_required': 'Authentication Required',
        'msg_auth_required': 'First-time connection requires Google account authentication in browser.\n\nWhen browser opens, select your Google account and complete authentication.\n(Timeout: 60 seconds)\n\nContinue?',
        'dlg_import_error': 'Import Error',
        'msg_import_error_colab': 'Required libraries are not installed:\n\n{0}\n\nPlease install with: pip install pydrive2 google-auth google-auth-oauthlib pyyaml',
        'dlg_auth_timeout': 'Authentication Timeout',
        'msg_auth_timeout': '{0}\n\nThis occurs when the browser is closed or authentication takes too long.\nPlease click the "Connection Test" button again.',
        'dlg_connection_test_error': 'Connection Test Error',
        'msg_connection_test_error': 'An error occurred during connection test:\n\n{0}',

        # Display Settings Dialog
        'label_current_size': 'Current size: {0} x {1}',
        'label_current_font_size': 'Current font size: {0}pt',
        'status_display_settings_applied': 'Display settings applied - Window: {0}x{1}, Font: {2}pt',

        # Databricks/Colab Status
        'status_disabled': 'Disabled',
        'status_config_error': 'Configuration Error',
        'status_configured': 'Configured',
        'msg_databricks_disabled': 'Databricks integration is disabled (using local MLflow)\n\nTo enable, set environment variables:\n  DATABRICKS_ENABLED=true\n  DATABRICKS_HOST=https://...\n  DATABRICKS_TOKEN=dapi...',
        'msg_databricks_workspace': 'Databricks workspace: {0}',
        'msg_env_host_not_set': 'Environment variable DATABRICKS_HOST is not set',
        'msg_env_host_https_required': 'DATABRICKS_HOST must start with https://',
        'msg_env_token_not_set': 'Environment variable DATABRICKS_TOKEN is not set',
        'msg_colab_disabled': 'Google Colab integration is disabled\n\nTo enable, set environment variables:\n  COLAB_ENABLED=true\n  GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json',
        'msg_colab_workspace': 'Google Drive Folder: {0}',
        'msg_env_client_secrets_not_set': 'Environment variable GOOGLE_CLIENT_SECRETS is not set',
        'msg_env_client_secrets_not_found': 'Client secrets file not found: {0}',
        'status_authenticated': 'Authenticated',
        'status_not_authenticated': 'Not Authenticated',
        'msg_colab_authenticated': '\nAuthenticated',
        'msg_colab_auth_required': '\nAuthentication required (browser auth on first transfer)',
        'msg_databricks_env_template': '''# Databricks Environment Variables

# Windows (PowerShell):
$env:DATABRICKS_ENABLED = "true"
$env:DATABRICKS_HOST = "https://your-workspace.cloud.databricks.com"
$env:DATABRICKS_TOKEN = "dapi..."
$env:DATABRICKS_EXPERIMENT_PREFIX = "/Users/your-email@example.com/experiments"

# Windows (Command Prompt):
set DATABRICKS_ENABLED=true
set DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
set DATABRICKS_TOKEN=dapi...
set DATABRICKS_EXPERIMENT_PREFIX=/Users/your-email@example.com/experiments

# Linux/Mac:
export DATABRICKS_ENABLED="true"
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_EXPERIMENT_PREFIX="/Users/your-email@example.com/experiments"

# .env file format:
DATABRICKS_ENABLED=true
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
DATABRICKS_EXPERIMENT_PREFIX=/Users/your-email@example.com/experiments
''',
        'msg_colab_env_template': '''# Google Colab Environment Variables

# Windows (PowerShell):
$env:COLAB_ENABLED = "true"
$env:GOOGLE_CLIENT_SECRETS = "C:\\path\\to\\client_secrets.json"
$env:COLAB_DRIVE_FOLDER_NAME = "annotation_data"

# Windows (Command Prompt):
set COLAB_ENABLED=true
set GOOGLE_CLIENT_SECRETS=C:\\path\\to\\client_secrets.json
set COLAB_DRIVE_FOLDER_NAME=annotation_data

# Linux/Mac:
export COLAB_ENABLED="true"
export GOOGLE_CLIENT_SECRETS="/path/to/client_secrets.json"
export COLAB_DRIVE_FOLDER_NAME="annotation_data"

# .env file format:
COLAB_ENABLED=true
GOOGLE_CLIENT_SECRETS=/path/to/client_secrets.json
COLAB_DRIVE_FOLDER_NAME=annotation_data
''',
        'msg_oauth_setup_guide_full': '''================================================================================
OAuth Setup Guide for Google Cloud Console
================================================================================

1. Access Google Cloud Console
   https://console.cloud.google.com/

2. Create or select a project
   - Click the project selector at the top and choose "New Project"
   - Enter project name and create

3. Enable Google Drive API
   - From the left menu, go to "APIs & Services" → "Library"
   - Search for "Google Drive API"
   - Click "Enable"

4. Configure OAuth consent screen
   - Go to "APIs & Services" → "OAuth consent screen"
   - User type: Select "External" (for personal use)
   - Enter app name and email address
   - No need to add scopes (will be auto-configured later)
   - Add your email address as a test user

5. Create OAuth Client ID
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Application type: "Desktop app"
   - Enter a name and click "Create"

6. Download client_secrets.json
   - Click the download icon next to the created client ID
   - Click "Download JSON"
   - Rename the file to "client_secrets.json" and save

7. Set environment variables
   COLAB_ENABLED=true
   GOOGLE_CLIENT_SECRETS=path/to/saved/client_secrets.json

================================================================================
Notes
================================================================================

- client_secrets.json contains sensitive information. Do not commit to Git
- Recommended to add client_secrets.json to .gitignore
- Browser authentication will be required on first transfer
- Credentials are saved to .google_credentials.json and reused automatically

================================================================================''',

        # MLflow UI
        'dlg_mlflow_ui': 'MLflow UI',
        'msg_mlflow_ui_started': 'Local MLflow UI has been started.\n\nAccess http://localhost:5000 in your browser to view experiment results.\n\nClose the command window to stop the UI.',
        'msg_mlflow_ui_failed': 'Failed to start MLflow UI:\n\n{0}\n\nPlease verify MLflow is installed: pip install mlflow',
        'dlg_databricks_not_enabled': 'Databricks Not Enabled',
        'msg_databricks_not_enabled': 'Databricks integration is not enabled.\n\nPlease turn ON the "Databricks Integration" checkbox.',
        'msg_databricks_enable_confirm': 'Databricks integration is not enabled.\n\nWould you like to enable and connect?',
        'dlg_databricks_connection_failed': 'Databricks Connection Failed',
        'msg_databricks_connection_failed': 'Failed to connect to Databricks.\n\nPlease check your environment variable settings.',
        'dlg_databricks_connection_success': 'Databricks Connection Success',
        'msg_databricks_connection_success': 'Successfully connected to Databricks.',
        'dlg_databricks_connection_error': 'Databricks Connection Error',
        'msg_databricks_connection_error_env': 'Failed to connect to Databricks.\n\nPlease check your environment variable settings:\n- DATABRICKS_HOST\n- DATABRICKS_TOKEN\n\nFalling back to local MLflow mode.',
        'status_disconnected': 'Disconnected',
        'dlg_volumes_path_not_exist': 'Volumes Path Does Not Exist',
        'msg_volumes_path_not_exist': 'The transfer destination Volumes path does not exist:\n\n{0}\n\nDetails: {1}\n\nPlease create this path in Databricks and try again.\n\nYou can also specify a different path using\nthe DATABRICKS_VOLUMES_PATH environment variable.\n\nExample: /Volumes/workspace/default/test',
        'dlg_transfer_confirm': 'Transfer Confirmation',
        'msg_transfer_confirm_databricks': 'Transfer the following to Databricks:\n\nAnnotation count: {0}\nFilename: {1}\nDestination: {2}/{1}\n\nContinue?',
        'dlg_colab_not_enabled': 'Google Colab Not Enabled',
        'msg_colab_not_enabled': 'Google Colab integration is not enabled.\n\nTo enable, set environment variables:\n  COLAB_ENABLED=true\n  GOOGLE_CLIENT_SECRETS=path/to/client_secrets.json\n\nCheck the Settings button for details.',

        # Sync Dialog
        'label_local_runs': 'Local Runs: {0}',
        'label_databricks_runs': 'Databricks Runs: {0}',
        'label_unsynced_runs': 'Estimated Unsynced Runs: {0}',
        'label_orphaned_runs': 'Runs only in Databricks: {0}',
        'dlg_delete_confirm': 'Delete Confirmation',
        'msg_delete_runs_confirm': 'Delete {0} runs from Databricks.\n\nThis action cannot be undone. Continue?',

        # Transfer Complete/Error
        'dlg_transfer_complete': 'Transfer Complete',
        'msg_transfer_complete_databricks': 'Transfer to Databricks completed.\n\nAnnotation count: {0}\nZIP size: {1:.2f} MB\nDestination: {2}',
        'dlg_transfer_error': 'Transfer Error',
        'msg_transfer_error': 'An error occurred during transfer:\n\n{0}',
        'msg_unknown_error': 'Unknown error',
        'label_google_drive_models': 'Models on Google Drive: {0}',
        'label_unknown_date': 'Unknown',

        # Auto-training pipeline
        'chk_auto_train_after_transfer': 'Start auto-training after transfer',
        'tip_auto_train_cluster_required': 'Cluster ID is required for auto-training (env: DATABRICKS_CLUSTER_ID)',
        'msg_auto_train_started': 'Auto-training started (Run ID: {0})',
        'msg_auto_train_failed': 'Transfer succeeded but failed to start training:\n{0}',
        'msg_transfer_and_train_complete': 'Transfer to Databricks completed.\n\nAnnotations: {0}\nZIP size: {1:.1f} MB\nDestination: {2}\n\nAuto-training started (Run ID: {3})',
        'label_cluster_id': 'Cluster ID',
        'label_notebook_path': 'Notebook Path',
        'label_auto_train_settings': 'Auto-training Pipeline Settings',
        'label_set_via_env': 'Set via environment variable',
    },
}


# =============================================================================
# キーマッピング（旧キー → 新キー）
# =============================================================================
# 既存コードとの互換性のため、旧キー名でもアクセス可能にする
# 注意: 新規コードでは新しいキー名を使用すること

_KEY_MAPPING = {
    # app_
    'window_title': 'app_title',
    'language': 'app_language',
    'language_ja': 'app_language_ja',
    'language_en': 'app_language_en',
    'language_switch': 'app_language_switch',
    'language_changed': 'app_language_changed',
    'restart_required': 'app_restart_required',

    # section_
    'data_load_section': 'section_data_load',
    'save_annotation_section': 'section_save_annotation',
    'pilot_model_section': 'section_pilot_model',
    'object_detection_section': 'section_object_detection',
    'model_management_section': 'section_model_management',
    'display_settings_section': 'section_display_settings',

    # btn_
    'browse': 'btn_browse',
    'load_images': 'btn_load_images',
    'load_annotations': 'btn_load_annotations',
    'create_video': 'btn_create_video',
    'train_and_save': 'btn_train_save',
    'load_model': 'btn_load_model',
    'auto_annotate': 'btn_auto_annotate',
    'batch_inference': 'btn_batch_inference',
    'load_yolo_annotation': 'btn_load_yolo_annotation',
    'preset': 'btn_preset',
    'apply': 'btn_apply',
    'train_yolo': 'btn_train_yolo',
    'yolo_auto_annotate': 'btn_yolo_auto_annotate',
    'open_mlflow': 'btn_open_mlflow',
    'open_databricks': 'btn_open_databricks',
    'sync': 'btn_sync',
    'transfer': 'btn_transfer',
    'settings': 'btn_settings',
    'open_colab': 'btn_open_colab',
    'download': 'btn_download',
    'window_font_settings': 'btn_window_font_settings',
    'reverse_play': 'btn_reverse_play',
    'forward_play': 'btn_forward_play',
    'delete_current': 'btn_delete_current',
    'restore_deleted': 'btn_restore_deleted',
    'restore_all_deleted': 'btn_restore_all_deleted',
    'current_position': 'btn_current_position',
    'range_delete': 'btn_range_delete',
    'detect': 'btn_detect',
    'redetect': 'btn_redetect',
    'clear': 'btn_clear',
    'analysis': 'btn_analysis',

    # label_
    'annotated_count': 'label_annotated_count',
    'image_source': 'label_image_source',
    'pilot_model_select': 'label_pilot_model_select',
    'detection_classes': 'label_detection_classes',
    'classes_example': 'label_classes_example',
    'yolo_model': 'label_yolo_model',
    'mlflow_local': 'label_mlflow_local',
    'databricks_integration': 'label_databricks_integration',
    'colab_integration': 'label_colab_integration',
    'canvas_zoom_label': 'label_canvas_zoom_label',
    'canvas_zoom_tooltip': 'label_canvas_zoom_tooltip',
    'image_seek': 'label_image_seek',
    'play': 'label_play',
    'delete_restore': 'label_delete_restore',
    'delete_range': 'label_delete_range',
    'from': 'label_from',
    'downsampling': 'label_downsampling',
    'angle_range': 'label_angle_range',
    'throttle_range': 'label_throttle_range',
    'consecutive': 'label_consecutive',
    'interval': 'label_interval',
    'items': 'label_items',
    'items_added': 'label_items_added',
    'cam_method': 'label_cam_method',
    'cam_target': 'label_cam_target',
    'cam_direction': 'label_cam_direction',
    'image_info': 'label_image_info',
    'data_distribution': 'label_data_distribution',
    'no_annotation': 'label_no_annotation',
    'no_image_selected': 'label_no_image_selected',
    'inference_label': 'label_inference',
    'select_folder_prompt': 'label_select_folder_prompt',

    # chk_
    'show_future_annotation': 'chk_show_future_annotation',
    'show_inference': 'chk_show_inference',
    'show_diff_vector': 'chk_show_diff_vector',
    'show_detection_inference': 'chk_show_detection_inference',
    'show_segmentation_inference': 'chk_show_segmentation_inference',
    'dark_mode': 'chk_dark_mode',

    # placeholder_
    'folder_placeholder': 'placeholder_folder',
    'classes_placeholder': 'placeholder_classes',

    # tip_
    'load_model_tooltip': 'tip_load_model',
    'show_future_annotation_tooltip': 'tip_show_future_annotation',
    'cam_method_tooltip': 'tip_cam_method',
    'cam_target_tooltip': 'tip_cam_target',
    'cam_direction_tooltip': 'tip_cam_direction',
    'train_yolo_tooltip': 'tip_train_yolo',
    'open_mlflow_tooltip': 'tip_open_mlflow',
    'open_databricks_tooltip': 'tip_open_databricks',
    'sync_tooltip': 'tip_sync',
    'transfer_tooltip': 'tip_transfer',
    'open_colab_tooltip': 'tip_open_colab',
    'colab_transfer_tooltip': 'tip_colab_transfer',
    'colab_download_tooltip': 'tip_colab_download',
    'set_start_tooltip': 'tip_set_start',
    'set_end_tooltip': 'tip_set_end',
    'consecutive_tooltip': 'tip_consecutive',
    'interval_tooltip': 'tip_interval',
    'detect_tooltip': 'tip_detect',
    'clear_downsampling_tooltip': 'tip_clear_downsampling',
    'analysis_tooltip': 'tip_analysis',

    # status_
    'click_to_reannotate': 'status_click_to_reannotate',
    'model_not_loaded': 'status_model_not_loaded',
    'detection_model_not_loaded': 'status_detection_model_not_loaded',
    'segmentation_model_not_loaded': 'status_segmentation_model_not_loaded',

    # dlg_
    'dialog_warning': 'dlg_warning',
    'dialog_error': 'dlg_error',
    'dialog_info': 'dlg_info',
    'dialog_complete': 'dlg_complete',
    'dialog_confirm': 'dlg_confirm',
    'dialog_export_complete': 'dlg_export_complete',
    'dialog_load_complete': 'dlg_load_complete',
    'dialog_load_complete_warning': 'dlg_load_complete_warning',
    'dialog_load_error': 'dlg_load_error',
    'dialog_sync_complete': 'dlg_sync_complete',
    'dialog_sync_cancel': 'dlg_sync_cancel',
    'dialog_transfer_complete': 'dlg_transfer_complete',
    'dialog_transfer_cancel': 'dlg_transfer_cancel',
    'dialog_add_complete': 'dlg_add_complete',
    'dialog_copy_complete': 'dlg_copy_complete',
}

# 旧キーを新キーに変換するヘルパー関数（get_text内で使用）
def _resolve_key(key: str) -> str:
    """旧キー名を新キー名に変換。存在しなければそのまま返す"""
    return _KEY_MAPPING.get(key, key)

# get_text関数を更新してキーマッピングを適用
_original_get_text = get_text

def get_text(key: str, *args, **kwargs) -> str:
    """翻訳テキストを取得（旧キー名もサポート）"""
    resolved_key = _resolve_key(key)
    return _original_get_text(resolved_key, *args, **kwargs)
