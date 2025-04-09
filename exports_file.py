"""
エクスポート関連ユーティリティ - アノテーションをエクスポートする関数
"""

import os
import json
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Callable, Optional

def export_to_donkey(
    folder_path: str, 
    annotations: Dict[str, Dict[str, Any]], 
    inference_results: Optional[Dict[str, Dict[str, Any]]] = None,
    deleted_indexes: Optional[List[int]] = None
) -> str:
    """アノテーションをDonkeycar形式でエクスポートする（1000件ごとに分割）

    Args:
        folder_path: 出力先のフォルダパス
        annotations: アノテーション辞書
        inference_results: 推論結果辞書（オプション）
        deleted_indexes: 削除されたインデックスのリスト（オプション）

    Returns:
        作成されたマニフェストファイルのパス
    """
    import time
    from datetime import datetime
    
    # 出力フォルダを作成
    output_folder = folder_path
    os.makedirs(output_folder, exist_ok=True)
    
    # 画像を保存するimagesフォルダを作成
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    # 現在の日時を取得してセッションIDを作成
    current_date = datetime.now().strftime("%y-%m-%d")
    session_id = f"{current_date}_0"
    
    # タイムスタンプを記録（マニフェスト用）
    created_timestamp = time.time()
    
    # アノテーション情報をインデックス順に整理
    # オリジナルのインデックスを復元するか、または新しいインデックスを割り当てる
    indexed_annotations = []
    
    for img_path, annotation in annotations.items():
        if not annotation:
            continue
        
        # オリジナルのインデックスがあればそれを使用
        original_index = annotation.get("original_index")
        
        # ファイル名からインデックスを抽出
        extracted_index = None
        try:
            import re
            basename = os.path.basename(img_path)
            match = re.match(r'^(\d+)_', basename)
            if match:
                extracted_index = int(match.group(1))
        except:
            pass
        
        # インデックス優先順位: オリジナル > ファイル名から抽出 > なし（後で連番割り当て）
        index = original_index if original_index is not None else extracted_index
        
        # アノテーション情報と画像パスを保存
        indexed_annotations.append({
            "index": index,
            "img_path": img_path,
            "annotation": annotation
        })
    
    # インデックスがないエントリに連番を割り当て
    next_index = 0
    for entry in indexed_annotations:
        if entry["index"] is None:
            while any(e["index"] == next_index for e in indexed_annotations if e["index"] is not None):
                next_index += 1
            entry["index"] = next_index
            next_index += 1
    
    # インデックス順にソート
    indexed_annotations.sort(key=lambda x: x["index"] if x["index"] is not None else float('inf'))
    
    # インデックスを最終的に割り当て（連続した番号になるように）
    catalog_entries = []
    
    for i, entry in enumerate(indexed_annotations):
        img_path = entry["img_path"]
        annotation = entry["annotation"]
        assigned_index = i  # 連番を割り当て
        
        # 画像ファイル名を作成（インデックスをプレフィックスに）
        new_img_name = f"{assigned_index}_cam_image_array_.jpg"
        
        # 画像をimagesフォルダにコピー
        dest_path = os.path.join(images_folder, new_img_name)
        shutil.copy2(img_path, dest_path)
        
        # タイムスタンプ
        timestamp_ms = int(time.time() * 1000)
        
        # エントリを作成 - 画像ファイル名のみを使用（パスなし）
        catalog_entry = {
            "_index": assigned_index,
            "_session_id": session_id,
            "_timestamp_ms": timestamp_ms,
            "cam/image_array": new_img_name,
            "user/angle": annotation["angle"],
            "user/mode": "user",
            "user/throttle": annotation["throttle"]
        }
        
        # 位置情報があれば追加
        if 'loc' in annotation:
            catalog_entry["user/loc"] = annotation["loc"]
        
        # 推論結果があれば追加
        if inference_results and img_path in inference_results:
            inference = inference_results[img_path]
            # 新しいキー形式確認
            if "pilot/angle" in inference and "pilot/throttle" in inference:
                catalog_entry["pilot/angle"] = inference["pilot/angle"]
                catalog_entry["pilot/throttle"] = inference["pilot/throttle"]
            else:
                catalog_entry["pilot/angle"] = inference["angle"]
                catalog_entry["pilot/throttle"] = inference["throttle"]
                
            # 推論結果に位置情報があれば追加
            if "loc" in inference or "pilot/loc" in inference:
                catalog_entry["pilot/loc"] = inference.get("pilot/loc", inference.get("loc", 0))
        
        catalog_entries.append(catalog_entry)
    
    if not catalog_entries:
        return None
    
    # 1000件ごとに分割してカタログファイルを作成
    catalog_files = []
    
    for i in range(0, len(catalog_entries), 1000):
        batch = catalog_entries[i:i+1000]
        catalog_path = os.path.join(output_folder, f"catalog_{i//1000}.catalog")
        catalog_files.append(os.path.basename(catalog_path))
        
        batch_line_lengths = []  # このバッチの行長さ
        
        with open(catalog_path, 'w') as f:
            for entry in batch:
                json_line = json.dumps(entry)
                f.write(json_line + '\n')
                batch_line_lengths.append(len(json_line))
        
        # カタログマニフェストファイルを作成
        manifest_path = os.path.join(output_folder, f"catalog_{i//1000}.catalog_manifest")
        manifest_data = {
            "created_at": created_timestamp,
            "line_lengths": batch_line_lengths,
            "path": os.path.basename(catalog_path),
            "start_index": i
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f)
    
    # 削除されたインデックスを確認し、設定（Noneの場合は空リスト）
    if deleted_indexes is None:
        deleted_indexes = []
    
    # カスタム列の設定（存在するデータを確認）
    column_names = ["cam/image_array", "user/angle", "user/throttle", "user/mode"]
    column_types = ["image_array", "float", "float", "str"]
    
    # 位置情報や推論結果のカラムが使用されていれば追加
    has_loc = any('loc' in anno for anno in annotations.values())
    has_pilot = inference_results is not None and len(inference_results) > 0
    
    if has_pilot:
        column_names.extend(["pilot/angle", "pilot/throttle"])
        column_types.extend(["float", "float"])
    
    if has_loc:
        column_names.extend(["user/loc"])
        column_types.extend(["int"])
        if has_pilot:
            column_names.extend(["pilot/loc"])
            column_types.extend(["int"])
    
    # manifest.json ファイルを作成
    manifest_data = [
        # 列名のリスト
        column_names,
        # データ型のリスト
        column_types,
        # 追加設定（空の辞書）
        {},
        # セッション情報
        {
            "created_at": created_timestamp,
            "sessions": {
                "all_full_ids": [session_id],
                "last_id": 0,
                "last_full_id": session_id
            }
        },
        # カタログファイル情報（削除済みインデックスを含む）
        {
            "paths": catalog_files,
            "current_index": len(catalog_entries),
            "max_len": 1000,
            "deleted_indexes": deleted_indexes
        }
    ]
    
    manifest_path = os.path.join(output_folder, "manifest.json")
    with open(manifest_path, 'w') as f:
        for item in manifest_data:
            f.write(json.dumps(item) + '\n')
    
    return manifest_path

def export_to_jetracer(
    folder_path: str, 
    annotations: Dict[str, Dict[str, Any]], 
    inference_results: Optional[Dict[str, Dict[str, Any]]] = None
) -> str:
    """アノテーションをJetracer形式でエクスポートする

    Args:
        folder_path: 出力先のフォルダパス
        annotations: アノテーション辞書
        inference_results: 推論結果辞書（オプション）

    Returns:
        作成されたカタログファイルのパス
    """
    import time
    from datetime import datetime
    
    # フォルダを作成
    output_folder = folder_path
    os.makedirs(output_folder, exist_ok=True)
    
    # カタログファイルを作成
    current_date = datetime.now().strftime("%y-%m-%d")
    session_id = f"{current_date}_{0}"
    catalog_path = os.path.join(output_folder, "catalog_0.catalog")
    
    # カタログエントリを作成
    catalog_entries = []
    
    for img_path, annotation in annotations.items():
        if not annotation:
            continue
            
        img_name = os.path.basename(img_path)
        
        # 画像をコピー
        dest_path = os.path.join(output_folder, img_name)
        shutil.copy2(img_path, dest_path)
        
        # タイムスタンプ
        timestamp_ms = int(time.time() * 1000)
        
        # エントリを作成
        entry = {
            "_index": len(catalog_entries),
            "_session_id": session_id,
            "_timestamp_ms": timestamp_ms,
            "cam/image_array": img_name,
            "user/angle": annotation["angle"],
            "user/mode": "user",
            "user/throttle": annotation["throttle"]
        }
        
        # 推論結果があれば追加
        if inference_results and img_path in inference_results:
            inference = inference_results[img_path]
            if "auto/angle" in inference and "auto/throttle" in inference:
                entry["auto/angle"] = inference["auto/angle"]
                entry["auto/throttle"] = inference["auto/throttle"]
            else:
                entry["auto/angle"] = inference["angle"]
                entry["auto/throttle"] = inference["throttle"]
        
        catalog_entries.append(entry)
    
    # カタログファイルに書き込み
    with open(catalog_path, 'w') as f:
        for entry in catalog_entries:
            f.write(json.dumps(entry) + '\n')
    
    return catalog_path

def export_to_yolo(
    folder_path: str,
    bbox_annotations: Dict[str, List[Dict[str, Any]]],
    output_subfolder: str = "yolo_annotations"
) -> str:
    """バウンディングボックスアノテーションをYOLO形式でエクスポートする

    Args:
        folder_path: 出力先のフォルダパス
        bbox_annotations: バウンディングボックスアノテーション辞書
        output_subfolder: 出力先のサブフォルダ名（デフォルト: "yolo_annotations"）

    Returns:
        作成されたデータセット設定ファイル（dataset.yaml）のパス
    """
    import os
    import shutil
    import json
    
    # YOLO形式のアノテーション用フォルダを作成
    yolo_folder = os.path.join(folder_path, output_subfolder)
    os.makedirs(yolo_folder, exist_ok=True)
    
    # 画像とラベルのフォルダを作成
    images_folder = os.path.join(yolo_folder, "images")
    labels_folder = os.path.join(yolo_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    
    # クラスリストを取得
    all_classes = set()
    for bboxes in bbox_annotations.values():
        for bbox in bboxes:
            all_classes.add(bbox.get('class', 'unknown'))
    
    class_list = sorted(list(all_classes))
    
    # クラス名ファイルを作成
    class_file_path = os.path.join(yolo_folder, "classes.txt")
    with open(class_file_path, 'w') as f:
        for cls in class_list:
            f.write(f"{cls}\n")
    
    # データセット設定用のYAMLファイルを作成
    yaml_content = f"""
# YOLO形式のデータセット設定
path: {yolo_folder}  # データセットのルートディレクトリ
train: images  # 訓練用画像の相対パス
val: images    # 検証用画像の相対パス

nc: {len(class_list)}  # クラス数
names: {class_list}  # クラス名
"""
    
    yaml_file_path = os.path.join(yolo_folder, "dataset.yaml")
    with open(yaml_file_path, 'w') as f:
        f.write(yaml_content)
    
    # 各画像のアノテーションを処理
    processed_count = 0
    total_bboxes = 0
    
    for img_path, bboxes in bbox_annotations.items():
        if not bboxes:
            continue
            
        # 画像ファイル名を取得
        img_filename = os.path.basename(img_path)
        img_basename = os.path.splitext(img_filename)[0]
        
        # 画像をコピー
        dst_img_path = os.path.join(images_folder, img_filename)
        shutil.copy2(img_path, dst_img_path)
        
        # YOLOフォーマットのラベルファイルを作成
        label_path = os.path.join(labels_folder, f"{img_basename}.txt")
        
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                # クラスIDを取得
                class_name = bbox.get('class', 'unknown')
                class_id = class_list.index(class_name)
                
                # バウンディングボックスの座標を取得
                x1 = bbox['x1']
                y1 = bbox['y1']
                x2 = bbox['x2']
                y2 = bbox['y2']
                
                # YOLO形式に変換（中心x, 中心y, 幅, 高さ）
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # YOLOフォーマットでファイルに書き込み
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                total_bboxes += 1
        
        processed_count += 1
    
    # README.txtファイルを作成して使用方法を説明
    readme_content = f"""# YOLO形式アノテーションデータ

このフォルダには、YOLO形式でエクスポートされたアノテーションデータが含まれています。

## フォルダ構成
- images/: アノテーション付き画像
- labels/: YOLOフォーマットのアノテーションファイル（各画像に対応）
- classes.txt: クラス名のリスト
- dataset.yaml: YOLOv5/v8用のデータセット設定ファイル

## クラス情報
検出クラス数: {len(class_list)}
クラス: {', '.join(class_list)}

## 統計情報
アノテーション画像数: {processed_count}
合計バウンディングボックス数: {total_bboxes}

## YOLOフォーマット
各行の形式: <class_id> <x_center> <y_center> <width> <height>
※すべての座標値は画像サイズで正規化されています（0～1の範囲）
"""
    
    with open(os.path.join(yolo_folder, "README.txt"), 'w') as f:
        f.write(readme_content)
    
    return yaml_file_path

def export_to_video(
    annotations: Dict[str, Dict[str, Any]], 
    inference_results: Dict[str, Dict[str, Any]], 
    output_path: str,
    show_inference: bool = True,
    skip_count: int = 1,
    fps: int = 30,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> int:
    """アノテーションを動画として出力する

    Args:
        annotations: アノテーション辞書
        inference_results: 推論結果辞書
        output_path: 出力ファイルパス
        show_inference: 推論結果を表示するかどうか
        skip_count: 何枚ごとに動画に含めるか
        fps: フレームレート
        progress_callback: 進捗コールバック関数

    Returns:
        処理されたフレーム数
    """
    # アノテーション画像のパスをスキップ設定で抽出
    image_paths = list(annotations.keys())
    image_paths.sort()  # パスをソート
    
    if skip_count > 1:
        image_paths = image_paths[::skip_count]
    
    if not image_paths:
        return 0
        
    # 最初の画像からビデオサイズを決定
    first_img = cv2.imread(image_paths[0])
    height, width, channels = first_img.shape
    
    # 動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデック
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_processed = 0
    
    try:
        for i, img_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i, len(image_paths))
                
            # 画像を読み込む
            cv_img = cv2.imread(img_path)
            
            if cv_img is None:
                continue
                
            # PIL画像に変換（アノテーションの描画用）
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # アノテーションを描画（赤色）
            if img_path in annotations:
                anno = annotations[img_path]
                x, y = anno["x"], anno["y"]
                # 赤い円を描画
                draw.ellipse((x-15, y-15, x+15, y+15), outline='red', width=3)
            
            # 推論結果を描画（青色）
            if show_inference and img_path in inference_results:
                inference = inference_results[img_path]
                x, y = inference["x"], inference["y"]
                # 青い円を描画
                draw.ellipse((x-15, y-15, x+15, y+15), outline='blue', width=3)
            
            # PIL画像をOpenCV形式に戻す
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # ビデオに追加
            video.write(cv_img)
            frames_processed += 1
            
    finally:
        # ビデオを閉じる
        video.release()
    
    return frames_processed