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