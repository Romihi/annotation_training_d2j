"""
エクスポート関連ユーティリティ - アノテーションをエクスポートする関数
"""

import os
import json
import shutil
import time
import re
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Callable, Optional, Union

def export_to_donkey(
    folder_path: str, 
    annotations: Dict[Union[str, int], Dict[str, Any]], 
    inference_results: Optional[Dict[Union[str, int], Dict[str, Any]]] = None,
    deleted_indexes: Optional[List[int]] = None,
    images_list: Optional[List[str]] = None,  # 互換性のために残す
    image_map: Optional[Dict[int, Dict[str, str]]] = None,  # 新しいパラメータ：{index: {variant: image_path, ...}, ...}
    variant_keys: Optional[Dict[str, str]] = None,  # 新しいパラメータ：{variant: key_name, ...}
    diff_vectors: Optional[Dict[Union[str, int], Dict[str, Any]]] = None  # 追加: 差分ベクトルデータ
) -> str:
    
    """アノテーションをDonkeycar形式でエクスポートする（1000件ごとに分割） - 複数画像ソース対応

    Args:
        folder_path: 出力先のフォルダパス
        annotations: アノテーション辞書（キーがインデックスまたは画像パス）
        inference_results: 推論結果辞書（オプション）
        deleted_indexes: 削除されたインデックスのリスト（オプション）
        images_list: 画像パスのリスト（互換性のため残す）
        image_map: インデックスごとの画像ソース別パスのマップ
        variant_keys: 画像ソース別のカタログキー名

    Returns:
        作成されたマニフェストファイルのパス
    """    
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
    indexed_annotations = []
    
    for key, annotation in annotations.items():
        if not annotation:
            continue
        
        # キーの型に基づいて元のインデックスを取得
        if isinstance(key, int):
            original_index = key
        else:
            # パスからインデックスを抽出
            original_index = annotation.get("original_index")
            if original_index is None:
                try:
                    basename = os.path.basename(key)
                    match = re.match(r'^(\d+)_', basename)
                    if match:
                        original_index = int(match.group(1))
                except:
                    pass
            
        # アノテーション情報とインデックスを保存
        indexed_annotations.append({
            "index": original_index,
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
    
    # 使用されるカタログキーの一覧を取得
    catalog_keys = []
    if variant_keys:
        catalog_keys = list(variant_keys.values())
    
    # インデックスを最終的に割り当て（連続した番号になるように）
    catalog_entries = []
    
    for i, entry in enumerate(indexed_annotations):
        original_index = entry["index"]
        annotation = entry["annotation"]
        assigned_index = i  # 連番を割り当て
        
        # 画像マップからこのインデックスの画像パスを取得
        variant_images = {}
        if image_map and original_index in image_map:
            variant_images = image_map[original_index]
        elif isinstance(original_index, int) and images_list and 0 <= original_index < len(images_list):
            # 後方互換性のため単一リストからも取得
            img_path = images_list[original_index]
            variant = "cam"  # デフォルトバリアント
            try:
                basename = os.path.basename(img_path)
                match = re.match(r'^\d+_([a-zA-Z0-9]+)_', basename)
                if match:
                    variant = match.group(1)
            except:
                pass
            variant_images[variant] = img_path
        
        if not variant_images:
            print(f"警告: インデックス {original_index} の画像が見つかりません。このエントリはスキップします。")
            continue
        
        # タイムスタンプ
        timestamp_ms = int(time.time() * 1000)
        
        # 基本エントリを作成
        catalog_entry = {
            "_index": assigned_index,
            "_session_id": session_id,
            "_timestamp_ms": timestamp_ms,
            "user/angle": annotation["angle"],
            "user/mode": "user",
            "user/throttle": annotation["throttle"]
        }
        
        # 位置情報があれば追加
        if 'loc' in annotation:
            catalog_entry["user/loc"] = annotation["loc"]
        
        # 推論結果があれば追加
        if inference_results:
            inference = None
            if isinstance(original_index, int) and original_index in inference_results:
                inference = inference_results[original_index]
            
            if inference:
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

        # 追加: 差分ベクトル情報を追加
        if diff_vectors:
            # インデックスまたはパスで差分ベクトルデータを探す
            diff_data = None
            if isinstance(original_index, int) and original_index in diff_vectors:
                diff_data = diff_vectors[original_index]
            
            # パスでも探す（複数のバリアントがある場合）
            if not diff_data:
                for variant, img_path in variant_images.items():
                    if img_path in diff_vectors:
                        diff_data = diff_vectors[img_path]
                        break
            
            if diff_data:
                catalog_entry["diff/angle"] = diff_data['angle_diff']
                catalog_entry["diff/throttle"] = diff_data['throttle_diff']
                catalog_entry["diff/magnitude"] = diff_data['vector_magnitude']
                catalog_entry["diff/angle_rad"] = diff_data['vector_angle']
                        
        # 各バリアントの画像をコピーしてエントリに追加
        for variant, img_path in variant_images.items():
            if not os.path.exists(img_path):
                print(f"警告: 画像ファイル {img_path} が存在しません。")
                continue
            
            # 画像ファイル名を作成
            new_img_name = f"{assigned_index}_{variant}_image_array_.jpg"
            
            try:
                # 画像をimagesフォルダにコピー
                dest_path = os.path.join(images_folder, new_img_name)
                shutil.copy2(img_path, dest_path)
                
                # カタログキー名を決定
                catalog_key = f"{variant}/image_array"  # デフォルト
                if variant_keys and variant in variant_keys:
                    catalog_key = variant_keys[variant]
                
                # カタログキーを記録（カラム名として使用）
                if catalog_key not in catalog_keys:
                    catalog_keys.append(catalog_key)
                
                # エントリに画像情報を追加
                catalog_entry[catalog_key] = new_img_name
                
            except Exception as e:
                print(f"警告: 画像 {img_path} のコピー中にエラーが発生しました: {e}")
        
        # 少なくとも1つの画像がエントリに追加された場合のみカタログに追加
        if any(key in catalog_entry for key in catalog_keys):
            catalog_entries.append(catalog_entry)
    
    if not catalog_entries:
        print("警告: エクスポート可能なエントリがありません。")
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
    # 画像カラム
    column_names = catalog_keys + ["user/angle", "user/throttle", "user/mode"]
    column_types = ["image_array"] * len(catalog_keys) + ["float", "float", "str"]
    
    # 位置情報や推論結果のカラムが使用されていれば追加
    has_loc = any('loc' in anno for anno in annotations.values())
    has_pilot = inference_results is not None and len(inference_results) > 0
    has_diff = diff_vectors is not None and len(diff_vectors) > 0  # 追加

    if has_pilot:
        column_names.extend(["pilot/angle", "pilot/throttle"])
        column_types.extend(["float", "float"])
    
    if has_loc:
        column_names.extend(["user/loc"])
        column_types.extend(["int"])
        if has_pilot:
            column_names.extend(["pilot/loc"])
            column_types.extend(["int"])

    # 追加: 差分ベクトルのカラムを追加
    if has_diff:
        column_names.extend(["diff/angle", "diff/throttle", "diff/magnitude", "diff/angle_rad"])
        column_types.extend(["float", "float", "float", "float"])

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
    annotations: Dict[Union[str, int], Dict[str, Any]], 
    inference_results: Optional[Dict[Union[str, int], Dict[str, Any]]] = None
) -> str:
    """アノテーションをJetracer形式でエクスポートする - Donkeycar形式のディレクトリ構造に統一

    Args:
        folder_path: 出力先のフォルダパス
        annotations: アノテーション辞書（キーがインデックスまたは画像パス）
        inference_results: 推論結果辞書（オプション）

    Returns:
        作成されたマニフェストファイルのパス
    """
    import time
    from datetime import datetime
    import shutil
    import json
    import os
    import re
    from PIL import Image
    
    # 出力フォルダを作成
    output_folder = folder_path
    os.makedirs(output_folder, exist_ok=True)
    
    # 画像を保存するimagesフォルダを作成（Donkeycar形式に統一）
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    # 現在の日時を取得してセッションIDを作成
    current_date = datetime.now().strftime("%y-%m-%d")
    session_id = f"{current_date}_0"
    
    # タイムスタンプを記録（マニフェスト用）
    created_timestamp = time.time()
    
    # アノテーション情報をインデックス順に整理（Donkeycar形式と同じロジック）
    indexed_annotations = []
    
    for key, annotation in annotations.items():
        if not annotation:
            continue
        
        # キーの型に基づいて元のインデックスを取得
        if isinstance(key, int):
            original_index = key
        else:
            # パスからインデックスを抽出
            original_index = annotation.get("original_index")
            if original_index is None:
                try:
                    basename = os.path.basename(key)
                    match = re.match(r'^(\d+)_', basename)
                    if match:
                        original_index = int(match.group(1))
                except:
                    pass
            
        # アノテーション情報とインデックスを保存
        indexed_annotations.append({
            "index": original_index,
            "annotation": annotation,
            "img_path": key if isinstance(key, str) else None
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
    
    # 最初の画像から画像サイズを取得
    img_width, img_height = None, None
    for entry in indexed_annotations:
        img_path = entry["img_path"]
        if img_path and isinstance(img_path, str) and os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                    break
            except Exception as e:
                print(f"画像サイズ取得エラー ({img_path}): {e}")
                continue
    
    if img_width is None or img_height is None:
        print("エラー: 画像サイズを取得できませんでした。")
        return None
    
    print(f"検出された画像サイズ: {img_width}x{img_height}")
    
    # カタログエントリを作成（1000件ごとに分割可能な形式）
    catalog_entries = []
    
    for i, entry in enumerate(indexed_annotations):
        original_index = entry["index"]
        annotation = entry["annotation"]
        img_path = entry["img_path"]
        assigned_index = i  # 連番を割り当て
        
        if not img_path or not os.path.exists(img_path):
            print(f"警告: インデックス {original_index} の画像が見つかりません。このエントリはスキップします。")
            continue
        
        # アノテーションから座標情報を取得
        angle = annotation.get("angle", 0)
        throttle = annotation.get("throttle", 0)
        
        # -1～1の値を画像のピクセル座標に変換
        # angle: -1(左端) ～ 1(右端) → 0 ～ img_width
        # throttle: -1(下端) ～ 1(上端) → img_height ～ 0 (Jetracerでは通常Y軸は反転)
        x_pixel = int((angle + 1) * img_width / 2)
        y_pixel = int((1 - throttle) * img_height / 2)  # throttleは反転
        
        # 範囲チェック
        x_pixel = max(0, min(x_pixel, img_width - 1))
        y_pixel = max(0, min(y_pixel, img_height - 1))
        
        # Jetracer形式のファイル名を作成: x_y_index_cam_image_array_.jpg
        jetracer_filename = f"{x_pixel}_{y_pixel}_{assigned_index}_cam_image_array_.jpg"
        
        try:
            # 画像をimagesフォルダ内にJetracer形式のファイル名でコピー
            dest_path = os.path.join(images_folder, jetracer_filename)
            shutil.copy2(img_path, dest_path)
            print(f"コピー完了: {os.path.basename(img_path)} -> images/{jetracer_filename}")
        except Exception as e:
            print(f"警告: 画像のコピー中にエラーが発生しました ({img_path}): {e}")
            continue
        
        # タイムスタンプ
        timestamp_ms = int(time.time() * 1000)
        
        # カタログエントリを作成（Donkeycar形式に準拠）
        catalog_entry = {
            "_index": assigned_index,
            "_session_id": session_id,
            "_timestamp_ms": timestamp_ms,
            "cam/image_array": jetracer_filename,  # imagesフォルダ内の相対パス
            "user/angle": angle,
            "user/mode": "user",
            "user/throttle": throttle,
            "x_pixel": x_pixel,
            "y_pixel": y_pixel
        }
        
        # 位置情報があれば追加
        if 'loc' in annotation:
            catalog_entry["user/loc"] = annotation["loc"]
        
        # 推論結果があれば追加
        if inference_results:
            inference = None
            if isinstance(original_index, int) and original_index in inference_results:
                inference = inference_results[original_index]
            elif img_path in inference_results:
                inference = inference_results[img_path]
            
            if inference:
                # 新しいキー形式確認
                if "pilot/angle" in inference and "pilot/throttle" in inference:
                    catalog_entry["pilot/angle"] = inference["pilot/angle"]
                    catalog_entry["pilot/throttle"] = inference["pilot/throttle"]
                else:
                    catalog_entry["pilot/angle"] = inference.get("angle", 0)
                    catalog_entry["pilot/throttle"] = inference.get("throttle", 0)
                    
                # 推論結果に位置情報があれば追加
                if "loc" in inference or "pilot/loc" in inference:
                    catalog_entry["pilot/loc"] = inference.get("pilot/loc", inference.get("loc", 0))
        
        catalog_entries.append(catalog_entry)
    
    if not catalog_entries:
        print("警告: エクスポート可能なエントリがありません。")
        return None
    
    # 1000件ごとに分割してカタログファイルを作成（Donkeycar形式と同じ）
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
    
    # 推論結果があるかチェック
    has_pilot = inference_results is not None and len(inference_results) > 0
    has_loc = any('loc' in anno for anno in annotations.values())
    
    # カラム名とデータ型を定義（Donkeycar形式に準拠）
    column_names = ["cam/image_array", "user/angle", "user/throttle", "user/mode", "x_pixel", "y_pixel"]
    column_types = ["image_array", "float", "float", "str", "int", "int"]
    
    # 位置情報や推論結果のカラムが使用されていれば追加
    if has_pilot:
        column_names.extend(["pilot/angle", "pilot/throttle"])
        column_types.extend(["float", "float"])
    
    if has_loc:
        column_names.extend(["user/loc"])
        column_types.extend(["int"])
        if has_pilot:
            column_names.extend(["pilot/loc"])
            column_types.extend(["int"])
    
    # manifest.json ファイルを作成（Donkeycar形式と完全に統一）
    manifest_data = [
        # 列名のリスト
        column_names,
        # データ型のリスト
        column_types,
        # 追加設定（Jetracer固有の情報を追加）
        {
            "image_size": [img_width, img_height],
            "coordinate_mapping": "angle->x_pixel, throttle->y_pixel (inverted)",
            "format": "jetracer"
        },
        # セッション情報
        {
            "created_at": created_timestamp,
            "sessions": {
                "all_full_ids": [session_id],
                "last_id": 0,
                "last_full_id": session_id
            }
        },
        # カタログファイル情報
        {
            "paths": catalog_files,
            "current_index": len(catalog_entries),
            "max_len": 1000,
            "deleted_indexes": []  # Jetracerでは削除インデックスは空
        }
    ]
    
    manifest_path = os.path.join(output_folder, "manifest.json")
    with open(manifest_path, 'w') as f:
        for item in manifest_data:
            f.write(json.dumps(item) + '\n')
    
    # Jetracer用の座標情報ファイルを作成（追加情報として）
    coordinates_file = os.path.join(output_folder, "coordinates.txt")
    with open(coordinates_file, 'w') as f:
        f.write("# Jetracer座標情報\n")
        f.write(f"# 画像サイズ: {img_width}x{img_height}\n")
        f.write("# フォーマット: filename, x_pixel, y_pixel, angle, throttle\n")
        for entry in catalog_entries:
            f.write(f"{entry['cam/image_array']}, {entry['x_pixel']}, {entry['y_pixel']}, {entry['user/angle']:.4f}, {entry['user/throttle']:.4f}\n")
    
    # README.txtファイルを作成
    readme_content = f"""# Jetracer形式アノテーションデータ（Donkeycar構造準拠）

このフォルダには、Jetracer形式でエクスポートされたアノテーションデータが含まれています。
ディレクトリ構造とカタログファイル形式はDonkeycar形式に統一されています。

## ディレクトリ構造
```
{os.path.basename(output_folder)}/
├── images/                     # 画像フォルダ（Donkeycar形式に統一）
│   ├── 200_100_0_cam_image_array_.jpg
│   ├── 150_120_1_cam_image_array_.jpg
│   └── ...
├── catalog_0.catalog          # カタログファイル（JSON Lines形式）
├── catalog_0.catalog_manifest # カタログマニフェスト
├── manifest.json              # メインマニフェスト（Donkeycar形式準拠）
├── coordinates.txt            # 座標情報の一覧（Jetracer固有）
└── README.txt                 # このファイル
```

## ファイル名形式
画像ファイル名: x_y_index_cam_image_array_.jpg
例: images/200_100_2_cam_image_array_.jpg

- x: X座標のピクセル値 (0～{img_width-1})
- y: Y座標のピクセル値 (0～{img_height-1})  
- index: 画像のインデックス番号（連番）

## 座標変換
元のアノテーション値(-1～1)から画像ピクセル座標への変換:
- angle (-1～1) → X座標 (0～{img_width-1})
- throttle (-1～1) → Y座標 ({img_height-1}～0) ※Y軸は反転

## カタログファイル形式
- Donkeycar形式と同じJSON Lines形式
- 1000件ごとに分割可能
- `cam/image_array` キーには `images/` フォルダからの相対パス

## 読み込み互換性
- Donkeycarの読み込み処理と同じ方法で読み込み可能
- `manifest.json` の第3要素に `"format": "jetracer"` フラグで識別
- インデックスベースのアノテーション辞書として再構築可能

## 統計情報
画像サイズ: {img_width}x{img_height}
エクスポート画像数: {len(catalog_entries)}枚
作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
セッションID: {session_id}
"""
    
    with open(os.path.join(output_folder, "README.txt"), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Jetracerエクスポート完了: {len(catalog_entries)}枚の画像を処理しました")
    print(f"出力フォルダ: {output_folder}")
    print(f"画像フォルダ: {os.path.join(output_folder, 'images')}")
    
    return manifest_path

def draw_arrow_on_image(draw, start_x, start_y, end_x, end_y, color='green', width=2):
    """PIL.ImageDrawで矢印を描画するヘルパー関数"""
    import math
    
    # 矢印の線を描画
    draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)
    
    # ベクトルの角度を計算
    dx = end_x - start_x
    dy = end_y - start_y
    
    # 矢印が短すぎる場合は矢印の先端を描画しない
    vector_length = math.sqrt(dx*dx + dy*dy)
    if vector_length < 5:
        return
        
    angle = math.atan2(dy, dx)
    
    # 矢印の先端のサイズ
    arrow_length = 10
    arrow_angle = math.pi / 6  # 30度
    
    # 矢印の先端の座標を計算
    arrow_x1 = end_x - arrow_length * math.cos(angle - arrow_angle)
    arrow_y1 = end_y - arrow_length * math.sin(angle - arrow_angle)
    arrow_x2 = end_x - arrow_length * math.cos(angle + arrow_angle)
    arrow_y2 = end_y - arrow_length * math.sin(angle + arrow_angle)
    
    # 矢印の先端を線で描画
    draw.line([(end_x, end_y), (arrow_x1, arrow_y1)], fill=color, width=width)
    draw.line([(end_x, end_y), (arrow_x2, arrow_y2)], fill=color, width=width)

def export_to_video(
    annotations: Dict[Union[str, int], Dict[str, Any]], 
    inference_results: Dict[Union[str, int], Dict[str, Any]], 
    output_path: str,
    show_inference: bool = True,
    skip_count: int = 1,
    fps: int = 30,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
    images_list: Optional[List[str]] = None,  # 画像パスのリストを明示的に指定
    diff_vectors: Optional[Dict[Union[str, int], Dict[str, Any]]] = None  # 追加: 差分ベクトルデータ
) -> int:
    """アノテーションを動画として出力する - インデックスベースのアノテーションに対応

    Args:
        annotations: アノテーション辞書（キーがインデックスまたは画像パス）
        inference_results: 推論結果辞書（キーがインデックスまたは画像パス）
        output_path: 出力ファイルパス
        show_inference: 推論結果を表示するかどうか
        skip_count: 何枚ごとに動画に含めるか
        fps: フレームレート
        progress_callback: 進捗コールバック関数
        images_list: 画像パスのリスト（指定しない場合はアノテーションから抽出）

    Returns:
        処理されたフレーム数
    """
    # デバッグ情報の出力
    print(f"アノテーション数: {len(annotations)}")
    print(f"推論結果数: {len(inference_results) if inference_results else 0}")
    print(f"画像リスト: {'あり' if images_list else 'なし'} ({len(images_list) if images_list else 0}枚)")
    
    # 画像パスとアノテーションのインデックスを整理
    indexed_data = []
    
    # 進捗表示
    if progress_callback:
        progress_callback(0, 100, "アノテーションデータを準備中...")
    
    # インデックスとパスの検出方法を改良
    for key, annotation in annotations.items():
        if not annotation:
            continue
        
        # キーの型に基づいて元のインデックスとパスを取得
        original_index = None
        img_path = None
        
        if isinstance(key, int):
            # キーが数値の場合
            original_index = key
            
            # images_listから画像パスを取得
            if images_list and 0 <= original_index < len(images_list):
                img_path = images_list[original_index]
        else:
            # キーが文字列（画像パス）の場合
            img_path = key
            original_index = annotation.get("original_index")
            
            # インデックスがない場合はパスからの抽出を試みる
            if original_index is None:
                try:
                    basename = os.path.basename(img_path)
                    match = re.match(r'^(\d+)_', basename)
                    if match:
                        original_index = int(match.group(1))
                except:
                    pass
        
        # デバッグ出力
        print(f"処理中: キー={key}, インデックス={original_index}, パス={img_path}")
        
        # 有効な画像パスがなく、インデックスがある場合はimages_listから探す
        if not img_path and original_index is not None and images_list:
            if 0 <= original_index < len(images_list):
                img_path = images_list[original_index]
                print(f"  インデックスから画像パスを取得: {img_path}")
        
        # インデックスがなく、パスがある場合はパス自体をそのまま使用
        if original_index is None and img_path:
            print(f"  インデックスが見つからないため、キーをそのまま使用")
            try:
                # パスからインデックスを抽出する最終試行
                basename = os.path.basename(img_path)
                match = re.match(r'^(\d+)_', basename)
                if match:
                    original_index = int(match.group(1))
                else:
                    # 何も見つからなければ仮のインデックスを割り当て
                    original_index = len(indexed_data)
            except:
                original_index = len(indexed_data)
        
        # 有効な画像パスがあるエントリのみ追加
        if img_path and os.path.exists(img_path):
            indexed_data.append({
                "index": original_index,
                "path": img_path,
                "annotation": annotation
            })
            print(f"  データ追加: インデックス={original_index}, パス={os.path.basename(img_path)}")
        else:
            print(f"  画像パスが無効なためスキップ: {img_path}")
    
    print(f"処理対象データ数: {len(indexed_data)}")
    
    # インデックス順にソート
    indexed_data.sort(key=lambda x: x["index"] if x["index"] is not None else float('inf'))
    
    # 進捗表示
    if progress_callback:
        progress_callback(10, 100, "インデックス順にデータをソート中...")
    
    # スキップ設定を適用
    if skip_count > 1:
        indexed_data = indexed_data[::skip_count]
    
    if not indexed_data:
        print("エラー: エクスポート可能なアノテーションデータがありません。")
        return 0
    
    # 進捗表示
    if progress_callback:
        progress_callback(15, 100, "動画出力設定を準備中...")
    
    # 最初の画像からビデオサイズを決定
    first_img_path = indexed_data[0]["path"]
    first_img = cv2.imread(first_img_path)
    
    if first_img is None:
        print(f"エラー: 画像 {first_img_path} を読み込めませんでした。")
        return 0
        
    height, width, channels = first_img.shape
    
    # 動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデック
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_processed = 0
    total_frames = len(indexed_data)
    
    try:
        for i, data in enumerate(indexed_data):
            # 進捗報告 - プログレスバーでは10%～95%の範囲を使用
            progress_percent = 20 + int((i / total_frames) * 75)
            if progress_callback:
                progress_callback(
                    progress_percent, 
                    100, 
                    f"フレーム処理中: {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)"
                )
            
            img_path = data["path"]
            annotation = data["annotation"]
            index = data["index"]
            
            # 画像を読み込む
            cv_img = cv2.imread(img_path)
            
            if cv_img is None:
                print(f"警告: 画像 {img_path} を読み込めませんでした。スキップします。")
                continue
                
            # PIL画像に変換（アノテーションの描画用）
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # アノテーションを描画（赤色）
            if annotation:
                x, y = annotation["x"], annotation["y"]
                # 赤い円を描画
                draw.ellipse((x-15, y-15, x+15, y+15), outline='red', width=3)
                
                # 角度と速度の情報をテキストとして表示
                angle = annotation.get("angle", 0)
                throttle = annotation.get("throttle", 0)
                draw.text((10, 10), f"Angle: {angle:.2f}", fill='red')
                draw.text((10, 30), f"Throttle: {throttle:.2f}", fill='red')
                
                # 位置情報があれば表示
                if 'loc' in annotation:
                    loc = annotation['loc']
                    draw.text((10, 50), f"Loc: {loc}", fill='red')
            
            # 推論結果を描画（青色）
            if show_inference:
                inference = None
                # インデックスまたはパスで推論結果を探す
                if index is not None and index in inference_results:
                    inference = inference_results[index]
                elif img_path in inference_results:
                    inference = inference_results[img_path]
                
                if inference:
                    x, y = inference["x"], inference["y"]
                    # 青い円を描画
                    draw.ellipse((x-15, y-15, x+15, y+15), outline='cyan', width=3)
                    
                    # 推論結果の角度と速度を表示
                    if "pilot/angle" in inference and "pilot/throttle" in inference:
                        p_angle = inference["pilot/angle"]
                        p_throttle = inference["pilot/throttle"]
                    else:
                        p_angle = inference.get("angle", 0)
                        p_throttle = inference.get("throttle", 0)
                    
                    draw.text((width - 150, 10), f"P.Angle: {p_angle:.2f}", fill='cyan')
                    draw.text((width - 150, 30), f"P.Throttle: {p_throttle:.2f}", fill='cyan')
                    
                    # 位置情報があれば表示
                    if "pilot/loc" in inference or "loc" in inference:
                        p_loc = inference.get("pilot/loc", inference.get("loc", 0))
                        draw.text((width - 150, 50), f"P.Loc: {p_loc}", fill='cyan')
            
           # 追加: 差分ベクトル矢印を描画
            if annotation and x is not None and y is not None and diff_vectors:
                # 差分ベクトル情報を取得
                diff_data = None
                if index is not None and index in diff_vectors:
                    diff_data = diff_vectors[index]
                
                if diff_data:
                    # 教師データと推論結果の座標
                    anno_x, anno_y = annotation["x"], annotation["y"]
                    
                    # 矢印を描画（緑色）
                    draw_arrow_on_image(draw, anno_x, anno_y, x, y, color='green', width=2)
                    
                    # 差分ベクトルの情報を表示
                    draw.text((10, 70), f"Diff Mag: {diff_data['vector_magnitude']:.3f}", fill='green')
                    draw.text((10, 90), f"Angle Diff: {diff_data['angle_diff']:+.3f}", fill='green')
                    draw.text((10, 110), f"Throttle Diff: {diff_data['throttle_diff']:+.3f}", fill='green')

            # インデックス情報を表示
            draw.text((width // 2 - 50, height - 30), f"Index: {index}", fill='white')
            
            # PIL画像をOpenCV形式に戻す
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # ビデオに追加
            video.write(cv_img)
            frames_processed += 1
            
            # 進捗表示（コンソール用）
            if frames_processed % 10 == 0 or frames_processed == total_frames:
                print(f"\r動画作成中: {frames_processed}/{total_frames} フレーム処理済み ({frames_processed/total_frames*100:.1f}%)", end="")
            
    except Exception as e:
        print(f"\n動画作成中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()  # スタックトレースを出力
    finally:
        # 改行を出力
        if frames_processed > 0:
            print()
            
        # 完了メッセージを表示
        if progress_callback:
            progress_callback(100, 100, f"完了: {frames_processed}フレームを処理しました。")
            
        # ビデオを閉じる
        video.release()
    
    return frames_processed
    
def export_to_video_multi_source(
    annotations: Dict[Union[str, int], Dict[str, Any]], 
    inference_results: Dict[Union[str, int], Dict[str, Any]], 
    output_path: str,
    source_images_lists: List[List[str]],  # 複数ソースの画像リスト
    source_names: List[str],  # ソース名のリスト
    show_inference: bool = True,
    skip_count: int = 1,
    fps: int = 30,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
    diff_vectors: Optional[Dict[Union[str, int], Dict[str, Any]]] = None  # 追加: 差分ベクトルデータ
) -> int:
    """複数画像ソースを横に並べて動画として出力する

    Args:
        annotations: アノテーション辞書
        inference_results: 推論結果辞書
        output_path: 出力ファイルパス
        source_images_lists: 複数ソースの画像リスト
        source_names: ソース名のリスト
        show_inference: 推論結果を表示するか
        skip_count: 何枚ごとに動画に含めるか
        fps: フレームレート
        progress_callback: 進捗コールバック関数

    Returns:
        処理されたフレーム数
    """
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw
    import re
    
    # 進捗表示
    if progress_callback:
        progress_callback(0, 100, "複数ソース動画の作成準備中...")
    
    # 各ソースの画像数をチェック
    if not source_images_lists or not source_names:
        print("エラー: 画像ソースが指定されていません。")
        return 0
    
    # 各ソースの画像数の最小値を取得
    min_images_count = min(len(images) for images in source_images_lists)
    
    if min_images_count == 0:
        print("エラー: 画像ソースのいずれかに画像がありません。")
        return 0
    
    # スキップを適用したインデックスリストを作成
    if skip_count > 1:
        indices = list(range(0, min_images_count, skip_count))
    else:
        indices = list(range(min_images_count))
    
    if not indices:
        print("エラー: スキップ設定後のフレーム数がゼロになりました。")
        return 0
    
    # 進捗表示
    if progress_callback:
        progress_callback(5, 100, "最初のフレームを処理中...")
    
    # すべてのソースの最初の画像からサイズを取得
    first_frames = []
    for source_images in source_images_lists:
        if indices and indices[0] < len(source_images):
            first_img_path = source_images[indices[0]]
            first_img = cv2.imread(first_img_path)
            if first_img is not None:
                first_frames.append(first_img)
    
    if not first_frames:
        print("エラー: 最初のフレームを読み込めませんでした。")
        return 0
    
    # すべてのフレームを同じサイズにリサイズ
    # 最も小さい高さを使用
    min_height = min(frame.shape[0] for frame in first_frames)
    resized_frames = []
    
    for frame in first_frames:
        # アスペクト比を維持しながらリサイズ
        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_width = int(min_height * aspect_ratio)
        resized = cv2.resize(frame, (new_width, min_height))
        resized_frames.append(resized)
    
    # 横に並べた時の合計幅と高さ
    total_width = sum(frame.shape[1] for frame in resized_frames)
    height = min_height
    
    # 進捗表示
    if progress_callback:
        progress_callback(10, 100, f"動画設定: {len(source_names)}ソース, 幅{total_width}px, 高さ{height}px")
    
    # 動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデック
    video = cv2.VideoWriter(output_path, fourcc, fps, (total_width, height))
    
    frames_processed = 0
    total_frames = len(indices)
    
    try:
        for i, idx in enumerate(indices):
            # 進捗報告 - プログレスバーでは15%～95%の範囲を使用
            progress_percent = 15 + int((i / total_frames) * 80)
            if progress_callback:
                progress_callback(
                    progress_percent, 
                    100, 
                    f"フレーム処理中: {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)"
                )
            
            # 各ソースの画像を取得
            current_frames = []
            for source_idx, source_images in enumerate(source_images_lists):
                if idx < len(source_images):
                    img_path = source_images[idx]
                    cv_img = cv2.imread(img_path)
                    
                    if cv_img is None:
                        print(f"警告: ソース {source_names[source_idx]} の画像 {img_path} を読み込めませんでした。")
                        # 黒い画像を代わりに使用
                        cv_img = np.zeros((min_height, resized_frames[source_idx].shape[1], 3), dtype=np.uint8)
                    else:
                        # PIL画像に変換してアノテーションを描画
                        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_img)
                        
                        # インデックスからパスへのマッピングを試みる
                        original_index = None
                        try:
                            basename = os.path.basename(img_path)
                            match = re.match(r'^(\d+)_', basename)
                            if match:
                                original_index = int(match.group(1))
                        except:
                            pass
                        
                        # アノテーションを描画（赤色）
                        annotation = None
                        if img_path in annotations:
                            annotation = annotations[img_path]
                        elif original_index is not None and original_index in annotations:
                            annotation = annotations[original_index]
                            
                        if annotation:
                            x, y = annotation["x"], annotation["y"]
                            # 赤い円を描画
                            draw.ellipse((x-15, y-15, x+15, y+15), outline='red', width=3)
                            
                            # 角度と速度の情報をテキストとして表示
                            angle = annotation.get("angle", 0)
                            throttle = annotation.get("throttle", 0)
                            draw.text((10, 10), f"Angle: {angle:.2f}", fill='red')
                            draw.text((10, 30), f"Throttle: {throttle:.2f}", fill='red')
                            
                            # 位置情報があれば表示
                            if 'loc' in annotation:
                                loc = annotation['loc']
                                draw.text((10, 50), f"Loc: {loc}", fill='red')
                        
                        # 推論結果を描画（青色）
                        if show_inference:
                            inference = None
                            if img_path in inference_results:
                                inference = inference_results[img_path]
                            elif original_index is not None and original_index in inference_results:
                                inference = inference_results[original_index]
                            
                            if inference:
                                x, y = inference["x"], inference["y"]
                                # 青い円を描画
                                draw.ellipse((x-15, y-15, x+15, y+15), outline='cyan', width=3)
                                
                                # 推論結果の角度と速度を表示
                                if "pilot/angle" in inference and "pilot/throttle" in inference:
                                    p_angle = inference["pilot/angle"]
                                    p_throttle = inference["pilot/throttle"]
                                else:
                                    p_angle = inference.get("angle", 0)
                                    p_throttle = inference.get("throttle", 0)
                                
                                width, height = pil_img.size
                                draw.text((width - 150, 10), f"P.Angle: {p_angle:.2f}", fill='cyan')
                                draw.text((width - 150, 30), f"P.Throttle: {p_throttle:.2f}", fill='cyan')
                                
                                # 位置情報があれば表示
                                if "pilot/loc" in inference or "loc" in inference:
                                    p_loc = inference.get("pilot/loc", inference.get("loc", 0))
                                    draw.text((width - 150, 50), f"P.Loc: {p_loc}", fill='cyan')
                        
                        # ソース名を表示
                        draw.text((10, height - 30), f"Source: {source_names[source_idx]}", fill='white')
                        
                        # OpenCV形式に戻す
                        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    
                    # アスペクト比を維持しながらリサイズ
                    aspect_ratio = cv_img.shape[1] / cv_img.shape[0]
                    new_width = int(min_height * aspect_ratio)
                    resized = cv2.resize(cv_img, (new_width, min_height))
                    current_frames.append(resized)
                else:
                    # インデックスが範囲外の場合、黒い画像を使用
                    black_img = np.zeros((min_height, resized_frames[source_idx].shape[1], 3), dtype=np.uint8)
                    current_frames.append(black_img)
            
            # フレームを横に結合
            if current_frames:
                combined_frame = np.hstack(current_frames)
                video.write(combined_frame)
                frames_processed += 1
            
            # 進捗表示（コンソール用）
            if frames_processed % 10 == 0 or frames_processed == total_frames:
                print(f"\r複数ソース動画作成中: {frames_processed}/{total_frames} フレーム処理済み ({frames_processed/total_frames*100:.1f}%)", end="")
            
    except Exception as e:
        print(f"\n複数ソース動画作成中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 改行を出力
        if frames_processed > 0:
            print()
            
        # 完了メッセージを表示
        if progress_callback:
            progress_callback(100, 100, f"完了: {frames_processed}フレームを処理しました。")
            
        # ビデオを閉じる
        video.release()
    
    return frames_processed

def export_segmentation_to_yolo(output_folder, segmentation_annotations, class_names=None, images_list=None):
    """セグメンテーションアノテーションをYOLO形式でエクスポート
    Args:
        output_folder: 出力フォルダ
        segmentation_annotations: アノテーションデータ (パスベースまたはインデックスベース)
        class_names: クラス名のリスト
        images_list: インデックスベースの場合の画像パスリスト
    """
    if class_names is None:
        # 全クラスを収集
        all_classes = set()
        for annotations in segmentation_annotations.values():
            for annotation in annotations:
                all_classes.add(annotation.get('class', 'unknown'))
        class_names = sorted(list(all_classes))
    
    # フォルダ構造を作成
    images_dir = os.path.join(output_folder, 'images')
    labels_dir = os.path.join(output_folder, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 各画像のアノテーションを処理
    for key, annotations in segmentation_annotations.items():
        # インデックスベースかパスベースか判定
        if images_list is not None and isinstance(key, int):
            # インデックスベース: インデックスからパスを取得
            img_index = key
            if img_index >= len(images_list):
                continue
            img_path = images_list[img_index]
        else:
            # パスベース: キーがパス
            img_path = key
            
        # 画像をコピー
        img_filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(images_dir, img_filename))
        
        # ラベルファイルを作成
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        # 画像サイズを取得
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        with open(label_path, 'w') as f:
            for annotation in annotations:
                class_name = annotation.get('class', 'unknown')
                points = annotation.get('points', [])
                
                if class_name in class_names and len(points) >= 3:
                    class_id = class_names.index(class_name)
                    
                    # ポリゴンの座標を正規化
                    normalized_points = []
                    for x, y in points:
                        norm_x = x / img_width
                        norm_y = y / img_height
                        normalized_points.extend([norm_x, norm_y])
                    
                    # YOLO形式: class_id x1 y1 x2 y2 x3 y3 ...
                    line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_points)
                    f.write(line + '\n')
    
    # classes.txtを作成
    classes_path = os.path.join(output_folder, 'classes.txt')
    with open(classes_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # dataset.yamlを作成
    yaml_content = f"""path: {output_folder}
train: images
val: images
test: images

nc: {len(class_names)}
names: {class_names}
"""
    
    yaml_path = os.path.join(output_folder, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path