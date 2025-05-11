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

#TODO:リファクタリング
def export_to_donkey(
    folder_path: str, 
    annotations: Dict[Union[str, int], Dict[str, Any]], 
    inference_results: Optional[Dict[Union[str, int], Dict[str, Any]]] = None,
    deleted_indexes: Optional[List[int]] = None,
    images_list: Optional[List[str]] = None,  # 互換性のために残す
    image_map: Optional[Dict[int, Dict[str, str]]] = None,  # 新しいパラメータ：{index: {variant: image_path, ...}, ...}
    variant_keys: Optional[Dict[str, str]] = None  # 新しいパラメータ：{variant: key_name, ...}
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
        
        print(annotation)
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
    output_subfolder: str = "data_yolo",
    split_ratio: float = 0.8,  # Train/val split ratio (80% for training, 20% for validation)
    include_empty_images: bool = True,  # バウンディングボックスがない画像も含めるかどうか
    all_images_list: Optional[List[str]] = None,  # バウンディングボックスがない画像も含めるための全画像リスト
    deleted_indexes: Optional[List[int]] = None,  # 削除済みのインデックスリスト
    index_to_path_map: Optional[Dict[int, str]] = None  # インデックスからパスへのマッピング
) -> str:
    """バウンディングボックスアノテーションをYOLO形式でエクスポートする - Ultralytics HUB互換

    Args:
        folder_path: 出力先のフォルダパス
        bbox_annotations: バウンディングボックスアノテーション辞書
        output_subfolder: 出力先のサブフォルダ名（デフォルト: "data_yolo"）
        split_ratio: 訓練/検証データの分割比率（デフォルト: 0.8）
        include_empty_images: バウンディングボックスがない画像も含めるかどうか（デフォルト: True）
        all_images_list: 全画像リスト（バウンディングボックスがない画像も含める場合に使用）
        deleted_indexes: 削除済みのインデックスリスト（削除された画像をスキップするために使用）
        index_to_path_map: インデックスからパスへのマッピング（削除された画像をスキップするために使用）

    Returns:
        作成されたデータセット設定ファイル（dataset.yaml）のパス
    """
    import os
    import shutil
    import json
    import random
    import re
    
    # YOLO形式のアノテーション用フォルダを作成
    yolo_folder = os.path.join(folder_path, output_subfolder)
    
    # 画像とラベルのフォルダを作成 - Ultralytics HUB の構造に合わせる
    images_train_folder = os.path.join(yolo_folder, "images", "train")
    images_val_folder = os.path.join(yolo_folder, "images", "val")
    labels_train_folder = os.path.join(yolo_folder, "labels", "train")
    labels_val_folder = os.path.join(yolo_folder, "labels", "val")
    
    os.makedirs(images_train_folder, exist_ok=True)
    os.makedirs(images_val_folder, exist_ok=True)
    os.makedirs(labels_train_folder, exist_ok=True)
    os.makedirs(labels_val_folder, exist_ok=True)
    
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
    
    # 削除済みのインデックスがなければ空リストを使用
    if deleted_indexes is None:
        deleted_indexes = []
    
    # パスからインデックスを抽出する関数（パスからのインデックス抽出に使用）
    def extract_index_from_path(path):
        try:
            # ファイル名を取得
            basename = os.path.basename(path)
            # インデックスを抽出（例: 10900_cam_image_array_.jpg から 10900 を抽出）
            match = re.match(r'^(\d+)_', basename)
            if match:
                return int(match.group(1))
        except:
            pass
        return None
    
    # 全画像パスのリストを取得
    img_paths = []
    
    if include_empty_images and all_images_list:
        # バウンディングボックスがない画像も含める場合、all_images_listを使用
        img_paths = all_images_list
    else:
        # バウンディングボックスがある画像のみを使用
        img_paths = list(bbox_annotations.keys())
    
    # 削除された画像を除外
    filtered_img_paths = []
    for path in img_paths:
        # パスからインデックスを抽出
        index = None
        
        # 1. インデックスからパスへのマッピングが提供されている場合
        if index_to_path_map is not None:
            # reverse_mapを作成（パスからインデックスへのマッピング）
            reverse_map = {v: k for k, v in index_to_path_map.items()}
            if path in reverse_map:
                index = reverse_map[path]
        
        # 2. 上記で見つからない場合はパスからインデックスを抽出
        if index is None:
            index = extract_index_from_path(path)
        
        # インデックスが見つかり、削除されていない場合のみ追加
        if index is not None and index not in deleted_indexes:
            filtered_img_paths.append(path)
        # インデックスが見つからない場合はとりあえず追加
        elif index is None:
            filtered_img_paths.append(path)
    
    # 画像を重複なくリストに保持（セットを使用）
    unique_img_paths = list(set(filtered_img_paths))
    
    # ランダムにシャッフル（訓練/検証に分割するため）
    random.shuffle(unique_img_paths)
    
    # 訓練/検証データに分割
    split_idx = int(len(unique_img_paths) * split_ratio)
    train_paths = unique_img_paths[:split_idx]
    val_paths = unique_img_paths[split_idx:]
    
    # 各画像のアノテーションを処理
    processed_train_count = 0
    processed_val_count = 0
    total_bboxes = 0
    
    # 統計情報用の変数
    train_with_boxes = 0
    train_no_boxes = 0
    val_with_boxes = 0
    val_no_boxes = 0
    skipped_deleted = 0
    
    # 訓練データの処理
    for img_path in train_paths:
        # バウンディングボックスの取得
        bboxes = bbox_annotations.get(img_path, [])
        
        # 画像ファイル名を取得
        img_filename = os.path.basename(img_path)
        img_basename = os.path.splitext(img_filename)[0]
        
        try:
            # 画像をコピー
            dst_img_path = os.path.join(images_train_folder, img_filename)
            shutil.copy2(img_path, dst_img_path)
            
            # YOLOフォーマットのラベルファイルを作成（バウンディングボックスがなくても空ファイルを作成）
            label_path = os.path.join(labels_train_folder, f"{img_basename}.txt")
            
            if bboxes:
                # バウンディングボックスがある場合
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
                
                train_with_boxes += 1
            else:
                # バウンディングボックスがない場合、空のファイルを作成
                with open(label_path, 'w') as f:
                    pass  # 空ファイルを作成
                train_no_boxes += 1
            
            processed_train_count += 1
            
        except Exception as e:
            print(f"警告: 訓練画像 {img_path} の処理中にエラーが発生しました: {e}")
            continue
    
    # 検証データの処理
    for img_path in val_paths:
        # バウンディングボックスの取得
        bboxes = bbox_annotations.get(img_path, [])
        
        # 画像ファイル名を取得
        img_filename = os.path.basename(img_path)
        img_basename = os.path.splitext(img_filename)[0]
        
        try:
            # 画像をコピー
            dst_img_path = os.path.join(images_val_folder, img_filename)
            shutil.copy2(img_path, dst_img_path)
            
            # YOLOフォーマットのラベルファイルを作成（バウンディングボックスがなくても空ファイルを作成）
            label_path = os.path.join(labels_val_folder, f"{img_basename}.txt")
            
            if bboxes:
                # バウンディングボックスがある場合
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
                
                val_with_boxes += 1
            else:
                # バウンディングボックスがない場合、空のファイルを作成
                with open(label_path, 'w') as f:
                    pass  # 空ファイルを作成
                val_no_boxes += 1
            
            processed_val_count += 1
            
        except Exception as e:
            print(f"警告: 検証画像 {img_path} の処理中にエラーが発生しました: {e}")
            continue
    
    # スキップされた削除済みの画像数を計算
    if include_empty_images and all_images_list:
        skipped_deleted = len(all_images_list) - len(filtered_img_paths)
    
    # Ultralytics HUB 形式のYAMLファイルを作成
    # クラス名の辞書形式に変換
    class_dict = {i: class_name for i, class_name in enumerate(class_list)}
    
    yaml_content = f"""# Ultralytics YOLOv8, AGPL-3.0 license
# {output_subfolder} dataset by Custom Annotation Tool
# Example usage: yolo train data={output_subfolder}.yaml

# Train/val/test sets
path:  # dataset root dir (leave empty for HUB)
train: images/train  # train images (relative to 'path') {processed_train_count} images
val: images/val  # val images (relative to 'path') {processed_val_count} images
test:  # test images (optional)

# Classes
names: 
"""
    
    # クラス名を辞書形式で追加
    for idx, name in class_dict.items():
        yaml_content += f"  {idx}: {name}\n"
    
    # 空のダウンロードセクションを追加
    yaml_content += "\n# Download script/URL (optional)\ndownload:\n"
    
    yaml_file_path = os.path.join(yolo_folder, f"{output_subfolder}.yaml")
    with open(yaml_file_path, 'w') as f:
        f.write(yaml_content)
    
    # README.txtファイルを作成して使用方法を説明
    readme_content = f"""# YOLO形式アノテーションデータ（Ultralytics HUB 互換）

このフォルダには、YOLO形式でエクスポートされたアノテーションデータが含まれています。
Ultralytics HUB と互換性があります。

## フォルダ構成
- images/train/: 訓練用画像 ({processed_train_count}枚)
- images/val/: 検証用画像 ({processed_val_count}枚)
- labels/train/: 訓練用アノテーションファイル
- labels/val/: 検証用アノテーションファイル
- classes.txt: クラス名のリスト
- {output_subfolder}.yaml: YOLOv8用のデータセット設定ファイル

## クラス情報
検出クラス数: {len(class_list)}
クラス: {', '.join(class_list)}

## 統計情報
訓練画像数: {processed_train_count}枚 (バウンディングボックスあり: {train_with_boxes}枚, なし: {train_no_boxes}枚)
検証画像数: {processed_val_count}枚 (バウンディングボックスあり: {val_with_boxes}枚, なし: {val_no_boxes}枚)
合計バウンディングボックス数: {total_bboxes}個
スキップされた削除済み画像数: {skipped_deleted}枚

## 使用方法
1. このフォルダをYOLOv8のプロジェクトフォルダに配置します。
2. 次のコマンドで学習を開始できます:
   yolo train data={output_subfolder}.yaml

## YOLOフォーマット
各行の形式: <class_id> <x_center> <y_center> <width> <height>
※すべての座標値は画像サイズで正規化されています（0～1の範囲）
※バウンディングボックスがない画像のラベルファイルは空ファイルとして作成されています
"""
    
    with open(os.path.join(yolo_folder, "README.txt"), 'w') as f:
        f.write(readme_content)
    
    return yaml_file_path

def export_to_video(
    annotations: Dict[Union[str, int], Dict[str, Any]], 
    inference_results: Dict[Union[str, int], Dict[str, Any]], 
    output_path: str,
    show_inference: bool = True,
    skip_count: int = 1,
    fps: int = 30,
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None,
    images_list: Optional[List[str]] = None  # 画像パスのリストを明示的に指定
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
                    draw.ellipse((x-15, y-15, x+15, y+15), outline='blue', width=3)
                    
                    # 推論結果の角度と速度を表示
                    if "pilot/angle" in inference and "pilot/throttle" in inference:
                        p_angle = inference["pilot/angle"]
                        p_throttle = inference["pilot/throttle"]
                    else:
                        p_angle = inference.get("angle", 0)
                        p_throttle = inference.get("throttle", 0)
                    
                    draw.text((width - 150, 10), f"P.Angle: {p_angle:.2f}", fill='blue')
                    draw.text((width - 150, 30), f"P.Throttle: {p_throttle:.2f}", fill='blue')
                    
                    # 位置情報があれば表示
                    if "pilot/loc" in inference or "loc" in inference:
                        p_loc = inference.get("pilot/loc", inference.get("loc", 0))
                        draw.text((width - 150, 50), f"P.Loc: {p_loc}", fill='blue')
            
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
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None
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
                                draw.ellipse((x-15, y-15, x+15, y+15), outline='blue', width=3)
                                
                                # 推論結果の角度と速度を表示
                                if "pilot/angle" in inference and "pilot/throttle" in inference:
                                    p_angle = inference["pilot/angle"]
                                    p_throttle = inference["pilot/throttle"]
                                else:
                                    p_angle = inference.get("angle", 0)
                                    p_throttle = inference.get("throttle", 0)
                                
                                width, height = pil_img.size
                                draw.text((width - 150, 10), f"P.Angle: {p_angle:.2f}", fill='blue')
                                draw.text((width - 150, 30), f"P.Throttle: {p_throttle:.2f}", fill='blue')
                                
                                # 位置情報があれば表示
                                if "pilot/loc" in inference or "loc" in inference:
                                    p_loc = inference.get("pilot/loc", inference.get("loc", 0))
                                    draw.text((width - 150, 50), f"P.Loc: {p_loc}", fill='blue')
                        
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