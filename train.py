#!/usr/bin/env python3
"""
train.py - マルチ画像入力モデルの学習スクリプト
"""
import os
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# 既存のコード
from model_catalog import MODEL_REGISTRY, get_model
from model_training import train_model

# マルチ画像モデル拡張
from multi_image_models import (
    register_multi_image_models, 
    create_multi_image_datasets,
    run_model_with_multi_input_support
)

def load_catalog_data(catalog_file):
    """カタログファイルからデータをロード"""
    image_paths = []
    annotations = []
    
    # カタログディレクトリを取得
    catalog_dir = os.path.dirname(catalog_file)
    # 画像ディレクトリはカタログファイルと同じ階層のimagesフォルダ
    image_dir = os.path.join(os.path.dirname(catalog_dir), "images")
    
    # imagesフォルダが存在しない場合は、カタログと同じ階層を探す
    if not os.path.exists(image_dir):
        image_dir = os.path.join(catalog_dir, "images")
    
    # それでも見つからない場合は、カタログと同じディレクトリを使用
    if not os.path.exists(image_dir):
        image_dir = catalog_dir
        
    print(f"画像ディレクトリ: {image_dir}")
    
    with open(catalog_file, 'r') as f:
        for line in f:
            try:
                # JSONとして解析
                entry = json.loads(line)
                
                # 必要なフィールドが存在するか確認
                if "cam/image_array" in entry and "user/angle" in entry and "user/throttle" in entry:
                    # 画像パスを構築
                    image_path = os.path.join(image_dir, entry["cam/image_array"])
                    
                    # ファイルが存在するか確認
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        
                        # アノテーションを追加
                        annotations.append({
                            "angle": entry["user/angle"],
                            "throttle": entry["user/throttle"]
                        })
                    else:
                        print(f"警告: 画像ファイルが存在しません: {image_path}")
                        
            except json.JSONDecodeError:
                print(f"警告: 無効なJSON行をスキップします: {line[:50]}...")
    
    print(f"カタログからロードしたデータ: {len(image_paths)}件のサンプル")
    return image_paths, annotations

def load_multi_camera_catalog(catalog_file, num_inputs=3, use_lidar=False):
    """複数カメラ+LiDARデータをカタログからロード"""
    image_paths = []
    annotations = []
    
    # カタログディレクトリを取得
    catalog_dir = os.path.dirname(catalog_file)
    # 画像ディレクトリはカタログファイルと同じ階層のimagesフォルダ
    image_dir = os.path.join(os.path.dirname(catalog_dir), "images")
    
    # imagesフォルダが存在しない場合は、カタログと同じ階層を探す
    if not os.path.exists(image_dir):
        image_dir = os.path.join(catalog_dir, "images")
    
    # それでも見つからない場合は、カタログと同じディレクトリを使用
    if not os.path.exists(image_dir):
        image_dir = catalog_dir
        
    print(f"画像ディレクトリ: {image_dir}")
    
    # 使用するカメラの順序（最後が現在のフレーム）
    camera_keys = []
    
    # LiDARを使用する場合
    if use_lidar:
        if num_inputs == 3:
            camera_keys = ["cam/image_array", "lidar/image_array", "cam1/image_array"]
        elif num_inputs == 2:
            camera_keys = ["cam/image_array", "lidar/image_array"]
        else:
            # LiDARに対応していない場合は標準のカメラのみ使用
            camera_keys = ["cam/image_array", "cam1/image_array"]
    else:
        if num_inputs == 3:
            camera_keys = ["cam/image_array", "cam/image_array", "cam1/image_array"]
        elif num_inputs == 2:
            camera_keys = ["cam/image_array", "cam1/image_array"]
        else:
            camera_keys = ["cam/image_array"]
    
    print(f"使用するカメラキー: {camera_keys}")

    # 各エントリで、必要なすべてのカメラ画像が存在することを確認
    entries = []
    with open(catalog_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                
                # すべてのキーが存在するか確認
                all_keys_exist = all(key in entry for key in camera_keys)
                if all_keys_exist and "user/angle" in entry and "user/throttle" in entry:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    
    # 各エントリの画像パスとアノテーションを収集
    for entry in entries:
        # 各カメラの画像パスを取得
        entry_images = []
        for key in camera_keys:
            image_path = os.path.join(image_dir, entry[key])
            if os.path.exists(image_path):
                entry_images.append(image_path)
            else:
                print(f"警告: 画像ファイルが存在しません: {image_path}")
                break
        
        # すべての画像が存在する場合のみ追加
        if len(entry_images) == len(camera_keys):
            # 複数画像の場合はリストとして保存（最新のフレームが最後）
            if num_inputs > 1:
                image_paths.append(entry_images)
            else:
                image_paths.append(entry_images[0])
            
            # アノテーションを追加
            annotations.append({
                "angle": entry["user/angle"],
                "throttle": entry["user/throttle"]
            })
    
    print(f"マルチカメラデータ: {len(image_paths)}件のサンプル ({num_inputs}枚の画像入力)")
    return image_paths, annotations

def simple_progress_callback(current, total, message):
    """シンプルな進捗コールバック"""
    print(f"[{current}/{total}] {message}")
    return True

def main():
    parser = argparse.ArgumentParser(description='マルチ画像入力モデルの学習')
    parser.add_argument('--catalog_file', type=str, required=True, 
                        help='カタログファイルのパス')
    parser.add_argument('--model', type=str, default='donkey_multi3_concat',
                        help='モデルタイプ (例: donkey_multi3_concat, donkey_multi3_channel, donkey_multi3_attention)')
    parser.add_argument('--num_inputs', type=int, default=None,
                        help='入力画像数 (Noneの場合は自動検出)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='学習エポック数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='バッチサイズ')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学習率')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='モデル保存ディレクトリ')
    parser.add_argument('--use_lidar', action='store_true',
                        help='LiDAR画像を使用する (3入力の場合: カメラ1, LiDAR, カメラ2)')
    parser.add_argument('--use_multi_camera', action='store_true',
                        help='複数カメラの画像を使用する (マルチ画像入力モデル用)')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='データ拡張を使用する')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='検証データの割合')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Early Stoppingを使用する')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early Stoppingの忍耐値')
    
    args = parser.parse_args()
    
    # マルチ画像モデルをレジストリに登録
    register_multi_image_models()
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データのロード
    if args.use_multi_camera:
        image_paths, annotations = load_multi_camera_catalog(
            args.catalog_file, 
            num_inputs=args.num_inputs or 3,
            use_lidar=args.use_lidar
        )
    else:
        image_paths, annotations = load_catalog_data(args.catalog_file)
    
    if len(image_paths) == 0:
        print("エラー: 有効なデータが見つかりませんでした。")
        return
    
    # データセットとデータローダーの作成
    train_loader, val_loader, dataset_info = create_multi_image_datasets(
        image_paths=image_paths,
        annotations=annotations,
        model_name=args.model,
        num_inputs=args.num_inputs,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=4,
        use_augmentation=args.use_augmentation
    )
    
    print(f"データセット情報:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")

    #m
    from multi_image_models import train_multi_image_model

    training_results = train_multi_image_model(
        model_name=args.model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        save_dir=args.save_dir,
        device=device,
        progress_callback=simple_progress_callback,
        pretrained=True,
        use_early_stopping=args.early_stopping,
        patience=args.patience
    )


    # トレーニング結果の表示
    print("\nトレーニング結果:")
    print(f"モデル名: {training_results['model_name']}")
    print(f"完了エポック数: {training_results['completed_epochs']}/{training_results['num_epochs']}")
    print(f"最終トレーニング損失: {training_results['train_losses'][-1]:.6f}")
    print(f"最終検証損失: {training_results['val_losses'][-1]:.6f}")
    print(f"最良検証損失: {training_results['best_val_loss']:.6f}")
    print(f"最良モデルのパス: {training_results['best_model_path']}")
    
    if training_results['early_stopped']:
        print(f"Early Stoppingによりエポック {training_results['stopped_epoch']} で学習終了")
    
    # モデルの保存パスを表示
    print(f"\nモデルファイル:")
    print(f"最終モデル: {training_results['model_path']}")
    print(f"最良モデル: {training_results['best_model_path']}")
    
    # 学習完了のメッセージ
    print(f"\n{args.model} モデルの学習が完了しました！")
    print(f"評価するには次のコマンドを実行してください:")
    print(f"python evaluate.py --catalog_file {args.catalog_file} --model {args.model} --model_path {training_results['best_model_path']} --num_inputs {dataset_info['num_inputs']}")

if __name__ == "__main__":
    main()