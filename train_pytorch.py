#!/usr/bin/env python3
"""
Script for training Donkeycar models using PyTorch Lightning

Usage:
    train_pytorch.py --tub=<tub_path> [--tub=<tub_path>] --model=<model_path> [--batch=<batch_size>]
                    [--epochs=<number_of_epochs>] [--transfer=<transfer_model_path>]

Options:
    -h --help                       Show this help screen
    --tub=<tub_path>                Path to tub directory containing training data
    --model=<model_path>            Path where trained model will be saved
    --batch=<batch_size>            Batch size [default: 64]
    --epochs=<number_of_epochs>     Number of epochs [default: 20]
    --transfer=<transfer_model_path> Path to base model for transfer learning
"""

import os
import sys
import glob
import json
import torch
import numpy as np
from PIL import Image
from docopt import docopt
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Donkeycarのパスを追加
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import donkeycar as dk
from donkeycar.config import load_config

# ResNet18モデルをインポート
from resnet import ResNet18

def get_image_transform():
    """画像変換パイプラインを定義"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

class TubDataset(Dataset):
    """Donkeycarのデータを読み込むデータセット（Donkeycar v5形式対応）"""
    def __init__(self, tub_paths, transform=None):
        self.records = []
        self.transform = transform or get_image_transform()
        self.tub_paths = tub_paths
        self.data = []
        
        # すべてのtubからレコードを読み込む
        for tub_path in tub_paths:
            print(f"Tubパスの探索: {tub_path}")
            
            # Donkeycar v5形式のcatalogファイルを検索
            catalog_files = sorted(glob.glob(os.path.join(tub_path, 'catalog_*.catalog')))
            
            if catalog_files:
                print(f"Donkeycar v5形式のカタログファイルを検出: {len(catalog_files)}個")
                
                # マニフェストファイルからレコード情報を読み込む
                manifest_path = os.path.join(tub_path, 'manifest.json')
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                            print(f"マニフェスト情報: {manifest.keys()}")
                    except Exception as e:
                        print(f"マニフェスト読み込みエラー: {e}")
                
                # 各カタログファイルを読み込む
                for catalog_file in catalog_files:
                    try:
                        print(f"カタログファイル読み込み中: {catalog_file}")
                        with open(catalog_file, 'r') as f:
                            catalog_content = f.read()
                            try:
                                catalog_data = json.loads(catalog_content)
                                # カタログ構造を確認
                                if isinstance(catalog_data, list):
                                    print(f"カタログデータは配列形式: {len(catalog_data)}項目")
                                    for item in catalog_data:
                                        if isinstance(item, dict) and 'cam/image_array' in item and 'user/angle' in item and 'user/throttle' in item:
                                            self.data.append(item)
                                elif isinstance(catalog_data, dict):
                                    print(f"カタログデータは辞書形式: {len(catalog_data.keys())}項目")
                                    # 辞書構造を調査
                                    sample_keys = list(catalog_data.keys())[:5]
                                    print(f"サンプルキー: {sample_keys}")
                                    
                                    # レコードデータの特定
                                    for key, value in catalog_data.items():
                                        if isinstance(value, dict) and 'cam/image_array' in value and 'user/angle' in value and 'user/throttle' in value:
                                            self.data.append(value)
                            except json.JSONDecodeError:
                                # JSON形式でない場合、各行をJSONとして解析
                                print("JSONとして解析できないため、行ごとに処理します")
                                for line in catalog_content.split('\n'):
                                    if line.strip():
                                        try:
                                            item = json.loads(line)
                                            if isinstance(item, dict) and 'cam/image_array' in item and 'user/angle' in item and 'user/throttle' in item:
                                                self.data.append(item)
                                        except:
                                            pass
                    except Exception as e:
                        print(f"カタログファイル読み込みエラー: {e}")
            
            else:
                # Donkeycar v3/v4形式を試す
                record_paths = glob.glob(os.path.join(tub_path, 'record_*.json'))
                if record_paths:
                    print(f"従来のレコードファイルを検出: {len(record_paths)}個")
                    for record_path in sorted(record_paths):
                        try:
                            with open(record_path, 'r') as f:
                                record = json.load(f)
                                if 'cam/image_array' in record and 'user/angle' in record and 'user/throttle' in record:
                                    self.data.append(record)
                        except Exception as e:
                            print(f"レコードファイル読み込みエラー: {e}")
                else:
                    # ディレクトリ内のすべてのファイルを一覧表示して確認
                    print(f"ディレクトリ内容を確認します: {tub_path}")
                    files = os.listdir(tub_path)
                    print(f"ファイル一覧: {files[:10]}..." if len(files) > 10 else f"ファイル一覧: {files}")
        
        print(f"合計 {len(self.data)} レコードを読み込みました")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            record = self.data[idx]
            
            # 画像の読み込み
            img_path = os.path.join(self.tub_paths[0], 'images', record['cam/image_array'])
            if not os.path.exists(img_path):
                print(f"警告: 画像ファイルが見つかりません: {img_path}")
                # インデックスを1つずらして再試行
                if idx + 1 < len(self.data):
                    return self.__getitem__(idx + 1)
                else:
                    return self.__getitem__(0)
            
            img = Image.open(img_path)
            
            # 画像の変換
            if self.transform:
                img = self.transform(img)
            
            # ターゲットデータの取得（steering, throttle）
            angle = float(record['user/angle'])
            throttle = float(record['user/throttle'])
            
            # [-1, 1]から[0, 1]へのスケーリング（必要に応じて）
            angle = (angle + 1) / 2
            throttle = (throttle + 1) / 2
            
            target = torch.tensor([angle, throttle], dtype=torch.float)
            
            return img, target
        
        except Exception as e:
            print(f"レコード {idx} の読み込み中にエラーが発生しました: {e}")
            # インデックスを1つずらして再試行
            if idx + 1 < len(self.data):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)

def train_model(tub_paths, model_path, batch_size=64, epochs=20, transfer_model=None):
    """PyTorch Lightningを使用してモデルをトレーニング"""
    
    # データセットの作成
    print(f"データディレクトリの検索: {tub_paths}")
    
    # 絶対パスに変換
    abs_tub_paths = [os.path.abspath(path) for path in tub_paths]
    print(f"絶対パス: {abs_tub_paths}")
    
    dataset = TubDataset(abs_tub_paths)
    
    # データセットが空の場合はエラー
    if len(dataset) == 0:
        print("エラー: データセットが空です。tubディレクトリの内容を確認してください。")
        sys.exit(1)
    
    # トレーニング/検証データの分割（80/20）
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"トレーニングデータ: {train_size}レコード")
    print(f"検証データ: {val_size}レコード")
    
    # データローダーの設定
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # マルチプロセス無効化
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # マルチプロセス無効化
        persistent_workers=False
    )
    
    # モデルのインスタンス化
    model = ResNet18(
        input_shape=(batch_size, 3, 224, 224),
        output_size=(2,)  # 角度とスロットルの2出力
    )
    
    # 転移学習の設定
    if transfer_model and os.path.exists(transfer_model):
        print(f"転移学習のためのモデルを読み込み中: {transfer_model}")
        checkpoint = torch.load(transfer_model)
        model.load_state_dict(checkpoint)
    
    # モデル保存ディレクトリの確認と作成
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"モデル保存ディレクトリを作成しました: {model_dir}")
    
    # チェックポイントコールバックの設定
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir if model_dir else '.',
        filename=os.path.basename(model_path).split('.')[0],
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # トレーナーの設定
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=os.path.dirname(model_path),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision=16 if torch.cuda.is_available() else 32,  # 半精度学習によるメモリ削減
        gradient_clip_val=0.5,  # 勾配クリッピングでメモリ使用量を抑制
        accumulate_grad_batches=2  # 勾配蓄積によるバッチサイズの実質的な削減
    )
    
    print(f"{'GPUを使用' if torch.cuda.is_available() else 'CPUを使用'}してトレーニングを開始します")
    
    # トレーニングの実行
    trainer.fit(model, train_loader, val_loader)
    
    # 最終モデルの保存
    torch.save(model.state_dict(), model_path)
    
    print(f"モデルを保存しました: {model_path}")
    return model

if __name__ == "__main__":
    args = docopt(__doc__)
    
    # 引数の解析
    tub_paths = args['--tub']
    if not isinstance(tub_paths, list):
        tub_paths = [tub_paths]
    
    model_path = args['--model']
    batch_size = int(args['--batch']) if args['--batch'] else 64
    epochs = int(args['--epochs']) if args['--epochs'] else 20
    transfer_model = args['--transfer']
    
    print("""
推奨:
- メモリエラーが発生する場合はバッチサイズを小さくしてください (例: --batch=16)
- データ読み込みエラーが発生する場合は --batch=8 --epochs=10 などで試してください
""")
    print(f"モデル保存先: {model_path}")
    print(f"バッチサイズ: {batch_size}")
    print(f"エポック数: {epochs}")
    if transfer_model:
        print(f"転移学習元: {transfer_model}")
    
    # GPUの確認
    if torch.cuda.is_available():
        print(f"GPU検出: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU未検出: CPUを使用します")
    
    # 設定の読み込み
    try:
        cfg = load_config()
    except Exception as e:
        print(f"設定読み込みエラー: {e}")
        print("デフォルト設定を使用します")
        cfg = type('Config', (), {})
        cfg.IMAGE_H = 120
        cfg.IMAGE_W = 160
        cfg.IMAGE_DEPTH = 3
        cfg.TRAIN_TEST_SPLIT = 0.8
        cfg.BATCH_SIZE = batch_size
        
    # カタログファイルの内容を確認
    tub_path = tub_paths[0]
    sample_catalog = glob.glob(os.path.join(tub_path, 'catalog_*.catalog'))
    if sample_catalog:
        print(f"サンプルカタログファイルの検査: {sample_catalog[0]}")
        try:
            with open(sample_catalog[0], 'r') as f:
                catalog_data = json.load(f)
                if isinstance(catalog_data, dict):
                    print(f"カタログデータ形式: 辞書（キー数: {len(catalog_data.keys())}）")
                    sample_keys = list(catalog_data.keys())[:5]
                    print(f"サンプルキー: {sample_keys}")
                    if sample_keys:
                        sample_entry = catalog_data[sample_keys[0]]
                        print(f"サンプルエントリ: {sample_entry.keys() if isinstance(sample_entry, dict) else type(sample_entry)}")
                elif isinstance(catalog_data, list):
                    print(f"カタログデータ形式: リスト（項目数: {len(catalog_data)}）")
                    if catalog_data:
                        sample_entry = catalog_data[0]
                        print(f"サンプルエントリ: {sample_entry.keys() if isinstance(sample_entry, dict) else type(sample_entry)}")
        except Exception as e:
            print(f"カタログファイル検査エラー: {e}")
    
    # トレーニングの実行
    try:
        model = train_model(
            tub_paths, 
            model_path, 
            batch_size=batch_size, 
            epochs=epochs, 
            transfer_model=transfer_model
        )
    except Exception as e:
        print(f"トレーニングエラー: {e}")
        import traceback
        traceback.print_exc()