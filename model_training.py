"""
モデルトレーニングユーティリティ - TIMMベースのモデルのトレーニングと評価
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime

from model_catalog import get_model, AnnotationDataset

import random
from PIL import Image, ImageOps, ImageEnhance

def create_augmentation_transform(
    use_flip=True,
    flip_prob=0.5,
    use_color=True,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    use_geometry=True,
    rotation_degrees=5,
    translate_ratio=0.1,
    use_erase=True,
    erase_prob=0.5,
    erase_min_ratio=0.02,
    erase_max_ratio=0.2,
    base_transform=None
) -> transforms.Compose:
    """詳細設定可能なデータオーグメンテーション変換を作成する

    Args:
        use_flip: 水平反転を使用するかどうか
        flip_prob: 水平反転の確率
        use_color: 色調整を使用するかどうか
        brightness: 明るさの調整範囲
        contrast: コントラストの調整範囲
        saturation: 彩度の調整範囲
        use_geometry: 幾何変換を使用するかどうか
        rotation_degrees: 回転角度の範囲
        translate_ratio: 平行移動の比率
        use_erase: ランダムイレースを使用するかどうか
        erase_prob: イレースの確率
        erase_min_ratio: イレースの最小比率
        erase_max_ratio: イレースの最大比率
        base_transform: ベース変換（モデルの前処理）

    Returns:
        変換のCompose
    """
    transform_list = []
    
    # 水平反転
    if use_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
    
    # 色調整
    if use_color:
        transform_list.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation
            )
        )
    
    # 幾何変換
    if use_geometry:
        transform_list.append(
            transforms.RandomAffine(
                degrees=rotation_degrees,
                translate=(translate_ratio, translate_ratio)
            )
        )
    
    # ランダムイレース
    if use_erase:
        transform_list.append(
            transforms.RandomErasing(
                p=erase_prob,
                scale=(erase_min_ratio, erase_max_ratio),
                ratio=(0.3, 3.3),
                value=0
            )
        )
    
    # ベース変換（モデルの前処理）を追加
    if base_transform is not None:
        transform_list.append(base_transform)
        
    return transforms.Compose(transform_list)

def generate_augmentation_samples(
    image_path,
    num_samples=4,
    use_flip=True,
    flip_prob=0.5,
    use_color=True,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    use_geometry=True,
    rotation_degrees=5,
    translate_ratio=0.1,
    use_erase=True,
    erase_prob=0.5,
    erase_min_ratio=0.02,
    erase_max_ratio=0.2
) -> list:
    """指定された画像に対してオーグメンテーションのサンプルを生成する
    
    Args:
        image_path: 画像パス
        num_samples: 生成するサンプル数
        各種オーグメンテーションのパラメータ
        
    Returns:
        PIL.Imageのリスト（オリジナル画像を含む）
    """
    # 画像を読み込む
    original_img = Image.open(image_path).convert('RGB')
    samples = [original_img]  # オリジナル画像を含める
    
    # 変換用コンポーネント
    transform_components = []
    
    # 水平反転
    if use_flip:
        transform_components.append(
            (lambda img: ImageOps.mirror(img), "水平反転", flip_prob)
        )
    
    # 色調整（明るさ、コントラスト、彩度）
    if use_color:
        # 明るさ
        transform_components.append(
            (lambda img: ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-brightness, brightness)),
             "明るさ調整", 1.0)
        )
        # コントラスト
        transform_components.append(
            (lambda img: ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-contrast, contrast)),
             "コントラスト調整", 1.0)
        )
        # 彩度
        transform_components.append(
            (lambda img: ImageEnhance.Color(img).enhance(1.0 + random.uniform(-saturation, saturation)),
             "彩度調整", 1.0)
        )
    
    # 幾何変換（回転、平行移動）
    if use_geometry:
        # 回転
        transform_components.append(
            (lambda img: img.rotate(random.uniform(-rotation_degrees, rotation_degrees), 
                                    resample=Image.BICUBIC, expand=False),
             "回転", 1.0)
        )
        # 平行移動
        def translate_img(img):
            width, height = img.size
            dx = int(random.uniform(-translate_ratio, translate_ratio) * width)
            dy = int(random.uniform(-translate_ratio, translate_ratio) * height)
            return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
        
        transform_components.append(
            (translate_img, "平行移動", 1.0)
        )
    
    # ランダムイレース
    if use_erase:
        def erase_img(img):
            img_arr = np.array(img)
            h, w, _ = img_arr.shape
            
            # イレースする領域のサイズを計算
            area = h * w
            erase_area = random.uniform(erase_min_ratio, erase_max_ratio) * area
            aspect_ratio = random.uniform(0.3, 3.3)
            
            h_erase = int(np.sqrt(erase_area * aspect_ratio))
            w_erase = int(np.sqrt(erase_area / aspect_ratio))
            
            # 領域をランダムに選択
            x = random.randint(0, w - w_erase)
            y = random.randint(0, h - h_erase)
            
            # 領域を黒で塗りつぶす
            img_arr[y:y+h_erase, x:x+w_erase, :] = 0
            return Image.fromarray(img_arr)
        
        transform_components.append(
            (erase_img, "ランダムイレース", erase_prob)
        )
    
    # サンプル生成
    for _ in range(num_samples - 1):  # オリジナルを除いて指定数を生成
        img = original_img.copy()
        augmentation_applied = []
        
        # 各変換をランダムに適用
        for transform_func, transform_name, prob in transform_components:
            if random.random() < prob:
                img = transform_func(img)
                augmentation_applied.append(transform_name)
        
        # 画像と適用した変換の説明をタプルで保存
        samples.append((img, ', '.join(augmentation_applied)))
    
    return samples

def create_datasets(
    data_dir: str = None,
    annotation_file: str = None,
    image_paths: List[str] = None,
    annotations: List[Dict] = None,
    val_split: float = 0.2, 
    model_name: str = 'resnet18',
    batch_size: int = 32,
    num_workers: int = 4,
    use_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """トレーニングとバリデーション用のデータローダーを作成する"""
    # 引数チェック
    if image_paths is None or annotations is None or len(image_paths) == 0 or len(annotations) == 0:
        raise ValueError("有効な画像パスとアノテーションが必要です。")
    
    # サンプル画像から実際のサイズを取得
    sample_img = Image.open(image_paths[0]).convert('RGB')
    actual_size = (sample_img.height, sample_img.width)
    print(f"実際の画像サイズ: {actual_size}")
    
    # モデルの前処理を取得（実際のサイズを指定）
    model = get_model(model_name, pretrained=False, input_size=actual_size)
    base_transform = model.get_preprocess()
    
    # データオーグメンテーションの設定
    if use_augmentation:
        if isinstance(use_augmentation, dict):
            # 詳細設定が提供されている場合
            aug_params = use_augmentation
            # まず明示的にToTensorを入れる
            transform_list = [transforms.ToTensor()]
            
            # 水平反転
            if aug_params.get('use_flip', True):
                transform_list.append(transforms.RandomHorizontalFlip(p=aug_params.get('flip_prob', 0.5)))
            
            # 色調整
            if aug_params.get('use_color', True):
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=aug_params.get('brightness', 0.2),
                        contrast=aug_params.get('contrast', 0.2),
                        saturation=aug_params.get('saturation', 0.2)
                    )
                )
            
            # 幾何変換
            if aug_params.get('use_geometry', True):
                transform_list.append(
                    transforms.RandomAffine(
                        degrees=aug_params.get('rotation_degrees', 5),
                        translate=(aug_params.get('translate_ratio', 0.1), 
                                aug_params.get('translate_ratio', 0.1))
                    )
                )
            
            # ランダムイレース（テンソル変換後に適用）
            if aug_params.get('use_erase', True):
                transform_list.append(
                    transforms.RandomErasing(
                        p=aug_params.get('erase_prob', 0.5),
                        scale=(aug_params.get('erase_min_ratio', 0.02), 
                            aug_params.get('erase_max_ratio', 0.2)),
                        ratio=(0.3, 3.3),
                        value=0
                    )
                )
            
            # 正規化（ベース変換の一部として）
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
            
            transform = transforms.Compose(transform_list)
        else:
            # 従来の単純な有効化の場合
            transform = transforms.Compose([
                transforms.Resize(actual_size),  # 実際のサイズにリサイズ
                transforms.ToTensor(),  # 明示的にToTensorを最初に
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        # オーグメンテーションなしの場合も明示的なToTensorを含める
        transform = transforms.Compose([
            transforms.Resize(actual_size),  # 実際のサイズにリサイズ
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # データセットの作成
    dataset = AnnotationDataset(image_paths, annotations, transform=transform)
    
    # バッチサイズが小さすぎる場合の対策
    if batch_size < 2:
        batch_size = 2
        print("警告: バッチサイズが小さすぎるため、2に調整されました")
    
    # トレーニングセットと検証セットに分割
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # データセット情報
    dataset_info = {
        'total_samples': len(dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'batch_size': batch_size,
        'num_classes': 2,  # angle, throttle
        'use_augmentation': use_augmentation,
        'actual_image_size': actual_size
    }
    
    return train_loader, val_loader, dataset_info

def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    save_dir: str = './saved_models',
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, int, str], bool]] = None
) -> Dict[str, Any]:
    """モデルをトレーニングする

    Args:
        model_name: トレーニングするモデル名
        train_loader: トレーニングデータローダー
        val_loader: 検証用データローダー
        num_epochs: エポック数
        learning_rate: 学習率
        weight_decay: 重み減衰
        save_dir: モデル保存ディレクトリ
        device: 使用するデバイス (Noneの場合は自動選択)
        progress_callback: 進捗コールバック関数 (current, total, message) -> continue

    Returns:
        トレーニング結果の辞書
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    model = get_model(model_name, pretrained=True)
    model = model.to(device)
    
    # 損失関数と最適化アルゴリズム
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # トレーニングループ
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 保存ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    
    # タイムスタンプを使用してファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'{model_name}_model_{timestamp}.pth')
    best_model_path = os.path.join(save_dir, f'{model_name}_best_{timestamp}.pth')
    
    for epoch in range(num_epochs):
        # 進捗コールバック - エポック開始
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs} 開始"
            should_continue = progress_callback(epoch, num_epochs, message)
            if not should_continue:
                break
        
        model.train()
        epoch_loss = 0.0
        
        # トレーニングステップ
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 勾配のリセット
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 逆伝播と最適化
            loss.backward()
            optimizer.step()
            
            # 損失の記録
            epoch_loss += loss.item() * inputs.size(0)
            
            # バッチごとの進捗コールバック（10%ごと）
            if progress_callback and (i % max(1, len(train_loader) // 10) == 0):
                batch_progress = i / len(train_loader)
                total_progress = (epoch + batch_progress) / num_epochs
                message = f"エポック {epoch+1}/{num_epochs}, バッチ {i}/{len(train_loader)}, 損失: {loss.item():.4f}"
                should_continue = progress_callback(int(total_progress * num_epochs), num_epochs, message)
                if not should_continue:
                    break
        
        # エポック損失の計算
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 学習率の調整
        scheduler.step(val_loss)
        
        # 進捗コールバック - エポック終了
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs}, 学習損失: {epoch_loss:.4f}, 検証損失: {val_loss:.4f}"
            should_continue = progress_callback(epoch + 1, num_epochs, message)
            if not should_continue:
                break
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, best_model_path)
    
    # 最終モデルの保存
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
    }, model_path)
    
    # トレーニング結果
    training_results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'model_path': model_path,
        'best_model_path': best_model_path,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }
    
    # トレーニング結果の可視化
    plot_training_results(training_results, save_dir,timestamp)
    
    return training_results
#m
def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    save_dir: str = './saved_models',
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, int, str], bool]] = None,
    pretrained: bool = True,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """モデルをトレーニングする

    Args:
        model_name: トレーニングするモデル名
        train_loader: トレーニングデータローダー
        val_loader: 検証用データローダー
        num_epochs: エポック数
        learning_rate: 学習率
        weight_decay: 重み減衰
        save_dir: モデル保存ディレクトリ
        device: 使用するデバイス (Noneの場合は自動選択)
        progress_callback: 進捗コールバック関数 (current, total, message) -> continue
        pretrained: 事前学習済みの重みを使用するかどうか
        model_path: 特定のモデルファイルから重みをロードする場合のパス

    Returns:
        トレーニング結果の辞書
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    if progress_callback:
        progress_callback(0, num_epochs, "モデルをロード中...")
    
    # まず事前学習済みの重みでモデルを初期化（またはランダム初期化）
    model = get_model(model_name, pretrained=pretrained)
    
    # 特定のモデルファイルから重みをロードする場合
    if model_path and os.path.exists(model_path):
        if progress_callback:
            progress_callback(0, num_epochs, f"保存済みモデル '{os.path.basename(model_path)}' から重みをロード中...")
        
        try:
            # モデルチェックポイントをロード
            checkpoint = torch.load(model_path, map_location=device)
            
            # state_dictがあるかチェック
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"モデル重みを '{model_path}' からロードしました")
            else:
                # 直接state_dictが保存されている場合
                model.load_state_dict(checkpoint)
                print(f"モデル重みを '{model_path}' からロードしました")
                
        except Exception as e:
            print(f"モデル重みのロードに失敗しました: {e}")
            print("事前学習済みモデルまたはランダム初期化を使用します")
    
    model = model.to(device)
    
    # 損失関数と最適化アルゴリズム
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # トレーニングループ
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 保存ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    
    # タイムスタンプを使用してファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'{model_name}_model_{timestamp}.pth')
    best_model_path = os.path.join(save_dir, f'{model_name}_best_{timestamp}.pth')
    
    for epoch in range(num_epochs):
        # 進捗コールバック - エポック開始
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs} 開始"
            should_continue = progress_callback(epoch, num_epochs, message)
            if not should_continue:
                break
        
        model.train()
        epoch_loss = 0.0
        
        # トレーニングステップ
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 勾配のリセット
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 逆伝播と最適化
            loss.backward()
            optimizer.step()
            
            # 損失の記録
            epoch_loss += loss.item() * inputs.size(0)
            
            # バッチごとの進捗コールバック（10%ごと）
            if progress_callback and (i % max(1, len(train_loader) // 10) == 0):
                batch_progress = i / len(train_loader)
                total_progress = (epoch + batch_progress) / num_epochs
                message = f"エポック {epoch+1}/{num_epochs}, バッチ {i}/{len(train_loader)}, 損失: {loss.item():.4f}"
                should_continue = progress_callback(int(total_progress * num_epochs), num_epochs, message)
                if not should_continue:
                    break
        
        # エポック損失の計算
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 学習率の調整
        scheduler.step(val_loss)
        
        # 進捗コールバック - エポック終了
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs}, 学習損失: {epoch_loss:.4f}, 検証損失: {val_loss:.4f}"
            should_continue = progress_callback(epoch + 1, num_epochs, message)
            if not should_continue:
                break
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, best_model_path)
    
    # 最終モデルの保存
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
    }, model_path)
    
    # トレーニング結果
    training_results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'model_path': model_path,
        'best_model_path': best_model_path,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'pretrained': pretrained,
        'loaded_weights': model_path is not None and os.path.exists(model_path)
    }
    
    # トレーニング結果の可視化
    plot_training_results(training_results, save_dir, timestamp)
    
    return training_results
#m
def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    save_dir: str = './saved_models',
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, int, str], bool]] = None,
    pretrained: bool = True,
    model_path: Optional[str] = None,
    use_early_stopping: bool = False,
    patience: int = 5
) -> Dict[str, Any]:
    """モデルをトレーニングする

    Args:
        model_name: トレーニングするモデル名
        train_loader: トレーニングデータローダー
        val_loader: 検証用データローダー
        num_epochs: エポック数
        learning_rate: 学習率
        weight_decay: 重み減衰
        save_dir: モデル保存ディレクトリ
        device: 使用するデバイス (Noneの場合は自動選択)
        progress_callback: 進捗コールバック関数 (current, total, message) -> continue
        pretrained: 事前学習済みの重みを使用するかどうか
        model_path: 特定のモデルファイルから重みをロードする場合のパス
        use_early_stopping: Early Stoppingを使用するかどうか
        patience: Early Stoppingの忍耐値（検証損失が改善しなくなってから待機するエポック数）

    Returns:
        トレーニング結果の辞書
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    if progress_callback:
        progress_callback(0, num_epochs, "モデルをロード中...")
    
    # まず事前学習済みの重みでモデルを初期化（またはランダム初期化）
    model = get_model(model_name, pretrained=pretrained)
    
    # 特定のモデルファイルから重みをロードする場合
    if model_path and os.path.exists(model_path):
        if progress_callback:
            progress_callback(0, num_epochs, f"保存済みモデル '{os.path.basename(model_path)}' から重みをロード中...")
        
        try:
            # モデルチェックポイントをロード
            checkpoint = torch.load(model_path, map_location=device)
            
            # state_dictがあるかチェック
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"モデル重みを '{model_path}' からロードしました")
            else:
                # 直接state_dictが保存されている場合
                model.load_state_dict(checkpoint)
                print(f"モデル重みを '{model_path}' からロードしました")
                
        except Exception as e:
            print(f"モデル重みのロードに失敗しました: {e}")
            print("事前学習済みモデルまたはランダム初期化を使用します")
    
    model = model.to(device)
    
    # 損失関数と最適化アルゴリズム
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # トレーニングループ
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Early Stopping用の変数
    early_stopping_counter = 0
    early_stopped = False
    stopped_epoch = 0
    
    # 保存ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    
    # タイムスタンプを使用してファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'{model_name}_model_{timestamp}.pth')
    best_model_path = os.path.join(save_dir, f'{model_name}_best_{timestamp}.pth')
    
    completed_epochs = 0
    for epoch in range(num_epochs):
        # 進捗コールバック - エポック開始
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs} 開始"
            should_continue = progress_callback(epoch, num_epochs, message)
            if not should_continue:
                break
        
        model.train()
        epoch_loss = 0.0
        
        # トレーニングステップ
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 勾配のリセット
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 逆伝播と最適化
            loss.backward()
            optimizer.step()
            
            # 損失の記録
            epoch_loss += loss.item() * inputs.size(0)
            
            # バッチごとの進捗コールバック（10%ごと）
            if progress_callback and (i % max(1, len(train_loader) // 10) == 0):
                batch_progress = i / len(train_loader)
                total_progress = (epoch + batch_progress) / num_epochs
                message = f"エポック {epoch+1}/{num_epochs}, バッチ {i}/{len(train_loader)}, 損失: {loss.item():.4f}"
                should_continue = progress_callback(int(total_progress * num_epochs), num_epochs, message)
                if not should_continue:
                    break
        
        # エポック損失の計算
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 学習率の調整
        scheduler.step(val_loss)
        
        # エポックの完了をカウント
        completed_epochs = epoch + 1
        
        # 進捗コールバック - エポック終了
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs}, 学習損失: {epoch_loss:.4f}, 検証損失: {val_loss:.4f}"
            should_continue = progress_callback(epoch + 1, num_epochs, message)
            if not should_continue:
                break
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # カウンタをリセット
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, best_model_path)
            
            if progress_callback:
                progress_callback(epoch + 1, num_epochs, 
                                f"エポック {epoch+1}/{num_epochs}: 新しい最良モデルを保存しました（損失: {best_val_loss:.6f}）")
        else:
            # 検証損失が改善しなかった場合
            if use_early_stopping:
                early_stopping_counter += 1
                if progress_callback:
                    progress_callback(epoch + 1, num_epochs, 
                                    f"エポック {epoch+1}/{num_epochs}: 検証損失が改善しませんでした（カウンタ: {early_stopping_counter}/{patience}）")
                
                # Early Stoppingの判定
                if early_stopping_counter >= patience:
                    if progress_callback:
                        progress_callback(epoch + 1, num_epochs, 
                                        f"エポック {epoch+1}/{num_epochs}: Early Stoppingによりトレーニングを終了します")
                    early_stopped = True
                    stopped_epoch = epoch + 1
                    break
    
    # 最終モデルの保存
    torch.save({
        'epoch': completed_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'early_stopped': early_stopped,
        'stopped_epoch': stopped_epoch if early_stopped else completed_epochs,
    }, model_path)
    
    # トレーニング結果
    training_results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'model_path': model_path,
        'best_model_path': best_model_path,
        'num_epochs': num_epochs,
        'completed_epochs': completed_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'pretrained': pretrained,
        'loaded_weights': model_path is not None and os.path.exists(model_path),
        'early_stopped': early_stopped,
        'stopped_epoch': stopped_epoch if early_stopped else completed_epochs,
        'patience': patience if use_early_stopping else 0,
    }
    
    # トレーニング結果の可視化
    plot_training_results(training_results, save_dir, timestamp)
    
    return training_results


def validate_model(model, dataloader, criterion, device):
    """モデルの検証を行う"""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
    
    return val_loss / len(dataloader.dataset)

def plot_training_results(results, save_dir,timestamp):
    """トレーニング結果をプロットする

    Args:
        results: トレーニング結果の辞書
        save_dir: プロット保存ディレクトリ
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['train_losses'], label='Training Loss')
    plt.plot(results['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Training Results: {results['model_name']}")
    plt.legend()
    plt.grid(True)
    
    # プロットの保存
    plot_path = os.path.join(save_dir, f"{results['model_name']}_{timestamp}_training_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f'Training plot saved: {plot_path}')

def evaluate_model(
    model_name: str,
    test_loader: DataLoader,
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """モデルを評価する

    Args:
        model_name: 評価するモデル名
        test_loader: テストデータローダー
        model_path: 評価するモデルのパス (Noneの場合は新しくロード)
        device: 使用するデバイス (Noneの場合は自動選択)

    Returns:
        評価結果の辞書
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    model = get_model(model_name, pretrained=False)
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Model loaded from: {model_path}')
    
    model = model.to(device)
    model.eval()
    
    # 評価指標
    mse_loss = 0.0
    mae_loss = 0.0
    angle_errors = []
    throttle_errors = []
    
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    # 推論時間の計測
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 推論
            outputs = model(inputs)
            
            # 損失の計算
            mse_loss += criterion_mse(outputs, targets).item() * inputs.size(0)
            mae_loss += criterion_mae(outputs, targets).item() * inputs.size(0)
            
            # 角度と速度の誤差を個別に記録
            for i in range(outputs.size(0)):
                angle_errors.append(abs(outputs[i, 0].item() - targets[i, 0].item()))
                throttle_errors.append(abs(outputs[i, 1].item() - targets[i, 1].item()))
    
    # 平均誤差の計算
    num_samples = len(test_loader.dataset)
    avg_mse = mse_loss / num_samples
    avg_mae = mae_loss / num_samples
    avg_angle_error = sum(angle_errors) / len(angle_errors)
    avg_throttle_error = sum(throttle_errors) / len(throttle_errors)
    
    # 推論時間（バッチ処理全体の時間）
    inference_time = time.time() - start_time
    avg_inference_time_per_sample = inference_time / num_samples
    
    # 評価結果
    eval_results = {
        'model_name': model_name,
        'mse': avg_mse,
        'mae': avg_mae,
        'angle_error': avg_angle_error,
        'throttle_error': avg_throttle_error,
        'inference_time': inference_time,
        'inference_time_per_sample': avg_inference_time_per_sample,
        'num_samples': num_samples
    }
    
    # 結果の表示
    print(f'Evaluation Results for {model_name}:')
    print(f'MSE: {avg_mse:.6f}')
    print(f'MAE: {avg_mae:.6f}')
    print(f'Average Angle Error: {avg_angle_error:.6f}')
    print(f'Average Throttle Error: {avg_throttle_error:.6f}')
    print(f'Total Inference Time: {inference_time:.4f} seconds')
    print(f'Average Inference Time per Sample: {avg_inference_time_per_sample*1000:.4f} ms')
    
    return eval_results

def compare_models(
    model_names: List[str],
    test_loader: DataLoader,
    model_dir: str = './saved_models',
    use_best: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, List[Any]]:
    """複数のモデルを比較する

    Args:
        model_names: 比較するモデル名のリスト
        test_loader: テストデータローダー
        model_dir: モデルディレクトリ
        use_best: 最良モデルを使用するか (Falseの場合は最終モデル)
        device: 使用するデバイス (Noneの場合は自動選択)

    Returns:
        モデル比較結果の辞書
    """
    results = {
        'model_names': [],
        'mse': [],
        'mae': [],
        'angle_error': [],
        'throttle_error': [],
        'inference_time': [],
        'inference_time_per_sample': [],
        'params_count': []
    }
    
    for model_name in model_names:
        # モデルパスの設定
        suffix = 'best' if use_best else 'final'
        model_path = os.path.join(model_dir, f'{model_name}_{suffix}.pth')
        
        if not os.path.exists(model_path):
            print(f'Warning: Model file not found: {model_path}. Skipping {model_name}.')
            continue
        
        # モデル評価
        eval_result = evaluate_model(model_name, test_loader, model_path, device)
        
        # パラメータ数の取得
        model = get_model(model_name)
        params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 結果の記録
        results['model_names'].append(model_name)
        results['mse'].append(eval_result['mse'])
        results['mae'].append(eval_result['mae'])
        results['angle_error'].append(eval_result['angle_error'])
        results['throttle_error'].append(eval_result['throttle_error'])
        results['inference_time'].append(eval_result['inference_time'])
        results['inference_time_per_sample'].append(eval_result['inference_time_per_sample'])
        results['params_count'].append(params_count / 1e6)  # 百万単位
    
    # 結果の表示（テーブル形式）
    print('\nModel Comparison:')
    print('-' * 100)
    print(f"{'Model':<25} {'MSE':<10} {'MAE':<10} {'Angle Err':<10} {'Throttle Err':<12} {'Inf. Time (ms)':<15} {'Params (M)':<10}")
    print('-' * 100)
    
    for i, model_name in enumerate(results['model_names']):
        print(f"{model_name:<25} {results['mse'][i]:<10.6f} {results['mae'][i]:<10.6f} {results['angle_error'][i]:<10.6f} {results['throttle_error'][i]:<12.6f} {results['inference_time_per_sample'][i]*1000:<15.4f} {results['params_count'][i]:<10.2f}")
    
    # 結果をプロット
    plot_model_comparison(results)
    
    return results

def plot_model_comparison(results):
    """モデル比較結果をプロットする"""
    # 1. 精度プロット（MSEとMAE）
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(results['model_names'], results['mse'])
    plt.title('Mean Squared Error')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('MSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.6f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(results['model_names'], results['mae'])
    plt.title('Mean Absolute Error')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('MAE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.6f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    # 2. 角度とスロットルの誤差
    plt.subplot(2, 2, 3)
    bars = plt.bar(results['model_names'], results['angle_error'])
    plt.title('Average Angle Error')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Error')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.6f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.subplot(2, 2, 4)
    bars = plt.bar(results['model_names'], results['throttle_error'])
    plt.title('Average Throttle Error')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Error')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.6f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    plt.close()
    
    # 3. 推論時間とパラメータ数の関係
    plt.figure(figsize=(10, 6))
    plt.scatter(results['params_count'], [t*1000 for t in results['inference_time_per_sample']], s=80, alpha=0.7)
    
    # モデル名をプロット
    for i, model_name in enumerate(results['model_names']):
        plt.annotate(model_name, 
                     (results['params_count'][i], results['inference_time_per_sample'][i]*1000),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Parameters Count (Millions)')
    plt.ylabel('Inference Time per Sample (ms)')
    plt.title('Model Efficiency: Inference Time vs Model Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # X軸を対数スケールに
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('model_efficiency_comparison.png')
    plt.close()

def visualize_predictions(
    model_name: str,
    test_loader: DataLoader,
    model_path: Optional[str] = None,
    num_samples: int = 5,
    device: Optional[torch.device] = None
):
    """モデルの予測を視覚化する

    Args:
        model_name: 評価するモデル名
        test_loader: テストデータローダー
        model_path: 評価するモデルのパス (Noneの場合は新しくロード)
        num_samples: 視覚化するサンプル数
        device: 使用するデバイス (Noneの場合は自動選択)
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    model = get_model(model_name, pretrained=False)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Model loaded from: {model_path}')
    
    model = model.to(device)
    model.eval()
    
    # データの取得
    data_iter = iter(test_loader)
    inputs, targets = next(data_iter)
    
    # サンプル数の調整
    num_samples = min(num_samples, inputs.size(0))
    
    # 予測
    with torch.no_grad():
        inputs_device = inputs[:num_samples].to(device)
        outputs = model(inputs_device).cpu().numpy()
    
    # ターゲットをNumPy配列に変換
    targets = targets[:num_samples].numpy()
    
    # 予測の視覚化
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    for i in range(num_samples):
        # 画像の表示
        img = inputs[i].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axs[i, 0].imshow(img)
        axs[i, 0].set_title(f'Sample {i+1}')
        axs[i, 0].axis('off')
        
        # ステアリングホイールの描画
        axs[i, 1].set_xlim(-1.2, 1.2)
        axs[i, 1].set_ylim(-1.2, 1.2)
        
        # 円の描画
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        axs[i, 1].add_patch(circle)
        
        # 予測角度の描画（-1から1の範囲を-π/4からπ/4に変換）
        pred_angle = outputs[i, 0] * np.pi/4
        target_angle = targets[i, 0] * np.pi/4
        
        # 予測角度の矢印
        axs[i, 1].arrow(0, 0, np.cos(pred_angle), np.sin(pred_angle), 
                       head_width=0.1, head_length=0.1, fc='red', ec='red', label='Predicted')
        
        # ターゲット角度の矢印
        axs[i, 1].arrow(0, 0, np.cos(target_angle), np.sin(target_angle), 
                       head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='Target')
        
        # スロットル情報をテキストで表示
        pred_throttle = outputs[i, 1]
        target_throttle = targets[i, 1]
        axs[i, 1].text(-1.1, -1.1, f'Pred Throttle: {pred_throttle:.2f}', color='red')
        axs[i, 1].text(-1.1, -1.0, f'Target Throttle: {target_throttle:.2f}', color='blue')
        
        # 凡例を表示
        if i == 0:
            axs[i, 1].legend()
        
        axs[i, 1].set_title(f'Steering Prediction')
        axs[i, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions.png')
    plt.close()
    print(f'Predictions visualization saved: {model_name}_predictions.png')

# #def export_model_to_h5(model_path, output_path, model_type):
#     """PyTorchモデルをDonkeycar用のH5形式に変換する
    
#     Args:
#         model_path: 変換するPyTorchモデルのパス
#         output_path: 出力するH5ファイルのパス
#         model_type: モデルの種類
        
#     Returns:
#         出力されたH5ファイルのパス
#     """
#     import torch
#     import numpy as np
#     import h5py
#     import os
#     from datetime import datetime
    
#     from model_catalog import get_model
    
#     try:
#         # PyTorchモデルを読み込む
#         device = torch.device('cpu')  # CPU上で変換
#         model = get_model(model_type, pretrained=False)
        
#         # 保存されたモデルの状態を読み込む
#         checkpoint = torch.load(model_path, map_location=device)
        
#         # state_dictを取得
#         if 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             # 直接state_dictが保存されている場合
#             model.load_state_dict(checkpoint)
        
#         # 評価モードに設定
#         model.eval()
        
#         # モデルの構造を取得（レイヤー名と重み）
#         layers = []
        
#         # モデルの構造をトラバースして重みを抽出
#         for name, param in model.named_parameters():
#             # 名前をDonkeycar互換の形式に変換
#             # 例: "features.0.weight" → "features_0_weight"
#             h5_name = name.replace('.', '_')
            
#             # パラメータをNumPy配列に変換
#             weight_np = param.data.cpu().numpy()
            
#             # レイヤー情報を記録
#             layers.append({
#                 'name': h5_name,
#                 'weight': weight_np,
#                 'shape': weight_np.shape
#             })
        
#         # H5ファイルに保存
#         with h5py.File(output_path, 'w') as f:
#             # メタデータを保存
#             f.attrs['model_type'] = model_type
#             f.attrs['created_date'] = np.string_(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#             f.attrs['pytorch_model'] = np.string_(os.path.basename(model_path))
            
#             # モデルのアーキテクチャ情報を保存
#             arch_group = f.create_group('architecture')
#             arch_group.attrs['name'] = model_type
            
#             # レイヤーごとの重みを保存
#             weights_group = f.create_group('weights')
#             for layer in layers:
#                 # レイヤーのデータセットを作成
#                 dataset = weights_group.create_dataset(
#                     layer['name'], 
#                     data=layer['weight'],
#                     compression="gzip", 
#                     compression_opts=9
#                 )
#                 # 形状情報も属性として保存
#                 dataset.attrs['shape'] = layer['shape']
            
#             # Donkeycar互換のモデル構成情報を追加
#             config_group = f.create_group('model_config')
#             #config_group.attrs['input_shape'] = np.array([120, 160, 3])  # 一般的なDonkeycarの入力サイズ
#             config_group.attrs['input_shape'] = np.array([224, 224, 3])  # 一般的なDonkeycarの入力サイズ
#             config_group.attrs['output_shape'] = np.array([2])  # angle, throttle
#             config_group.attrs['type'] = 'pytorch_linear'
            
#             print(f"モデルを.h5形式に変換し、{output_path}に保存しました")
#             return output_path
            
#     except Exception as e:
#         print(f"H5変換エラー: {str(e)}")
#         raise e

# def export_model_to_h5(model_path, output_path, model_type='linear'):
#     """PyTorchモデルをDonkeycar互換のh5形式に変換する"""
#     try:
#         import tensorflow as tf
#         from tensorflow import keras
#         import numpy as np
        
#         # Donkeycarの標準入力サイズ
#         input_shape = (224, 224, 3)
#         #input_shape = (120, 160, 3)
        
#         # Donkeycar互換のモデルを直接作成
#         img_in = keras.layers.Input(shape=input_shape, name='img_in')
        
#         # モデルタイプに基づいて適切なCNNレイヤーを構築
#         drop = 0.2
#         x = img_in
#         x = keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', name='conv2d_1')(x)
#         x = keras.layers.Dropout(drop)(x)
#         x = keras.layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', name='conv2d_2')(x)
#         x = keras.layers.Dropout(drop)(x)
#         x = keras.layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu', name='conv2d_3')(x)
#         x = keras.layers.Dropout(drop)(x)
#         x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv2d_4')(x)
#         x = keras.layers.Dropout(drop)(x)
#         x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv2d_5')(x)
#         x = keras.layers.Dropout(drop)(x)
#         x = keras.layers.Flatten(name='flattened')(x)
#         x = keras.layers.Dense(100, activation='relu', name='dense_1')(x)
#         x = keras.layers.Dropout(drop)(x)
#         x = keras.layers.Dense(50, activation='relu', name='dense_2')(x)
#         x = keras.layers.Dropout(drop)(x)
        
#         # リニアモデルの場合は2つの出力（角度とスロットル）
#         outputs = []
#         outputs.append(keras.layers.Dense(1, activation='linear', name='n_outputs0')(x))
#         outputs.append(keras.layers.Dense(1, activation='linear', name='n_outputs1')(x))
        
#         # Kerasモデルを作成
#         model = keras.Model(inputs=[img_in], outputs=outputs, name='linear')
        
#         # モデルをコンパイル - Donkeycarと互換性のある設定
#         model.compile(optimizer='adam', loss='mse')
        
#         # モデルの構造と重みを保存
#         model.save(output_path, include_optimizer=True)
        
#         print(f"Donkeycar互換のモデルを保存しました: {output_path}")
#         return True
        
#     except Exception as e:
#         print(f"モデル変換中にエラーが発生しました: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

def export_model_to_h5(model_path, output_path, model_type='linear'):
    """PyTorchモデルをDonkeycar互換のh5形式に変換する"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np
        
        # TensorFlow/Kerasのバージョンを確認
        tf_version = tf.__version__
        keras_version = keras.__version__
        print(f"TensorFlow version: {tf_version}")
        print(f"Keras version: {keras_version}")
        
        # Donkeycarの標準入力サイズ
        input_shape = (120, 160, 3)
        
        # Donkeycar互換のモデルを直接作成
        img_in = keras.layers.Input(shape=input_shape, name='img_in')
        
        # モデルタイプに基づいて適切なCNNレイヤーを構築
        drop = 0.2
        x = img_in
        x = keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', name='conv2d_1')(x)
        x = keras.layers.Dropout(drop)(x)
        x = keras.layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', name='conv2d_2')(x)
        x = keras.layers.Dropout(drop)(x)
        x = keras.layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu', name='conv2d_3')(x)
        x = keras.layers.Dropout(drop)(x)
        x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv2d_4')(x)
        x = keras.layers.Dropout(drop)(x)
        x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv2d_5')(x)
        x = keras.layers.Dropout(drop)(x)
        x = keras.layers.Flatten(name='flattened')(x)
        x = keras.layers.Dense(100, activation='relu', name='dense_1')(x)
        x = keras.layers.Dropout(drop)(x)
        x = keras.layers.Dense(50, activation='relu', name='dense_2')(x)
        x = keras.layers.Dropout(drop)(x)
        
        # リニアモデルの場合は2つの出力（角度とスロットル）
        outputs = []
        outputs.append(keras.layers.Dense(1, activation='linear', name='n_outputs0')(x))
        outputs.append(keras.layers.Dense(1, activation='linear', name='n_outputs1')(x))
        
        # Kerasモデルを作成
        model = keras.Model(inputs=[img_in], outputs=outputs, name='linear')
        
        # モデルをコンパイル - Donkeycarと互換性のある設定
        # 古いKerasバージョンに対応するためにlrパラメータを使用
        try:
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        except:
            # 古いKerasバージョンでは'lr'を使用
            model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
        
        # モデルの保存 - バージョン互換性のために基本的なオプションを使用
        try:
            # 新しい方法での保存を試みる
            model.save(output_path, include_optimizer=True, save_format='h5')
        except:
            try:
                # 古い方法での保存を試みる
                model.save(output_path, include_optimizer=True)
            except:
                # さらに古い方法
                model.save(output_path)
        
        print(f"Donkeycar互換のモデルを保存しました: {output_path}")
        
        # モデルの読み込みテスト
        try:
            test_model = keras.models.load_model(output_path)
            print("保存したモデルを正常に読み込めることを確認しました。")
        except Exception as test_error:
            print(f"保存したモデルの読み込みテスト中にエラーが発生しました: {test_error}")
            print("モデルは保存されましたが、Donkeycarで読み込めない可能性があります。")
        
        return True
        
    except Exception as e:
        print(f"モデル変換中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

# モジュールが直接実行された場合のサンプル処理（オプション）
if __name__ == "__main__":
    import argparse
    
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='モデルトレーニングユーティリティ')
    parser.add_argument('--data_dir', type=str, required=True, help='データディレクトリ')
    parser.add_argument('--model', type=str, default='mobilenetv3_small_100', help='モデルタイプ')
    parser.add_argument('--epochs', type=int, default=30, help='エポック数')
    parser.add_argument('--batch_size', type=int, default=8, help='バッチサイズ')
    args = parser.parse_args()
    
    # サンプルのトレーニング実行
    try:
        # データローダーの作成
        annotation_file = os.path.join(args.data_dir, "annotation", "catalog_0.catalog")
        train_loader, val_loader, dataset_info = create_datasets(
            data_dir=args.data_dir,
            annotation_file=annotation_file,
            val_split=0.2,
            model_name=args.model,
            batch_size=args.batch_size
        )
        
        print(f"データセット情報: {dataset_info}")
        
        # モデルのトレーニング
        training_results = train_model(
            model_name=args.model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_dir=os.path.join(args.data_dir, "annotation", "annotation_models")
        )
        
        print(f"トレーニング結果: {training_results}")
        
        # 検証用にモデルの評価
        test_loader = val_loader  # 同じデータを使用
        eval_results = evaluate_model(
            model_name=args.model,
            test_loader=test_loader,
            model_path=training_results['best_model_path']
        )
        
        print(f"評価結果: {eval_results}")
        
        # 予測の可視化
        visualize_predictions(
            model_name=args.model,
            test_loader=test_loader,
            model_path=training_results['best_model_path'],
            num_samples=5
        )
        
    except Exception as e:
        print(f"エラー: {str(e)}")