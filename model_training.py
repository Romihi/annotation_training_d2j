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

# Add these functions to model_training.py for location classification
###
class LocationClassificationDataset(torch.utils.data.Dataset):
    """位置分類用のカスタムデータセット"""
    def __init__(self, image_paths, location_labels, transform=None):
        self.image_paths = image_paths
        self.location_labels = location_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # PILで画像を読み込む
        img = Image.open(img_path).convert('RGB')
        
        # 変換を適用
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                img_np = np.array(img)
                img = self.transform(img_np)
        
        # 位置ラベルをターゲットとして使用
        target = torch.tensor(self.location_labels[idx], dtype=torch.long)
        
        return img, target

def create_location_datasets(
    image_paths: List[str] = None,
    location_labels: List[int] = None,
    val_split: float = 0.2, 
    model_name: str = 'resnet18_location',
    batch_size: int = 32,
    num_workers: int = 4,
    use_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """位置分類用のデータセットを作成する

    Args:
        image_paths: 画像パスのリスト
        location_labels: 位置ラベルのリスト
        val_split: 検証用データの割合
        model_name: モデル名
        batch_size: バッチサイズ
        num_workers: ワーカー数
        use_augmentation: データ拡張を使用するかどうか

    Returns:
        トレーニング用DataLoader, 検証用DataLoader, データセット情報
    """
    if image_paths is None or location_labels is None or len(image_paths) == 0 or len(location_labels) == 0:
        raise ValueError("有効な画像パスと位置ラベルが必要です。")

    # 入力サイズを取得
    sample_img = Image.open(image_paths[0]).convert('RGB')
    actual_size = (sample_img.height, sample_img.width)
    print(f"実際の画像サイズ: {actual_size}")

    # モデルから前処理を取得
    model = get_model(model_name, pretrained=False, input_size=actual_size)
    base_transform = model.get_preprocess()

    # データ拡張
    if use_augmentation:
        transform = transforms.Compose([
            transforms.Resize(actual_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(actual_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    # データセット作成
    dataset = LocationClassificationDataset(image_paths, location_labels, transform=transform)

    # データ分割
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ユニークなクラス数を取得
    num_classes = len(set(location_labels))

    dataset_info = {
        'total_samples': len(dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'batch_size': batch_size,
        'num_classes': num_classes,
        'use_augmentation': use_augmentation,
        'actual_image_size': actual_size
    }

    return train_loader, val_loader, dataset_info

def train_location_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 8,
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
    """位置分類モデルをトレーニングする

    Args:
        model_name: トレーニングするモデル名
        train_loader: トレーニングデータローダー
        val_loader: 検証用データローダー
        num_classes: クラス数
        num_epochs: エポック数
        learning_rate: 学習率
        weight_decay: 重み減衰
        save_dir: モデル保存ディレクトリ
        device: 使用するデバイス (Noneの場合は自動選択)
        progress_callback: 進捗コールバック関数 (current, total, message) -> continue
        pretrained: 事前学習済みの重みを使用するかどうか
        model_path: 特定のモデルファイルから重みをロードする場合のパス
        use_early_stopping: Early Stoppingを使用するかどうか
        patience: Early Stoppingの忍耐値

    Returns:
        トレーニング結果の辞書
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    if progress_callback:
        progress_callback(0, num_epochs, "モデルをロード中...")
    
    # モデルを初期化（クラス数を引数に追加）
    if 'donkey_location' in model_name:
        model = get_model(model_name, pretrained=pretrained)
        model.classifier = nn.Linear(50, num_classes)  # 出力層を置き換え
    elif 'resnet18_location' in model_name:
        model = get_model(model_name, pretrained=pretrained)
        # TIMMベースモデルは初期化時にnum_outputsを設定するので
        # コンストラクタで置き換える必要はないが、確認のため
        if hasattr(model, 'regressor'):
            in_features = model.regressor.in_features if hasattr(model.regressor, 'in_features') else model.regressor[0].in_features
            model.regressor = nn.Linear(in_features, num_classes)
    else:
        # その他のモデル対応
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # トレーニングループ
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
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
        correct = 0
        total = 0
        
        # トレーニングステップ
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # 統計情報を更新
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # バッチごとの進捗コールバック（10%ごと）
            if progress_callback and (i % max(1, len(train_loader) // 10) == 0):
                batch_progress = i / len(train_loader)
                total_progress = (epoch + batch_progress) / num_epochs
                message = f"エポック {epoch+1}/{num_epochs}, バッチ {i}/{len(train_loader)}, 損失: {loss.item():.4f}"
                should_continue = progress_callback(int(total_progress * num_epochs), num_epochs, message)
                if not should_continue:
                    break
        
        # エポック損失と精度の計算
        epoch_loss /= len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # 検証
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 統計情報を更新
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # 検証損失と精度の計算
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 学習率の調整
        scheduler.step(val_loss)
        
        # エポックの完了をカウント
        completed_epochs = epoch + 1
        
        # 進捗コールバック - エポック終了
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs}, 学習損失: {epoch_loss:.4f}, 検証損失: {val_loss:.4f}, "
            message += f"学習精度: {epoch_accuracy:.2f}%, 検証精度: {val_accuracy:.2f}%"
            should_continue = progress_callback(epoch + 1, num_epochs, message)
            if not should_continue:
                break
        
        # 最良モデルの保存（検証損失の改善）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # カウンタをリセット
            
            # 最良精度も更新
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'accuracy': best_val_acc,
                'num_classes': num_classes
            }, best_model_path)
            
            if progress_callback:
                progress_callback(epoch + 1, num_epochs, 
                                f"エポック {epoch+1}/{num_epochs}: 新しい最良モデルを保存しました"
                                f"（損失: {best_val_loss:.6f}, 精度: {best_val_acc:.2f}%）")
        # 検証精度のみ改善した場合
        elif val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            
            # 精度が改善した場合も保存
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': best_val_acc,
                'num_classes': num_classes
            }, best_model_path)
            
            if progress_callback:
                progress_callback(epoch + 1, num_epochs, 
                                f"エポック {epoch+1}/{num_epochs}: 新しい最良精度を保存しました"
                                f"（精度: {best_val_acc:.2f}%, 損失: {val_loss:.6f}）")
        else:
            # 検証損失が改善しなかった場合
            if use_early_stopping:
                early_stopping_counter += 1
                if progress_callback:
                    progress_callback(epoch + 1, num_epochs, 
                                    f"エポック {epoch+1}/{num_epochs}: 検証損失が改善しませんでした"
                                    f"（カウンタ: {early_stopping_counter}/{patience}）")
                
                # Early Stoppingの判定
                if early_stopping_counter >= patience:
                    if progress_callback:
                        progress_callback(epoch + 1, num_epochs, 
                                        f"エポック {epoch+1}/{num_epochs}: Early Stoppingにより"
                                        f"トレーニングを終了します")
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
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'early_stopped': early_stopped,
        'stopped_epoch': stopped_epoch if early_stopped else completed_epochs,
        'num_classes': num_classes
    }, model_path)
    
    # トレーニング結果
    training_results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
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
        'num_classes': num_classes
    }
    
    # トレーニング結果の可視化
    plot_location_training_results(training_results, save_dir, timestamp)
    
    return training_results

def plot_location_training_results(results, save_dir, timestamp):
    """位置分類モデルのトレーニング結果をプロットする

    Args:
        results: トレーニング結果の辞書
        save_dir: プロット保存ディレクトリ
        timestamp: タイムスタンプ
    """
    # 2x1のサブプロットを作成（損失と精度）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 損失のプロット
    ax1.plot(results['train_losses'], label='Training Loss')
    ax1.plot(results['val_losses'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f"Training Losses: {results['model_name']}")
    ax1.legend()
    ax1.grid(True)
    
    # 精度のプロット
    ax2.plot(results['train_accuracies'], label='Training Accuracy')
    ax2.plot(results['val_accuracies'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f"Training Accuracies: {results['model_name']}")
    ax2.legend()
    ax2.grid(True)
    
    # プロットの保存
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{results['model_name']}_{timestamp}_training_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f'Training plot saved: {plot_path}')

def evaluate_location_model(
    model_name: str,
    test_loader: DataLoader,
    num_classes: int = 8,
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """位置分類モデルを評価する

    Args:
        model_name: 評価するモデル名
        test_loader: テストデータローダー
        num_classes: クラス数
        model_path: 評価するモデルのパス (Noneの場合は新しくロード)
        device: 使用するデバイス (Noneの場合は自動選択)

    Returns:
        評価結果の辞書
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    if 'donkey_location' in model_name:
        model = get_model(model_name, pretrained=False)
        model.classifier = nn.Linear(50, num_classes)  # 出力層を置き換え
    elif 'resnet18_location' in model_name:
        model = get_model(model_name, pretrained=False)
        if hasattr(model, 'regressor'):
            in_features = model.regressor.in_features if hasattr(model.regressor, 'in_features') else model.regressor[0].in_features
            model.regressor = nn.Linear(in_features, num_classes)
    else:
        # その他のモデル対応
        model = get_model(model_name, pretrained=False)
    
    if model_path:
        # モデルのチェックポイントをロード
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f'Model loaded from: {model_path}')
    
    model = model.to(device)
    model.eval()
    
    # 評価指標
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 混同行列の初期化
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    # 推論時間の計測
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 推論
            outputs = model(inputs)
            
            # 損失の計算
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            # 予測クラスの取得
            _, predicted = torch.max(outputs, 1)
            
            # 正解数のカウント
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # 混同行列の更新
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # 評価指標の計算
    test_loss /= total
    accuracy = 100.0 * correct / total
    
    # クラスごとの精度
    class_accuracy = 100.0 * confusion_matrix.diag() / confusion_matrix.sum(1)
    
    # 推論時間
    inference_time = time.time() - start_time
    avg_inference_time_per_sample = inference_time / total
    
    # 評価結果
    eval_results = {
        'model_name': model_name,
        'loss': test_loss,
        'accuracy': accuracy,
        'class_accuracy': class_accuracy.cpu().numpy(),
        'confusion_matrix': confusion_matrix.cpu().numpy(),
        'inference_time': inference_time,
        'inference_time_per_sample': avg_inference_time_per_sample,
        'num_samples': total
    }
    
    # 結果の表示
    print(f'Evaluation Results for {model_name}:')
    print(f'Loss: {test_loss:.6f}')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Total Inference Time: {inference_time:.4f} seconds')
    print(f'Average Inference Time per Sample: {avg_inference_time_per_sample*1000:.4f} ms')
    
    return eval_results

def visualize_location_predictions(
    model_name: str,
    test_loader: DataLoader,
    num_classes: int = 8,
    model_path: Optional[str] = None,
    num_samples: int = 5,
    device: Optional[torch.device] = None
):
    """位置分類モデルの予測を視覚化する

    Args:
        model_name: 評価するモデル名
        test_loader: テストデータローダー
        num_classes: クラス数
        model_path: 評価するモデルのパス (Noneの場合は新しくロード)
        num_samples: 視覚化するサンプル数
        device: 使用するデバイス (Noneの場合は自動選択)
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    if 'donkey_location' in model_name:
        model = get_model(model_name, pretrained=False)
        model.classifier = nn.Linear(50, num_classes)  # 出力層を置き換え
    elif 'resnet18_location' in model_name:
        model = get_model(model_name, pretrained=False)
        if hasattr(model, 'regressor'):
            in_features = model.regressor.in_features if hasattr(model.regressor, 'in_features') else model.regressor[0].in_features
            model.regressor = nn.Linear(in_features, num_classes)
    else:
        # その他のモデル対応
        model = get_model(model_name, pretrained=False)
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
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
        outputs = model(inputs_device)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    
    # ターゲットをNumPyに変換
    targets = targets[:num_samples].cpu().numpy()
    predicted = predicted.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # 予測の視覚化
    fig, axs = plt.subplots(num_samples, 1, figsize=(10, 3*num_samples))
    if num_samples == 1:
        axs = [axs]  # 1つの場合は配列に変換
    
    # 色のマップ - 位置番号ごとに異なる色で表示
    # get_location_colorと一致した色をRGBで定義
    color_map = [
        (1.0, 0.0, 0.0),      # 赤 - 位置0
        (0.0, 0.6, 0.0),      # 緑 - 位置1
        (0.0, 0.0, 1.0),      # 青 - 位置2
        (1.0, 0.65, 0.0),     # オレンジ - 位置3
        (0.5, 0.0, 0.5),      # 紫 - 位置4
        (0.0, 0.5, 0.5),      # ティール - 位置5
        (1.0, 0.0, 1.0),      # マゼンタ - 位置6
        (0.5, 0.5, 0.0)       # オリーブ - 位置7
    ]
    
    for i in range(num_samples):
        # 画像の表示
        img = inputs[i].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axs[i].imshow(img)
        
        # 予測とターゲットの表示
        target_class = targets[i]
        pred_class = predicted[i]
        prob = probabilities[i, pred_class]
        
        # 予測とターゲットが一致するかどうかで色を変更
        match_color = 'green' if pred_class == target_class else 'red'
        
        # タイトルの設定
        axs[i].set_title(f'Sample {i+1} - Target: {target_class} (位置 {target_class}), '
                        f'Prediction: {pred_class} (位置 {pred_class}, 確信度: {prob:.2f})',
                        color=match_color)
        
        # 各クラスの確率をバーグラフで表示
        for j in range(num_classes):
            # バーの色を設定（位置番号による色）
            if j < len(color_map):
                bar_color = color_map[j]
            else:
                bar_color = (0.5, 0.5, 0.5)  # デフォルト：グレー
                
            # 左側にバーを表示（画像上に重ねる）
            bar_width = 0.05
            bar_height = probabilities[i, j] * 0.5  # 画像高さの50%まで
            
            # バーの位置調整
            bar_x = 0.05 + j * bar_width * 1.5
            bar_y = 0.95 - bar_height
            
            # 透明度付きでバーを描画
            rect = plt.Rectangle((bar_x, bar_y), bar_width, bar_height,
                              transform=axs[i].transAxes, alpha=0.7,
                              color=bar_color)
            axs[i].add_patch(rect)
            
            # 確率値のテキスト表示
            prob_text = f'{probabilities[i, j]:.2f}'
            axs[i].text(bar_x + bar_width/2, bar_y - 0.02, f'{j}',
                     ha='center', va='top', transform=axs[i].transAxes,
                     color='black', fontsize=8)
            
            # 位置番号=jのバーが一番高い場合（予測クラス）、または正解クラスの場合は強調表示
            if j == pred_class or j == target_class:
                edge_color = 'white' if j == pred_class else 'yellow'
                highlight = plt.Rectangle((bar_x, bar_y), bar_width, bar_height,
                                      transform=axs[i].transAxes, fill=False,
                                      edgecolor=edge_color, linewidth=2)
                axs[i].add_patch(highlight)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_location_predictions.png')
    plt.close()
    print(f'Location predictions visualization saved: {model_name}_location_predictions.png')

def prepare_location_data_from_annotations(annotations, location_annotations):
    """アノテーションから位置分類用のデータを準備する

    Args:
        annotations: 自動運転アノテーション辞書
        location_annotations: 位置アノテーション辞書

    Returns:
        画像パスのリスト、位置ラベルのリスト
    """
    image_paths = []
    location_labels = []
    
    # 位置情報があるアノテーションを収集
    for img_path, location in location_annotations.items():
        if img_path in annotations:  # 自動運転アノテーションもある場合のみ
            image_paths.append(img_path)
            location_labels.append(location)
    
    return image_paths, location_labels

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
                
    except Exception as e:
        print(f"エラー: {str(e)}")