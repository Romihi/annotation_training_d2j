"""
モデルトレーニングユーティリティ - TIMMベースのモデルのトレーニングと評価
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import traceback


from model_catalog import get_model, AnnotationDataset
from managers.trajectory_training_manager import TrajectoryTrainingManager

import random
from PIL import Image, ImageOps, ImageEnhance


class EarlyStopping:
    """Early Stopping の実装"""

    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        Args:
            patience: 改善が見られない連続エポック数
            min_delta: 改善とみなす最小変化量
            verbose: ログ出力するかどうか
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        検証損失をチェックして早期停止するかどうかを判定

        Args:
            val_loss: 現在の検証損失

        Returns:
            bool: 早期停止するかどうか
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            # 損失が改善した場合
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 損失が改善しなかった場合
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def format_time(seconds):
    """秒数を時:分:秒の形式にフォーマット"""
    if seconds < 0:
        return "計算中..."
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}時間{minutes:02d}分{secs:02d}秒"
    elif minutes > 0:
        return f"{minutes}分{secs:02d}秒"
    else:
        return f"{secs}秒"


def get_eta(remaining_seconds):
    """残り時間から終了予定時刻を計算"""
    if remaining_seconds < 0:
        return "計算中..."
    
    eta = datetime.now() + timedelta(seconds=remaining_seconds)
    return eta.strftime("%H:%M:%S")


def create_unified_progress_message(epoch, num_epochs, elapsed_time, epoch_times=None,
                                  batch_info=None, epoch_loss=None, val_loss=None, current_loss=None, is_epoch_start=False,
                                  steering_loss=None, throttle_loss=None, val_steering_loss=None, val_throttle_loss=None,
                                  speed_loss=None, val_speed_loss=None):
    """統一された進捗メッセージを作成"""
    # 基本情報
    if batch_info:
        batch_i, batch_total = batch_info
        status_line = f"エポック {epoch+1}/{num_epochs} - バッチ {batch_i}/{batch_total}"
    elif is_epoch_start:
        status_line = f"エポック {epoch+1}/{num_epochs} 開始"
    else:
        status_line = f"エポック {epoch+1}/{num_epochs} 完了"

    # 損失情報
    loss_line = ""
    if current_loss is not None:
        if isinstance(current_loss, dict) and 'steering' in current_loss and 'throttle' in current_loss and 'total' in current_loss:
            base_loss = (f"損失 - Total: {current_loss['total']:.4f} "
                        f"(Steering: {current_loss['steering']:.4f}, Throttle: {current_loss['throttle']:.4f}")
            if 'speed' in current_loss:
                base_loss += f", Speed: {current_loss['speed']:.4f}"
            loss_line = base_loss + ")"

            # 将来予測損失を個別に改行して表示
            if 'future_5' in current_loss and isinstance(current_loss['future_5'], dict):
                f5 = current_loss['future_5']
                loss_line += f"\n  t+5  - Angle: {f5['angle']:.4f}, Throttle: {f5['throttle']:.4f}"
                if 'speed' in f5:
                    loss_line += f", Speed: {f5['speed']:.4f}"
            if 'future_10' in current_loss and isinstance(current_loss['future_10'], dict):
                f10 = current_loss['future_10']
                loss_line += f"\n  t+10 - Angle: {f10['angle']:.4f}, Throttle: {f10['throttle']:.4f}"
                if 'speed' in f10:
                    loss_line += f", Speed: {f10['speed']:.4f}"
        elif isinstance(current_loss, dict) and 'steering' in current_loss and 'throttle' in current_loss:
            loss_line = f"損失 - Steering: {current_loss['steering']:.4f}, Throttle: {current_loss['throttle']:.4f}"
            if 'speed' in current_loss:
                loss_line += f", Speed: {current_loss['speed']:.4f}"

            # 将来予測損失を個別に改行して表示
            if 'future_5' in current_loss and isinstance(current_loss['future_5'], dict):
                f5 = current_loss['future_5']
                loss_line += f"\n  t+5  - Angle: {f5['angle']:.4f}, Throttle: {f5['throttle']:.4f}"
                if 'speed' in f5:
                    loss_line += f", Speed: {f5['speed']:.4f}"
            if 'future_10' in current_loss and isinstance(current_loss['future_10'], dict):
                f10 = current_loss['future_10']
                loss_line += f"\n  t+10 - Angle: {f10['angle']:.4f}, Throttle: {f10['throttle']:.4f}"
                if 'speed' in f10:
                    loss_line += f", Speed: {f10['speed']:.4f}"
        else:
            loss_line = f"現在の損失: {current_loss:.4f}"
    elif epoch_loss is not None and val_loss is not None:
        if steering_loss is not None and throttle_loss is not None and val_steering_loss is not None and val_throttle_loss is not None:
            train_loss_detail = f"Steering: {steering_loss:.4f}, Throttle: {throttle_loss:.4f}"
            val_loss_detail = f"Steering: {val_steering_loss:.4f}, Throttle: {val_throttle_loss:.4f}"
            if speed_loss is not None and val_speed_loss is not None:
                train_loss_detail += f", Speed: {speed_loss:.4f}"
                val_loss_detail += f", Speed: {val_speed_loss:.4f}"
            loss_line = (f"学習損失 - Total: {epoch_loss:.4f} ({train_loss_detail})\n"
                        f"検証損失 - Total: {val_loss:.4f} ({val_loss_detail})")
        else:
            loss_line = f"学習損失: {epoch_loss:.4f}, 検証損失: {val_loss:.4f}"
    
    # 時間情報
    elapsed_str = format_time(elapsed_time)
    time_line = f"経過時間: {elapsed_str}"
    
    # 残り時間推定（エポック履歴がある場合）
    if epoch_times and len(epoch_times) > 0:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        if batch_info:
            # バッチ進捗を考慮した残り時間計算
            batch_progress = batch_info[0] / batch_info[1]
            current_epoch_remaining = avg_epoch_time * (1 - batch_progress)
            remaining_epochs = num_epochs - epoch - 1
            estimated_remaining_time = current_epoch_remaining + (avg_epoch_time * remaining_epochs)
        elif is_epoch_start:
            # エポック開始時の残り時間計算
            remaining_epochs = num_epochs - epoch
            estimated_remaining_time = avg_epoch_time * remaining_epochs
        else:
            # エポック完了時の残り時間計算
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        if estimated_remaining_time > 0:
            remaining_str = format_time(estimated_remaining_time)
            eta_str = get_eta(estimated_remaining_time)
            time_line += f" | 残り時間: {remaining_str} | 終了予定: {eta_str}"
    
    # メッセージ組み立て
    message_parts = [status_line]
    if loss_line:
        message_parts.append(loss_line)
    message_parts.append(time_line)
    
    return "\n".join(message_parts)


def calculate_individual_losses(outputs, targets, criterion):
    """steering（角度）、throttle（スロットル）、speed（速度）、将来予測の個別損失を計算"""
    # outputs と targets の形状: [batch_size, 2 or 3 or 9] （0: angle, 1: throttle, 2: speed（オプション））
    # 9出力の場合: [angle, throttle, speed, t+5_angle, t+5_throttle, t+5_speed, t+10_angle, t+10_throttle, t+10_speed]
    steering_outputs = outputs[:, 0:1]  # steering (angle)
    throttle_outputs = outputs[:, 1:2]  # throttle

    steering_targets = targets[:, 0:1]
    throttle_targets = targets[:, 1:2]

    steering_loss = criterion(steering_outputs, steering_targets).item()
    throttle_loss = criterion(throttle_outputs, throttle_targets).item()

    # 3つ目の出力（speed）がある場合
    speed_loss = None
    if outputs.shape[1] >= 3 and targets.shape[1] >= 3:
        speed_outputs = outputs[:, 2:3]
        speed_targets = targets[:, 2:3]
        speed_loss = criterion(speed_outputs, speed_targets).item()

    # 将来予測の損失（個別にangle, throttle, speedを計算）
    # 9出力の場合（speed有り）: [angle, throttle, speed, t+5_angle, t+5_throttle, t+5_speed, t+10_angle, t+10_throttle, t+10_speed]
    # 6出力の場合（speed無し）: [angle, throttle, t+5_angle, t+5_throttle, t+10_angle, t+10_throttle]
    future_5_losses = None  # {angle, throttle, speed}の辞書
    future_10_losses = None  # {angle, throttle, speed}の辞書

    if outputs.shape[1] >= 9 and targets.shape[1] >= 9:
        # speed有りの将来予測（9出力）
        # t+5の損失（インデックス3,4,5: angle, throttle, speed）
        future_5_losses = {
            'angle': criterion(outputs[:, 3:4], targets[:, 3:4]).item(),
            'throttle': criterion(outputs[:, 4:5], targets[:, 4:5]).item(),
            'speed': criterion(outputs[:, 5:6], targets[:, 5:6]).item()
        }

        # t+10の損失（インデックス6,7,8: angle, throttle, speed）
        future_10_losses = {
            'angle': criterion(outputs[:, 6:7], targets[:, 6:7]).item(),
            'throttle': criterion(outputs[:, 7:8], targets[:, 7:8]).item(),
            'speed': criterion(outputs[:, 8:9], targets[:, 8:9]).item()
        }
    elif outputs.shape[1] >= 6 and targets.shape[1] >= 6 and speed_loss is None:
        # speed無しの将来予測（6出力）
        # t+5の損失（インデックス2,3: angle, throttle）
        future_5_losses = {
            'angle': criterion(outputs[:, 2:3], targets[:, 2:3]).item(),
            'throttle': criterion(outputs[:, 3:4], targets[:, 3:4]).item()
        }

        # t+10の損失（インデックス4,5: angle, throttle）
        future_10_losses = {
            'angle': criterion(outputs[:, 4:5], targets[:, 4:5]).item(),
            'throttle': criterion(outputs[:, 5:6], targets[:, 5:6]).item()
        }

    return steering_loss, throttle_loss, speed_loss, future_5_losses, future_10_losses

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
    use_augmentation: bool = False,
    use_speed: bool = False,
    use_future: bool = False,
    num_outputs: int = 2
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """トレーニングとバリデーション用のデータローダーを作成する"""
    # 引数チェック
    if image_paths is None or annotations is None or len(image_paths) == 0 or len(annotations) == 0:
        raise ValueError("有効な画像パスとアノテーションが必要です。")

    # サンプル画像から実際のサイズを取得
    sample_img = Image.open(image_paths[0]).convert('RGB')
    actual_size = (sample_img.height, sample_img.width)
    print(f"実際の画像サイズ: {actual_size}")

    # モデルの前処理を取得（実際のサイズとnum_outputsを指定）
    model = get_model(model_name, pretrained=False, input_size=actual_size, num_outputs=num_outputs)
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
            
            
            transform = transforms.Compose(transform_list)
        else:
            # 従来の単純な有効化の場合
            transform = transforms.Compose([
                transforms.Resize(actual_size),  # 実際のサイズにリサイズ
                transforms.ToTensor(),  # 明示的にToTensorを最初に
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
            ])
    else:
        # オーグメンテーションなしの場合も明示的なToTensorを含める
        transform = transforms.Compose([
            transforms.Resize(actual_size),  # 実際のサイズにリサイズ
            transforms.ToTensor()
        ])

    # データセットの作成（use_speed, use_futureパラメータを追加）
    dataset = AnnotationDataset(image_paths, annotations, transform=transform, use_speed=use_speed, use_future=use_future)

    # バッチサイズが小さすぎる場合の対策
    if batch_size < 2:
        batch_size = 2
        print("警告: バッチサイズが小さすぎるため、2に調整されました")

    # トレーニングセットと検証セットに分割
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # GPU使用時の最適化設定
    use_cuda = torch.cuda.is_available()
    pin_memory = use_cuda
    # Windows環境では num_workers=0 が推奨される場合が多い
    actual_num_workers = 0 if os.name == 'nt' and use_cuda else num_workers

    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_num_workers,
        pin_memory=pin_memory
    )

    # データセット情報
    dataset_info = {
        'total_samples': len(dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'batch_size': batch_size,
        'num_classes': num_outputs,  # 2 (angle, throttle) or 3 (angle, throttle, speed) or more with future
        'use_augmentation': use_augmentation,
        'use_speed': use_speed,
        'use_future': use_future,
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
    patience: int = 5,
    min_delta: float = 0.0001,
    optimizer_name: str = 'Adam',
    scheduler_name: str = 'ReduceLROnPlateau',
    custom_model_name: Optional[str] = None,
    num_outputs: int = 2,
    input_size: Optional[Tuple[int, int]] = None
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
        min_delta: Early Stoppingの最小改善量（この値以上の改善がないと改善とみなさない）
        optimizer_name: 最適化アルゴリズム名（Adam, AdamW, SGD）
        scheduler_name: 学習率スケジューラ名（ReduceLROnPlateau, StepLR, CosineAnnealingLR, None）
        num_outputs: 出力数（2=angle/throttle, 3=angle/throttle/speed）
        input_size: 入力画像サイズ (高さ, 幅)。Noneの場合はデータローダーから推定

    Returns:
        トレーニング結果の辞書
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input_sizeが未指定の場合、データローダーから推定
    if input_size is None:
        sample_batch = next(iter(train_loader))
        input_size = (sample_batch[0].shape[2], sample_batch[0].shape[3])  # (H, W)
        print(f"データローダーから入力サイズを推定: {input_size}")

    # モデルのロード
    if progress_callback:
        progress_callback(0, num_epochs, "モデルをロード中...")

    # まず事前学習済みの重みでモデルを初期化（またはランダム初期化）
    model = get_model(model_name, pretrained=pretrained, input_size=input_size, num_outputs=num_outputs)

    # 入力サイズをモデルから確認
    model_input_size = model.input_size if hasattr(model, 'input_size') else input_size
    print(f"Model input size: {model_input_size}")
    
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

    # 損失関数
    criterion = nn.MSELoss()

    # 最適化アルゴリズムの選択
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学習率スケジューラの選択
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    elif scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None
    
    # トレーニングループ
    train_losses = []
    val_losses = []
    train_steering_losses = []
    train_throttle_losses = []
    train_speed_losses = []
    val_steering_losses = []
    val_throttle_losses = []
    val_speed_losses = []
    best_val_loss = float('inf')

    # Early Stopping用の変数
    early_stopping_counter = 0
    early_stopped = False
    stopped_epoch = 0

    # キャンセル用の変数
    cancelled = False

    # 保存ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)

    # ファイル名に使用する名前を決定（カスタム名が指定されていればそれを使用）
    save_name = custom_model_name if custom_model_name else model_name

    # タイムスタンプを使用してファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'{save_name}.pth')
    best_model_path = os.path.join(save_dir, f'{save_name}_best.pth')
    
    # 時間計測用の変数
    training_start_time = time.time()
    epoch_times = []
    
    completed_epochs = 0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 進捗コールバック - エポック開始（統一フォーマット）
        if progress_callback:
            elapsed_time = time.time() - training_start_time

            # エポック開始メッセージ
            message = create_unified_progress_message(
                epoch=epoch,
                num_epochs=num_epochs,
                elapsed_time=elapsed_time,
                epoch_times=epoch_times if epoch > 0 else None,
                is_epoch_start=True
            )

            should_continue = progress_callback(epoch, num_epochs, message)
            if not should_continue:
                cancelled = True
                break

        # キャンセルされた場合はエポックループを抜ける
        if cancelled:
            break

        model.train()
        epoch_loss = 0.0
        epoch_steering_loss = 0.0
        epoch_throttle_loss = 0.0
        epoch_speed_loss = 0.0

        # トレーニングステップ
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 勾配のリセット
            optimizer.zero_grad()

            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 個別損失の計算
            steering_loss, throttle_loss, speed_loss, future_5_losses, future_10_losses = calculate_individual_losses(outputs, targets, criterion)

            # 逆伝播と最適化
            loss.backward()
            optimizer.step()

            # 損失の記録
            epoch_loss += loss.item() * inputs.size(0)
            epoch_steering_loss += steering_loss * inputs.size(0)
            epoch_throttle_loss += throttle_loss * inputs.size(0)
            if speed_loss is not None:
                epoch_speed_loss += speed_loss * inputs.size(0)

            # バッチごとの進捗コールバック（10%ごと）
            if progress_callback and (i % max(1, len(train_loader) // 10) == 0):
                batch_progress = i / len(train_loader)
                total_progress = (epoch + batch_progress) / num_epochs

                elapsed_time = time.time() - training_start_time

                # 統一フォーマットでバッチ進捗メッセージを作成（統合+個別損失付き）
                current_losses = {
                    'total': loss.item(),
                    'steering': steering_loss,
                    'throttle': throttle_loss
                }
                if speed_loss is not None:
                    current_losses['speed'] = speed_loss
                if future_5_losses is not None:
                    current_losses['future_5'] = future_5_losses  # 辞書形式
                if future_10_losses is not None:
                    current_losses['future_10'] = future_10_losses  # 辞書形式

                message = create_unified_progress_message(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    elapsed_time=elapsed_time,
                    epoch_times=epoch_times,
                    batch_info=(i, len(train_loader)),
                    current_loss=current_losses
                )

                should_continue = progress_callback(int(total_progress * num_epochs), num_epochs, message)
                if not should_continue:
                    cancelled = True
                    break

        # バッチレベルでキャンセルされた場合はエポックループを抜ける
        if cancelled:
            break

        # エポック損失の計算
        epoch_loss /= len(train_loader.dataset)
        epoch_steering_loss /= len(train_loader.dataset)
        epoch_throttle_loss /= len(train_loader.dataset)
        if num_outputs >= 3:
            epoch_speed_loss /= len(train_loader.dataset)
            train_speed_losses.append(epoch_speed_loss)
        train_losses.append(epoch_loss)
        train_steering_losses.append(epoch_steering_loss)
        train_throttle_losses.append(epoch_throttle_loss)

        # 検証
        model.eval()
        val_loss = 0.0
        val_steering_loss = 0.0
        val_throttle_loss = 0.0
        val_speed_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 検証時の個別損失計算
                batch_steering_loss, batch_throttle_loss, batch_speed_loss, _, _ = calculate_individual_losses(outputs, targets, criterion)

                val_loss += loss.item() * inputs.size(0)
                val_steering_loss += batch_steering_loss * inputs.size(0)
                val_throttle_loss += batch_throttle_loss * inputs.size(0)
                if batch_speed_loss is not None:
                    val_speed_loss += batch_speed_loss * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_steering_loss /= len(val_loader.dataset)
        val_throttle_loss /= len(val_loader.dataset)
        if num_outputs >= 3:
            val_speed_loss /= len(val_loader.dataset)
            val_speed_losses.append(val_speed_loss)
        val_losses.append(val_loss)
        val_steering_losses.append(val_steering_loss)
        val_throttle_losses.append(val_throttle_loss)

        # 学習率の調整
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # エポックの完了をカウント
        completed_epochs = epoch + 1

        # エポック時間を記録
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # 進捗コールバック - エポック終了（統一フォーマット、個別損失付き）
        if progress_callback:
            elapsed_time = time.time() - training_start_time

            # エポック完了メッセージ（個別損失付き）
            message = create_unified_progress_message(
                epoch=epoch,
                num_epochs=num_epochs,
                elapsed_time=elapsed_time,
                epoch_times=epoch_times,
                epoch_loss=epoch_loss,
                val_loss=val_loss,
                steering_loss=epoch_steering_loss,
                throttle_loss=epoch_throttle_loss,
                val_steering_loss=val_steering_loss,
                val_throttle_loss=val_throttle_loss,
                speed_loss=epoch_speed_loss if num_outputs >= 3 else None,
                val_speed_loss=val_speed_loss if num_outputs >= 3 else None
            )

            should_continue = progress_callback(epoch + 1, num_epochs, message)
            if not should_continue:
                cancelled = True
                break

        # 最良モデルの保存（min_deltaを考慮した改善判定）
        improved = val_loss < best_val_loss - min_delta
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if improved:
            early_stopping_counter = 0  # カウンタをリセット

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'input_size': model_input_size,
            }, best_model_path)

            if progress_callback:
                progress_callback(epoch + 1, num_epochs,
                                f"エポック {epoch+1}/{num_epochs}: 新しい最良モデルを保存しました（損失: {best_val_loss:.6f}）")
        else:
            # 検証損失が改善しなかった場合（min_delta以上の改善なし）
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
    
    # 学習時間を計算
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    # キャンセルされた場合の処理
    if cancelled:
        print("学習がキャンセルされました")
        training_results = {
            'model_name': model_name,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'model_path': None,
            'best_model_path': best_model_path if os.path.exists(best_model_path) else None,
            'num_epochs': num_epochs,
            'completed_epochs': completed_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'pretrained': pretrained,
            'loaded_weights': False,
            'early_stopped': False,
            'stopped_epoch': completed_epochs,
            'patience': patience if use_early_stopping else 0,
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'epoch_times': epoch_times,
            'train_steering_losses': train_steering_losses,
            'train_throttle_losses': train_throttle_losses,
            'val_steering_losses': val_steering_losses,
            'val_throttle_losses': val_throttle_losses,
            'cancelled': True
        }
        return training_results

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
        'input_size': model_input_size,
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
        'total_training_time': total_training_time,
        'avg_epoch_time': avg_epoch_time,
        'epoch_times': epoch_times,
        'train_steering_losses': train_steering_losses,
        'train_throttle_losses': train_throttle_losses,
        'val_steering_losses': val_steering_losses,
        'val_throttle_losses': val_throttle_losses,
        'cancelled': False
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

class LocationModelManager:
    def __init__(self, app_dir_path, models_dir_name):
        self.APP_DIR_PATH = app_dir_path
        self.MODELS_DIR_NAME = models_dir_name
        self.model = None
        self.model_type = None
        self.model_path = None
        self.num_classes = 8  # 固定で8クラス
        
    def get_model_list(self, model_type=None):
        """利用可能な位置モデルまたはウェイポイントモデルのリストを取得 - モデルタイプでフィルタリング

        Args:
            model_type (str, optional): フィルタリングするモデルタイプ。指定しない場合はすべての位置モデルを返す。

        Returns:
            list: モデルのファイルパスリスト（モデルタイプでフィルタリング済み）
        """
        models_dir = os.path.join(self.APP_DIR_PATH, self.MODELS_DIR_NAME)
        os.makedirs(models_dir, exist_ok=True)

        # モデルファイルを検索
        all_model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

        # モデルタイプが指定されている場合はそれでフィルタリング
        model_files = []
        if model_type:
            # モデルタイプがファイル名に含まれているか確認
            for model_file in all_model_files:
                if model_type.lower() in model_file.lower():
                    model_files.append(os.path.join(models_dir, model_file))
        else:
            # モデルタイプが指定されていない場合は位置モデルのみを返す
            for model_file in all_model_files:
                if any(keyword in model_file.lower() for keyword in ['location', 'loc_model']):
                    model_files.append(os.path.join(models_dir, model_file))

        # モデルファイルを作成日時順にソート（新しいものが上）
        # カスタムサフィックスが追加された場合でも正しくソートされるよう、mtimeを使用
        model_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        return model_files

    def load_model(self, model_type, model_path, progress_callback=None):
        """位置モデルを読み込む"""
        try:
            # 進捗表示コールバック
            if progress_callback:
                progress_callback(30, "モデルチェックポイントを読み込み中...")
            
            # モデルチェックポイントをロード
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # クラス数を取得（チェックポイントから）
            num_classes = None
            if 'model_state_dict' in checkpoint:
                # classifierの重みを確認
                for key, value in checkpoint['model_state_dict'].items():
                    if 'classifier.weight' in key:
                        num_classes = value.shape[0]  # 出力層の最初の次元がクラス数
                        break
                    if 'regressor.weight' in key:
                        num_classes = value.shape[0]
                        break
            else:
                # 直接state_dictの場合
                for key, value in checkpoint.items():
                    if 'classifier.weight' in key:
                        num_classes = value.shape[0]
                        break
                    if 'regressor.weight' in key:
                        num_classes = value.shape[0]
                        break
            
            # クラス数がまだ特定できない場合はデフォルト値
            if num_classes is None:
                num_classes = checkpoint.get('num_classes', 8)  # デフォルト8
            
            self.num_classes = num_classes
            
            if progress_callback:
                progress_callback(50, f"モデル '{model_type}' をロード中... (クラス数: {num_classes})")
            
            # モデルを初期化
            if model_type == 'donkey_location':
                from model_catalog import DonkeyLocationModel
                self.model = DonkeyLocationModel(num_classes=num_classes)
            elif model_type == 'resnet18_location':
                from model_catalog import ResNet18LocationModel
                self.model = ResNet18LocationModel(num_classes=num_classes)
            else:
                # その他のモデル対応
                from model_catalog import get_model
                self.model = get_model(model_type, num_classes=num_classes)
            
            if progress_callback:
                progress_callback(70, "モデルの重みをロード中...")
            
            # モデルの重みをロード
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # デバイスを設定
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()
            
            # モデル情報を保存
            self.model_path = model_path
            self.model_type = model_type
            
            return True, num_classes
            
        except Exception as e:
            traceback.print_exc()
            return False, str(e)
    
    def run_inference(self, img_path):
        """指定された画像に対して位置推論を実行"""
        if self.model is None:
            return None
        
        try:
            # 画像を読み込む
            img = Image.open(img_path).convert('RGB')
            
            # モデルの前処理を取得
            if not hasattr(self.model, '_preprocess') or self.model._preprocess is None:
                self.model._preprocess = self.model.get_preprocess()
            
            # 前処理を適用
            tensor_image = self.model._preprocess(img)
            tensor_image = tensor_image.unsqueeze(0)
            
            # デバイスを取得
            device = next(self.model.parameters()).device
            tensor_image = tensor_image.to(device)
            
            # 推論実行
            with torch.no_grad():
                logits = self.model(tensor_image)
                probs = torch.softmax(logits, dim=1)
                
                # クラスインデックスと確率を取得
                max_prob, pred_class = torch.max(probs, dim=1)
                
                # 全クラスの確率をリストとして取得
                all_probs = probs[0].cpu().numpy().tolist()
            
            # 推論結果を返す
            return {
                'pred_class': pred_class.item(),
                'confidence': max_prob.item(),
                'all_probs': all_probs
            }
            
        except Exception as e:
            print(f"位置推論実行エラー: {e}")
            traceback.print_exc()
            return None
    
    def batch_inference(self, img_paths, progress_callback=None, use_index_keys=False):
        """複数の画像に対してバッチ推論を実行
        Args:
            img_paths: 画像パスのリスト
            progress_callback: 進捗コールバック関数
            use_index_keys: Trueの場合、インデックスをキーにして結果を返す
        """
        results = {}
        total = len(img_paths)
        
        for i, img_path in enumerate(img_paths):
            if progress_callback:
                progress_callback(i, total, f"画像 {i+1}/{total} を処理中...")
            
            result = self.run_inference(img_path)
            if result:
                key = i if use_index_keys else img_path
                results[key] = result
        
        return results
    
    def is_model_loaded(self):
        """位置モデルが読み込まれているかチェック"""
        return hasattr(self, 'model') and self.model is not None

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
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(actual_size),
            transforms.ToTensor()
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
    patience: int = 5,
    min_delta: float = 0.0001,
    optimizer_name: str = 'Adam',
    scheduler_name: str = 'ReduceLROnPlateau',
    custom_model_name: Optional[str] = None
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
        min_delta: Early Stoppingの最小改善量（この値以上の改善がないと改善とみなさない）

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

    # 損失関数
    criterion = nn.CrossEntropyLoss()

    # 最適化アルゴリズムの選択
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学習率スケジューラの選択
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    elif scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None

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

    # キャンセル用の変数
    cancelled = False

    # 保存ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)

    # ファイル名に使用する名前を決定（カスタム名が指定されていればそれを使用）
    save_name = custom_model_name if custom_model_name else model_name

    # タイムスタンプを使用してファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'{save_name}.pth')
    best_model_path = os.path.join(save_dir, f'{save_name}_best.pth')

    # 時間計測用の変数
    training_start_time = time.time()
    epoch_times = []

    completed_epochs = 0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # 進捗コールバック - エポック開始（統一フォーマット）
        if progress_callback:
            elapsed_time = time.time() - training_start_time

            # エポック開始メッセージ
            message = create_unified_progress_message(
                epoch=epoch,
                num_epochs=num_epochs,
                elapsed_time=elapsed_time,
                epoch_times=epoch_times if epoch > 0 else None,
                is_epoch_start=True
            )

            should_continue = progress_callback(epoch, num_epochs, message)
            if not should_continue:
                cancelled = True
                break

        # キャンセルされた場合はエポックループを抜ける
        if cancelled:
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
                
                elapsed_time = time.time() - training_start_time
                
                # 統一フォーマットでバッチ進捗メッセージを作成
                message = create_unified_progress_message(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    elapsed_time=elapsed_time,
                    epoch_times=epoch_times,
                    batch_info=(i, len(train_loader)),
                    current_loss=loss.item()
                )
                
                should_continue = progress_callback(int(total_progress * num_epochs), num_epochs, message)
                if not should_continue:
                    cancelled = True
                    break

        # バッチレベルでキャンセルされた場合はエポックループを抜ける
        if cancelled:
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
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # エポックの完了をカウント
        completed_epochs = epoch + 1
        
        # 進捗コールバック - エポック終了
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs}, 学習損失: {epoch_loss:.4f}, 検証損失: {val_loss:.4f}, "
            message += f"学習精度: {epoch_accuracy:.2f}%, 検証精度: {val_accuracy:.2f}%"
            should_continue = progress_callback(epoch + 1, num_epochs, message)
            if not should_continue:
                break
        
        # 最良モデルの保存（min_deltaを考慮した改善判定）
        improved = val_loss < best_val_loss - min_delta
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if improved:
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
            early_stopping_counter = 0  # 精度が改善した場合もカウンタをリセット

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
            # 検証損失・精度ともに改善しなかった場合
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
    
    # 学習時間を計算
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    # キャンセルされた場合の処理
    if cancelled:
        print("学習がキャンセルされました")
        training_results = {
            'model_name': model_name,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'model_path': None,
            'best_model_path': best_model_path if os.path.exists(best_model_path) else None,
            'num_epochs': num_epochs,
            'completed_epochs': completed_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'pretrained': pretrained,
            'loaded_weights': False,
            'early_stopped': False,
            'stopped_epoch': completed_epochs,
            'patience': patience if use_early_stopping else 0,
            'num_classes': num_classes,
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'epoch_times': epoch_times,
            'cancelled': True
        }
        return training_results

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
        'num_classes': num_classes,
        'total_training_time': total_training_time,
        'avg_epoch_time': avg_epoch_time,
        'epoch_times': epoch_times,
        'cancelled': False
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
            save_dir=os.path.join(args.data_dir, "annotation", "annotation_models"),
            input_size=dataset_info.get('actual_image_size')
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


class WaypointRegressionDataset(Dataset):
    """ウェイポイント回帰用のカスタムデータセット"""

    def __init__(self, image_paths, waypoint_coordinates, transform=None):
        """
        Args:
            image_paths: 画像パスのリスト
            waypoint_coordinates: ウェイポイント座標のリスト [[x1,y1,x2,y2,...], ...]
            transform: 画像変換処理
        """
        self.image_paths = image_paths
        self.waypoint_coordinates = waypoint_coordinates
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 画像をロード
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception as e:
            print(f"画像読み込みエラー {self.image_paths[idx]}: {e}")
            # デフォルト画像を返す
            img = Image.new('RGB', (224, 224), color='black')

        # 変換を適用
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                img_np = np.array(img)
                img = self.transform(img_np)

        # ウェイポイント座標をターゲットとして使用
        target = torch.tensor(self.waypoint_coordinates[idx], dtype=torch.float32)

        return img, target


def create_waypoint_datasets(
    image_paths: List[str] = None,
    waypoint_labels: List[List[float]] = None,
    val_split: float = 0.2,
    model_name: str = 'donkey_waypoint',
    batch_size: int = 8,
    num_workers: int = 4,
    use_augmentation: bool = False,
    num_waypoints: int = 4
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """ウェイポイント回帰用のデータセットを作成する

    Args:
        image_paths: 画像パスのリスト
        waypoint_labels: ウェイポイント座標のリスト [[x1,y1,x2,y2,...], ...]
        val_split: 検証用データの割合
        model_name: モデル名
        batch_size: バッチサイズ
        num_workers: ワーカー数
        use_augmentation: データ拡張を使用するかどうか
        num_waypoints: ウェイポイント数

    Returns:
        トレーニング用DataLoader, 検証用DataLoader, データセット情報
    """
    if image_paths is None or waypoint_labels is None or len(image_paths) == 0 or len(waypoint_labels) == 0:
        raise ValueError("有効な画像パスとウェイポイント座標が必要です。")

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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    # データセット作成
    dataset = WaypointRegressionDataset(image_paths, waypoint_labels, transform=transform)

    # データ分割
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataset_info = {
        'total_samples': len(dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'batch_size': batch_size,
        'num_waypoints': num_waypoints,
        'use_augmentation': use_augmentation,
        'actual_image_size': actual_size
    }

    return train_loader, val_loader, dataset_info


# =========================================================================
# 時系列軌道予測モデルの学習
# =========================================================================

def train_trajectory_model(
    valid_indexes,
    annotations,
    images,
    source_images_map,
    selected_sources,
    config,
    models_dir='./saved_models',
    mlflow_manager=None,
    progress_callback=None,
):
    """時系列軌道予測モデルを学習する（TrajectoryTrainingManagerのラッパー）

    Args:
        valid_indexes: 有効なインデックスリスト
        annotations: Dict[int, dict]
        images: List[str]
        source_images_map: Dict[variant_name, List[str]]
        selected_sources: List[str]
        config: dict — 学習設定
            model_arch: "gru" | "tcn" | "causal_cnn"
            seq_len, pred_horizon, stride, hidden_dim, num_layers,
            dropout, img_size, epochs, batch_size, learning_rate,
            val_split, augment,
            (TCN固有) tcn_channels, kernel_size
            (CausalCNN固有) cnn_channels, kernel_size
        models_dir: モデル保存ディレクトリ
        mlflow_manager: MLflowManagerインスタンス（任意）
        progress_callback: (current, total, message) -> bool

    Returns:
        dict — 学習結果
    """
    manager = TrajectoryTrainingManager(models_dir, mlflow_manager)
    return manager.train(
        valid_indexes=valid_indexes,
        annotations=annotations,
        images=images,
        source_images_map=source_images_map,
        selected_sources=selected_sources,
        config=config,
        progress_callback=progress_callback,
    )


def predict_trajectory_model(
    model_path,
    valid_indexes,
    annotations,
    images,
    source_images_map,
    models_dir='./saved_models',
    progress_callback=None,
):
    """学習済み時系列モデルで予測を実行する（TrajectoryTrainingManagerのラッパー）

    Args:
        model_path: モデルファイルパス
        valid_indexes: 有効なインデックスリスト
        annotations: Dict[int, dict]
        images: List[str]
        source_images_map: Dict[variant_name, List[str]]
        models_dir: モデル保存ディレクトリ
        progress_callback: (current, total, message) -> bool

    Returns:
        dict — {"status", "predictions", "config", "total_predictions"}
    """
    manager = TrajectoryTrainingManager(models_dir)
    return manager.predict(
        model_path=model_path,
        valid_indexes=valid_indexes,
        annotations=annotations,
        images=images,
        source_images_map=source_images_map,
        progress_callback=progress_callback,
    )


def load_trajectory_model(model_path, device=None):
    """保存済み時系列モデルをロードする（後方互換対応）

    Args:
        model_path: モデルファイルパス
        device: 使用するデバイス

    Returns:
        tuple — (model, config_dict, selected_sources)
    """
    return TrajectoryTrainingManager.load_model(model_path, device)