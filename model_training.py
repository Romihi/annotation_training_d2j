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
from model_info import get_model_input_size
from managers.sequence_training_manager import SequenceTrainingManager

import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw


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

# =====================================================================
# ドメインランダマイゼーション カスタム変換（PIL ベース）
# =====================================================================

def _dr_motion_blur(img: Image.Image, kernel_size: int = 9) -> Image.Image:
    """ランダム方向のモーションブラー"""
    k = max(3, int(kernel_size) | 1)
    angle = random.uniform(0, 180)
    rotated = img.rotate(angle, expand=False, resample=Image.BILINEAR)
    blurred = rotated.filter(ImageFilter.BoxBlur(k // 2))
    return blurred.rotate(-angle, expand=False, resample=Image.BILINEAR)


def _dr_fog(img: Image.Image, intensity: float = 0.4) -> Image.Image:
    """霧・霞: 白オーバーレイのアルファブレンド"""
    fog = Image.new('RGBA', img.size, (220, 220, 220, int(intensity * 255)))
    base = img.convert('RGBA')
    base.alpha_composite(fog)
    return base.convert('RGB')


def _dr_sunspot(img: Image.Image, num_spots: int = 3, intensity: float = 0.6) -> Image.Image:
    """路面サンスポット: 放射状グラデーション楕円 + 光のストリーク"""
    W, H = img.size
    road_top = H // 3
    base_arr = np.array(img.convert('RGB'), dtype=np.float32)
    # アキュムレータ: RGB寄与量とアルファ重み
    acc_rgb = np.zeros((H, W, 3), dtype=np.float32)
    acc_alpha = np.zeros((H, W), dtype=np.float32)

    ys, xs = np.mgrid[0:H, 0:W]

    def _add_gradient_ellipse(cx, cy, rx, ry, color_rgb, peak_alpha, sigma=2.5):
        """Gaussianフォールオフ楕円をアキュムレータに加算"""
        dx = (xs - cx) / max(rx, 1)
        dy = (ys - cy) / max(ry, 1)
        falloff = np.exp(-sigma * (dx * dx + dy * dy))
        a = falloff * peak_alpha
        for c, col in enumerate(color_rgb):
            acc_rgb[:, :, c] += a * col
        acc_alpha[:] += a

    for _ in range(num_spots):
        cx = random.randint(W // 8, 7 * W // 8)
        cy = random.randint(road_top, H - H // 10)
        rx = random.randint(W // 10, W // 4)
        ry = random.randint(H // 16, H // 7)
        # メイン楕円: 中心が明るい白→薄い黄色
        _add_gradient_ellipse(cx, cy, rx, ry, (255, 252, 220), intensity)

        # 光のストリーク: 細長い楕円を2〜4本
        for _ in range(random.randint(2, 4)):
            scx = cx + random.randint(-rx, rx)
            scy = cy + random.randint(-ry // 2, ry // 2)
            srx = random.randint(int(rx * 0.6), int(rx * 1.5))
            sry = random.randint(max(2, ry // 8), ry // 4)
            # 回転: 座標を事前に変換してから楕円距離を計算
            angle_rad = random.uniform(-0.5, 0.5)
            cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))
            dxr = (xs - scx) * cos_a + (ys - scy) * sin_a
            dyr = -(xs - scx) * sin_a + (ys - scy) * cos_a
            dist_s = (dxr / max(srx, 1)) ** 2 + (dyr / max(sry, 1)) ** 2
            falloff_s = np.exp(-3.0 * dist_s)
            streak_alpha = intensity * random.uniform(0.35, 0.7)
            a_s = falloff_s * streak_alpha
            for c, col in enumerate((255, 255, 245)):
                acc_rgb[:, :, c] += a_s * col
            acc_alpha[:] += a_s

    # アルファを[0,1]にクランプしてブレンド
    alpha_ch = np.clip(acc_alpha, 0, 1)[:, :, np.newaxis]
    overlay_rgb = np.clip(acc_rgb / np.where(acc_alpha[:, :, np.newaxis] > 0, acc_alpha[:, :, np.newaxis], 1), 0, 255)
    out_arr = np.clip(base_arr * (1 - alpha_ch) + overlay_rgb * alpha_ch, 0, 255).astype(np.uint8)
    return Image.fromarray(out_arr, 'RGB')


def _dr_gamma(img: Image.Image, gamma_min: float = 0.5, gamma_max: float = 1.8) -> Image.Image:
    """ガンマ補正: 露出シミュレーション"""
    gamma = random.uniform(gamma_min, gamma_max)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.power(arr, gamma)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))


def _dr_shadow(img: Image.Image, darkness: float = 0.5) -> Image.Image:
    """影コントラスト: 左右どちらかの帯をグラデーションで暗くする"""
    W, H = img.size
    arr = np.array(img, dtype=np.float32)
    shadow_w = int(random.uniform(0.25, 0.65) * W)
    factor_dark = 1.0 - darkness
    if random.random() < 0.5:
        fade = np.linspace(factor_dark, 1.0, shadow_w)
        arr[:, :shadow_w, :] *= fade[np.newaxis, :, np.newaxis]
    else:
        fade = np.linspace(1.0, factor_dark, shadow_w)
        arr[:, W - shadow_w:, :] *= fade[np.newaxis, :, np.newaxis]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _dr_gaussian_noise_tensor(tensor, std: float = 0.03):
    """ガウシアンノイズ: テンソルへのランダムノイズ付加"""
    import torch
    return (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)


def create_augmentation_transform(
    use_flip=True, flip_prob=0.5,
    use_color=True, brightness=0.2, contrast=0.2, saturation=0.2,
    use_geometry=True, rotation_degrees=5, translate_ratio=0.1,
    use_erase=True, erase_prob=0.5, erase_min_ratio=0.02, erase_max_ratio=0.2,
    # --- ドメインランダマイゼーション (A-D) ---
    use_hue=False, hue_range=0.1, hue_prob=0.5,
    use_grayscale=False, grayscale_prob=0.1,
    use_blur=False, blur_kernel=5, blur_prob=0.5,
    use_motion_blur=False, motion_kernel=9, motion_prob=0.5,
    use_fog=False, fog_intensity=0.3, fog_prob=0.5,
    use_noise=False, noise_std=0.03, noise_prob=0.5,
    # --- 強光・路面グレア ---
    use_sunspot=False, sunspot_intensity=0.5, sunspot_num=1, sunspot_prob=0.5,
    use_gamma=False, gamma_min=0.5, gamma_max=1.8, gamma_prob=0.5,
    use_shadow=False, shadow_darkness=0.5, shadow_prob=0.5,
    base_transform=None
) -> transforms.Compose:
    """詳細設定可能なデータオーグメンテーション変換を作成する"""
    pil_list = []   # ToTensor 前の PIL 変換
    tensor_list = [transforms.ToTensor()]  # ToTensor + その後のテンソル変換

    # --- PIL 変換群 ---
    # 色調整 (明るさ/コントラスト/彩度)
    if use_color:
        pil_list.append(transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
        ))

    # 色相シフト (C) — 独立した確率で適用
    if use_hue:
        _hr = hue_range
        pil_list.append(transforms.RandomApply(
            [transforms.ColorJitter(hue=_hr)],
            p=hue_prob
        ))

    # グレースケール化 (C)
    if use_grayscale:
        pil_list.append(transforms.RandomGrayscale(p=grayscale_prob))

    # 水平反転
    if use_flip:
        pil_list.append(transforms.RandomHorizontalFlip(p=flip_prob))

    # 幾何変換
    if use_geometry:
        pil_list.append(transforms.RandomAffine(
            degrees=rotation_degrees,
            translate=(translate_ratio, translate_ratio)
        ))

    # ガウシアンブラー (B)
    if use_blur:
        k = max(3, int(blur_kernel) | 1)
        pil_list.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=k, sigma=(0.5, 2.0))],
            p=blur_prob
        ))

    # モーションブラー (B)
    if use_motion_blur:
        _mk = motion_kernel
        pil_list.append(transforms.RandomApply(
            [transforms.Lambda(lambda img: _dr_motion_blur(img, _mk))],
            p=motion_prob
        ))

    # 霧 (D)
    if use_fog:
        _fi = fog_intensity
        pil_list.append(transforms.RandomApply(
            [transforms.Lambda(lambda img: _dr_fog(img, _fi))],
            p=fog_prob
        ))

    # サンスポット (強光)
    if use_sunspot:
        _si, _sn = sunspot_intensity, sunspot_num
        pil_list.append(transforms.RandomApply(
            [transforms.Lambda(lambda img: _dr_sunspot(img, _sn, _si))],
            p=sunspot_prob
        ))

    # ガンマ補正 (強光)
    if use_gamma:
        _gmin, _gmax = gamma_min, gamma_max
        pil_list.append(transforms.RandomApply(
            [transforms.Lambda(lambda img: _dr_gamma(img, _gmin, _gmax))],
            p=gamma_prob
        ))

    # 影コントラスト (強光)
    if use_shadow:
        _sd = shadow_darkness
        pil_list.append(transforms.RandomApply(
            [transforms.Lambda(lambda img: _dr_shadow(img, _sd))],
            p=shadow_prob
        ))

    # --- テンソル変換群（ToTensor 後）---
    # ガウシアンノイズ (A)
    if use_noise:
        _ns = noise_std
        tensor_list.append(transforms.RandomApply(
            [transforms.Lambda(lambda t: _dr_gaussian_noise_tensor(t, _ns))],
            p=noise_prob
        ))

    # ランダムイレース
    if use_erase:
        tensor_list.append(transforms.RandomErasing(
            p=erase_prob,
            scale=(erase_min_ratio, erase_max_ratio),
            ratio=(0.3, 3.3),
            value=0
        ))

    if base_transform is not None:
        tensor_list.append(base_transform)

    return transforms.Compose(pil_list + tensor_list)

def generate_augmentation_samples(
    image_path,
    num_samples=4,
    use_flip=True, flip_prob=0.5,
    use_color=True, brightness=0.2, contrast=0.2, saturation=0.2,
    use_geometry=True, rotation_degrees=5, translate_ratio=0.1,
    use_erase=True, erase_prob=0.5, erase_min_ratio=0.02, erase_max_ratio=0.2,
    use_hue=False, hue_range=0.1, hue_prob=0.5,
    use_grayscale=False, grayscale_prob=0.1,
    use_blur=False, blur_kernel=5, blur_prob=0.5,
    use_motion_blur=False, motion_kernel=9, motion_prob=0.5,
    use_fog=False, fog_intensity=0.3, fog_prob=0.5,
    use_noise=False, noise_std=0.03, noise_prob=0.5,
    use_sunspot=False, sunspot_intensity=0.5, sunspot_num=1, sunspot_prob=0.5,
    use_gamma=False, gamma_min=0.5, gamma_max=1.8, gamma_prob=0.5,
    use_shadow=False, shadow_darkness=0.5, shadow_prob=0.5,
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

    # --- ドメインランダマイゼーション ---
    if use_hue:
        _hr = hue_range
        transform_components.append(
            (lambda img: transforms.functional.adjust_hue(img, random.uniform(-_hr, _hr)),
             "色相シフト", hue_prob)
        )
    if use_grayscale:
        transform_components.append(
            (lambda img: img.convert('L').convert('RGB'), "グレースケール", grayscale_prob)
        )
    if use_blur:
        _bk = blur_kernel
        transform_components.append(
            (lambda img: img.filter(ImageFilter.GaussianBlur(radius=max(1, _bk // 2))),
             "ガウシアンブラー", blur_prob)
        )
    if use_motion_blur:
        _mk = motion_kernel
        transform_components.append(
            (lambda img: _dr_motion_blur(img, _mk), "モーションブラー", motion_prob)
        )
    if use_fog:
        _fi = fog_intensity
        transform_components.append(
            (lambda img: _dr_fog(img, _fi), "霧", fog_prob)
        )
    if use_noise:
        _ns = noise_std
        def _noise_pil(img):
            arr = np.array(img, dtype=np.float32)
            arr += np.random.randn(*arr.shape) * (_ns * 255)
            return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        transform_components.append((_noise_pil, "ガウシアンノイズ", noise_prob))
    if use_sunspot:
        _si, _sn = sunspot_intensity, sunspot_num
        transform_components.append(
            (lambda img: _dr_sunspot(img, _sn, _si), "サンスポット", sunspot_prob)
        )
    if use_gamma:
        _gmin, _gmax = gamma_min, gamma_max
        transform_components.append(
            (lambda img: _dr_gamma(img, _gmin, _gmax), "ガンマ補正", gamma_prob)
        )
    if use_shadow:
        _sd = shadow_darkness
        transform_components.append(
            (lambda img: _dr_shadow(img, _sd), "影コントラスト", shadow_prob)
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
    speed_normalize: float = None,
    mask_polygon: List[Tuple[float, float]] = None,
    future_offsets: List[int] = None,
    pip_paths: List[Optional[str]] = None,
    pip_rect: Tuple[float, float, float, float] = None,
    num_outputs: int = 2,
    multi_source_paths: List[List[str]] = None,
    num_sources: int = 1,
    fusion_method: str = 'concat',
    virtual_source_type: Optional[str] = None,
    downscale_factor: float = 1.0,
    downscale_mode: str = 'pixelate',
    temporal_interval: int = 10
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """トレーニングとバリデーション用のデータローダーを作成する

    マルチソースモードの場合:
        multi_source_paths: [[source1, source2, ...], ...] 形式のグループ化パス
        num_sources: 画像ソース数 (>1 でマルチソースモード)
        fusion_method: 融合方法 ('concat' or 'attention')
    """
    # 引数チェック
    if image_paths is None or annotations is None or len(image_paths) == 0 or len(annotations) == 0:
        raise ValueError("有効な画像パスとアノテーションが必要です。")

    is_virtual_source = virtual_source_type is not None and num_sources > 1
    is_multi_source = num_sources > 1 and multi_source_paths is not None and not is_virtual_source

    # サンプル画像から実際のサイズを取得
    if is_multi_source:
        sample_path = multi_source_paths[0][0]
    else:
        sample_path = image_paths[0]
    sample_img = Image.open(sample_path).convert('RGB')
    actual_size = (sample_img.height, sample_img.width)
    print(f"実際の画像サイズ: {actual_size}")
    if is_multi_source:
        print(f"マルチソースモード: {num_sources}ソース, 融合方法: {fusion_method}")

    # サイズ縮小モード: actual_size自体をダウンスケールしてモデルを構築
    if downscale_factor < 1.0 and downscale_mode == 'resize':
        actual_size = (max(1, int(actual_size[0] * downscale_factor)),
                       max(1, int(actual_size[1] * downscale_factor)))
        print(f"入力サイズ縮小: {actual_size} (係数: {downscale_factor:.2f})")

    # モデルの前処理を取得（実際のサイズとnum_outputsを指定）
    model = get_model(model_name, pretrained=False, input_size=actual_size, num_outputs=num_outputs)
    base_transform = model.get_preprocess()

    # ピクセレーションモード: 元サイズのまま内容を劣化させる
    if downscale_factor < 1.0 and downscale_mode == 'pixelate':
        _factor = downscale_factor
        def _pixelate(img):
            W, H = img.size
            sw = max(1, int(W * _factor))
            sh = max(1, int(H * _factor))
            return img.resize((sw, sh), Image.NEAREST).resize((W, H), Image.NEAREST)
        base_transform = transforms.Compose([transforms.Lambda(_pixelate), base_transform])

    # データオーグメンテーションの設定
    if use_augmentation:
        if isinstance(use_augmentation, dict):
            p = use_augmentation
            transform = create_augmentation_transform(
                use_flip=p.get('use_flip', True),
                flip_prob=p.get('flip_prob', 0.5),
                use_color=p.get('use_color', True),
                brightness=p.get('brightness', 0.2),
                contrast=p.get('contrast', 0.2),
                saturation=p.get('saturation', 0.2),
                use_geometry=p.get('use_geometry', True),
                rotation_degrees=p.get('rotation_degrees', 5),
                translate_ratio=p.get('translate_ratio', 0.1),
                use_erase=p.get('use_erase', True),
                erase_prob=p.get('erase_prob', 0.5),
                erase_min_ratio=p.get('erase_min_ratio', 0.02),
                erase_max_ratio=p.get('erase_max_ratio', 0.2),
                use_hue=p.get('use_hue', False),
                hue_range=p.get('hue_range', 0.1),
                hue_prob=p.get('hue_prob', 0.5),
                use_grayscale=p.get('use_grayscale', False),
                grayscale_prob=p.get('grayscale_prob', 0.1),
                use_blur=p.get('use_blur', False),
                blur_kernel=p.get('blur_kernel', 5),
                blur_prob=p.get('blur_prob', 0.5),
                use_motion_blur=p.get('use_motion_blur', False),
                motion_kernel=p.get('motion_kernel', 9),
                motion_prob=p.get('motion_prob', 0.5),
                use_fog=p.get('use_fog', False),
                fog_intensity=p.get('fog_intensity', 0.3),
                fog_prob=p.get('fog_prob', 0.5),
                use_noise=p.get('use_noise', False),
                noise_std=p.get('noise_std', 0.03),
                noise_prob=p.get('noise_prob', 0.5),
                use_sunspot=p.get('use_sunspot', False),
                sunspot_intensity=p.get('sunspot_intensity', 0.5),
                sunspot_num=p.get('sunspot_num', 1),
                sunspot_prob=p.get('sunspot_prob', 0.5),
                use_gamma=p.get('use_gamma', False),
                gamma_min=p.get('gamma_min', 0.5),
                gamma_max=p.get('gamma_max', 1.8),
                gamma_prob=p.get('gamma_prob', 0.5),
                use_shadow=p.get('use_shadow', False),
                shadow_darkness=p.get('shadow_darkness', 0.5),
                shadow_prob=p.get('shadow_prob', 0.5),
            )
        else:
            # 従来の単純な有効化の場合
            transform = transforms.Compose([
                transforms.Resize(actual_size),
                transforms.ToTensor(),
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

    # データセットの作成
    if is_virtual_source:
        from model_catalog import VirtualSourceDataset
        dataset = VirtualSourceDataset(
            image_paths=image_paths,
            annotations=annotations,
            num_virtual_sources=num_sources,
            virtual_type=virtual_source_type,
            transform=transform,
            use_speed=use_speed,
            use_future=use_future,
            temporal_interval=temporal_interval,
            speed_normalize=speed_normalize,
            mask_polygon=mask_polygon,
            future_offsets=future_offsets
        )
        print(f"VirtualSourceDataset作成: {len(dataset)}サンプル, {num_sources}仮想ソース, タイプ={virtual_source_type}, 時間差={temporal_interval}")
    elif is_multi_source:
        from model_catalog import MultiSourceDataset
        dataset = MultiSourceDataset(
            grouped_image_paths=multi_source_paths,
            annotations=annotations,
            num_sources=num_sources,
            transform=transform,
            use_speed=use_speed,
            use_future=use_future,
            speed_normalize=speed_normalize,
            mask_polygon=mask_polygon,
            future_offsets=future_offsets
        )
        print(f"MultiSourceDataset作成: {len(dataset)}サンプル, {num_sources}ソース")
    else:
        dataset = AnnotationDataset(image_paths, annotations, transform=transform, use_speed=use_speed, use_future=use_future,
                                    speed_normalize=speed_normalize, mask_polygon=mask_polygon,
                                    future_offsets=future_offsets,
                                    pip_paths=pip_paths, pip_rect=pip_rect)

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
        'actual_image_size': actual_size,
        'num_sources': num_sources,
        'fusion_method': fusion_method if is_multi_source else None
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
    input_size: Optional[Tuple[int, int]] = None,
    num_sources: int = 1,
    fusion_method: str = 'concat',
    selected_sources: Optional[List[str]] = None,
    virtual_source_type: Optional[str] = None,
    temporal_interval: int = 10,
    speed_normalize: Optional[float] = None,
    vehicle_mask: Optional[List[Tuple[float, float]]] = None,
    future_offsets: Optional[List[int]] = None,
    pip_embed: Optional[Dict[str, Any]] = None
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

    # マルチソース判定
    is_multi_source = num_sources > 1

    # モデルのロード
    if progress_callback:
        progress_callback(0, num_epochs, "モデルをロード中...")

    if is_multi_source:
        # マルチソースモデルを作成
        from model_catalog import create_multi_source_model
        model = create_multi_source_model(
            base_model_name=model_name,
            num_sources=num_sources,
            fusion_method=fusion_method,
            pretrained=pretrained,
            num_outputs=num_outputs,
            input_size=input_size
        )
        print(f"マルチソースモデル作成: {model.name} ({num_sources}ソース, {fusion_method}融合)")
    else:
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
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            if is_multi_source:
                # マルチソース同士の互換性チェック
                ckpt_num_sources = checkpoint.get('num_sources', 1)
                ckpt_fusion = checkpoint.get('fusion_method', 'concat')
                arch_match = (ckpt_num_sources == num_sources and ckpt_fusion == fusion_method)

                if arch_match:
                    model.load_state_dict(state_dict)
                    print(f"マルチソースモデル重みをロードしました: {os.path.basename(model_path)}")
                else:
                    # アーキテクチャ不一致: エンコーダ重みのみ転移
                    print(f"アーキテクチャ不一致 (ソース数: {ckpt_num_sources}→{num_sources}, "
                          f"融合: {ckpt_fusion}→{fusion_method})")
                    print("エンコーダ重みのみ転移します (strict=False)")
                    model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict)
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

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'input_size': model_input_size,
                'selected_sources': selected_sources,
            }
            if speed_normalize:
                save_dict['speed_normalize'] = speed_normalize
            if vehicle_mask:
                save_dict['vehicle_mask'] = [list(p) for p in vehicle_mask]
            if future_offsets:
                save_dict['future_offsets'] = [int(v) for v in future_offsets]
            if pip_embed:
                save_dict['pip_embed'] = pip_embed
            if is_multi_source:
                save_dict['num_sources'] = num_sources
                save_dict['fusion_method'] = fusion_method
                save_dict['base_model_name'] = model_name
                if virtual_source_type:
                    save_dict['virtual_source_type'] = virtual_source_type
                    save_dict['temporal_interval'] = temporal_interval
            torch.save(save_dict, best_model_path)

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
    final_save_dict = {
        'epoch': completed_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'early_stopped': early_stopped,
        'stopped_epoch': stopped_epoch if early_stopped else completed_epochs,
        'input_size': model_input_size,
        'selected_sources': selected_sources,
    }
    if speed_normalize:
        final_save_dict['speed_normalize'] = speed_normalize
    if vehicle_mask:
        final_save_dict['vehicle_mask'] = [list(p) for p in vehicle_mask]
    if future_offsets:
        final_save_dict['future_offsets'] = [int(v) for v in future_offsets]
    if pip_embed:
        final_save_dict['pip_embed'] = pip_embed
    if is_multi_source:
        final_save_dict['num_sources'] = num_sources
        final_save_dict['fusion_method'] = fusion_method
        final_save_dict['base_model_name'] = model_name
        if virtual_source_type:
            final_save_dict['virtual_source_type'] = virtual_source_type
            final_save_dict['temporal_interval'] = temporal_interval
    torch.save(final_save_dict, model_path)

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
        # 位置モデルの入出力構成（複数画像入力 / 座標・姿勢出力）。
        # 旧形式チェックポイントでは単一画像・クラス分類として扱う。
        self.location_config = self._default_location_config()

    @staticmethod
    def _default_location_config(base_model_name=None, num_classes=8):
        return {
            'base_model_name': base_model_name,
            'num_sources': 1,
            'fusion_method': 'concat',
            'selected_sources': None,
            'virtual_source_type': None,
            'temporal_interval': 10,
            'output_mode': 'class',
            'num_classes': num_classes,
            'pose_dim': 0,
            'include_heading': False,
            'pose_norm': None,
            'pose_source': None,
            'input_size': None,
            'downscale_factor': 1.0,
            'downscale_mode': 'resize',
            'grid_config': None,
            'num_grid_classes': 0,
        }

    # 格子分類の推論結果に保持する上位セル数（表示側の Top-N はこの範囲で選択）
    GRID_TOP_KEEP = 10

    @property
    def output_mode(self):
        return self.location_config.get('output_mode', 'class')

    @property
    def num_sources(self):
        return int(self.location_config.get('num_sources', 1) or 1)

    @property
    def has_class_output(self):
        return 'class' in self.output_mode.split('_')

    @property
    def has_pose_output(self):
        return 'pose' in self.output_mode.split('_')

    @property
    def has_grid_output(self):
        return 'grid' in self.output_mode.split('_')

    @property
    def grid_config(self):
        return self.location_config.get('grid_config')

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
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # 新形式（複数画像入力 / 座標・姿勢出力）のチェックポイント
            location_config = checkpoint.get('location_config') if isinstance(checkpoint, dict) else None
            if location_config:
                from model_catalog import create_multi_source_location_model
                cfg = dict(self._default_location_config(model_type))
                cfg.update(location_config)
                num_classes = int(cfg.get('num_classes') or checkpoint.get('num_classes', 8))
                cfg['num_classes'] = num_classes
                if progress_callback:
                    progress_callback(50, f"モデル '{model_type}' をロード中... "
                                          f"(入力{cfg['num_sources']}枚 / 出力: {cfg['output_mode']})")
                self.model = create_multi_source_location_model(
                    base_model_name=cfg.get('base_model_name') or model_type,
                    num_sources=int(cfg['num_sources']),
                    fusion_method=cfg.get('fusion_method') or 'concat',
                    num_classes=num_classes,
                    output_mode=cfg['output_mode'],
                    pose_dim=int(cfg.get('pose_dim') or 4),
                    pretrained=False,
                    input_size=tuple(cfg['input_size']) if cfg.get('input_size') else None,
                    num_grid_classes=int(cfg.get('num_grid_classes') or 0),
                )
                if progress_callback:
                    progress_callback(70, "モデルの重みをロード中...")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(device)
                self.model.eval()
                self.model_path = model_path
                self.model_type = model_type
                self.num_classes = num_classes
                self.location_config = cfg
                return True, num_classes

            # 旧形式: 単一画像入力・クラス分類
            self.location_config = self._default_location_config(model_type)
            legacy_input_size = None
            if isinstance(checkpoint, dict):
                # 実画像サイズで学習したモデルは input_size を持つ（無ければ既定サイズ）
                if checkpoint.get('input_size'):
                    legacy_input_size = tuple(int(v) for v in checkpoint['input_size'])
                    self.location_config['input_size'] = list(legacy_input_size)
                self.location_config['downscale_factor'] = float(checkpoint.get('downscale_factor', 1.0) or 1.0)
                self.location_config['downscale_mode'] = checkpoint.get('downscale_mode', 'resize') or 'resize'

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
            self.location_config['num_classes'] = num_classes

            if progress_callback:
                progress_callback(50, f"モデル '{model_type}' をロード中... (クラス数: {num_classes})")
            
            # モデルを初期化（学習時の入力サイズがあればそのサイズで構築）
            if model_type == 'donkey_location':
                from model_catalog import DonkeyLocationModel
                if legacy_input_size:
                    self.model = DonkeyLocationModel(num_classes=num_classes, input_size=legacy_input_size)
                else:
                    self.model = DonkeyLocationModel(num_classes=num_classes)
            else:
                # TIMMバックボーン系の位置推論モデル（resnet18_location含む）
                from model_catalog import create_location_model
                self.model = create_location_model(model_type, num_classes=num_classes,
                                                   input_size=legacy_input_size)
            
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
        """指定された画像（または画像パスのリスト）に対して位置推論を実行

        Args:
            img_path: 画像パス、または複数画像入力モデル用のパスリスト
                      （selected_sources / 時間差スタックの順）。仮想ソース
                      crop/scale モデルは1枚から内部で生成する。

        Returns:
            dict: クラス出力があれば 'pred_class', 'confidence', 'all_probs'、
                  座標・姿勢出力があれば 'pose': {'x', 'y', 'theta'} を含む。
        """
        if self.model is None:
            return None

        try:
            paths = list(img_path) if isinstance(img_path, (list, tuple)) else [img_path]
            cfg = self.location_config
            num_sources = self.num_sources
            virtual_type = cfg.get('virtual_source_type')

            # 入力枚数の整合（crop/scale は1枚から生成、それ以外は不足分を先頭で補う）
            if virtual_type in ('crop', 'scale'):
                paths = paths[:1]
            elif len(paths) < num_sources:
                paths = paths + [paths[-1]] * (num_sources - len(paths))
            elif len(paths) > num_sources:
                paths = paths[:num_sources]

            from model_catalog import (MultiSourceLocationModel, location_virtual_sources,
                                       split_location_outputs, denormalize_pose_output,
                                       pixelate_image)

            images = [Image.open(p).convert('RGB') for p in paths]
            # 学習時にピクセレーションモードだった場合は推論でも同じ劣化を適用する
            # （resize モードはモデルの入力サイズに含まれているため前処理で自動的に一致する）
            factor = float(cfg.get('downscale_factor', 1.0) or 1.0)
            if cfg.get('downscale_mode') == 'pixelate' and factor < 1.0:
                images = [pixelate_image(img, factor) for img in images]

            # モデルの前処理を取得
            if not hasattr(self.model, '_preprocess') or self.model._preprocess is None:
                self.model._preprocess = self.model.get_preprocess()
            if isinstance(self.model, MultiSourceLocationModel):
                if virtual_type in ('crop', 'scale') and num_sources > 1:
                    images = location_virtual_sources(images[0], virtual_type, num_sources)
                tensors = [self.model._preprocess(img) for img in images]
                tensor_image = torch.cat(tensors, dim=0).unsqueeze(0)
            else:
                tensor_image = self.model._preprocess(images[0]).unsqueeze(0)

            device = next(self.model.parameters()).device
            tensor_image = tensor_image.to(device)

            result = {}
            with torch.no_grad():
                outputs = self.model(tensor_image)
                logits, pose, grid = split_location_outputs(outputs, self.output_mode)

                if logits is not None:
                    probs = torch.softmax(logits, dim=1)
                    max_prob, pred_class = torch.max(probs, dim=1)
                    result['pred_class'] = pred_class.item()
                    result['confidence'] = max_prob.item()
                    result['all_probs'] = probs[0].cpu().numpy().tolist()

                if pose is not None and cfg.get('pose_norm'):
                    vec = pose[0].float().cpu().numpy()
                    x, y, theta = denormalize_pose_output(
                        vec, cfg['pose_norm'], include_heading=bool(cfg.get('include_heading')))
                    result['pose'] = {'x': x, 'y': y, 'theta': theta}
                    result['pose_vec'] = vec.tolist()

                if grid is not None and cfg.get('grid_config'):
                    from model_catalog import grid_topn, grid_weighted_position
                    gprobs = torch.softmax(grid, dim=1)[0].cpu().numpy()
                    top = grid_topn(gprobs, cfg['grid_config'], n=self.GRID_TOP_KEEP)
                    wx, wy = grid_weighted_position(top, n=3)
                    # top: 確率降順の上位セル（表示側で Top-N / 重み付きを選んで使う）
                    result['grid'] = {
                        'top': top,
                        'top1': {'cell': top[0]['cell'], 'x': top[0]['x'], 'y': top[0]['y'],
                                 'prob': top[0]['prob']},
                        'weighted': {'x': wx, 'y': wy, 'n': 3},
                    }

            result['input_paths'] = paths
            return result

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

class LocationMultiSourceDataset(torch.utils.data.Dataset):
    """位置推論用データセット（複数画像入力・クラス / 座標姿勢ターゲット）

    grouped_image_paths: [[src1, src2, ...], ...]。各サンプルの画像をチャネル方向に
    連結した [num_sources*3, H, W] テンソルを返す。virtual_source_type が
    crop / scale の場合は各サンプル1枚目から仮想ソースを生成する
    （temporal は呼び出し側で過去フレームのパスを組にして渡す）。

    戻り値: (image_tensor,
             class_target [long, クラスラベルが無い場合は -1],
             pose_target  [float, pose_dim。姿勢ターゲットが無い場合は長さ0])
    """

    def __init__(self, grouped_image_paths, class_labels=None, pose_vectors=None,
                 num_sources=1, virtual_source_type=None, transform=None, mask_polygon=None,
                 pixelate_factor=None, grid_labels=None, grid_xy=None):
        self.grouped_paths = grouped_image_paths
        self.class_labels = class_labels
        self.pose_vectors = pose_vectors
        self.grid_labels = grid_labels     # 格子セル index（格子分類時）
        self.grid_xy = grid_xy             # 格子分類の真値座標 [x, y]（位置誤差の評価用）
        self.num_sources = num_sources
        self.virtual_source_type = virtual_source_type
        self.transform = transform
        self.mask_polygon = mask_polygon
        self.pixelate_factor = pixelate_factor  # ピクセレーションモード時の係数（<1.0 で有効）

    def _load(self, path):
        from model_catalog import apply_vehicle_mask, pixelate_image
        img = apply_vehicle_mask(Image.open(path).convert('RGB'), self.mask_polygon)
        return pixelate_image(img, self.pixelate_factor)

    def __len__(self):
        return len(self.grouped_paths)

    def _apply_transform(self, img):
        if self.transform is None:
            return transforms.ToTensor()(img)
        try:
            return self.transform(img)
        except Exception:
            return self.transform(np.array(img))

    def __getitem__(self, idx):
        from model_catalog import location_virtual_sources
        paths = self.grouped_paths[idx]
        if self.virtual_source_type in ('crop', 'scale') and self.num_sources > 1:
            images = location_virtual_sources(self._load(paths[0]), self.virtual_source_type, self.num_sources)
        else:
            images = [self._load(p) for p in paths]
        stacked = torch.cat([self._apply_transform(im) for im in images], dim=0)

        cls_value = self.class_labels[idx] if self.class_labels is not None else -1
        class_target = torch.tensor(int(cls_value), dtype=torch.long)
        if self.pose_vectors is not None:
            pose_target = torch.tensor(self.pose_vectors[idx], dtype=torch.float)
        else:
            pose_target = torch.zeros(0, dtype=torch.float)
        grid_value = self.grid_labels[idx] if self.grid_labels is not None else -1
        grid_target = torch.tensor(int(grid_value), dtype=torch.long)
        if self.grid_xy is not None:
            grid_xy = torch.tensor(self.grid_xy[idx], dtype=torch.float)
        else:
            grid_xy = torch.zeros(0, dtype=torch.float)
        return stacked, class_target, pose_target, grid_target, grid_xy


def resolve_location_input_size(raw_size, downscale_factor=1.0, downscale_mode='resize',
                                input_size=None):
    """位置モデルの学習入力サイズと pixelate 係数を決める（自動運転モデルと同じ規則）

    - 既定は実画像サイズ（input_size 指定時はそれ）でモデルを構築する
    - downscale_factor < 1.0 かつ resize モード: サイズ自体を縮小する
    - downscale_factor < 1.0 かつ pixelate モード: サイズは変えず内容を劣化させる

    Returns: (input_size (H, W), pixelate_factor or None)
    """
    size = tuple(input_size) if input_size else tuple(raw_size)
    pixelate_factor = None
    if downscale_factor is not None and downscale_factor < 1.0:
        if downscale_mode == 'resize':
            size = (max(1, int(size[0] * downscale_factor)), max(1, int(size[1] * downscale_factor)))
        else:
            pixelate_factor = float(downscale_factor)
    return size, pixelate_factor


def compute_pose_norm(pose_targets, margin_ratio=0.05):
    """[x, y, theta] リストから座標正規化の min/max を求める（少し余白を持たせる）"""
    arr = np.asarray(pose_targets, dtype=np.float64).reshape(-1, 3)
    x_min, x_max = float(arr[:, 0].min()), float(arr[:, 0].max())
    y_min, y_max = float(arr[:, 1].min()), float(arr[:, 1].max())
    x_margin = max((x_max - x_min) * margin_ratio, 0.05)
    y_margin = max((y_max - y_min) * margin_ratio, 0.05)
    return {'x_min': x_min - x_margin, 'x_max': x_max + x_margin,
            'y_min': y_min - y_margin, 'y_max': y_max + y_margin}


def create_location_datasets(
    image_paths: List[str] = None,
    location_labels: List[int] = None,
    val_split: float = 0.2,
    model_name: str = 'resnet18_location',
    batch_size: int = 32,
    num_workers: int = 4,
    use_augmentation: bool = False,
    grouped_image_paths: Optional[List[List[str]]] = None,
    pose_targets: Optional[List[List[float]]] = None,
    output_mode: str = 'class',
    num_sources: int = 1,
    virtual_source_type: Optional[str] = None,
    include_heading: bool = True,
    mask_polygon=None,
    pose_norm: Optional[Dict[str, float]] = None,
    downscale_factor: float = 1.0,
    downscale_mode: str = 'resize',
    input_size: Optional[Tuple[int, int]] = None,
    grid_cell_size: float = 0.5,
    grid_config: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """位置推論用のデータセットを作成する

    従来の引数（image_paths + location_labels）のみを渡した場合は、これまでどおり
    単一画像・クラス分類用データセットを返す。

    grouped_image_paths を渡すと複数画像入力（実カメラの複数ソース、または
    仮想ソース crop / scale / temporal）用のデータセットになる。output_mode に
    応じて class / pose / class_pose のターゲットを返す。

    Args:
        image_paths: 画像パスのリスト（単一入力・従来互換）
        location_labels: 位置クラスラベル（インデックス化済み）のリスト。pose のみの場合は None
        val_split: 検証用データの割合
        model_name: モデル名（"<backbone>_location"）
        batch_size: バッチサイズ
        num_workers: ワーカー数
        use_augmentation: データ拡張を使用するかどうか
        grouped_image_paths: [[src1, src2, ...], ...] 各サンプルの入力画像パス
        pose_targets: [[x, y, theta], ...] 各サンプルの座標・姿勢（map座標系[m], [rad]）
        output_mode: 'class' / 'pose' / 'class_pose'
        num_sources: 入力画像枚数
        virtual_source_type: None / 'crop' / 'scale' / 'temporal'
        include_heading: 姿勢(theta)をターゲットに含めるか
        mask_polygon: 車両マスク（正規化座標ポリゴン）
        pose_norm: 座標正規化の min/max（None のとき pose_targets から算出）
        downscale_factor: 解像度スライダーの係数（1.0 = フル解像度）
        downscale_mode: 'resize'（入力サイズ自体を縮小）/ 'pixelate'（サイズは維持し内容を劣化）
        input_size: 学習入力サイズ (H, W) の明示指定。None なら実画像サイズを使う
        grid_cell_size: 格子分類（output_mode に 'grid' を含む）のセル一辺 [m]
        grid_config: 格子定義の明示指定（None なら pose_targets から算出）

    Returns:
        トレーニング用DataLoader, 検証用DataLoader, データセット情報
    """
    from model_catalog import PixelateTransform

    legacy_mode = grouped_image_paths is None and output_mode == 'class' and num_sources == 1
    if legacy_mode:
        if image_paths is None or location_labels is None or len(image_paths) == 0 or len(location_labels) == 0:
            raise ValueError("有効な画像パスと位置ラベルが必要です。")

        # 入力サイズ: 実画像サイズ（解像度設定に応じて縮小 / ピクセレーション）
        sample_img = Image.open(image_paths[0]).convert('RGB')
        raw_size = (sample_img.height, sample_img.width)
        actual_size, pixelate_factor = resolve_location_input_size(
            raw_size, downscale_factor, downscale_mode, input_size)
        print(f"実際の画像サイズ: {raw_size} / 学習入力サイズ: {actual_size}"
              + (f" / pixelate x{pixelate_factor:.2f}" if pixelate_factor else ""))

        # データ拡張
        head_ops = [PixelateTransform(pixelate_factor)] if pixelate_factor else []
        if use_augmentation:
            transform = transforms.Compose(head_ops + [
                transforms.Resize(actual_size),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
            ])
        else:
            transform = transforms.Compose(head_ops + [
                transforms.Resize(actual_size),
                transforms.ToTensor()
            ])

        dataset = LocationClassificationDataset(image_paths, location_labels, transform=transform)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_classes = len(set(location_labels))
        dataset_info = {
            'total_samples': len(dataset),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'batch_size': batch_size,
            'num_classes': num_classes,
            'use_augmentation': use_augmentation,
            'actual_image_size': actual_size,
            'raw_image_size': raw_size,
            'downscale_factor': downscale_factor,
            'downscale_mode': downscale_mode,
            'output_mode': 'class',
            'num_sources': 1,
            'virtual_source_type': None,
            'pose_dim': 0,
            'pose_norm': None,
            'include_heading': False,
        }
        return train_loader, val_loader, dataset_info

    # --- 複数画像入力 / 座標・姿勢出力 / 格子分類 ---
    from model_catalog import (location_heads, normalize_pose_targets, make_grid_config,
                               grid_cell_index)
    heads = location_heads(output_mode)   # 不正な output_mode はここで ValueError
    if grouped_image_paths is None:
        if not image_paths:
            raise ValueError("有効な画像パスが必要です。")
        grouped_image_paths = [[p] for p in image_paths]
    if len(grouped_image_paths) == 0:
        raise ValueError("有効な学習サンプルがありません。")

    use_class = 'class' in heads
    use_pose = 'pose' in heads
    use_grid = 'grid' in heads
    if use_class and (location_labels is None or len(location_labels) != len(grouped_image_paths)):
        raise ValueError("クラス分類には全サンプル分の位置ラベルが必要です。")
    if (use_pose or use_grid) and (pose_targets is None or len(pose_targets) != len(grouped_image_paths)):
        raise ValueError("座標・姿勢回帰 / 格子分類には全サンプル分の pose ターゲットが必要です。")

    # 入力サイズ: 実画像サイズ（解像度設定に応じて縮小 / ピクセレーション）。
    # 自動運転モデルと同様に、このサイズでモデルを構築しチェックポイントに保存する
    sample_img = Image.open(grouped_image_paths[0][0]).convert('RGB')
    raw_size = (sample_img.height, sample_img.width)
    input_size, pixelate_factor = resolve_location_input_size(
        raw_size, downscale_factor, downscale_mode, input_size)
    print(f"元画像サイズ: {raw_size} / 学習入力サイズ: {input_size}"
          + (f" / pixelate x{pixelate_factor:.2f}" if pixelate_factor else ""))

    # データ拡張（座標・姿勢回帰 / 格子分類では左右反転を使わない: 世界座標との対応が崩れるため）
    if use_augmentation:
        aug_ops = [transforms.Resize(input_size), transforms.ToTensor()]
        if not (use_pose or use_grid):
            aug_ops.append(transforms.RandomHorizontalFlip())
        aug_ops.extend([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=3, translate=(0.05, 0.05)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])
        transform = transforms.Compose(aug_ops)
    else:
        transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])

    pose_vectors = None
    pose_dim = 0
    if use_pose:
        if pose_norm is None:
            pose_norm = compute_pose_norm(pose_targets)
        pose_vectors = normalize_pose_targets(pose_targets, pose_norm, include_heading=include_heading)
        pose_dim = int(pose_vectors.shape[1])
    else:
        pose_norm = None
        include_heading = False

    # 格子分類: x, y を格子セル index に離散化（格子定義は学習データの範囲から作る）
    grid_labels = None
    grid_xy = None
    if use_grid:
        if grid_config is None:
            grid_config = make_grid_config(pose_targets, cell_size=grid_cell_size)
        grid_labels = [grid_cell_index(p[0], p[1], grid_config) for p in pose_targets]
        grid_xy = [[float(p[0]), float(p[1])] for p in pose_targets]
        print(f"格子分類: セル {grid_config['cell_size']}m x {grid_config['nx']}x{grid_config['ny']}"
              f" = {grid_config['num_cells']}セル（サンプルのあるセル: {len(grid_config.get('occupied', {}))}）")
    else:
        grid_config = None

    dataset = LocationMultiSourceDataset(
        grouped_image_paths,
        class_labels=list(location_labels) if use_class else None,
        pose_vectors=pose_vectors,
        num_sources=num_sources,
        virtual_source_type=virtual_source_type,
        transform=transform,
        mask_polygon=mask_polygon,
        pixelate_factor=pixelate_factor,
        grid_labels=grid_labels,
        grid_xy=grid_xy,
    )

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataset_info = {
        'total_samples': len(dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'batch_size': batch_size,
        'num_classes': len(set(location_labels)) if use_class else 0,
        'use_augmentation': use_augmentation,
        'actual_image_size': input_size,
        'raw_image_size': raw_size,
        'downscale_factor': downscale_factor,
        'downscale_mode': downscale_mode,
        'output_mode': output_mode,
        'num_sources': num_sources,
        'virtual_source_type': virtual_source_type,
        'pose_dim': pose_dim,
        'pose_norm': pose_norm,
        'include_heading': include_heading,
        'grid_config': grid_config,
        'num_grid_classes': int(grid_config['num_cells']) if grid_config else 0,
    }
    return train_loader, val_loader, dataset_info


def _unpack_location_batch(batch):
    """位置データセットのバッチを
    (inputs, class_targets|None, pose_targets|None, grid_targets|None, grid_xy|None) に分解"""
    if len(batch) == 2:
        return batch[0], batch[1], None, None, None
    inputs, cls_t, pose_t = batch[0], batch[1], batch[2]
    grid_t = batch[3] if len(batch) > 3 else None
    grid_xy = batch[4] if len(batch) > 4 else None
    if pose_t is not None and pose_t.dim() == 2 and pose_t.shape[1] == 0:
        pose_t = None
    if cls_t is not None and (cls_t < 0).all():
        cls_t = None
    if grid_t is not None and (grid_t < 0).all():
        grid_t = None
    if grid_xy is not None and grid_xy.dim() == 2 and grid_xy.shape[1] == 0:
        grid_xy = None
    return inputs, cls_t, pose_t, grid_t, grid_xy


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
    custom_model_name: Optional[str] = None,
    num_sources: int = 1,
    fusion_method: str = 'concat',
    output_mode: str = 'class',
    pose_dim: int = 4,
    pose_loss_weight: float = 1.0,
    location_config: Optional[Dict[str, Any]] = None,
    input_size: Optional[Tuple[int, int]] = None,
    save_plot: bool = True,
    num_grid_classes: int = 0,
    grid_loss_weight: float = 1.0,
    grid_top_n: int = 3,
    grid_label_sigma: float = 1.0,
    grid_class_balance: bool = True,
) -> Dict[str, Any]:
    """位置推論モデルをトレーニングする（クラス分類 / 座標・姿勢回帰 / 両方）

    Args:
        model_name: トレーニングするモデル名（"<backbone>_location"）
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
        min_delta: Early Stoppingの最小改善量
        optimizer_name / scheduler_name: 最適化アルゴリズム / 学習率スケジューラ
        custom_model_name: 保存ファイル名（Noneなら model_name）
        num_sources: 入力画像枚数（1 = 従来の単一画像入力）
        fusion_method: 複数入力時の特徴融合 'concat' / 'attention'
        output_mode: 出力ヘッドの組合せ。'class'（クラス分類）/ 'pose'（座標・姿勢回帰）/
                     'grid'（格子分類）を '_' で連結（例: 'class_pose', 'pose_grid'）
        pose_dim: 座標・姿勢出力の次元（x, y, cos, sin = 4 / x, y = 2）
        pose_loss_weight: 複合損失における座標・姿勢損失の重み
        location_config: チェックポイントへ保存する入出力構成（pose_norm, grid_config,
                         selected_sources 等）
        input_size: donkey_location の入力サイズ
        save_plot: 学習曲線PNGを save_dir に保存するか
        num_grid_classes: 格子分類のセル数（location_config['grid_config'] と対応）
        grid_loss_weight: 複合損失における格子分類損失の重み
        grid_top_n: 格子分類の評価で重み付き位置に使う上位セル数
        grid_label_sigma: 格子ラベルの平滑化幅（セル単位）。真値座標を中心とするガウス分布で
                          近傍セルにも確率を配る（0 で従来の one-hot）。空間構造を学びやすくし、
                          Top-N の重み付き平均が意味を持つようにする
        grid_class_balance: サンプル数の多いセル（停車中など）へ予測が偏らないよう、
                            真値セルの出現頻度の逆数（平方根）で損失を重み付けする

    Returns:
        トレーニング結果の辞書
    """
    from model_catalog import (MultiSourceLocationModel, create_multi_source_location_model,
                               split_location_outputs, pose_errors, location_heads,
                               grid_position_errors)

    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    heads = location_heads(output_mode)
    use_class = 'class' in heads
    use_pose = 'pose' in heads
    use_grid = 'grid' in heads
    location_config = dict(location_config or {})
    pose_norm = location_config.get('pose_norm')
    include_heading = bool(location_config.get('include_heading', pose_dim >= 4))
    grid_config = location_config.get('grid_config')
    if use_pose and not pose_norm:
        raise ValueError("座標・姿勢回帰には location_config['pose_norm'] が必要です。")
    if use_grid:
        if not grid_config:
            raise ValueError("格子分類には location_config['grid_config'] が必要です。")
        num_grid_classes = int(num_grid_classes or grid_config.get('num_cells') or 0)

    # モデルのロード
    if progress_callback:
        progress_callback(0, num_epochs, "モデルをロード中...")

    # input_size が未指定ならデータローダーから推定（自動運転モデルと同じ）
    if input_size is None:
        try:
            sample_batch = next(iter(train_loader))
            input_size = (int(sample_batch[0].shape[2]), int(sample_batch[0].shape[3]))
            print(f"データローダーから入力サイズを推定: {input_size}")
        except Exception as e:
            print(f"入力サイズの推定に失敗（既定サイズを使用）: {e}")

    legacy_model = (output_mode == 'class' and num_sources == 1)
    if legacy_model:
        # 従来どおりの単一画像・クラス分類モデル（チェックポイント形式も従来互換）。
        # input_size を渡して実画像サイズ（縮小サイズ）で構築する
        from model_catalog import create_location_model
        model = create_location_model(model_name, num_classes=num_classes,
                                      pretrained=pretrained, input_size=input_size)
    else:
        model = create_multi_source_location_model(
            base_model_name=model_name, num_sources=num_sources, fusion_method=fusion_method,
            num_classes=num_classes, output_mode=output_mode, pose_dim=pose_dim,
            pretrained=pretrained, input_size=input_size,
            num_grid_classes=num_grid_classes if use_grid else 0)

    # 特定のモデルファイルから重みをロードする場合
    loaded_weights = False
    if model_path and os.path.exists(model_path):
        if progress_callback:
            progress_callback(0, num_epochs, f"保存済みモデル '{os.path.basename(model_path)}' から重みをロード中...")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            missing, unexpected = model.load_state_dict(state, strict=False)
            loaded_weights = True
            print(f"モデル重みを '{model_path}' からロードしました"
                  f"（未ロード: {len(missing)}, 不一致: {len(unexpected)}）")
        except Exception as e:
            print(f"モデル重みのロードに失敗しました: {e}")
            print("事前学習済みモデルまたはランダム初期化を使用します")

    model = model.to(device)
    model_input_size = tuple(model.input_size) if hasattr(model, 'input_size') else input_size
    print(f"Model input size: {model_input_size}")

    # 損失関数
    class_criterion = nn.CrossEntropyLoss()

    # 格子分類の損失: 近傍セルへ平滑化したソフトラベル ＋ セル頻度による重み付け。
    # 停車区間などで特定セルにフレームが集中すると、one-hot + 単純CE では
    # 「どのフレームでも最頻セルを予測する」崩れが起きやすいため。
    grid_cell_centers = None
    grid_sample_weights = None
    if use_grid:
        from model_catalog import grid_cell_center
        centers = [grid_cell_center(c, grid_config) for c in range(num_grid_classes)]
        grid_cell_centers = torch.tensor(centers, dtype=torch.float, device=device)   # [C, 2]
        if grid_class_balance:
            counts = torch.ones(num_grid_classes, dtype=torch.float)
            for c, n in (grid_config.get('occupied') or {}).items():
                if 0 <= int(c) < num_grid_classes:
                    counts[int(c)] = float(n)
            w = 1.0 / torch.sqrt(counts)
            occupied_mask = counts > 1.0
            if occupied_mask.any():
                w = w / w[occupied_mask].mean()   # 出現セルの平均が 1 になるよう正規化
            grid_sample_weights = w.to(device)

    def grid_loss_fn(grid_out, grid_t, grid_xy):
        """格子分類の損失（ソフトラベル・頻度重み付き。grid_xy が無ければ通常の CE）"""
        log_p = torch.log_softmax(grid_out, dim=1)
        if grid_xy is not None and grid_label_sigma > 0 and grid_cell_centers is not None:
            sigma = float(grid_label_sigma) * float(grid_config['cell_size'])
            d2 = ((grid_xy.to(device)[:, None, :] - grid_cell_centers[None, :, :]) ** 2).sum(-1)
            target = torch.softmax(-d2 / (2.0 * sigma * sigma), dim=1)          # [B, C]
            per_sample = -(target * log_p).sum(1)
        else:
            per_sample = -log_p.gather(1, grid_t[:, None]).squeeze(1)
        if grid_sample_weights is not None:
            w = grid_sample_weights[grid_t]
            return (per_sample * w).sum() / w.sum().clamp_min(1e-6)
        return per_sample.mean()
    pose_criterion = nn.SmoothL1Loss()

    # 最適化アルゴリズムの選択
    if optimizer_name == 'AdamW':
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

    def run_epoch(loader, train: bool):
        """1エポック分の学習または検証を実行し、統計を返す"""
        nonlocal cancelled
        model.train(train)
        total_loss = 0.0
        n_samples = 0
        correct = 0
        n_class = 0
        pos_err_sum = 0.0
        head_err_sum = 0.0
        n_pose = 0
        grid_correct = 0
        grid_err1_sum = 0.0
        grid_errw_sum = 0.0
        n_grid = 0
        for i, batch in enumerate(loader):
            inputs, cls_t, pose_t, grid_t, grid_xy = _unpack_location_batch(batch)
            inputs = inputs.to(device)
            if cls_t is not None:
                cls_t = cls_t.to(device)
            if pose_t is not None:
                pose_t = pose_t.to(device)
            if grid_t is not None:
                grid_t = grid_t.to(device)

            with torch.set_grad_enabled(train):
                outputs = model(inputs)
                logits, pose_out, grid_out = split_location_outputs(outputs, output_mode)
                loss = torch.zeros((), device=device)
                if logits is not None and cls_t is not None:
                    loss = loss + class_criterion(logits, cls_t)
                if pose_out is not None and pose_t is not None:
                    loss = loss + pose_loss_weight * pose_criterion(pose_out, pose_t)
                if grid_out is not None and grid_t is not None:
                    loss = loss + grid_loss_weight * grid_loss_fn(grid_out, grid_t, grid_xy)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            bs = inputs.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            if logits is not None and cls_t is not None:
                _, predicted = torch.max(logits, 1)
                correct += (predicted == cls_t).sum().item()
                n_class += bs
            if pose_out is not None and pose_t is not None:
                p_err, h_err = pose_errors(pose_out.detach().cpu().numpy(), pose_t.cpu().numpy(),
                                           pose_norm, include_heading=include_heading)
                pos_err_sum += float(p_err.sum())
                if h_err is not None:
                    head_err_sum += float(np.degrees(h_err).sum())
                n_pose += bs
            if grid_out is not None and grid_t is not None:
                _, g_pred = torch.max(grid_out, 1)
                grid_correct += (g_pred == grid_t).sum().item()
                if grid_xy is not None:
                    g_probs = torch.softmax(grid_out.detach(), dim=1).cpu().numpy()
                    e1, ew = grid_position_errors(g_probs, grid_xy.cpu().numpy(), grid_config,
                                                  top_n=grid_top_n)
                    grid_err1_sum += float(e1.sum())
                    grid_errw_sum += float(ew.sum())
                n_grid += bs

            # バッチごとの進捗コールバック（学習時のみ、10%ごと）
            if train and progress_callback and (i % max(1, len(loader) // 10) == 0):
                batch_progress = i / len(loader)
                total_progress = (epoch + batch_progress) / num_epochs
                message = create_unified_progress_message(
                    epoch=epoch, num_epochs=num_epochs,
                    elapsed_time=time.time() - training_start_time,
                    epoch_times=epoch_times, batch_info=(i, len(loader)),
                    current_loss=loss.item())
                if not progress_callback(int(total_progress * num_epochs), num_epochs, message):
                    cancelled = True
                    break

        return {
            'loss': total_loss / max(1, n_samples),
            'accuracy': 100.0 * correct / n_class if n_class else 0.0,
            'pos_error': pos_err_sum / n_pose if n_pose else 0.0,
            'heading_error': head_err_sum / n_pose if (n_pose and include_heading) else 0.0,
            'grid_accuracy': 100.0 * grid_correct / n_grid if n_grid else 0.0,
            'grid_error': grid_err1_sum / n_grid if n_grid else 0.0,          # Top1 セル中心の位置誤差[m]
            'grid_weighted_error': grid_errw_sum / n_grid if n_grid else 0.0, # Top-N 重み付き位置誤差[m]
        }

    # トレーニングループ
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_pos_errors, val_pos_errors = [], []
    train_heading_errors, val_heading_errors = [], []
    train_grid_accuracies, val_grid_accuracies = [], []
    train_grid_errors, val_grid_errors = [], []
    val_grid_weighted_errors = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_pos_error = None
    best_val_heading_error = None
    best_val_grid_acc = None
    best_val_grid_error = None
    best_val_grid_weighted_error = None

    early_stopping_counter = 0
    early_stopped = False
    stopped_epoch = 0
    cancelled = False

    os.makedirs(save_dir, exist_ok=True)
    save_name = custom_model_name if custom_model_name else model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(save_dir, f'{save_name}.pth')
    best_model_path = os.path.join(save_dir, f'{save_name}_best.pth')

    # チェックポイントへ保存する入出力構成（新形式モデルのみ）
    checkpoint_config = None
    if isinstance(model, MultiSourceLocationModel):
        checkpoint_config = dict(location_config)
        checkpoint_config.update({
            'base_model_name': model_name,
            'num_sources': num_sources,
            'fusion_method': fusion_method,
            'output_mode': output_mode,
            'num_classes': num_classes,
            'pose_dim': pose_dim if use_pose else 0,
            'include_heading': include_heading if use_pose else False,
            'pose_norm': pose_norm if use_pose else None,
            'input_size': list(model.input_size),
            'grid_config': grid_config if use_grid else None,
            'num_grid_classes': num_grid_classes if use_grid else 0,
            'grid_label_sigma': float(grid_label_sigma) if use_grid else None,
            'grid_class_balance': bool(grid_class_balance) if use_grid else None,
        })

    def build_checkpoint(epoch_value, extra):
        ckpt = {
            'epoch': epoch_value,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'num_classes': num_classes,
            'output_mode': output_mode,
            'num_sources': num_sources,
            # 推論時に同じ入力サイズ / 解像度設定を再現するための情報（従来形式にも保存）
            'input_size': list(model_input_size) if model_input_size else None,
            'downscale_factor': location_config.get('downscale_factor', 1.0),
            'downscale_mode': location_config.get('downscale_mode', 'resize'),
        }
        if checkpoint_config is not None:
            ckpt['location_config'] = checkpoint_config
        ckpt.update(extra)
        return ckpt

    training_start_time = time.time()
    epoch_times = []
    completed_epochs = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        if progress_callback:
            message = create_unified_progress_message(
                epoch=epoch, num_epochs=num_epochs,
                elapsed_time=time.time() - training_start_time,
                epoch_times=epoch_times if epoch > 0 else None, is_epoch_start=True)
            if not progress_callback(epoch, num_epochs, message):
                cancelled = True
                break

        train_stats = run_epoch(train_loader, train=True)
        if cancelled:
            break
        val_stats = run_epoch(val_loader, train=False)

        train_losses.append(train_stats['loss'])
        val_losses.append(val_stats['loss'])
        train_accuracies.append(train_stats['accuracy'])
        val_accuracies.append(val_stats['accuracy'])
        train_pos_errors.append(train_stats['pos_error'])
        val_pos_errors.append(val_stats['pos_error'])
        train_heading_errors.append(train_stats['heading_error'])
        val_heading_errors.append(val_stats['heading_error'])
        train_grid_accuracies.append(train_stats['grid_accuracy'])
        val_grid_accuracies.append(val_stats['grid_accuracy'])
        train_grid_errors.append(train_stats['grid_error'])
        val_grid_errors.append(val_stats['grid_error'])
        val_grid_weighted_errors.append(val_stats['grid_weighted_error'])

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats['loss'])
            else:
                scheduler.step()

        completed_epochs = epoch + 1
        epoch_times.append(time.time() - epoch_start_time)

        # エポック終了メッセージ
        if progress_callback:
            message = (f"エポック {epoch+1}/{num_epochs}, 学習損失: {train_stats['loss']:.4f}, "
                       f"検証損失: {val_stats['loss']:.4f}")
            if use_class:
                message += f", 学習精度: {train_stats['accuracy']:.2f}%, 検証精度: {val_stats['accuracy']:.2f}%"
            if use_pose:
                message += f", 検証位置誤差: {val_stats['pos_error']:.3f}m"
                if include_heading:
                    message += f", 検証方位誤差: {val_stats['heading_error']:.1f}°"
            if use_grid:
                message += (f", 格子精度: {val_stats['grid_accuracy']:.1f}%"
                            f", 格子Top1誤差: {val_stats['grid_error']:.3f}m"
                            f", Top{grid_top_n}重み付き: {val_stats['grid_weighted_error']:.3f}m")
            if not progress_callback(epoch + 1, num_epochs, message):
                cancelled = True
                break

        # 最良モデルの保存（min_deltaを考慮した改善判定）
        val_loss = val_stats['loss']
        val_accuracy = val_stats['accuracy']
        improved = val_loss < best_val_loss - min_delta
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if improved:
            early_stopping_counter = 0
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
            best_val_pos_error = val_stats['pos_error'] if use_pose else None
            best_val_heading_error = val_stats['heading_error'] if (use_pose and include_heading) else None
            best_val_grid_acc = val_stats['grid_accuracy'] if use_grid else None
            best_val_grid_error = val_stats['grid_error'] if use_grid else None
            best_val_grid_weighted_error = val_stats['grid_weighted_error'] if use_grid else None
            torch.save(build_checkpoint(epoch, {
                'loss': best_val_loss, 'accuracy': best_val_acc,
                'pos_error': best_val_pos_error, 'heading_error': best_val_heading_error,
            }), best_model_path)
            if progress_callback:
                progress_callback(epoch + 1, num_epochs,
                                  f"エポック {epoch+1}/{num_epochs}: 新しい最良モデルを保存しました"
                                  f"（損失: {best_val_loss:.6f}）")
        elif use_class and val_accuracy > best_val_acc:
            # 検証精度のみ改善した場合も保存
            best_val_acc = val_accuracy
            early_stopping_counter = 0
            torch.save(build_checkpoint(epoch, {
                'loss': val_loss, 'accuracy': best_val_acc,
                'pos_error': val_stats['pos_error'] if use_pose else None,
                'heading_error': val_stats['heading_error'] if (use_pose and include_heading) else None,
            }), best_model_path)
            if progress_callback:
                progress_callback(epoch + 1, num_epochs,
                                  f"エポック {epoch+1}/{num_epochs}: 新しい最良精度を保存しました"
                                  f"（精度: {best_val_acc:.2f}%, 損失: {val_loss:.6f}）")
        else:
            if use_early_stopping:
                early_stopping_counter += 1
                if progress_callback:
                    progress_callback(epoch + 1, num_epochs,
                                      f"エポック {epoch+1}/{num_epochs}: 検証損失が改善しませんでした"
                                      f"（カウンタ: {early_stopping_counter}/{patience}）")
                if early_stopping_counter >= patience:
                    if progress_callback:
                        progress_callback(epoch + 1, num_epochs,
                                          f"エポック {epoch+1}/{num_epochs}: Early Stoppingにより"
                                          f"トレーニングを終了します")
                    early_stopped = True
                    stopped_epoch = epoch + 1
                    break

    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    training_results = {
        'model_name': model_name,
        'output_mode': output_mode,
        'num_sources': num_sources,
        'fusion_method': fusion_method if num_sources > 1 else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_pos_errors': train_pos_errors,
        'val_pos_errors': val_pos_errors,
        'train_heading_errors': train_heading_errors,
        'val_heading_errors': val_heading_errors,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_val_pos_error': best_val_pos_error,
        'best_val_heading_error': best_val_heading_error,
        'train_grid_accuracies': train_grid_accuracies,
        'val_grid_accuracies': val_grid_accuracies,
        'train_grid_errors': train_grid_errors,
        'val_grid_errors': val_grid_errors,
        'val_grid_weighted_errors': val_grid_weighted_errors,
        'best_val_grid_acc': best_val_grid_acc,
        'best_val_grid_error': best_val_grid_error,
        'best_val_grid_weighted_error': best_val_grid_weighted_error,
        'grid_config': grid_config if use_grid else None,
        'grid_top_n': grid_top_n,
        'model_path': None,
        'best_model_path': best_model_path if os.path.exists(best_model_path) else None,
        'num_epochs': num_epochs,
        'completed_epochs': completed_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'pretrained': pretrained,
        'loaded_weights': loaded_weights,
        'early_stopped': early_stopped,
        'stopped_epoch': stopped_epoch if early_stopped else completed_epochs,
        'patience': patience if use_early_stopping else 0,
        'num_classes': num_classes,
        'pose_norm': pose_norm if use_pose else None,
        'total_training_time': total_training_time,
        'avg_epoch_time': avg_epoch_time,
        'epoch_times': epoch_times,
        'cancelled': cancelled,
    }

    if cancelled:
        print("学習がキャンセルされました")
        training_results['early_stopped'] = False
        training_results['stopped_epoch'] = completed_epochs
        return training_results

    # 最終モデルの保存
    torch.save(build_checkpoint(completed_epochs, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_pos_errors': train_pos_errors,
        'val_pos_errors': val_pos_errors,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'early_stopped': early_stopped,
        'stopped_epoch': stopped_epoch if early_stopped else completed_epochs,
    }), final_model_path)
    training_results['model_path'] = final_model_path
    training_results['best_model_path'] = best_model_path

    # トレーニング結果の可視化
    if save_plot:
        try:
            plot_location_training_results(training_results, save_dir, timestamp)
        except Exception as e:
            print(f"学習曲線の保存に失敗しました: {e}")

    return training_results


def plot_location_training_results(results, save_dir, timestamp):
    """位置推論モデルのトレーニング結果をプロットする

    損失に加え、クラス分類なら精度、座標・姿勢回帰なら位置誤差[m]（と方位誤差[deg]）
    をプロットする。

    Args:
        results: トレーニング結果の辞書
        save_dir: プロット保存ディレクトリ
        timestamp: タイムスタンプ
    """
    output_mode = results.get('output_mode', 'class')
    heads = str(output_mode).split('_')
    use_class = 'class' in heads
    use_pose = 'pose' in heads
    use_grid = 'grid' in heads and bool(results.get('val_grid_accuracies'))
    has_heading = use_pose and any(v > 0 for v in results.get('val_heading_errors', []))

    panels = 1 + int(use_class) + int(use_pose) + int(has_heading) + 2 * int(use_grid)
    fig, axes = plt.subplots(panels, 1, figsize=(10, 4.5 * panels))
    if panels == 1:
        axes = [axes]
    ax_iter = iter(axes)

    # 損失のプロット
    ax = next(ax_iter)
    ax.plot(results['train_losses'], label='Training Loss')
    ax.plot(results['val_losses'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f"Training Losses: {results['model_name']}")
    ax.legend()
    ax.grid(True)

    # 精度のプロット
    if use_class:
        ax = next(ax_iter)
        ax.plot(results['train_accuracies'], label='Training Accuracy')
        ax.plot(results['val_accuracies'], label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f"Training Accuracies: {results['model_name']}")
        ax.legend()
        ax.grid(True)

    # 位置誤差のプロット
    if use_pose:
        ax = next(ax_iter)
        ax.plot(results.get('train_pos_errors', []), label='Training Position Error')
        ax.plot(results.get('val_pos_errors', []), label='Validation Position Error')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Position Error (m)')
        ax.set_title(f"Position Error: {results['model_name']}")
        ax.legend()
        ax.grid(True)

    if has_heading:
        ax = next(ax_iter)
        ax.plot(results.get('train_heading_errors', []), label='Training Heading Error')
        ax.plot(results.get('val_heading_errors', []), label='Validation Heading Error')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Heading Error (deg)')
        ax.set_title(f"Heading Error: {results['model_name']}")
        ax.legend()
        ax.grid(True)

    # 格子分類: セル精度と、Top1 セル中心 / Top-N 重み付き座標の位置誤差
    if use_grid:
        ax = next(ax_iter)
        ax.plot(results.get('train_grid_accuracies', []), label='Training Grid Accuracy')
        ax.plot(results.get('val_grid_accuracies', []), label='Validation Grid Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Grid Cell Accuracy (%)')
        ax.set_title(f"Grid Cell Accuracy: {results['model_name']}")
        ax.legend()
        ax.grid(True)

        ax = next(ax_iter)
        top_n = results.get('grid_top_n', 3)
        ax.plot(results.get('train_grid_errors', []), label='Training Top1 Cell Error')
        ax.plot(results.get('val_grid_errors', []), label='Validation Top1 Cell Error')
        ax.plot(results.get('val_grid_weighted_errors', []), '--',
                label=f'Validation Top{top_n} Weighted Error')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Position Error (m)')
        ax.set_title(f"Grid Position Error: {results['model_name']}")
        ax.legend()
        ax.grid(True)

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
# 時系列モデルの学習
# =========================================================================

def train_sequence_model(
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
    """時系列モデルを学習する（SequenceTrainingManagerのラッパー）

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
    manager = SequenceTrainingManager(models_dir, mlflow_manager)
    return manager.train(
        valid_indexes=valid_indexes,
        annotations=annotations,
        images=images,
        source_images_map=source_images_map,
        selected_sources=selected_sources,
        config=config,
        progress_callback=progress_callback,
    )


def predict_sequence_model(
    model_path,
    valid_indexes,
    annotations,
    images,
    source_images_map,
    models_dir='./saved_models',
    progress_callback=None,
):
    """学習済み時系列モデルで予測を実行する（SequenceTrainingManagerのラッパー）

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
    manager = SequenceTrainingManager(models_dir)
    return manager.predict(
        model_path=model_path,
        valid_indexes=valid_indexes,
        annotations=annotations,
        images=images,
        source_images_map=source_images_map,
        progress_callback=progress_callback,
    )


def load_sequence_model(model_path, device=None):
    """保存済み時系列モデルをロードする（後方互換対応）

    Args:
        model_path: モデルファイルパス
        device: 使用するデバイス

    Returns:
        tuple — (model, config_dict, selected_sources)
    """
    return SequenceTrainingManager.load_model(model_path, device)