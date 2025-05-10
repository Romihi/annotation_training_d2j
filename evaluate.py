#!/usr/bin/env python3
"""
evaluate.py - マルチ画像入力モデルの評価スクリプト
"""
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn import MSELoss, L1Loss
from tqdm import tqdm
import time


# 既存のコード
from model_catalog import MODEL_REGISTRY, get_model
# from multi_image_models import visualize_multi_image_predictions


# マルチ画像モデル拡張
from multi_image_models import (
    register_multi_image_models,
    create_multi_image_datasets,
    run_model_with_multi_input_support,
    MultiImageDataset,
    collate_multi_images
)

# トレーニング用のデータロード関数を再利用
from train import load_catalog_data, load_multi_camera_catalog

def visualize_multi_image_predictions(
    model,
    test_loader,
    model_path=None,
    num_samples=3,
    device=None,
    save_path=None
):
    """マルチ画像モデルの予測を視覚化"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルをロード
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"モデルをロードしました: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    # データを取得
    data_iter = iter(test_loader)
    batch = next(data_iter)
    images, targets = batch
    
    # サンプル数を調整
    num_samples = min(num_samples, len(targets))
    
    # マルチ画像モデルの場合、入力画像の数を取得
    num_inputs = len(images[0]) if isinstance(images, list) and isinstance(images[0], list) else 1
    
    # 予測を実行
    with torch.no_grad():
        if num_inputs > 1:
            # サンプル数を制限
            batch_images = images[:num_samples]
            batch_targets = targets[:num_samples].to(device)
            
            # バッチのサブセットを作成し、run_model_with_multi_input_supportを使用
            outputs = run_model_with_multi_input_support(model, batch_images, device).cpu().numpy()
        else:
            # 単一画像モデルの場合
            batch_images = images[:num_samples].to(device)
            batch_targets = targets[:num_samples].to(device)
            
            # 通常の推論
            outputs = model(batch_images).cpu().numpy()
    
    # ターゲットをNumPy配列に変換
    targets_np = batch_targets.cpu().numpy()
    
    # 予測の視覚化
    fig, axs = plt.subplots(num_samples, num_inputs + 1, figsize=(4 * (num_inputs + 1), 3 * num_samples))
    
    # 単一サンプルの場合、axsを2次元にリシェイプ
    if num_samples == 1:
        axs = np.array([axs])
    
    for i in range(num_samples):
        # 各入力画像を表示
        for j in range(num_inputs):
            if num_inputs > 1:
                # マルチ画像の場合
                img = images[i][j].numpy().transpose(1, 2, 0)
            else:
                # 単一画像の場合
                img = images[i].numpy().transpose(1, 2, 0)
            
            # 正規化を元に戻す
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # 画像を表示
            axs[i, j].imshow(img)
            axs[i, j].set_title(f"入力画像 {j+1}")
            axs[i, j].axis('off')
        
        # ステアリングホイールの描画
        axs[i, num_inputs].set_xlim(-1.2, 1.2)
        axs[i, num_inputs].set_ylim(-1.2, 1.2)
        
        # 円の描画
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        axs[i, num_inputs].add_patch(circle)
        
        # 予測角度の描画（-1~1に変換）
        pred_angle = (outputs[i, 0] * 2 - 1) * np.pi/4
        target_angle = (targets_np[i, 0] * 2 - 1) * np.pi/4
        
        # スロットル情報をテキストで表示（-1~1に変換）
        pred_throttle = outputs[i, 1] * 2 - 1
        target_throttle = targets_np[i, 0] * 2 - 1
        
        # 予測角度の矢印
        axs[i, num_inputs].arrow(0, 0, np.cos(pred_angle), np.sin(pred_angle), 
                               head_width=0.1, head_length=0.1, fc='red', ec='red', label='予測')
        
        # ターゲット角度の矢印
        axs[i, num_inputs].arrow(0, 0, np.cos(target_angle), np.sin(target_angle), 
                               head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='実際')
        
        # スロットル情報をテキストで表示
        axs[i, num_inputs].text(-1.1, -1.1, f'予測スロットル: {pred_throttle:.2f}', color='red')
        axs[i, num_inputs].text(-1.1, -1.0, f'実際スロットル: {target_throttle:.2f}', color='blue')
        
        # 凡例を表示
        if i == 0:
            axs[i, num_inputs].legend()
        
        axs[i, num_inputs].set_title(f'ステアリング予測')
        axs[i, num_inputs].axis('equal')
    
    plt.tight_layout()
    
    # 保存パスが指定されていない場合、モデル名を使用
    if save_path is None:
        save_path = f'{model.name}_multi_predictions.png'
    
    plt.savefig(save_path)
    plt.close()
    print(f'予測の可視化を保存しました: {save_path}')

def test_inference_speed(model, test_loader, model_path=None, num_iterations=100, device=None):
    """モデルの推論速度をテストする"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルをロード
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # テスト用データバッチを取得
    data_iter = iter(test_loader)
    batch = next(data_iter)
    images, _ = batch
    
    # 推論速度の計測
    import time
    total_time = 0.0
    
    print(f"推論速度テスト: {num_iterations}回の反復...")
    with torch.no_grad():
        for i in tqdm(range(num_iterations)):
            # ウォームアップ（最初の数回は除外）
            if i >= 10:  
                start_time = time.time()
                outputs = run_model_with_multi_input_support(model, images, device)
                total_time += time.time() - start_time
            else:
                outputs = run_model_with_multi_input_support(model, images, device)
    
    # 平均推論時間を計算
    avg_time = total_time / (num_iterations - 10)
    fps = 1.0 / avg_time * images[0].size(0) if isinstance(images, list) else 1.0 / avg_time * images.size(0)
    
    print(f"推論速度結果:")
    print(f"平均バッチ推論時間: {avg_time*1000:.2f} ms")
    print(f"1サンプルあたりの平均推論時間: {avg_time*1000/images[0].size(0) if isinstance(images, list) else avg_time*1000/images.size(0):.2f} ms")
    print(f"推論速度: {fps:.2f} FPS")
    
    return {
        'avg_batch_time_ms': avg_time * 1000,
        'avg_sample_time_ms': avg_time * 1000 / (images[0].size(0) if isinstance(images, list) else images.size(0)),
        'fps': fps
    }

# def evaluate_loaded_model(model, test_loader, device):
#     """既にロードされたモデルを評価する関数"""
#     model.eval()
    
#     # 評価指標
#     mse_loss = 0.0
#     mae_loss = 0.0
#     angle_errors = []
#     throttle_errors = []
    
#     criterion_mse = MSELoss()
#     criterion_mae = L1Loss()
    
#     # 推論時間の計測
#     start_time = time.time()
    
#     with torch.no_grad():
#         for inputs, targets in tqdm(test_loader, desc='Evaluating'):
#             # マルチ画像入力対応の実行関数を使用
#             outputs = run_model_with_multi_input_support(model, inputs, device)
#             targets = targets.to(device)
            
#             # 損失の計算
#             mse_loss += criterion_mse(outputs, targets).item() * targets.size(0)
#             mae_loss += criterion_mae(outputs, targets).item() * targets.size(0)
            
#             # 角度と速度の誤差を個別に記録
#             for i in range(outputs.size(0)):
#                 # 0~1の予測値を-1~1に戻して比較
#                 pred_angle = outputs[i, 0].item() * 2 - 1
#                 pred_throttle = outputs[i, 1].item() * 2 - 1
#                 target_angle = targets[i, 0].item() * 2 - 1
#                 target_throttle = targets[i, 1].item() * 2 - 1
                
#                 angle_errors.append(abs(pred_angle - target_angle))
#                 throttle_errors.append(abs(pred_throttle - target_throttle))
    
#     # 平均誤差の計算
#     num_samples = len(test_loader.dataset)
#     avg_mse = mse_loss / num_samples
#     avg_mae = mae_loss / num_samples
#     avg_angle_error = sum(angle_errors) / len(angle_errors)
#     avg_throttle_error = sum(throttle_errors) / len(throttle_errors)
    
#     # 推論時間（バッチ処理全体の時間）
#     inference_time = time.time() - start_time
#     avg_inference_time_per_sample = inference_time / num_samples
    
#     # 評価結果
#     eval_results = {
#         'model_name': args.model,
#         'mse': avg_mse,
#         'mae': avg_mae,
#         'angle_error': avg_angle_error,
#         'throttle_error': avg_throttle_error,
#         'inference_time': inference_time,
#         'inference_time_per_sample': avg_inference_time_per_sample,
#         'num_samples': num_samples
#     }
    
#     return eval_results

# def evaluate_loaded_model(model, test_loader, device):
#     """既にロードされたモデルを評価する関数"""
#     model.eval()
    
#     # データローダーが適切に機能しているか確認
#     num_samples = len(test_loader.dataset)
#     print(f"データセットサイズ: {num_samples}")
#     if num_samples == 0:
#         print("警告: データセットが空です。データのロードに問題がある可能性があります。")
#         return {
#             'model_name': args.model,
#             'error': 'データセットが空です'
#         }
    
#     print(f"データローダーのバッチ数: {len(test_loader)}")
    
#     # 最初のバッチをテスト
#     try:
#         data_iter = iter(test_loader)
#         first_batch = next(data_iter)
#         inputs, targets = first_batch
#         print(f"最初のバッチの入力形式: {type(inputs)}, ターゲット形式: {type(targets)}")
#         if isinstance(inputs, list):
#             print(f"入力画像数: {len(inputs[0])}, バッチサイズ: {len(inputs)}")
#         else:
#             print(f"入力形式: {inputs.shape}") 
#         print(f"ターゲット形式: {targets.shape}")
#     except Exception as e:
#         print(f"データローダーの最初のバッチ取得時にエラーが発生しました: {e}")
#         return {
#             'model_name': args.model,
#             'error': f'データローダーエラー: {e}'
#         }
#         # 評価指標
#     mse_loss = 0.0
#     mae_loss = 0.0
#     angle_errors = []
#     throttle_errors = []
    
#     criterion_mse = MSELoss()
#     criterion_mae = L1Loss()
    
#     # 推論時間の計測
#     start_time = time.time()
    
#     with torch.no_grad():
#         for inputs, targets in tqdm(test_loader, desc='Evaluating'):
#             # マルチ画像入力対応の実行関数を使用
#             outputs = run_model_with_multi_input_support(model, inputs, device)
#             targets = targets.to(device)
            
#             # 損失の計算
#             mse_loss += criterion_mse(outputs, targets).item() * targets.size(0)
#             mae_loss += criterion_mae(outputs, targets).item() * targets.size(0)
            
#             # 角度と速度の誤差を個別に記録
#             for i in range(outputs.size(0)):
#                 # 0~1の予測値を-1~1に戻して比較
#                 pred_angle = outputs[i, 0].item() * 2 - 1
#                 pred_throttle = outputs[i, 1].item() * 2 - 1
#                 target_angle = targets[i, 0].item() * 2 - 1
#                 target_throttle = targets[i, 1].item() * 2 - 1
                
#                 angle_errors.append(abs(pred_angle - target_angle))
#                 throttle_errors.append(abs(pred_throttle - target_throttle))
    
#     # 平均誤差の計算
#     num_samples = len(test_loader.dataset)
#     avg_mse = mse_loss / num_samples
#     avg_mae = mae_loss / num_samples
#     avg_angle_error = sum(angle_errors) / len(angle_errors)
#     avg_throttle_error = sum(throttle_errors) / len(throttle_errors)
    
#     # 推論時間（バッチ処理全体の時間）
#     inference_time = time.time() - start_time
#     avg_inference_time_per_sample = inference_time / num_samples
    
#     # 評価結果
#     eval_results = {
#         'model_name': args.model,
#         'mse': avg_mse,
#         'mae': avg_mae,
#         'angle_error': avg_angle_error,
#         'throttle_error': avg_throttle_error,
#         'inference_time': inference_time,
#         'inference_time_per_sample': avg_inference_time_per_sample,
#         'num_samples': num_samples
#     }
    
#     return eval_results

def evaluate_loaded_model(model, test_loader, device, model_name):
    """既にロードされたモデルを評価する関数"""
    model.eval()
    
    # データローダーが適切に機能しているか確認
    num_samples = len(test_loader.dataset)
    print(f"データセットサイズ: {num_samples}")
    if num_samples == 0:
        print("警告: データセットが空です。データのロードに問題がある可能性があります。")
        return {
            'model_name': model_name,
            'error': 'データセットが空です'
        }
    
    print(f"データローダーのバッチ数: {len(test_loader)}")
    
    # 最初のバッチをテスト
    try:
        data_iter = iter(test_loader)
        first_batch = next(data_iter)
        inputs, targets = first_batch
        print(f"最初のバッチの入力形式: {type(inputs)}, ターゲット形式: {type(targets)}")
        if isinstance(inputs, list):
            print(f"入力画像数: {len(inputs[0])}, バッチサイズ: {len(inputs)}")
        else:
            print(f"入力形式: {inputs.shape}") 
        print(f"ターゲット形式: {targets.shape}")
    except Exception as e:
        print(f"データローダーの最初のバッチ取得時にエラーが発生しました: {e}")
        return {
            'model_name': model_name,
            'error': f'データローダーエラー: {e}'
        }
    
    # 評価指標
    mse_loss = 0.0
    mae_loss = 0.0
    angle_errors = []
    throttle_errors = []
    
    criterion_mse = MSELoss()
    criterion_mae = L1Loss()
    
    # 推論時間の計測
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            # マルチ画像入力対応の実行関数を使用
            outputs = run_model_with_multi_input_support(model, inputs, device)
            targets = targets.to(device)
            
            # 損失の計算
            mse_loss += criterion_mse(outputs, targets).item() * targets.size(0)
            mae_loss += criterion_mae(outputs, targets).item() * targets.size(0)
            
            # 角度と速度の誤差を個別に記録
            for i in range(outputs.size(0)):
                # 0~1の予測値を-1~1に戻して比較
                pred_angle = outputs[i, 0].item() * 2 - 1
                pred_throttle = outputs[i, 1].item() * 2 - 1
                target_angle = targets[i, 0].item() * 2 - 1
                target_throttle = targets[i, 1].item() * 2 - 1
                
                angle_errors.append(abs(pred_angle - target_angle))
                throttle_errors.append(abs(pred_throttle - target_throttle))
    
    # 平均誤差の計算
    avg_mse = mse_loss / num_samples if num_samples > 0 else 0
    avg_mae = mae_loss / num_samples if num_samples > 0 else 0
    avg_angle_error = sum(angle_errors) / len(angle_errors) if angle_errors else 0
    avg_throttle_error = sum(throttle_errors) / len(throttle_errors) if throttle_errors else 0
    
    # 推論時間（バッチ処理全体の時間）
    inference_time = time.time() - start_time
    avg_inference_time_per_sample = inference_time / num_samples if num_samples > 0 else 0
    
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
    
    return eval_results

def main():
    parser = argparse.ArgumentParser(description='マルチ画像入力モデルの評価')
    parser.add_argument('--catalog_file', type=str, required=True, 
                        help='カタログファイルのパス')
    parser.add_argument('--model', type=str, default='donkey_multi3_concat',
                        help='モデルタイプ')
    parser.add_argument('--model_path', type=str, required=True,
                        help='評価するモデルのパス')
    parser.add_argument('--num_inputs', type=int, default=None,
                        help='入力画像数 (Noneの場合は自動検出)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='バッチサイズ')
    parser.add_argument('--visualize', action='store_true',
                        help='予測を視覚化する')
    parser.add_argument('--visualize_samples', type=int, default=3,
                        help='視覚化するサンプル数')
    parser.add_argument('--use_lidar', action='store_true',
                        help='LiDAR画像を使用する')
    parser.add_argument('--use_multi_camera', action='store_true',
                        help='複数カメラの画像を使用する')
    parser.add_argument('--speed_test', action='store_true',
                        help='推論速度テストを実行する')
    parser.add_argument('--speed_test_iterations', type=int, default=100,
                        help='推論速度テストの反復回数')
    
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
    _, test_loader, dataset_info = create_multi_image_datasets(
        image_paths=image_paths,
        annotations=annotations,
        model_name=args.model,
        num_inputs=args.num_inputs,
        val_split=0.2, 
        batch_size=args.batch_size,
        num_workers=4,
        use_augmentation=False  # 評価時はデータ拡張を使用しない
    )
    
    print(f"データセット情報:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # モデルの作成
    model = get_model(args.model, pretrained=False)
    
    # checkpoint形式からロード
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"チェックポイント形式からモデルをロードしました: {args.model_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"state_dict形式からモデルをロードしました: {args.model_path}")

    model = model.to(device)


    # モデルの評価
    print(f"モデル '{args.model}' の評価を開始...")
    #eval_results = evaluate_loaded_model(model, test_loader, device)
    eval_results = evaluate_loaded_model(model, test_loader, device, args.model)

    # eval_results = evaluate_model(
    #     model_name=args.model,
    #     test_loader=test_loader,
    #     model_path=args.model_path,
    #     device=device
    # )
    
    # 評価結果の表示
    print("\n評価結果:")
    print(f"モデル名: {eval_results['model_name']}")
    print(f"MSE: {eval_results['mse']:.6f}")
    print(f"MAE: {eval_results['mae']:.6f}")
    print(f"平均角度誤差: {eval_results['angle_error']:.6f}")
    print(f"平均スロットル誤差: {eval_results['throttle_error']:.6f}")
    print(f"推論時間: {eval_results['inference_time']:.4f} 秒")
    print(f"サンプルあたりの平均推論時間: {eval_results['inference_time_per_sample']*1000:.4f} ms")
    
    # 予測の視覚化
    if args.visualize:
        print("\n予測の視覚化...")
        visualize_multi_image_predictions(
            model=model,
            test_loader=test_loader,
            model_path=args.model_path,
            num_samples=args.visualize_samples,
            device=device,
            save_path=f"{args.model}_predictions.png"
        )
    
    # 推論速度テスト
    if args.speed_test:
        print("\n推論速度テスト...")
        speed_results = test_inference_speed(
            model=model,
            test_loader=test_loader,
            model_path=args.model_path,
            num_iterations=args.speed_test_iterations,
            device=device
        )
    
    print("\n評価が完了しました！")

if __name__ == "__main__":
    main()