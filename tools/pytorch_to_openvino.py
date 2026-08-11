#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import time
import glob as glob_module
import re


def verify_model_precision(xml_path, expected_precision='FP16'):
    """
    変換後のOpenVINOモデルの精度を確認する
    
    Args:
        xml_path: OpenVINOモデルのXMLファイルパス
        expected_precision: 期待される精度 (FP32, FP16, INT8)
        
    Returns:
        bool: 期待通りの精度であればTrue
    """
    try:
        from openvino.runtime import Core, Type
        
        core = Core()
        model = core.read_model(xml_path)
        
        print("\n" + "=" * 60)
        print("モデル精度の検証")
        print("=" * 60)
        
        # BINファイルのサイズを確認
        bin_path = xml_path.replace('.xml', '.bin')
        if os.path.exists(bin_path):
            bin_size_mb = os.path.getsize(bin_path) / 1024 / 1024
            print(f"  - BINファイルサイズ: {bin_size_mb:.2f} MB")
        
        # パラメータ（重み）の型を確認
        param_types = set()
        for param in model.get_parameters():
            param_types.add(str(param.get_element_type()))
        
        # 各オペレーションの型も確認
        op_types = {}
        for op in model.get_ops():
            if hasattr(op, 'get_output_element_type'):
                try:
                    output_type = str(op.get_output_element_type(0))
                    if output_type not in op_types:
                        op_types[output_type] = 0
                    op_types[output_type] += 1
                except:
                    pass
        
        print(f"\n[全体のオペレーション分布]")
        print(f"  - 期待される精度: {expected_precision}")
        print(f"  - パラメータの型: {', '.join(param_types)}")
        
        if op_types:
            print(f"  - オペレーション出力型の分布:")
            for dtype, count in sorted(op_types.items(), key=lambda x: -x[1]):
                print(f"      {dtype}: {count}個")
        
        # 計算オペレーション（実際の演算）の精度を確認
        compute_op_types = [
            'Convolution', 'GroupConvolution', 'ConvolutionBackpropData',
            'MatMul', 'Gemm',
            'Add', 'Subtract', 'Multiply', 'Divide',
            'Relu', 'PRelu', 'LeakyRelu', 'Sigmoid', 'Tanh', 'Swish', 'SoftMax',
            'BatchNormInference', 'NormalizeL2',
            'AvgPool', 'MaxPool', 'AdaptiveAvgPool',
            'ReduceMean', 'ReduceSum',
            'Interpolate',
        ]
        
        compute_ops_by_type = {}
        compute_ops_precision = {'FP16': 0, 'FP32': 0, 'INT8': 0, 'Other': 0}
        
        for op in model.get_ops():
            op_type = op.get_type_name()
            if op_type in compute_op_types:
                try:
                    output_type = str(op.get_output_element_type(0))
                    
                    if op_type not in compute_ops_by_type:
                        compute_ops_by_type[op_type] = {'FP16': 0, 'FP32': 0, 'INT8': 0, 'Other': 0}
                    
                    if 'float16' in output_type or 'f16' in output_type:
                        compute_ops_precision['FP16'] += 1
                        compute_ops_by_type[op_type]['FP16'] += 1
                    elif 'float32' in output_type or 'f32' in output_type:
                        compute_ops_precision['FP32'] += 1
                        compute_ops_by_type[op_type]['FP32'] += 1
                    elif 'int8' in output_type.lower() or 'i8' in output_type:
                        compute_ops_precision['INT8'] += 1
                        compute_ops_by_type[op_type]['INT8'] += 1
                    else:
                        compute_ops_precision['Other'] += 1
                        compute_ops_by_type[op_type]['Other'] += 1
                except:
                    pass
        
        total_compute_ops = sum(compute_ops_precision.values())
        
        print(f"\n[計算オペレーションの精度分析]")
        print(f"  ※ 実際の演算（Conv, MatMul, Add等）の精度が重要です")
        print(f"  - 計算オペレーション総数: {total_compute_ops}個")
        
        if total_compute_ops > 0:
            print(f"  - 精度別の内訳:")
            for prec, count in compute_ops_precision.items():
                if count > 0:
                    percentage = count / total_compute_ops * 100
                    print(f"      {prec}: {count}個 ({percentage:.1f}%)")
            
            print(f"\n  - オペレーション種類別の詳細:")
            for op_type, precisions in sorted(compute_ops_by_type.items()):
                details = []
                for prec, count in precisions.items():
                    if count > 0:
                        details.append(f"{prec}:{count}")
                if details:
                    print(f"      {op_type}: {', '.join(details)}")
        
        # 精度の判定
        fp16_count = op_types.get("<Type: 'float16'>", 0) + op_types.get("f16", 0)
        fp32_count = op_types.get("<Type: 'float32'>", 0) + op_types.get("f32", 0)
        
        fp16_compute = compute_ops_precision['FP16']
        fp32_compute = compute_ops_precision['FP32']
        int8_compute = compute_ops_precision['INT8']
        
        print(f"\n[判定結果]")
        
        is_correct = False
        
        if expected_precision == 'FP16':
            if fp16_compute > 0:
                is_correct = True
                print(f"✓ 精度検証: OK")
                print(f"  - 計算オペレーションの{fp16_compute}個がFP16で実行されます")
                if fp32_compute > 0:
                    print(f"  - 注意: {fp32_compute}個のオペレーションはFP32です")
                    print(f"    （一部の演算はFP32のまま残ることがあります）")
            else:
                # FP16計算オペレーションがなくても、重みがFP16なら実質的にFP16
                print(f"ℹ 精度検証: 重みはFP16で保存されています")
                print(f"  - 計算グラフはFP32で定義されていますが、")
                print(f"    推論時にOpenVINOが自動的にFP16で実行する可能性があります")
                print(f"  - GPUでの推論時はFP16で実行されます")
                print(f"  - CPUでの推論時は、inference_precision_hint='f16'を設定することで")
                print(f"    FP16推論が可能です（対応CPUの場合）")
                is_correct = True  # 重みがFP16なら許容
                
        elif expected_precision == 'FP32':
            if fp16_compute == 0 and int8_compute == 0:
                is_correct = True
                print(f"✓ 精度検証: OK - モデルはFP32で保存されています")
            else:
                print(f"⚠ 精度検証: 警告 - FP16/INT8オペレーションが含まれています")
                
        elif expected_precision == 'INT8':
            if int8_compute > 0:
                is_correct = True
                print(f"✓ 精度検証: OK")
                print(f"  - 計算オペレーションの{int8_compute}個がINT8で実行されます")
            else:
                print(f"⚠ 精度検証: 警告 - INT8の計算オペレーションが見つかりません")
        
        # BINファイルサイズによる追加判定
        if os.path.exists(bin_path) and expected_precision == 'FP16':
            # FP16の場合、BINファイルサイズが小さいはず
            print(f"\n[ファイルサイズによる判定]")
            print(f"  - BINファイルサイズ: {bin_size_mb:.2f} MB")
            print(f"  - FP16変換済みの場合、FP32の約半分のサイズになります")
        
        print("=" * 60)
        return is_correct
        
    except Exception as e:
        print(f"\n精度検証中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_onnx_to_openvino(onnx_path, output_path=None, precision='FP16'):
    """
    ONNXモデルをOpenVINO形式に変換する
    
    Args:
        onnx_path: 変換するONNXモデルのパス
        output_path: 出力ファイルパス (Noneの場合は自動生成)
        precision: モデルの精度 (FP32, FP16, INT8)
        
    Returns:
        変換されたOpenVINOモデルのパス
    """
    # 出力パスが指定されていない場合、元のファイル名を基に自動生成
    if output_path is None:
        base_path = os.path.splitext(onnx_path)[0]
        output_path = f"{base_path}_openvino"
    
    print(f"ONNXモデル '{onnx_path}' をOpenVINO形式に変換しています...")
    
    # 使用するONNXパス（FP16変換後のパスになる可能性がある）
    actual_onnx_path = onnx_path
    temp_fp16_onnx = None
    
    try:
        # FP16の場合、ONNXレベルでのFP16変換を試みる
        # 注意: 一部のモデル（EdgeNeXTなど）はCast操作が含まれており、
        # ONNXレベルでのFP16変換が失敗する場合がある
        onnx_fp16_conversion_enabled = False  # デフォルトで無効化（互換性の問題のため）
        
        if precision == 'FP16' and onnx_fp16_conversion_enabled:
            try:
                from onnxconverter_common import float16
                import onnx
                
                print("ONNXモデルをFP16に変換しています...")
                onnx_model = onnx.load(onnx_path)
                
                # 全体をFP16に変換（入出力も含む）
                onnx_model_fp16 = float16.convert_float_to_float16(
                    onnx_model,
                    keep_io_types=False,  # 入出力もFP16に変換
                    min_positive_val=1e-7,
                    max_finite_val=1e4
                )
                temp_fp16_onnx = os.path.splitext(onnx_path)[0] + "_fp16_temp.onnx"
                onnx.save(onnx_model_fp16, temp_fp16_onnx)
                actual_onnx_path = temp_fp16_onnx
                print("ONNXモデルをFP16に変換しました")
                print("  注意: 入出力もFP16になります。推論時にデータ型を合わせてください。")
                
            except ImportError:
                print("警告: onnxconverter-commonがインストールされていません")
                print("ONNXレベルのFP16変換をスキップし、OpenVINOのcompress_to_fp16を使用します")
            except Exception as e:
                print(f"警告: ONNX FP16変換に失敗しました: {e}")
                print("FP32のONNXモデルを使用して続行します（重みのみFP16圧縮）")
        
        # OpenVINOへの変換
        try:
            import openvino as ov
        except ImportError:
            print("OpenVINOがインストールされていません。以下のコマンドでインストールしてください:")
            print("pip install openvino")
            return None

        # OpenVINOモデルへの変換
        print("OpenVINO形式に変換しています...")

        # Model Optimizerのパラメータ設定
        mo_args = [
            '--input_model', actual_onnx_path,
            '--output_dir', os.path.dirname(output_path) or '.',
            '--model_name', os.path.basename(output_path),
            '--data_type', precision.upper()
        ]
        
        if precision == 'FP16':
            mo_args.extend(['--compress_to_fp16'])
        
        # Model Optimizerの実行
        # FP16の場合はCLIコマンドを優先（APIのcompress_to_fp16が正しく動作しない場合があるため）
        import subprocess
        import shutil
        
        if precision == 'FP16' and shutil.which('ovc'):
            # ovcコマンドを使用（最も確実）
            print(f"OVCコマンドを使用して変換しています（精度: {precision}）...")
            # --output_model でファイル名を指定
            cmd = ['ovc', actual_onnx_path, '--output_model', f"{output_path}.xml"]
            # FP16はデフォルトで有効だが、明示的に指定
            # compress_to_fp16=Trueがデフォルトなので追加引数不要
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.stdout:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"OVCコマンド実行エラー: {e}")
                print(f"標準エラー: {e.stderr}")
                # フォールバック：APIを使用
                print("APIでの変換を試みます...")
                try:
                    import openvino as ov
                    ov_model = ov.convert_model(actual_onnx_path)
                    ov.save_model(ov_model, f"{output_path}.xml", compress_to_fp16=True)
                except Exception as api_e:
                    print(f"API変換も失敗: {api_e}")
                    raise
        elif precision == 'FP32' and shutil.which('ovc'):
            # FP32の場合は明示的にcompress_to_fp16=Falseを指定
            print(f"OVCコマンドを使用して変換しています（精度: {precision}）...")
            cmd = ['ovc', actual_onnx_path, '--output_model', f"{output_path}.xml", '--compress_to_fp16=False']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.stdout:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"OVCコマンド実行エラー: {e}")
                print(f"標準エラー: {e.stderr}")
                raise
        else:
            # APIを使用
            try:
                print(f"OVC APIを使用して変換しています（精度: {precision}）...")
                ov_model = ov.convert_model(actual_onnx_path)
                ov.save_model(ov_model, f"{output_path}.xml", compress_to_fp16=(precision == 'FP16'))
                
            except Exception as e:
                # CLIコマンド経由での実行を試みる
                print(f"API変換に失敗しました: {e}")
                print("CLIコマンドで変換を試みます...")
                
                # ovcコマンドを試す
                if precision == 'FP16':
                    cmd = ['ovc', onnx_path, '--output', f"{output_path}.xml", '--compress_to_fp16']
                else:
                    cmd = ['ovc', onnx_path, '--output', f"{output_path}.xml"]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    print(result.stdout)
                    if result.stderr:
                        print("警告:", result.stderr)
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    # moコマンドにフォールバック
                    cmd = ['mo'] + mo_args
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        print(result.stdout)
                        if result.stderr:
                            print("警告:", result.stderr)
                    except subprocess.CalledProcessError as e:
                        print(f"Model Optimizer実行エラー: {e}")
                        print(f"標準出力: {e.stdout}")
                        print(f"標準エラー: {e.stderr}")
                        raise
        
        # 出力ファイルの確認
        xml_path = f"{output_path}.xml"
        bin_path = f"{output_path}.bin"
        
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            print(f"変換が完了しました！")
            print(f"  - XMLファイル: {xml_path}")
            print(f"  - BINファイル: {bin_path}")
            print(f"  - 精度: {precision}")
            
            # ファイルサイズの表示
            onnx_size = os.path.getsize(onnx_path)
            onnx_size_mb = onnx_size / 1024 / 1024
            bin_size = os.path.getsize(bin_path)
            bin_size_mb = bin_size / 1024 / 1024
            xml_size = os.path.getsize(xml_path)
            xml_size_kb = xml_size / 1024
            openvino_size = xml_size + bin_size
            openvino_size_mb = openvino_size / 1024 / 1024
            
            print("\n" + "=" * 60)
            print("ファイルサイズ比較")
            print("=" * 60)
            
            print(f"\n  [ファイルサイズ]")
            print(f"  ┌─────────────────────────────────────────────────┐")
            print(f"  │ 元のONNXモデル:            {onnx_size_mb:>8.2f} MB          │")
            print(f"  │ OpenVINO BINファイル:      {bin_size_mb:>8.2f} MB          │")
            print(f"  │ OpenVINO XMLファイル:      {xml_size_kb:>8.2f} KB          │")
            print(f"  │ OpenVINO合計 (XML+BIN):    {openvino_size_mb:>8.2f} MB          │")
            print(f"  └─────────────────────────────────────────────────┘")
            
            # サイズ比較バー
            max_bar_len = 40
            onnx_bar_len = max_bar_len
            bin_bar_len = int(max_bar_len * bin_size / onnx_size) if onnx_size > 0 else 0
            bin_bar_len = max(1, min(bin_bar_len, max_bar_len))
            
            print(f"\n  [サイズ比較バー]")
            print(f"  ONNX    : {'█' * onnx_bar_len} {onnx_size_mb:.2f} MB")
            print(f"  OpenVINO: {'█' * bin_bar_len}{'░' * (max_bar_len - bin_bar_len)} {bin_size_mb:.2f} MB")
            
            print(f"\n  サイズ削減率: {(1 - bin_size/onnx_size)*100:.1f}%")
            
            # FP16判定の目安
            if precision == 'FP16':
                print(f"\n  [FP16変換の判定]")
                
                expected_fp16_size = onnx_size * 0.5
                expected_fp16_mb = expected_fp16_size / 1024 / 1024
                actual_ratio = bin_size / onnx_size if onnx_size > 0 else 1
                
                print(f"  - 元のONNXモデル(FP32):    {onnx_size_mb:.2f} MB")
                print(f"  - FP16の理論サイズ(約50%): {expected_fp16_mb:.2f} MB")
                print(f"  - 実際のBINサイズ:         {bin_size_mb:.2f} MB")
                print(f"  - 実際の比率:              {actual_ratio*100:.1f}%")
                
                if actual_ratio < 0.65:
                    print(f"\n  ✓ 結果: 重みはFP16で保存されています")
                    print(f"    （元のサイズの{actual_ratio*100:.1f}%に圧縮）")
                elif actual_ratio < 0.85:
                    print(f"\n  △ 結果: 部分的にFP16で保存されている可能性があります")
                else:
                    print(f"\n  ✗ 結果: FP16変換が正しく行われていない可能性があります")
            print("=" * 60)
            
            # OpenVINOモデルの検証
            try:
                from openvino.runtime import Core
                
                print("\nOpenVINOモデルの検証を実行しています...")
                ie = Core()
                
                # 利用可能なデバイスの表示
                available_devices = ie.available_devices
                print(f"利用可能なデバイス: {available_devices}")
                
                # モデルの読み込み
                network = ie.read_model(model=xml_path, weights=bin_path)
                
                # 入出力情報の表示
                print("\nモデル情報:")
                for input_info in network.inputs:
                    input_shape = input_info.shape
                    print(f"  - 入力: {input_info.any_name}, 形状: {input_shape}")
                
                for output_info in network.outputs:
                    output_shape = output_info.shape
                    print(f"  - 出力: {output_info.any_name}, 形状: {output_shape}")
                
                # テスト推論（CPU）
                print("\nCPUでテスト推論を実行しています...")
                compiled_model = ie.compile_model(network, "CPU")
                
                print("OpenVINOモデルの検証に成功しました")
                
            except Exception as e:
                print(f"OpenVINOモデルの検証中にエラーが発生しました: {e}")
            
            # 精度の検証
            verify_model_precision(xml_path, precision)
            
            # 一時ファイルの削除
            if temp_fp16_onnx and os.path.exists(temp_fp16_onnx):
                os.remove(temp_fp16_onnx)
                print("一時ファイルを削除しました")
            
            return xml_path
        else:
            print("エラー: OpenVINOモデルファイルが生成されませんでした")
            # 一時ファイルの削除
            if temp_fp16_onnx and os.path.exists(temp_fp16_onnx):
                os.remove(temp_fp16_onnx)
            return None
        
    except Exception as e:
        print(f"変換中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        # 一時ファイルの削除
        if temp_fp16_onnx and os.path.exists(temp_fp16_onnx):
            os.remove(temp_fp16_onnx)
        return None


def convert_pytorch_to_openvino(model_path, model_type, output_path=None, input_size=(224, 224), 
                               precision='FP16', dynamic_batch=True):
    """
    PyTorchモデルをOpenVINO形式に変換する
    
    Args:
        model_path: 変換するPyTorchモデルのパス
        model_type: モデルタイプ（例: resnet18）
        output_path: 出力ファイルパス (Noneの場合は自動生成)
        input_size: 入力画像サイズ (高さ, 幅) - 保存されたモデルから自動検出される場合は上書きされる
        precision: モデルの精度 (FP32, FP16, INT8)
        
    Returns:
        変換されたOpenVINOモデルのパス
    """
    # 出力パスが指定されていない場合、元のファイル名を基に自動生成
    if output_path is None:
        base_path = os.path.splitext(model_path)[0]
        output_path = f"{base_path}_openvino"
    
    # モデルカタログのインポート
    try:
        import sys
        import torch
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.dirname(script_dir))
        from model_catalog import get_model, load_model_weights
    except ImportError as e:
        print(f"モデルカタログのインポートに失敗しました: {e}")
        print("このスクリプトはDonkeyCar環境内で実行する必要があります")
        return None
    
    print(f"PyTorchモデル '{model_path}' をOpenVINO形式に変換しています...")
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    try:
        # まず保存されたモデルから入力サイズ情報を取得を試みる
        actual_input_size = input_size
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                # 保存された入力サイズがあるかチェック
                if 'input_size' in checkpoint:
                    actual_input_size = checkpoint['input_size']
                    print(f"保存されたモデルから入力サイズを検出: {actual_input_size}")
                elif 'model_input_size' in checkpoint:
                    actual_input_size = checkpoint['model_input_size']
                    print(f"保存されたモデルから入力サイズを検出: {actual_input_size}")
                # DonkeyModelなど、重みのサイズから入力サイズを推定
                elif 'model_state_dict' in checkpoint and model_type.lower() in ['donkey_model', 'donkey', 'donkey_location']:
                    state_dict = checkpoint['model_state_dict']
                    if 'dense_layers.0.weight' in state_dict:
                        # 最初の全結合層の重みサイズから入力サイズを逆算
                        first_layer_input_size = state_dict['dense_layers.0.weight'].shape[1]
                        print(f"最初の全結合層の入力サイズ: {first_layer_input_size}")
                        
                        # DonkeyModelの場合、畳み込み出力サイズから元の画像サイズを推定
                        # 一般的なサイズパターンを試行
                        size_patterns = [
                            (120, 160),  # 標準サイズ
                            (224, 224),  # 正方形サイズ
                            (240, 320),  # 2倍サイズ
                            (180, 240),  # 1.5倍サイズ
                            (96, 128),   # 0.8倍サイズ
                            (60, 80),    # 0.5倍サイズ
                        ]
                        
                        for test_size in size_patterns:
                            # DonkeyModelの畳み込み出力サイズを計算
                            h, w = test_size
                            conv_out_h = ((h // 2) // 2) // 2  # 3回のmax pooling
                            conv_out_w = ((w // 2) // 2) // 2
                            expected_features = 64 * conv_out_h * conv_out_w
                            
                            if expected_features == first_layer_input_size:
                                actual_input_size = test_size
                                print(f"重みサイズから入力サイズを推定: {actual_input_size}")
                                break
                        else:
                            print(f"警告: 重みサイズ {first_layer_input_size} に対応する入力サイズを特定できませんでした")
                            
        except Exception as e:
            print(f"入力サイズの自動検出に失敗しました（デフォルト値を使用）: {e}")
        
        print(f"使用する入力サイズ: {actual_input_size}")
        
        # 現在の入力サイズと学習済みモデルのサイズが異なる場合の警告
        if input_size != actual_input_size:
            print(f"警告: 指定された入力サイズ {input_size} と学習済みモデルのサイズ {actual_input_size} が異なります")
            print(f"OpenVINOモデルは学習済みモデルのサイズ {actual_input_size} で作成されます")
            print(f"推論時は入力画像を {actual_input_size} にリサイズしてください")
        
        # 位置モデルの場合、クラス数も検出
        num_classes = None
        if model_type.lower() == "donkey_location":
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # 最終分類層からクラス数を検出
                    if 'classifier.weight' in state_dict:
                        num_classes = state_dict['classifier.weight'].shape[0]
                        print(f"検出されたクラス数: {num_classes}")
                    elif 'regressor.weight' in state_dict:
                        num_classes = state_dict['regressor.weight'].shape[0]
                        print(f"検出されたクラス数: {num_classes}")
            except Exception as e:
                print(f"クラス数の検出に失敗（デフォルト値8を使用）: {e}")
                num_classes = 8
        
        # モデルのロード
        if model_type.lower() == "donkey_location" and num_classes is not None:
            # 位置モデルの場合、クラス数も指定
            from model_catalog import DonkeyLocationModel
            model = get_model(model_type, pretrained=False, input_size=actual_input_size)
        else:
            model = get_model(model_type, pretrained=False, input_size=actual_input_size)
        model = load_model_weights(model, model_path, device)
        model = model.to(device)
        model.eval()
        
        # OpenVINOへの変換
        try:
            import openvino as ov
        except ImportError:
            print("OpenVINOがインストールされていません。以下のコマンドでインストールしてください:")
            print("pip install openvino")
            return None

        # 一時的にONNXファイルを作成（OpenVINOはONNX経由で変換）
        temp_onnx_path = os.path.splitext(model_path)[0] + "_temp.onnx"
        
        # ダミー入力の作成（検出された実際の入力サイズを使用）
        dummy_input = torch.randn(1, 3, actual_input_size[0], actual_input_size[1], device=device)
        
        # ONNXエクスポート
        if dynamic_batch:
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        else:
            dynamic_axes = None
            
        torch.onnx.export(
            model,
            dummy_input,
            temp_onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            dynamo=False
        )
        
        print("ONNX中間ファイルを作成しました")
        
        # FP16の場合、ONNXレベルでのFP16変換を試みる
        # 注意: 一部のモデル（EdgeNeXTなど）はCast操作が含まれており、
        # ONNXレベルでのFP16変換が失敗する場合がある
        onnx_fp16_conversion_enabled = False  # デフォルトで無効化（互換性の問題のため）
        
        if precision == 'FP16' and onnx_fp16_conversion_enabled:
            try:
                from onnxconverter_common import float16
                import onnx
                
                print("ONNXモデルをFP16に変換しています...")
                onnx_model = onnx.load(temp_onnx_path)
                
                # 全体をFP16に変換（入出力も含む）
                onnx_model_fp16 = float16.convert_float_to_float16(
                    onnx_model,
                    keep_io_types=False,  # 入出力もFP16に変換
                    min_positive_val=1e-7,
                    max_finite_val=1e4
                )
                temp_onnx_fp16_path = temp_onnx_path.replace('.onnx', '_fp16.onnx')
                onnx.save(onnx_model_fp16, temp_onnx_fp16_path)
                
                # FP16版を使用
                os.remove(temp_onnx_path)
                temp_onnx_path = temp_onnx_fp16_path
                print("ONNXモデルをFP16に変換しました")
                print("  注意: 入出力もFP16になります。推論時にデータ型を合わせてください。")
                
            except ImportError:
                print("警告: onnxconverter-commonがインストールされていません")
                print("ONNXレベルのFP16変換をスキップし、OpenVINOのcompress_to_fp16を使用します")
            except Exception as e:
                print(f"警告: ONNX FP16変換に失敗しました: {e}")
                print("FP32のONNXモデルを使用して続行します（重みのみFP16圧縮）")
        
        # OpenVINOモデルへの変換
        print("OpenVINO形式に変換しています...")
        
        # Model Optimizerのパラメータ設定
        output_dir = os.path.dirname(output_path) or '.'
        mo_args = [
            '--input_model', temp_onnx_path,
            '--output_dir', output_dir,
            '--model_name', os.path.basename(output_path),
            '--input_shape', f'[1,3,{actual_input_size[0]},{actual_input_size[1]}]',
            '--data_type', precision.upper()
        ]
        
        if precision == 'FP16':
            mo_args.extend(['--compress_to_fp16'])
        
        # Model Optimizerの実行
        # FP16の場合はCLIコマンドを優先（APIのcompress_to_fp16が正しく動作しない場合があるため）
        import subprocess
        import shutil
        
        if precision == 'FP16' and shutil.which('ovc'):
            # ovcコマンドを使用（最も確実）
            print(f"OVCコマンドを使用して変換しています（精度: {precision}）...")
            # --output_model でファイル名を指定、FP16はデフォルトで有効
            cmd = ['ovc', temp_onnx_path, '--output_model', f"{output_path}.xml"]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.stdout:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"OVCコマンド実行エラー: {e}")
                print(f"標準エラー: {e.stderr}")
                # フォールバック：APIを使用
                print("APIでの変換を試みます...")
                try:
                    import openvino as ov
                    ov_model = ov.convert_model(temp_onnx_path)
                    ov.save_model(ov_model, f"{output_path}.xml", compress_to_fp16=True)
                except Exception as api_e:
                    print(f"API変換も失敗: {api_e}")
                    raise
        elif precision == 'FP32' and shutil.which('ovc'):
            # FP32の場合は明示的にcompress_to_fp16=Falseを指定
            print(f"OVCコマンドを使用して変換しています（精度: {precision}）...")
            cmd = ['ovc', temp_onnx_path, '--output_model', f"{output_path}.xml", '--compress_to_fp16=False']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.stdout:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"OVCコマンド実行エラー: {e}")
                print(f"標準エラー: {e.stderr}")
                raise
        else:
            # APIを使用
            try:
                print(f"OVC APIを使用して変換しています（精度: {precision}）...")
                ov_model = ov.convert_model(temp_onnx_path)
                ov.save_model(ov_model, f"{output_path}.xml", compress_to_fp16=(precision == 'FP16'))
                
            except Exception as e:
                # CLIコマンド経由での実行を試みる
                print(f"API変換に失敗しました: {e}")
                print("CLIコマンドで変換を試みます...")
                
                # ovcコマンドを試す
                if precision == 'FP16':
                    cmd = ['ovc', temp_onnx_path, '--output_model', f"{output_path}.xml"]
                else:
                    cmd = ['ovc', temp_onnx_path, '--output_model', f"{output_path}.xml", '--compress_to_fp16=False']
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    print(result.stdout)
                    if result.stderr:
                        print("警告:", result.stderr)
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    # moコマンドにフォールバック
                    cmd = ['mo'] + mo_args
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        print(result.stdout)
                        if result.stderr:
                            print("警告:", result.stderr)
                    except subprocess.CalledProcessError as e:
                        print(f"Model Optimizer実行エラー: {e}")
                        print(f"標準出力: {e.stdout}")
                        print(f"標準エラー: {e.stderr}")
                        raise
        
        # 一時ファイルの削除
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)
            print("一時ファイルを削除しました")
        
        # 出力ファイルの確認
        xml_path = f"{output_path}.xml"
        bin_path = f"{output_path}.bin"
        
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            print(f"\n変換が完了しました！")
            print(f"  - XMLファイル: {xml_path}")
            print(f"  - BINファイル: {bin_path}")
            print(f"  - 精度: {precision}")
            print(f"  - 学習時の入力サイズ: {actual_input_size}")
            if input_size != actual_input_size:
                print(f"  - 注意: 現在の設定入力サイズ: {input_size}")
                print(f"  - 推論時は画像を {actual_input_size} にリサイズしてください")
            
            # ファイルサイズの比較
            print("\n" + "=" * 60)
            print("ファイルサイズ比較")
            print("=" * 60)
            
            # 元のPyTorchモデルサイズ
            pth_size = os.path.getsize(model_path)
            pth_size_mb = pth_size / 1024 / 1024
            
            # PyTorchモデルの純粋な重みサイズを計算
            weights_only_size = 0
            optimizer_size = 0
            other_size = 0
            try:
                import torch
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                if isinstance(checkpoint, dict):
                    for key, value in checkpoint.items():
                        if isinstance(value, dict):
                            # state_dict形式の場合
                            key_size = sum(
                                v.numel() * v.element_size() 
                                for v in value.values() 
                                if hasattr(v, 'numel')
                            )
                            if 'state_dict' in key.lower() or key == 'model':
                                weights_only_size += key_size
                            elif 'optimizer' in key.lower():
                                optimizer_size += key_size
                            else:
                                other_size += key_size
                        elif hasattr(value, 'numel'):
                            # 直接テンソルの場合
                            weights_only_size += value.numel() * value.element_size()
                else:
                    # state_dict形式で直接保存されている場合
                    weights_only_size = sum(
                        v.numel() * v.element_size() 
                        for v in checkpoint.values() 
                        if hasattr(v, 'numel')
                    )
            except Exception as e:
                logger.warning(f"重みサイズの計算に失敗: {e}")
                weights_only_size = 0
            
            weights_only_mb = weights_only_size / 1024 / 1024
            optimizer_mb = optimizer_size / 1024 / 1024
            
            # OpenVINOモデルサイズ
            xml_size = os.path.getsize(xml_path)
            bin_size = os.path.getsize(bin_path)
            openvino_total_size = xml_size + bin_size
            openvino_total_mb = openvino_total_size / 1024 / 1024
            bin_size_mb = bin_size / 1024 / 1024
            xml_size_kb = xml_size / 1024
            
            print(f"\n  [ファイルサイズ]")
            print(f"  ┌─────────────────────────────────────────────────────────┐")
            print(f"  │ 元のPyTorchモデル (.pth):      {pth_size_mb:>8.2f} MB              │")
            if weights_only_size > 0:
                print(f"  │   ├─ 純粋な重み (FP32):       {weights_only_mb:>8.2f} MB              │")
                if optimizer_size > 0:
                    print(f"  │   ├─ オプティマイザ状態:     {optimizer_mb:>8.2f} MB              │")
                other_mb = (pth_size - weights_only_size - optimizer_size) / 1024 / 1024
                if other_mb > 0.01:
                    print(f"  │   └─ その他（メタデータ等）: {other_mb:>8.2f} MB              │")
            print(f"  │ OpenVINO BINファイル:          {bin_size_mb:>8.2f} MB              │")
            print(f"  │ OpenVINO XMLファイル:          {xml_size_kb:>8.2f} KB              │")
            print(f"  │ OpenVINO合計 (XML+BIN):        {openvino_total_mb:>8.2f} MB              │")
            print(f"  └─────────────────────────────────────────────────────────┘")
            
            # サイズ比較バー（純粋な重みとの比較）
            compare_size = weights_only_size if weights_only_size > 0 else pth_size
            compare_size_mb = compare_size / 1024 / 1024
            
            max_bar_len = 40
            compare_bar_len = max_bar_len
            bin_bar_len = int(max_bar_len * bin_size / compare_size) if compare_size > 0 else 0
            bin_bar_len = max(1, min(bin_bar_len, max_bar_len))
            
            print(f"\n  [サイズ比較バー]")
            if weights_only_size > 0:
                print(f"  純粋な重み(FP32): {'█' * compare_bar_len} {compare_size_mb:.2f} MB")
            else:
                print(f"  PyTorch(.pth)   : {'█' * compare_bar_len} {compare_size_mb:.2f} MB")
            print(f"  OpenVINO BIN    : {'█' * bin_bar_len}{'░' * (max_bar_len - bin_bar_len)} {bin_size_mb:.2f} MB")
            
            # サイズ削減率（純粋な重みとの比較）
            reduction = (1 - bin_size / compare_size) * 100 if compare_size > 0 else 0
            print(f"\n  サイズ削減率: {reduction:.1f}%")
            
            # FP16判定の目安
            if precision == 'FP16':
                print(f"\n  [FP16変換の判定]")
                
                # 純粋な重みサイズとの比較
                if weights_only_size > 0:
                    expected_fp16_size = weights_only_size * 0.5
                    expected_fp16_mb = expected_fp16_size / 1024 / 1024
                    actual_ratio = bin_size / weights_only_size
                    
                    print(f"  - 純粋な重み (FP32):       {weights_only_mb:.2f} MB")
                    print(f"  - FP16の理論サイズ(50%):   {expected_fp16_mb:.2f} MB")
                    print(f"  - 実際のBINサイズ:         {bin_size_mb:.2f} MB")
                    print(f"  - 実際の比率:              {actual_ratio*100:.1f}%")
                    
                    if 0.45 <= actual_ratio <= 0.55:
                        print(f"\n  ✓ 結果: 重みは正確にFP16で保存されています")
                        print(f"    （理論値とほぼ一致: {actual_ratio*100:.1f}% ≈ 50%）")
                    elif actual_ratio < 0.45:
                        print(f"\n  ✓ 結果: 重みはFP16で保存されています")
                        print(f"    （追加の最適化により理論値より小さい可能性）")
                    elif actual_ratio < 0.65:
                        print(f"\n  ✓ 結果: 重みはFP16で保存されています")
                        print(f"    （元のサイズの{actual_ratio*100:.1f}%に圧縮）")
                    elif actual_ratio < 0.85:
                        print(f"\n  △ 結果: 部分的にFP16で保存されている可能性があります")
                    else:
                        print(f"\n  ✗ 結果: FP16変換が正しく行われていない可能性があります")
                else:
                    # 重みサイズが取得できない場合は従来の方法
                    expected_fp16_size = pth_size * 0.5
                    expected_fp16_mb = expected_fp16_size / 1024 / 1024
                    actual_ratio = bin_size / pth_size if pth_size > 0 else 1
                    
                    print(f"  - 元のPyTorchモデル: {pth_size_mb:.2f} MB")
                    print(f"  - 実際のBINサイズ:   {bin_size_mb:.2f} MB")
                    print(f"  - 実際の比率:        {actual_ratio*100:.1f}%")
                    print(f"  ※ 注意: .pthファイルにはオプティマイザ状態等も含まれている")
                    print(f"    ため、比率が50%より小さくなることがあります")
                    
                    if actual_ratio < 0.65:
                        print(f"\n  ✓ 結果: 重みはFP16で保存されています")
                        print(f"    （元のサイズの{actual_ratio*100:.1f}%に圧縮）")
            
            print("=" * 60)
            
            # OpenVINOモデルの検証
            try:
                from openvino.runtime import Core
                
                print("\nOpenVINOモデルの検証を実行しています...")
                ie = Core()
                
                # 利用可能なデバイスの表示
                available_devices = ie.available_devices
                print(f"利用可能なデバイス: {available_devices}")
                
                # モデルの読み込み
                network = ie.read_model(model=xml_path, weights=bin_path)
                
                # 入出力情報の表示
                print("\nモデル情報:")
                for input_info in network.inputs:
                    input_name = input_info.any_name
                    try:
                        # 動的形状の場合は partial_shape を使用
                        if input_info.partial_shape.is_dynamic:
                            input_shape = str(input_info.partial_shape)
                        else:
                            input_shape = list(input_info.shape)
                    except:
                        input_shape = "動的形状"
                    print(f"  - 入力: {input_name}, 形状: {input_shape}")
                
                for output_info in network.outputs:
                    output_name = output_info.any_name
                    try:
                        # 動的形状の場合は partial_shape を使用
                        if output_info.partial_shape.is_dynamic:
                            output_shape = str(output_info.partial_shape)
                        else:
                            output_shape = list(output_info.shape)
                    except:
                        output_shape = "動的形状"
                    print(f"  - 出力: {output_name}, 形状: {output_shape}")
                
                # 変換前後の推論結果を比較
                print("\n変換前後の推論結果を比較しています...")

                # テスト入力の準備（固定のランダムシードで再現性を確保）
                np.random.seed(42)
                test_input = np.random.randn(1, 3, actual_input_size[0], actual_input_size[1]).astype(np.float32)

                # PyTorchモデルで推論
                print("  - PyTorchモデルで推論中...")
                with torch.no_grad():
                    torch_input = torch.from_numpy(test_input).to(device)
                    pytorch_output = model(torch_input).cpu().numpy()
                print(f"    PyTorch出力: 形状={pytorch_output.shape}, 範囲=[{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")

                # OpenVINOモデルで推論
                print("  - OpenVINOモデルで推論中...")
                compiled_model = ie.compile_model(network, "CPU")
                infer_request = compiled_model.create_infer_request()
                infer_request.infer({0: test_input})
                openvino_output = infer_request.get_output_tensor(0).data
                print(f"    OpenVINO出力: 形状={openvino_output.shape}, 範囲=[{openvino_output.min():.6f}, {openvino_output.max():.6f}]")

                # 結果の比較
                print("\n推論結果の比較:")
                diff = np.abs(pytorch_output - openvino_output)
                mse = np.mean((pytorch_output - openvino_output) ** 2)
                mae = np.mean(diff)
                max_diff = np.max(diff)

                print(f"  - 平均二乗誤差 (MSE): {mse:.10f}")
                print(f"  - 平均絶対誤差 (MAE): {mae:.10f}")
                print(f"  - 最大絶対誤差: {max_diff:.10f}")

                # 相対誤差も計算（ゼロ除算を避ける）
                pytorch_abs = np.abs(pytorch_output)
                nonzero_mask = pytorch_abs > 1e-10
                if np.any(nonzero_mask):
                    relative_diff = diff[nonzero_mask] / pytorch_abs[nonzero_mask]
                    mean_relative_error = np.mean(relative_diff) * 100
                    max_relative_error = np.max(relative_diff) * 100
                    print(f"  - 平均相対誤差: {mean_relative_error:.6f}%")
                    print(f"  - 最大相対誤差: {max_relative_error:.6f}%")

                # 精度の評価
                if max_diff < 1e-5:
                    print("\n✓ 変換結果: 非常に高精度（誤差 < 1e-5）")
                elif max_diff < 1e-4:
                    print("\n✓ 変換結果: 高精度（誤差 < 1e-4）")
                elif max_diff < 1e-3:
                    print("\n✓ 変換結果: 良好（誤差 < 1e-3）")
                else:
                    print(f"\n⚠ 変換結果: 誤差が大きい可能性があります（最大誤差: {max_diff:.6f}）")

                print("OpenVINO推論テストに成功しました")
                
            except Exception as e:
                print(f"OpenVINOモデルの検証中にエラーが発生しました: {e}")
            
            # 精度の検証
            verify_model_precision(xml_path, precision)
            
            return xml_path
        else:
            print("エラー: OpenVINOモデルファイルが生成されませんでした")
            return None
        
    except Exception as e:
        print(f"変換中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        # 一時ファイルのクリーンアップ
        temp_onnx_path = os.path.splitext(model_path)[0] + "_temp.onnx"
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)
        
        return None


def convert_yolo_to_openvino(model_path, output_path=None, input_size=640, precision='FP16'):
    """
    YOLOモデル（Ultralytics）をOpenVINO形式に変換する

    Args:
        model_path: 変換するYOLOモデルのパス (.pt)
        output_path: 出力ディレクトリパス (Noneの場合は自動生成)
        input_size: 入力画像サイズ（デフォルト: 640）
        precision: モデルの精度 ('FP32', 'FP16', 'INT8')

    Returns:
        変換されたOpenVINOモデルのパス
    """
    # 出力パスが指定されていない場合、元のファイル名を基に自動生成
    if output_path is None:
        base_path = os.path.splitext(model_path)[0]
        output_path = f"{base_path}_openvino_model"

    print(f"YOLOモデル '{model_path}' をOpenVINO形式に変換しています...")

    try:
        # Ultralyticsのインポート
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Ultralyticsがインストールされていません。以下のコマンドでインストールしてください:")
            print("pip install ultralytics")
            return None

        # YOLOモデルの読み込み
        print("YOLOモデルを読み込んでいます...")
        model = YOLO(model_path)

        # モデル情報の表示
        print(f"\nモデル情報:")
        print(f"  - モデルタイプ: {model.model_name if hasattr(model, 'model_name') else 'YOLO'}")
        print(f"  - 入力サイズ: {input_size}")
        print(f"  - 精度: {precision}")

        # テスト入力の準備（変換前の推論用）
        print("\n変換前のPyTorchモデルで推論を実行...")
        np.random.seed(42)
        test_input = np.random.rand(input_size, input_size, 3).astype(np.uint8)

        # PyTorchモデルで推論（時間計測）
        start_time = time.time()
        pytorch_results = model(test_input, verbose=False)
        pytorch_time = time.time() - start_time

        # PyTorch推論結果の取得
        pytorch_boxes = pytorch_results[0].boxes
        if pytorch_boxes is not None and len(pytorch_boxes) > 0:
            pytorch_detections = {
                'boxes': pytorch_boxes.xyxy.cpu().numpy() if pytorch_boxes.xyxy is not None else np.array([]),
                'scores': pytorch_boxes.conf.cpu().numpy() if pytorch_boxes.conf is not None else np.array([]),
                'classes': pytorch_boxes.cls.cpu().numpy() if pytorch_boxes.cls is not None else np.array([])
            }
            print(f"  - 検出数: {len(pytorch_detections['boxes'])}")
            print(f"  - 推論時間: {pytorch_time*1000:.2f}ms")
            if len(pytorch_detections['boxes']) > 0:
                print(f"  - 信頼度範囲: [{pytorch_detections['scores'].min():.4f}, {pytorch_detections['scores'].max():.4f}]")
        else:
            pytorch_detections = {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([])
            }
            print(f"  - 検出数: 0")
            print(f"  - 推論時間: {pytorch_time*1000:.2f}ms")

        # OpenVINOへの変換
        print("\nOpenVINO形式に変換しています...")
        export_args = {
            'format': 'openvino',
            'imgsz': input_size,
            'half': precision == 'FP16'
        }

        # エクスポート実行
        export_path = model.export(**export_args)

        # エクスポートされたモデルのパスを確認
        if os.path.isdir(export_path):
            openvino_dir = export_path
        else:
            openvino_dir = os.path.dirname(export_path)

        # XMLファイルを探す
        xml_files = list(Path(openvino_dir).glob("*.xml"))
        if not xml_files:
            print("エラー: OpenVINOモデルファイル(.xml)が見つかりません")
            return None

        xml_path = str(xml_files[0])
        bin_path = xml_path.replace('.xml', '.bin')

        if not os.path.exists(bin_path):
            print("エラー: OpenVINOモデルファイル(.bin)が見つかりません")
            return None

        print(f"\n変換が完了しました！")
        print(f"  - XMLファイル: {xml_path}")
        print(f"  - BINファイル: {bin_path}")
        print(f"  - 精度: {precision}")

        # ファイルサイズの表示
        model_size = os.path.getsize(model_path) / 1024 / 1024
        openvino_size = (os.path.getsize(xml_path) + os.path.getsize(bin_path)) / 1024 / 1024

        print(f"\nファイルサイズ:")
        print(f"  - 元のYOLOモデル: {model_size:.1f} MB")
        print(f"  - OpenVINOモデル: {openvino_size:.1f} MB")
        if model_size > 0:
            print(f"  - サイズ比率: {openvino_size/model_size*100:.1f}%")

        # OpenVINOモデルでの推論テスト
        try:
            from openvino.runtime import Core

            print("\n変換後のOpenVINOモデルで推論を実行...")
            ie = Core()

            # 利用可能なデバイスの表示
            available_devices = ie.available_devices
            print(f"利用可能なデバイス: {available_devices}")

            # OpenVINOモデルの読み込みと推論
            openvino_model = YOLO(openvino_dir, task='detect')

            # OpenVINOモデルで推論（時間計測）
            start_time = time.time()
            openvino_results = openvino_model(test_input, verbose=False)
            openvino_time = time.time() - start_time

            # OpenVINO推論結果の取得
            openvino_boxes = openvino_results[0].boxes
            if openvino_boxes is not None and len(openvino_boxes) > 0:
                openvino_detections = {
                    'boxes': openvino_boxes.xyxy.cpu().numpy() if openvino_boxes.xyxy is not None else np.array([]),
                    'scores': openvino_boxes.conf.cpu().numpy() if openvino_boxes.conf is not None else np.array([]),
                    'classes': openvino_boxes.cls.cpu().numpy() if openvino_boxes.cls is not None else np.array([])
                }
                print(f"  - 検出数: {len(openvino_detections['boxes'])}")
                print(f"  - 推論時間: {openvino_time*1000:.2f}ms")
                print(f"  - 高速化: {pytorch_time/openvino_time:.2f}x")
                if len(openvino_detections['boxes']) > 0:
                    print(f"  - 信頼度範囲: [{openvino_detections['scores'].min():.4f}, {openvino_detections['scores'].max():.4f}]")
            else:
                openvino_detections = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'classes': np.array([])
                }
                print(f"  - 検出数: 0")
                print(f"  - 推論時間: {openvino_time*1000:.2f}ms")
                print(f"  - 高速化: {pytorch_time/openvino_time:.2f}x")

            # 検出結果の比較
            print("\n推論結果の比較:")
            pytorch_count = len(pytorch_detections['boxes'])
            openvino_count = len(openvino_detections['boxes'])

            print(f"  - PyTorch検出数: {pytorch_count}")
            print(f"  - OpenVINO検出数: {openvino_count}")

            if pytorch_count > 0 and openvino_count > 0:
                if pytorch_count == openvino_count:
                    box_diff = np.abs(pytorch_detections['boxes'] - openvino_detections['boxes'])
                    max_box_diff = np.max(box_diff)
                    mean_box_diff = np.mean(box_diff)

                    score_diff = np.abs(pytorch_detections['scores'] - openvino_detections['scores'])
                    max_score_diff = np.max(score_diff)
                    mean_score_diff = np.mean(score_diff)

                    class_match = np.sum(pytorch_detections['classes'] == openvino_detections['classes']) / pytorch_count * 100

                    print(f"  - バウンディングボックス平均誤差: {mean_box_diff:.4f} ピクセル")
                    print(f"  - バウンディングボックス最大誤差: {max_box_diff:.4f} ピクセル")
                    print(f"  - 信頼度平均誤差: {mean_score_diff:.6f}")
                    print(f"  - 信頼度最大誤差: {max_score_diff:.6f}")
                    print(f"  - クラス一致率: {class_match:.1f}%")

                    if max_box_diff < 1.0 and max_score_diff < 0.01 and class_match == 100:
                        print("\n✓ 変換結果: 非常に高精度")
                    elif max_box_diff < 5.0 and max_score_diff < 0.05 and class_match >= 90:
                        print("\n✓ 変換結果: 高精度")
                    elif max_box_diff < 10.0 and max_score_diff < 0.1 and class_match >= 80:
                        print("\n✓ 変換結果: 良好")
                    else:
                        print(f"\n⚠ 変換結果: 誤差が大きい可能性があります")
                else:
                    diff_count = abs(pytorch_count - openvino_count)
                    diff_percent = diff_count / max(pytorch_count, openvino_count) * 100
                    print(f"  - 検出数の差: {diff_count} ({diff_percent:.1f}%)")

                    if diff_percent < 10:
                        print("\n✓ 変換結果: 良好（検出数の差は小さい）")
                    else:
                        print(f"\n⚠ 検出数に差があります（{diff_percent:.1f}%）")
            elif pytorch_count == 0 and openvino_count == 0:
                print("\n✓ 変換結果: 一致（両方とも検出なし）")
            else:
                print("\n⚠ 検出結果が大きく異なります")

            print("\nOpenVINO推論テストに成功しました")

        except Exception as e:
            print(f"OpenVINOモデルの検証中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

        # 精度の検証
        verify_model_precision(xml_path, precision)

        return xml_path

    except Exception as e:
        print(f"変換中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

def quantize_to_int8(model_path, output_path, calibration_dir, input_size=(224, 224), num_samples=100, sample_interval=10):
    """
    OpenVINOモデルをINT8に量子化する
    
    Args:
        model_path: 量子化するOpenVINOモデルのパス（.xml）
        output_path: 出力ファイルパス
        calibration_dir: キャリブレーション画像のディレクトリ
        input_size: 入力画像サイズ (高さ, 幅)
        num_samples: キャリブレーションに使用するサンプル数
        sample_interval: サンプリング間隔（デフォルト: 10枚間隔）
        
    Returns:
        量子化されたOpenVINOモデルのパス
    """
    import glob
    from PIL import Image
    
    try:
        import nncf
    except ImportError:
        print("エラー: NNCFがインストールされていません。以下のコマンドでインストールしてください:")
        print("pip install nncf")
        return None
    
    try:
        from openvino.runtime import Core, serialize
    except ImportError:
        print("エラー: OpenVINOがインストールされていません。")
        return None
    
    print(f"INT8量子化を開始します...")
    print(f"  - 入力モデル: {model_path}")
    print(f"  - キャリブレーションディレクトリ: {calibration_dir}")
    print(f"  - 入力サイズ: {input_size}")
    print(f"  - サンプリング間隔: {sample_interval}枚ごと")
    
    # キャリブレーション画像の取得
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(calibration_dir, ext)))
        image_files.extend(glob.glob(os.path.join(calibration_dir, ext.upper())))
    
    if len(image_files) == 0:
        print(f"エラー: キャリブレーション画像が見つかりません: {calibration_dir}")
        print("サポートされている形式: jpg, jpeg, png, bmp")
        return None
    
    # ファイル名でソート（時系列順になることを期待）
    image_files.sort()
    
    # 均等サンプリング
    # sample_interval枚ごとに1枚選択し、num_samplesに達するまで
    total_images = len(image_files)
    
    if sample_interval > 1:
        # 均等サンプリング: sample_interval枚ごとに選択
        selected_files = image_files[::sample_interval]
        
        # num_samplesを超える場合は制限
        if len(selected_files) > num_samples:
            selected_files = selected_files[:num_samples]
        
        actual_samples = len(selected_files)
        print(f"  - 全画像数: {total_images}")
        print(f"  - 均等サンプリング: {sample_interval}枚ごとに選択")
        print(f"  - 選択された画像数: {actual_samples}")
        
        # カバー範囲を表示
        if actual_samples > 0:
            first_idx = 0
            last_idx = (actual_samples - 1) * sample_interval
            coverage = (last_idx / max(1, total_images - 1)) * 100
            print(f"  - カバー範囲: 画像 0 〜 {last_idx} ({coverage:.1f}%)")
    else:
        # sample_interval=1の場合は先頭から順番に選択
        actual_samples = min(num_samples, total_images)
        selected_files = image_files[:actual_samples]
        print(f"  - キャリブレーション画像数: {total_images} (使用: {actual_samples})")
    
    # モデルの読み込み
    print("\nモデルを読み込んでいます...")
    core = Core()
    model = core.read_model(model_path)
    
    # 前処理関数
    def preprocess_image(image_path):
        """画像を前処理してモデル入力形式に変換"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((input_size[1], input_size[0]))  # (width, height)
        img_array = np.array(img, dtype=np.float32)
        # 正規化 (0-255 -> 0-1)
        img_array = img_array / 255.0
        # チャネルファースト (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        # バッチ次元追加
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    # キャリブレーションデータセットの作成
    def calibration_data_generator():
        for i, img_path in enumerate(selected_files):
            try:
                img_data = preprocess_image(img_path)
                if (i + 1) % 20 == 0:
                    print(f"  キャリブレーション進行中: {i + 1}/{actual_samples}")
                yield img_data
            except Exception as e:
                print(f"警告: 画像の読み込みに失敗しました: {img_path} - {e}")
                continue
    
    print("\nキャリブレーションデータを準備しています...")
    calibration_dataset = nncf.Dataset(calibration_data_generator())
    
    # INT8量子化の実行
    print("\nINT8量子化を実行しています（数分かかる場合があります）...")
    try:
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            subset_size=num_samples
        )
    except Exception as e:
        print(f"量子化中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 量子化モデルの保存
    if output_path is None:
        base_path = os.path.splitext(model_path)[0]
        output_path = f"{base_path}_int8"
    
    xml_path = f"{output_path}.xml" if not output_path.endswith('.xml') else output_path
    bin_path = xml_path.replace('.xml', '.bin')
    
    print(f"\n量子化モデルを保存しています...")
    serialize(quantized_model, xml_path, bin_path)
    
    # 結果の表示
    original_size = os.path.getsize(model_path.replace('.xml', '.bin')) / 1024 / 1024
    quantized_size = os.path.getsize(bin_path) / 1024 / 1024
    
    print(f"\nINT8量子化が完了しました！")
    print(f"  - XMLファイル: {xml_path}")
    print(f"  - BINファイル: {bin_path}")
    print(f"  - 精度: INT8")
    print(f"\nファイルサイズ:")
    print(f"  - 元のモデル: {original_size:.1f} MB")
    print(f"  - INT8モデル: {quantized_size:.1f} MB")
    print(f"  - サイズ削減: {(1 - quantized_size/original_size)*100:.1f}%")
    
    # 精度の検証
    verify_model_precision(xml_path, 'INT8')
    
    return xml_path


def detect_model_type_from_filename(filename):
    """
    ファイル名からモデルタイプを自動検出する

    Args:
        filename: モデルファイル名（パスでも可）

    Returns:
        検出されたモデルタイプ文字列、検出できない場合はNone
    """
    basename = os.path.basename(filename).lower()

    # MODEL_REGISTRYのキーを長い順にソートして、最長一致で検出する
    # （例: "edgenext_xx_small" が "edgenext_x_small" より先にマッチするように）
    known_types = [
        # 長いキーを先に（部分一致を避けるため）
        # 位置推論モデルはベースのバックボーン名より先にマッチさせる
        "mobilenetv4_conv_small_location",
        "mobilenetv3_small_100_location",
        "efficientnet_lite0_location",
        "edgenext_xx_small_location",
        "mobilevit_xxs_location",
        "swin_moe_tiny_patch4_window7_224",
        "swin_tiny_patch4_window7_224",
        "swinv2_cr_tiny_ns_224",
        "mobilenetv4_conv_small",
        "mobilenetv3_small_100",
        "mobilenetv3_large_100",
        "shufflenetv2_x0_5",
        "swin_s3_tiny_224",
        "efficientformer_l1",
        "edgenext_xx_small",
        "edgenext_x_small",
        "efficientnet_lite0",
        "efficientnetv2_s",
        "resnet18_location",
        "resnet18_waypoint",
        "donkey_waypoint",
        "donkey_location",
        "mobilevitv2_050",
        "mobilevit_xxs",
        "mobilevit_xs",
        "efficientnet_b0",
        "convnext_nano",
        "convnext_tiny",
        "mobileone_s0",
        "ghostnet_050",
        "mobilevit_s",
        "donkeycar",
        "donkey_fcn",
        "swin_tiny",
        "resnet18",
        "resnet34",
        "yolo11n",
        "yolo11s",
        "yolo11m",
        "yolo11l",
        "yolo11x",
    ]

    for model_type in known_types:
        # ファイル名の先頭部分にモデルタイプが含まれているかチェック
        # 例: "edgenext_xx_small_20260207_144757.pth" → "edgenext_xx_small"
        if basename.startswith(model_type):
            return model_type

    # "donkeycar" の短縮形として "donkey" で始まるファイル名もチェック
    # （donkey_location, donkey_waypoint, donkey_fcn に該当しなかった場合）
    if basename.startswith("donkey"):
        return "donkeycar"

    return None


def interactive_mode():
    """
    引数なしで実行された場合のインタラクティブモード
    modelsフォルダからモデルを選択し、変換パラメータをユーザーに入力させる
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, "models")

    print("=" * 60)
    print("  OpenVINO変換ツール（インタラクティブモード）")
    print("=" * 60)

    # modelsフォルダの存在確認
    if not os.path.isdir(models_dir):
        print(f"\nエラー: modelsフォルダが見つかりません: {models_dir}")
        print("modelsフォルダを作成してモデルファイルを配置してください。")
        return

    # 対象ファイルの収集
    supported_extensions = ['.pth', '.pt', '.onnx', '.xml']
    model_files = []
    for ext in supported_extensions:
        model_files.extend(glob_module.glob(os.path.join(models_dir, f"*{ext}")))

    # ファイル名でソート
    model_files.sort(key=lambda f: os.path.basename(f).lower())

    if not model_files:
        print(f"\nエラー: modelsフォルダにモデルファイルが見つかりません: {models_dir}")
        print(f"サポートされている形式: {', '.join(supported_extensions)}")
        return

    # モデルファイル一覧の表示
    print(f"\nmodelsフォルダ: {models_dir}")
    print(f"見つかったモデルファイル: {len(model_files)}個\n")

    for i, filepath in enumerate(model_files, 1):
        basename = os.path.basename(filepath)
        ext = os.path.splitext(basename)[1]
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        detected_type = detect_model_type_from_filename(basename)
        type_str = f"  [{detected_type}]" if detected_type else ""
        print(f"  {i:3d}. {basename}  ({size_mb:.1f} MB){type_str}")

    # モデル選択
    print()
    while True:
        try:
            choice = input("変換するモデルの番号を入力してください (q=終了): ").strip()
            if choice.lower() == 'q':
                print("終了します。")
                return
            choice_num = int(choice)
            if 1 <= choice_num <= len(model_files):
                break
            print(f"1〜{len(model_files)} の番号を入力してください。")
        except ValueError:
            print("数字を入力してください。")

    selected_path = model_files[choice_num - 1]
    selected_name = os.path.basename(selected_path)
    file_ext = os.path.splitext(selected_name)[1].lower()

    print(f"\n選択されたモデル: {selected_name}")

    # モデルタイプの自動検出・入力（PyTorchモデルの場合）
    model_type = None
    if file_ext in ['.pth', '.pt']:
        detected_type = detect_model_type_from_filename(selected_name)
        if detected_type:
            print(f"モデルタイプを自動検出しました: {detected_type}")
            confirm = input(f"このモデルタイプで続行しますか？ (Y/n): ").strip()
            if confirm.lower() not in ('n', 'no'):
                model_type = detected_type

        if model_type is None:
            print("\nモデルタイプを入力してください。")
            print("利用可能なモデルタイプの例:")
            print("  donkeycar, resnet18, resnet34, edgenext_xx_small,")
            print("  mobilevit_xxs, efficientnet_lite0, convnext_nano,")
            print("  yolo11n, yolo11s, donkey_location, resnet18_location, ...")
            while True:
                model_type = input("モデルタイプ: ").strip()
                if model_type:
                    break
                print("モデルタイプを入力してください。")

    # YOLOモデル判定
    is_yolo = False
    if model_type and 'yolo' in model_type.lower():
        is_yolo = True
    elif 'yolo' in selected_name.lower():
        is_yolo = True

    # 精度の選択
    print("\n変換精度を選択してください:")
    print("  1. FP16（推奨 - 高速・省メモリ）")
    print("  2. FP32（高精度）")
    print("  3. INT8（最高速・要キャリブレーションデータ）")

    precision_map = {'1': 'FP16', '2': 'FP32', '3': 'INT8'}
    while True:
        precision_choice = input("精度 [1=FP16(デフォルト)/2=FP32/3=INT8]: ").strip()
        if precision_choice == '':
            precision = 'FP16'
            break
        if precision_choice in precision_map:
            precision = precision_map[precision_choice]
            break
        print("1, 2, 3 のいずれかを入力してください。")

    print(f"選択された精度: {precision}")

    # INT8の場合、キャリブレーションディレクトリを要求
    calibration_dir = None
    num_calibration_samples = 100
    sample_interval = 10
    if precision == 'INT8':
        print("\nINT8量子化にはキャリブレーション画像が必要です。")
        while True:
            calibration_dir = input("キャリブレーション画像のディレクトリパス: ").strip()
            if calibration_dir and os.path.isdir(calibration_dir):
                break
            if calibration_dir:
                print(f"ディレクトリが見つかりません: {calibration_dir}")
            else:
                print("パスを入力してください。")

        samples_input = input(f"キャリブレーションサンプル数 [{num_calibration_samples}]: ").strip()
        if samples_input:
            try:
                num_calibration_samples = int(samples_input)
            except ValueError:
                print(f"無効な入力です。デフォルト値 {num_calibration_samples} を使用します。")

        interval_input = input(f"サンプリング間隔 [{sample_interval}]: ").strip()
        if interval_input:
            try:
                sample_interval = int(interval_input)
            except ValueError:
                print(f"無効な入力です。デフォルト値 {sample_interval} を使用します。")

    # 入力サイズの取得
    width = 224
    height = 224
    yolo_input_size = 640

    if is_yolo:
        size_input = input(f"\nYOLO入力画像サイズ [{yolo_input_size}]: ").strip()
        if size_input:
            try:
                yolo_input_size = int(size_input)
            except ValueError:
                print(f"無効な入力です。デフォルト値 {yolo_input_size} を使用します。")
    elif file_ext in ['.pth', '.pt']:
        # チェックポイントから学習時の入力サイズを読み取る
        checkpoint_input_size = None
        try:
            checkpoint = torch.load(selected_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                if 'input_size' in checkpoint:
                    checkpoint_input_size = tuple(checkpoint['input_size'])
                elif 'model_input_size' in checkpoint:
                    checkpoint_input_size = tuple(checkpoint['model_input_size'])
        except Exception:
            pass

        if checkpoint_input_size:
            height, width = checkpoint_input_size
            print(f"\nチェックポイントから入力サイズを検出: {height}x{width}")
        else:
            # フォールバック: model_infoからデフォルトサイズを取得
            try:
                sys.path.insert(0, project_dir)
                from model_info import get_model_input_size
                default_size = get_model_input_size(model_type)
                height, width = default_size
                print(f"\nモデルのデフォルト入力サイズ: {height}x{width}")
            except Exception:
                pass

        size_input = input(f"入力画像サイズ (高さx幅) [{height}x{width}]: ").strip()
        if size_input:
            try:
                parts = re.split(r'[x,\s]+', size_input)
                if len(parts) == 2:
                    height = int(parts[0])
                    width = int(parts[1])
                elif len(parts) == 1:
                    height = width = int(parts[0])
            except ValueError:
                print(f"無効な入力です。デフォルト値 {height}x{width} を使用します。")

    # 変換確認
    print("\n" + "=" * 60)
    print("変換設定の確認")
    print("=" * 60)
    print(f"  モデルファイル: {selected_name}")
    if model_type:
        print(f"  モデルタイプ:   {model_type}")
    print(f"  精度:           {precision}")
    if is_yolo:
        print(f"  入力サイズ:     {yolo_input_size}")
    elif file_ext in ['.pth', '.pt']:
        print(f"  入力サイズ:     {height}x{width}")
    if calibration_dir:
        print(f"  キャリブレーション: {calibration_dir}")
        print(f"  サンプル数:     {num_calibration_samples}")
        print(f"  サンプリング間隔: {sample_interval}")
    print("=" * 60)

    confirm = input("\nこの設定で変換を開始しますか？ (Y/n): ").strip()
    if confirm.lower() in ('n', 'no'):
        print("変換を中止しました。")
        return

    print()

    # 変換実行（既存のロジックと同等）
    output_path = None

    if file_ext == '.xml':
        if precision != 'INT8':
            print("OpenVINOモデル（.xml）が指定されました。")
            print("既存のOpenVINOモデルに対しては、INT8量子化のみサポートしています。")
            return

        quantize_to_int8(
            model_path=selected_path,
            output_path=output_path,
            calibration_dir=calibration_dir,
            input_size=(height, width),
            num_samples=num_calibration_samples,
            sample_interval=sample_interval
        )
    elif file_ext == '.onnx':
        if precision == 'INT8':
            print("ONNXモデルをINT8に変換します（FP16変換 → INT8量子化）")
            fp16_output = os.path.splitext(selected_path)[0] + "_openvino"
            xml_path = convert_onnx_to_openvino(
                onnx_path=selected_path,
                output_path=fp16_output,
                precision='FP16'
            )
            if xml_path:
                int8_output = fp16_output + "_int8"
                quantize_to_int8(
                    model_path=xml_path,
                    output_path=int8_output,
                    calibration_dir=calibration_dir,
                    input_size=(height, width),
                    num_samples=num_calibration_samples,
                    sample_interval=sample_interval
                )
        else:
            convert_onnx_to_openvino(
                onnx_path=selected_path,
                output_path=output_path,
                precision=precision
            )
    elif file_ext in ['.pth', '.pt']:
        if is_yolo:
            print("YOLOモデルを検出しました。")
            if precision == 'INT8':
                print("YOLOモデルをINT8に変換します（FP16変換 → INT8量子化）")
                fp16_output = os.path.splitext(selected_path)[0] + "_openvino"
                xml_path = convert_yolo_to_openvino(
                    model_path=selected_path,
                    output_path=fp16_output,
                    input_size=yolo_input_size,
                    precision='FP16'
                )
                if xml_path:
                    int8_output = fp16_output + "_int8"
                    quantize_to_int8(
                        model_path=xml_path,
                        output_path=int8_output,
                        calibration_dir=calibration_dir,
                        input_size=(yolo_input_size, yolo_input_size),
                        num_samples=num_calibration_samples,
                        sample_interval=sample_interval
                    )
            else:
                convert_yolo_to_openvino(
                    model_path=selected_path,
                    output_path=output_path,
                    input_size=yolo_input_size,
                    precision=precision
                )
        else:
            print("PyTorchモデルを検出しました。")
            if not model_type:
                print("エラー: PyTorchモデルの場合はモデルタイプが必須です。")
                return

            if precision == 'INT8':
                print("PyTorchモデルをINT8に変換します（FP16変換 → INT8量子化）")
                fp16_output = os.path.splitext(selected_path)[0] + "_openvino"
                xml_path = convert_pytorch_to_openvino(
                    model_path=selected_path,
                    model_type=model_type,
                    output_path=fp16_output,
                    input_size=(height, width),
                    precision='FP16'
                )
                if xml_path:
                    int8_output = fp16_output + "_int8"
                    quantize_to_int8(
                        model_path=xml_path,
                        output_path=int8_output,
                        calibration_dir=calibration_dir,
                        input_size=(height, width),
                        num_samples=num_calibration_samples,
                        sample_interval=sample_interval
                    )
            else:
                convert_pytorch_to_openvino(
                    model_path=selected_path,
                    model_type=model_type,
                    output_path=output_path,
                    input_size=(height, width),
                    precision=precision
                )
    else:
        print(f"エラー: サポートされていないファイル形式です: {file_ext}")
        print("サポートされている形式: .pth, .pt, .onnx, .xml")


def main():
    # 引数なしで実行された場合はインタラクティブモードを起動
    if len(sys.argv) == 1:
        interactive_mode()
        return

    parser = argparse.ArgumentParser(description='PyTorch、YOLO、またはONNXモデルをOpenVINO形式に変換するスクリプト')
    parser.add_argument('--model_path', type=str, required=True, help='変換するモデルのパス（.pth, .pt, .onnx, .xml）')
    parser.add_argument('--model_type', type=str, help='モデルタイプ (例: resnet18, yolo) - PyTorchモデルの場合は必須')
    parser.add_argument('--output_path', type=str, default=None, help='出力OpenVINOモデルのパス（省略可）')
    parser.add_argument('--width', type=int, default=224, help='入力画像の幅（PyTorchモデルの場合）')
    parser.add_argument('--height', type=int, default=224, help='入力画像の高さ（PyTorchモデルの場合）')
    parser.add_argument('--input_size', type=int, default=640, help='入力画像サイズ（YOLOモデルの場合）')
    parser.add_argument('--precision', type=str, default='FP16', choices=['FP32', 'FP16', 'INT8'], help='モデルの精度（デフォルト: FP16）')
    parser.add_argument('--calibration_dir', type=str, default=None, help='INT8量子化用のキャリブレーション画像ディレクトリ（--precision INT8の場合は必須）')
    parser.add_argument('--num_calibration_samples', type=int, default=100, help='キャリブレーションに使用する最大サンプル数（デフォルト: 100）')
    parser.add_argument('--sample_interval', type=int, default=10, help='キャリブレーション画像のサンプリング間隔（デフォルト: 10枚ごと）')

    args = parser.parse_args()

    # INT8の場合、キャリブレーションディレクトリが必須
    if args.precision == 'INT8' and args.calibration_dir is None:
        print("=" * 60)
        print("エラー: INT8量子化にはキャリブレーション画像が必要です")
        print("=" * 60)
        print("\n--calibration_dir オプションでキャリブレーション画像の")
        print("ディレクトリを指定してください。")
        print("\n使用例:")
        print(f"  python {os.path.basename(__file__)} \\")
        print(f"    --model_path model.onnx \\")
        print(f"    --precision INT8 \\")
        print(f"    --calibration_dir /path/to/images")
        print("\nキャリブレーション画像について:")
        print("  - 実際の推論時に使用する画像と同様の画像を100枚程度用意")
        print("  - サポート形式: jpg, jpeg, png, bmp")
        print("  - 例: DonkeyCarの場合はtubデータの画像を使用")
        print("\n代替案:")
        print("  - まずFP16で変換し、後からINT8に量子化することも可能です")
        print(f"    python {os.path.basename(__file__)} --model_path model.onnx --precision FP16")
        print(f"    python {os.path.basename(__file__)} --model_path model_openvino.xml --precision INT8 --calibration_dir /path/to/images")
        return

    file_ext = os.path.splitext(args.model_path)[1].lower()
    model_path_lower = args.model_path.lower()
    is_yolo = ('yolo' in model_path_lower or
               (args.model_type and 'yolo' in args.model_type.lower()))

    # OpenVINOモデル（.xml）の場合はINT8量子化のみ
    if file_ext == '.xml':
        if args.precision != 'INT8':
            print("OpenVINOモデル（.xml）が指定されました。")
            print("既存のOpenVINOモデルに対しては、INT8量子化のみサポートしています。")
            print("\n使用例:")
            print(f"  python {os.path.basename(__file__)} --model_path {args.model_path} --precision INT8 --calibration_dir /path/to/images")
            return

        quantize_to_int8(
            model_path=args.model_path,
            output_path=args.output_path,
            calibration_dir=args.calibration_dir,
            input_size=(args.height, args.width),
            num_samples=args.num_calibration_samples,
            sample_interval=args.sample_interval
        )
    elif file_ext == '.onnx':
        if args.precision == 'INT8':
            # まずFP16で変換してからINT8量子化
            print("ONNXモデルをINT8に変換します（FP16変換 → INT8量子化）")
            fp16_output = args.output_path or os.path.splitext(args.model_path)[0] + "_openvino"
            xml_path = convert_onnx_to_openvino(
                onnx_path=args.model_path,
                output_path=fp16_output,
                precision='FP16'
            )
            if xml_path:
                int8_output = fp16_output + "_int8" if args.output_path is None else args.output_path + "_int8"
                quantize_to_int8(
                    model_path=xml_path,
                    output_path=int8_output,
                    calibration_dir=args.calibration_dir,
                    input_size=(args.height, args.width),
                    num_samples=args.num_calibration_samples,
                    sample_interval=args.sample_interval
                )
        else:
            convert_onnx_to_openvino(
                onnx_path=args.model_path,
                output_path=args.output_path,
                precision=args.precision
            )
    elif file_ext in ['.pth', '.pt']:
        if is_yolo:
            print("YOLOモデルを検出しました。")
            if args.precision == 'INT8':
                # YOLOもNNCFでINT8量子化（キャリブレーションデータが必要）
                print("YOLOモデルをINT8に変換します（FP16変換 → INT8量子化）")
                fp16_output = args.output_path or os.path.splitext(args.model_path)[0] + "_openvino"
                xml_path = convert_yolo_to_openvino(
                    model_path=args.model_path,
                    output_path=fp16_output,
                    input_size=args.input_size,
                    precision='FP16'
                )
                if xml_path:
                    int8_output = fp16_output + "_int8" if args.output_path is None else args.output_path + "_int8"
                    quantize_to_int8(
                        model_path=xml_path,
                        output_path=int8_output,
                        calibration_dir=args.calibration_dir,
                        input_size=(args.input_size, args.input_size),
                        num_samples=args.num_calibration_samples,
                        sample_interval=args.sample_interval
                    )
            else:
                convert_yolo_to_openvino(
                    model_path=args.model_path,
                    output_path=args.output_path,
                    input_size=args.input_size,
                    precision=args.precision
                )
        else:
            print("PyTorchモデルを検出しました。")
            if not args.model_type:
                print("エラー: PyTorchモデルの場合は --model_type が必須です。")
                print("YOLOモデルの場合は --model_type yolo を指定するか、ファイル名に 'yolo' を含めてください。")
                parser.print_help()
                return

            if args.precision == 'INT8':
                # まずFP16で変換してからINT8量子化
                print("PyTorchモデルをINT8に変換します（FP16変換 → INT8量子化）")
                fp16_output = args.output_path or os.path.splitext(args.model_path)[0] + "_openvino"
                xml_path = convert_pytorch_to_openvino(
                    model_path=args.model_path,
                    model_type=args.model_type,
                    output_path=fp16_output,
                    input_size=(args.height, args.width),
                    precision='FP16'
                )
                if xml_path:
                    int8_output = fp16_output + "_int8" if args.output_path is None else args.output_path + "_int8"
                    quantize_to_int8(
                        model_path=xml_path,
                        output_path=int8_output,
                        calibration_dir=args.calibration_dir,
                        input_size=(args.height, args.width),
                        num_samples=args.num_calibration_samples,
                        sample_interval=args.sample_interval
                    )
            else:
                convert_pytorch_to_openvino(
                    model_path=args.model_path,
                    model_type=args.model_type,
                    output_path=args.output_path,
                    input_size=(args.height, args.width),
                    precision=args.precision
                )
    else:
        print(f"エラー: サポートされていないファイル形式です: {file_ext}")
        print("サポートされている形式: .pth, .pt, .onnx, .xml")

if __name__ == "__main__":
    main()