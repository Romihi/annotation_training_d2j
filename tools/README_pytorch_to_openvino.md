# PyTorch to OpenVINO 変換ツール

PyTorch、ONNX、YOLOモデルをOpenVINO形式（IR: Intermediate Representation）に変換するスクリプトです。

## 概要

このツールは以下の変換をサポートします：

| 入力形式 | 出力形式 | 対応精度 |
|----------|----------|----------|
| PyTorch (.pth, .pt) | OpenVINO IR (.xml, .bin) | FP32, FP16, INT8 |
| ONNX (.onnx) | OpenVINO IR (.xml, .bin) | FP32, FP16, INT8 |
| YOLO (.pt) | OpenVINO IR (.xml, .bin) | FP32, FP16, INT8 |
| OpenVINO IR (.xml) | OpenVINO IR (.xml, .bin) | INT8のみ |

## インストール

### 必要なパッケージ

```bash
pip install openvino openvino-dev torch numpy nncf pillow
```

### YOLOモデルを変換する場合

```bash
pip install ultralytics
```

## 基本的な使い方

### FP16変換（デフォルト・推奨）

```bash
# ONNXモデル
python pytorch_to_openvino.py --model_path model.onnx

# PyTorchモデル（model_typeが必須）
python pytorch_to_openvino.py --model_path model.pth --model_type edgenext_xx_small

# YOLOモデル（ファイル名にyoloが含まれるか、--model_type yoloを指定）
python pytorch_to_openvino.py --model_path yolov8n.pt
```

### FP32変換

```bash
python pytorch_to_openvino.py --model_path model.onnx --precision FP32
```

### INT8変換（キャリブレーション画像が必要）

```bash
python pytorch_to_openvino.py \
    --model_path model.onnx \
    --precision INT8 \
    --calibration_dir /path/to/images \
    --num_calibration_samples 100
```

## コマンドライン引数

| 引数 | 必須 | デフォルト | 説明 |
|------|------|------------|------|
| `--model_path` | ✓ | - | 変換するモデルのパス |
| `--model_type` | △ | - | PyTorchモデルの場合は必須（例: edgenext_xx_small, resnet18） |
| `--output_path` | | 自動生成 | 出力ファイルパス（拡張子なし） |
| `--precision` | | FP16 | 精度: FP32, FP16, INT8 |
| `--width` | | 224 | 入力画像の幅（PyTorchモデル用） |
| `--height` | | 224 | 入力画像の高さ（PyTorchモデル用） |
| `--input_size` | | 640 | YOLOモデルの入力サイズ |
| `--calibration_dir` | △ | - | INT8量子化用のキャリブレーション画像ディレクトリ |
| `--num_calibration_samples` | | 100 | キャリブレーションに使用する画像数 |

## 精度について

### FP32（32ビット浮動小数点）
- 最高精度、最大のファイルサイズ
- 推論速度は最も遅い

### FP16（16ビット浮動小数点）- **推奨**
- FP32の約半分のファイルサイズ
- ほとんどのモデルで精度低下は無視できるレベル
- 推論速度がFP32より向上

### INT8（8ビット整数）
- 最小のファイルサイズ（FP32の約1/4）
- 最速の推論速度
- キャリブレーションデータが必要
- 一部のモデルで精度低下の可能性あり

## ファイルサイズ比較と検証

変換完了後、自動的にファイルサイズの詳細比較とFP16変換の検証が行われます：

```
============================================================
ファイルサイズ比較
============================================================

  [ファイルサイズ]
  ┌─────────────────────────────────────────────────────────┐
  │ 元のPyTorchモデル (.pth):         13.47 MB              │
  │   ├─ 純粋な重み (FP32):            4.42 MB              │
  │   ├─ オプティマイザ状態:           8.84 MB              │
  │   └─ その他（メタデータ等）:       0.21 MB              │
  │ OpenVINO BINファイル:              2.21 MB              │
  │ OpenVINO XMLファイル:            454.69 KB              │
  │ OpenVINO合計 (XML+BIN):            2.65 MB              │
  └─────────────────────────────────────────────────────────┘

  [サイズ比較バー]
  純粋な重み(FP32): ████████████████████████████████████████ 4.42 MB
  OpenVINO BIN    : ████████████████████░░░░░░░░░░░░░░░░░░░░ 2.21 MB

  サイズ削減率: 50.0%

  [FP16変換の判定]
  - 純粋な重み (FP32):       4.42 MB
  - FP16の理論サイズ(50%):   2.21 MB
  - 実際のBINサイズ:         2.21 MB
  - 実際の比率:              50.0%

  ✓ 結果: 重みは正確にFP16で保存されています
    （理論値とほぼ一致: 50.0% ≈ 50%）
============================================================
```

### .pthファイルの内訳について

PyTorchの`.pth`ファイルには以下が含まれています：

| 項目 | 説明 | 備考 |
|------|------|------|
| 純粋な重み (model_state_dict) | モデルパラメータ | OpenVINOに変換される部分 |
| オプティマイザ状態 | Adam等の学習状態（momentum, variance） | 推論には不要 |
| メタデータ | epoch、loss、設定情報など | 推論には不要 |

**Adamオプティマイザの場合**、各パラメータに対してmomentumとvarianceを保持するため、オプティマイザ状態は重みの約2倍のサイズになります。

### FP16変換の判定基準

| 実際の比率 | 判定結果 |
|-----------|---------|
| 45%～55% | ✓ 正確にFP16で保存 |
| 45%未満 | ✓ FP16 + 追加最適化 |
| 55%～65% | ✓ FP16で保存 |
| 65%～85% | △ 部分的にFP16 |
| 85%以上 | ✗ FP16変換失敗の可能性 |

## 推論時のFP16実行

OpenVINOモデルの重みがFP16で保存されていても、計算グラフはFP32で定義されている場合があります。推論時にFP16で実行するには、以下の設定が必要です：

### GPUの場合（自動）

```python
from openvino.runtime import Core

core = Core()
model = core.read_model("model_openvino.xml")
compiled_model = core.compile_model(model, "GPU")  # 自動的にFP16で実行
```

### CPUの場合（明示的に設定）

```python
from openvino.runtime import Core
from openvino import properties

core = Core()
model = core.read_model("model_openvino.xml")

# FP16推論ヒントを設定
config = {properties.hint.inference_precision: "f16"}
compiled_model = core.compile_model(model, "CPU", config)
```

### model_switcher.pyでの設定例

```python
# FP16推論を有効化する修正箇所（load_openvino_unified_model関数内）

import openvino as ov
from openvino import properties

core = ov.Core()
model = core.read_model(model_path)

# FP16推論の設定
config = {}
use_fp16 = getattr(cfg, 'OPENVINO_USE_FP16', True)  # デフォルトでFP16を使用

if use_fp16:
    try:
        config[properties.hint.inference_precision] = ov.Type.f16
        logger.info(f"FP16 inference enabled for {device}")
    except Exception as e:
        logger.warning(f"Failed to set FP16 inference: {e}")

compiled_model = core.compile_model(model, device, config)
```

### myconfig.pyに追加可能なオプション

```python
# OpenVINO FP16推論を有効化（デフォルト: True）
OPENVINO_USE_FP16 = True
```

## INT8量子化について

### キャリブレーションデータの準備

INT8量子化には、実際の推論時と同様の画像が必要です。

**DonkeyCarの場合：**
```bash
# tubデータの画像を使用
python pytorch_to_openvino.py \
    --model_path model.pth \
    --model_type edgenext_xx_small \
    --precision INT8 \
    --calibration_dir ./data/tub_1/images \
    --height 120 --width 160
```

**推奨事項：**
- 100枚以上の画像を用意
- 様々な条件（明るさ、角度など）の画像を含める
- 対応形式: jpg, jpeg, png, bmp

### 既存のOpenVINOモデルをINT8に変換

```bash
python pytorch_to_openvino.py \
    --model_path model_openvino.xml \
    --precision INT8 \
    --calibration_dir /path/to/images
```

## 変換フロー

```
┌─────────────┐
│ PyTorch     │──┐
│ (.pth, .pt) │  │
└─────────────┘  │    ┌──────────┐    ┌─────────────────┐
                 ├───▶│  ONNX    │───▶│  OpenVINO IR    │
┌─────────────┐  │    │ (中間形式)│    │  (.xml + .bin)  │
│ ONNX        │──┤    └──────────┘    │  FP32 or FP16   │
│ (.onnx)     │  │                    └────────┬────────┘
└─────────────┘  │                             │
                 │                             ▼ INT8の場合
┌─────────────┐  │                    ┌─────────────────┐
│ YOLO        │──┘                    │  NNCF量子化     │
│ (.pt)       │                       │  (キャリブレーション)│
└─────────────┘                       └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │  OpenVINO IR    │
                                      │  INT8           │
                                      └─────────────────┘
```

## 出力ファイル

変換が成功すると、以下のファイルが生成されます：

- `{model_name}_openvino.xml` - モデル構造（ネットワークトポロジー）
- `{model_name}_openvino.bin` - 重み（パラメータ）

## 精度検証

変換完了後、自動的に精度検証が行われます：

```
==================================================
モデル精度の検証
==================================================
  - BINファイルサイズ: 2.21 MB

[全体のオペレーション分布]
  - 期待される精度: FP16
  - パラメータの型: <Type: 'float32'>
  - オペレーション出力型の分布:
      <Type: 'float32'>: 491個
      <Type: 'float16'>: 152個
      <Type: 'int64_t'>: 195個

[計算オペレーションの精度分析]
  ※ 実際の演算（Conv, MatMul, Add等）の精度が重要です
  - 計算オペレーション総数: 197個
  - 精度別の内訳:
      FP32: 185個 (93.9%)
      Other: 12個 (6.1%)

[判定結果]
ℹ 精度検証: 重みはFP16で保存されています
  - 計算グラフはFP32で定義されていますが、
    推論時にOpenVINOが自動的にFP16で実行する可能性があります
  - GPUでの推論時はFP16で実行されます
  - CPUでの推論時は、inference_precision_hint='f16'を設定することで
    FP16推論が可能です（対応CPUの場合）
==================================================
```

**注意:** OpenVINOの`compress_to_fp16`は重みをFP16で保存しますが、計算グラフはFP32のまま残る場合があります。これは正常な動作で、推論時にハードウェアに応じてFP16で実行されます。

## トラブルシューティング

### ovcコマンドが見つからない

```bash
pip install openvino-dev
```

### NNCFがインストールされていない（INT8変換時）

```bash
pip install nncf
```

### モデル変換でエラーが発生する

1. ONNXのopsetバージョンを確認（opset 12推奨）
2. PyTorchとOpenVINOのバージョン互換性を確認
3. 入力サイズが正しいか確認

### ONNX FP16変換でエラーが発生する

一部のモデル（EdgeNeXTなど）はCast操作が含まれており、ONNXレベルでのFP16変換が失敗する場合があります。この場合、自動的にフォールバックしてOpenVINOの`compress_to_fp16`で重みのみをFP16に圧縮します。

### FP16変換後も計算グラフがFP32と表示される

これはOpenVINOの仕様上、正常な動作です：

1. **重みはFP16で保存** - BINファイルサイズで確認可能
2. **計算グラフはFP32で定義** - 推論時にハードウェアに応じてFP16で実行
3. **推論時のFP16実行** - GPU使用時は自動、CPU使用時は`inference_precision_hint`を設定

## 使用例

### DonkeyCarモデルの変換

```bash
# FP16変換（推奨）
python pytorch_to_openvino.py \
    --model_path ./models/edgenext_xx_small_best.pth \
    --model_type edgenext_xx_small \
    --height 120 --width 160

# INT8変換（さらに高速化）
python pytorch_to_openvino.py \
    --model_path ./models/edgenext_xx_small_best.pth \
    --model_type edgenext_xx_small \
    --precision INT8 \
    --calibration_dir ./data/tub_1/images \
    --height 120 --width 160
```

### YOLOv8モデルの変換

```bash
# FP16変換
python pytorch_to_openvino.py \
    --model_path yolov8n.pt \
    --input_size 640

# INT8変換
python pytorch_to_openvino.py \
    --model_path yolov8n.pt \
    --precision INT8 \
    --calibration_dir ./images \
    --input_size 640
```

## 推論時の使用方法

変換後のモデルをOpenVINOで使用する例：

```python
from openvino.runtime import Core
from openvino import properties
import numpy as np

# モデルの読み込み
core = Core()
model = core.read_model("model_openvino.xml")

# FP16推論を有効化（CPU使用時）
config = {properties.hint.inference_precision: "f16"}
compiled_model = core.compile_model(model, "CPU", config)

# 推論
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = compiled_model([input_data])
output = result[0]
```

## ライセンス

このスクリプトはDonkeyCarプロジェクトの一部として提供されています。

## 更新履歴

- 2025-01-14: 
  - ファイルサイズ比較の改善（純粋な重みサイズの内訳表示）
  - FP16変換判定の精度向上（理論値との比較）
  - 推論時のFP16実行方法の説明追加
  - 計算オペレーションの精度分析機能追加
- 2025-01-13: INT8量子化サポート追加、精度検証機能追加
- 初版: FP32/FP16変換サポート
