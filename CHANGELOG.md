# 変更履歴 (Changelog)

このファイルはプロジェクトの主要な変更履歴を記録します。

---

## [Unreleased]

### 2025-12-09

#### YOLOデフォルトクラスの変更

**変更内容**: YOLOの検出クラスをCOCOデフォルトからカスタムクラスに変更

| 変更前（COCOデフォルト） | 変更後（カスタム） |
|----------------------|-----------------|
| person, bicycle, car, motorcycle, ... | car, red_sign, green_sign, dog |

**変更されたファイル**:
- `config.py`: `CLASS_COLORS`, `SEGMENTATION_CLASS_COLORS`, `DETECTION_INFERENCE_CLASS_COLORS`, `DETECTION_INFERENCE_TEXT_COLORS` を更新

---

#### RTX 5060 Laptop GPU (sm_120) CUDA対応

**背景**: RTX 5060 Laptop GPUはBlackwellアーキテクチャ（sm_120）を採用しており、PyTorchの標準リリースではCUDAカーネルが含まれていませんでした。

**問題**:
- PyTorch 2.7.0.dev+cu124では、GPU検出は成功するが、実際のYOLO学習時に `CUDA error: no kernel image is available for execution on the device` エラーが発生

**解決策**:
- PyTorch 2.10.0.dev+cu128（CUDA 12.8対応）で完全動作を確認

#### 追加されたファイル

| ファイル | 説明 |
|---------|------|
| `install_pytorch_cu128.bat` | PyTorch Nightly (CUDA 12.8) インストールスクリプト |
| `run_cpu_fallback.bat` | CPUフォールバックモード起動スクリプト |
| `fix_rtx5060_kernel.bat` | RTX 5060カーネル修復スクリプト |
| `CUDA_SETUP.md` | RTX 5060 CUDA対応セットアップガイド |

#### 更新されたファイル

| ファイル | 変更内容 |
|---------|---------|
| `requirements_cuda.txt` | CUDA 12.8 Nightlyに更新、RTX 5060対応情報を追加 |
| `run.bat` | RTX 5060 (sm_120) 互換性のための環境変数を追加 |

#### 動作確認済み構成

| コンポーネント | バージョン |
|--------------|-----------|
| PyTorch | 2.10.0.dev20251208+cu128 |
| TorchVision | 0.25.0.dev20251207+cu128 |
| CUDA | 12.8 |
| GPU | RTX 5060 Laptop (sm_120) |

#### 参考リンク
- [PyTorch Forum - RTX 5060 Ti Success Story](https://discuss.pytorch.org/t/how-do-i-use-pytorch-with-rtx-5060-ti/220926/8)

---

## 過去の変更

### 2025-03-10 以前

- 初期リリース
- Donkeycar/Jetracer対応
- YOLO物体検知・セグメンテーション機能
- MLflow統合
- 各種モデルアーキテクチャ対応
