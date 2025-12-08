# CUDA対応セットアップガイド - RTX 5060 Laptop GPU

## ✅ RTX 5060 Laptop GPU - 動作確認済み構成

### 🎯 推奨インストール方法（検証済み・動作確認済み）

**PyTorch 2.10.0.dev+cu128** で完全動作します！

```batch
install_pytorch_cu128.bat
```

**動作確認済み構成**:
- **PyTorch**: 2.10.0.dev20251208+cu128
- **TorchVision**: 0.25.0.dev20251207+cu128
- **CUDA**: 12.8
- **GPU**: RTX 5060 Laptop (sm_120)
- **状態**: ✅ 完全動作 - カーネルエラーなし

---

## RTX 5060 Laptop GPU特有の問題（過去の問題）

### ⚠️ 警告メッセージについて
```
NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120 is not compatible
with the current PyTorch install
```

**重要**: この警告は古いPyTorchバージョン（2.7.0.dev+cu124など）で表示されます。

**過去の問題**:
- PyTorch 2.7.0.dev20250310+cu124では、テンソル作成は成功するが、実際の学習時にカーネルエラーが発生
- エラー: `CUDA error: no kernel image is available for execution on the device`
- 原因: PyTorchがsm_120用のCUDAカーネルをビルドに含めていなかった

**✅ 解決済み**:
- **PyTorch 2.10.0.dev+cu128** でsm_120サポートが追加されました
- カーネルエラーは発生しません
- GPU学習が正常に動作します

---

## ⚠️ CUDA Kernelエラーへの対処（旧バージョンの場合）

### 問題: "no kernel image is available for execution"

古いPyTorchバージョン（CUDA 12.4以前）で以下のエラーが発生する場合：
```
CUDA error: no kernel image is available for execution on the device
```

### 解決策

#### 🔧 方法A: PyTorch CUDA 12.8にアップグレード（最も推奨）✅
```batch
install_pytorch_cu128.bat
```
**これが最良の解決策です。** sm_120完全サポートが含まれています。

#### 🔧 方法B: CPUフォールバックモード（動作は遅い）
```batch
run_cpu_fallback.bat
```
- CUDAを無効化してCPUで動作
- 学習速度は遅くなるが、安定動作
- アノテーション作業は通常通り

---

## 推奨インストール方法（RTX 5060専用）

### 🔧 方法1: PyTorch CUDA 12.8 Nightly（最も推奨）✅

**動作確認済み・検証済みの構成です。この方法を強く推奨します。**

#### 手順:
1. **`install_pytorch_cu128.bat`を実行**
   ```batch
   install_pytorch_cu128.bat
   ```

2. **インストール内容**:
   - PyTorch 2.10.0.dev ナイトリービルド（CUDA 12.8対応）
   - TorchVision 0.25.0.dev（互換バージョン）
   - 既存のPyTorchを自動アンインストール
   - CUDA動作確認を自動実行

3. **期待される結果**:
   ```
   PyTorch version: 2.10.0.dev20251208+cu128
   CUDA available: True
   Device: NVIDIA GeForce RTX 5060 Laptop GPU
   GPU test successful!
   ```

4. **重要**: この構成では以下が確認されています：
   - ✅ GPU検出が正常に動作
   - ✅ テンソル演算がGPUで実行可能
   - ✅ YOLO学習時にカーネルエラーが発生しない
   - ✅ sm_120完全サポート

---

### 🔧 方法2: PyTorch CUDA 12.4 Nightly（旧バージョン・非推奨）

**注意**: この方法はカーネルエラーが発生する可能性があります。CUDA 12.8を推奨します。

#### 手順:
1. **`install_pytorch_nightly.bat`を実行**
   ```batch
   install_pytorch_nightly.bat
   ```

2. **インストール内容**:
   - PyTorch 2.7.0 ナイトリービルド（CUDA 12.4対応）
   - TorchVision 0.22.0（互換バージョン）
   - 既存のPyTorchを自動アンインストール
   - CUDA動作確認を自動実行

3. **既知の問題**:
   - ⚠️ YOLO学習時に「no kernel image available」エラーが発生する可能性
   - ⚠️ sm_120警告が表示される
   - ⚠️ GPU学習が正常に動作しない場合がある

#### トラブルシューティング:
- **torchvisionのバージョン競合が発生した場合**:
  ```batch
  fix_torchvision.bat
  ```
  このスクリプトは依存関係チェックをバイパスして互換性のあるtorchvisionをインストールします。

- **シーケンシャルインストールを試す場合**:
  ```batch
  install_pytorch_sequential.bat
  ```
  torch → torchvisionの順に段階的にインストールします。

---

### 📋 インストール確認

#### 簡易確認スクリプト
```batch
check_yolo_ready.bat
```

このスクリプトは以下を自動確認します：
- PyTorchバージョン
- CUDA可用性
- GPUデバイス名
- torchvisionインストール状況
- Ultralytics YOLOライブラリ
- YOLOモデル読み込み
- GPU演算テスト

#### 期待される出力例:
```
Checking PyTorch...
PyTorch: 2.7.0.dev20250310+cu124
CUDA: True
Device: NVIDIA GeForce RTX 5060 Laptop GPU

Checking torchvision...
TorchVision: 0.22.0.dev20250226+cu124

Checking Ultralytics...
Ultralytics: OK

Testing YOLO model loading...
YOLO model loading: OK

GPU Test...
GPU Tensor: cuda:0
Ready for training!
```

**⚠️ sm_120警告について**: 上記の確認中にsm_120警告が表示されますが、「GPU Tensor: cuda:0」と「Ready for training!」が表示されていれば問題ありません。

---

## その他のインストール方法

### 方法2: 最新安定版CUDA 12.4
```batch
install_pytorch_latest_stable.bat
```
- より安定したリリース版
- sm_120警告が出る可能性あり
- 基本機能は動作

### 方法3: CUDA 12.1版
```batch
install_cuda_pytorch.bat
```
- CUDA 12.1対応
- 古いバージョンだが安定
- sm_120警告が出る

---

## 手動インストール方法

### PyTorch Nightly（推奨）
```batch
# 仮想環境をアクティベート
venv\Scripts\activate.bat

# 既存のPyTorchをアンインストール
pip uninstall -y torch torchvision torchaudio

# Nightly buildをインストール
pip install --pre --upgrade torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall

# 確認
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### torchvision修復（バージョン競合時）
```batch
# 依存関係チェックをバイパスしてインストール
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps
pip install pillow
```

---

## トラブルシューティング

### 問題1: CUDA availableがFalse

**確認事項**:

1. **NVIDIAドライバーの確認**
   ```batch
   nvidia-smi
   ```
   - GPUが認識されているか
   - ドライバーバージョンが表示されるか

2. **PyTorchのCUDAバージョン確認**
   ```batch
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
   ```

3. **再インストール**
   ```batch
   install_pytorch_nightly.bat
   ```

### 問題2: torchvisionのバージョン競合

**エラー例**:
```
ERROR: Cannot install torch and torchvision==0.22.0.dev20250226+cu124
because these package versions have conflicting dependencies
```

**解決策**:
```batch
fix_torchvision.bat
```

または手動で：
```batch
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps
pip install pillow
```

### 問題3: sm_120警告が表示される

**これは正常です！**

警告が表示されても、以下が確認できれば問題ありません：
- `CUDA available: True`
- `Device: NVIDIA GeForce RTX 5060 Laptop GPU`
- `GPU Tensor: cuda:0`

実際のGPU演算は正常に動作しています。

---

## 技術情報

### RTX 5060 Laptop GPU仕様
- **Compute Capability**: sm_120（最新世代・Blackwell アーキテクチャ）
- **必要なCUDA**: 12.8推奨（12.4以降）
- **必要なPyTorch**: 2.10.0.dev nightly以降（CUDA 12.8対応）
- **メモリ**: GDDR6
- **対応状況**: ✅ PyTorch 2.10.0.dev+cu128で完全サポート

### 動作確認済み構成

#### ✅ 推奨構成（完全動作）
| コンポーネント | バージョン | 状態 |
|--------------|-----------|------|
| PyTorch | 2.10.0.dev20251208+cu128 | ✅ 完全動作 |
| TorchVision | 0.25.0.dev20251207+cu128 | ✅ 完全動作 |
| CUDA | 12.8 | ✅ sm_120サポート |
| GPU | RTX 5060 Laptop | ✅ カーネルエラーなし |
| 演算 | cuda:0 | ✅ 正常動作 |
| YOLO学習 | Ultralytics YOLO | ✅ GPU高速学習可能 |

#### ⚠️ 旧構成（非推奨・カーネルエラーあり）
| コンポーネント | バージョン | 状態 |
|--------------|-----------|------|
| PyTorch | 2.7.0.dev20250310+cu124 | ⚠️ カーネルエラー発生 |
| TorchVision | 0.22.0.dev20250226+cu124 | ⚠️ |
| CUDA | 12.4 | ⚠️ sm_120不完全 |
| GPU | RTX 5060 Laptop | ⚠️ 学習時エラー |

### インストールスクリプト一覧
| スクリプト | 用途 | 推奨度 |
|-----------|------|--------|
| `install_pytorch_cu128.bat` | **Nightly build（CUDA 12.8）** | ⭐⭐⭐⭐⭐ |
| `check_yolo_ready.bat` | 動作確認 | ⭐⭐⭐⭐⭐ |
| `run_cpu_fallback.bat` | CPUフォールバック | ⭐⭐⭐ |
| `install_pytorch_nightly.bat` | Nightly build（CUDA 12.4）旧版 | ⭐ 非推奨 |
| `install_pytorch_sequential.bat` | 段階的インストール | ⭐ 非推奨 |
| `fix_torchvision.bat` | torchvision修復 | ⭐ 非推奨 |

---

## よくある質問（FAQ）

### Q1: sm_120警告が出るけど大丈夫？
**A**: PyTorch 2.10.0.dev+cu128を使用していれば、この警告は出ません。もし警告が出る場合は、古いPyTorchバージョンを使用しています。`install_pytorch_cu128.bat`で最新版にアップグレードしてください。

### Q2: YOLO学習は正常にできる？
**A**: はい！PyTorch 2.10.0.dev+cu128を使用すれば、GPUを使用した高速学習が完全に動作します。`check_yolo_ready.bat`で事前確認できます。

### Q3: カーネルエラーが出る場合はどうすればいい？
**A**: `install_pytorch_cu128.bat`を実行してPyTorch 2.10.0.dev+cu128にアップグレードしてください。CUDA 12.8対応版ではカーネルエラーは発生しません。

### Q4: 他のGPUでも使える？
**A**: はい。CUDA 12.8に対応しているNVIDIA GPUであれば動作します。RTX 50シリーズ（sm_120）の場合は特に推奨します。

### Q5: 安定版PyTorchじゃダメ？
**A**: RTX 5060（sm_120）の場合、現時点では**ナイトリービルド（2.10.0.dev+cu128）が必須**です。安定版ではsm_120のCUDAカーネルが含まれていません。

### Q6: PyTorch 2.10はまだ開発版だけど大丈夫？
**A**: はい。RTX 5060のようなsm_120 GPUでは、開発版を使用する必要があります。YOLO学習での動作確認済みです。

## 参考リンク

- [PyTorch公式インストールガイド](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit ダウンロード](https://developer.nvidia.com/cuda-downloads)
- [PyTorch Forum - RTX 5060 Ti Success Story](https://discuss.pytorch.org/t/how-do-i-use-pytorch-with-rtx-5060-ti/220926/8)

---

## 更新履歴

### 2025年12月8日
- ✅ **PyTorch 2.10.0.dev20251208+cu128 で完全動作を確認**
- RTX 5060 Laptop GPU (sm_120) のCUDA kernelエラーが解決
- `install_pytorch_cu128.bat` スクリプトを追加
- CUDA 12.8対応版を推奨構成として更新
- 動作確認済み構成テーブルを更新
