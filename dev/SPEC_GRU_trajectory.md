# 仕様書: GRUベース軌道生成モデル (Advanced Training Feature)

## 対象リポジトリ
- URL: https://github.com/Romihi/annotation_training_d2j (branch: latest)
- ベースシステム: Donkeycarデータビューア + PyTorch学習ツール (PyQt5 デスクトップアプリ)

---

## 1. 概要

既存の学習機能（単純MLP）に加え、**GRUを使った軌道生成モデル**をAdvanced Training Featureとして追加する。
過去Tフレームの画像・ego_stateを時系列入力とし、将来Nステップの `(steering, throttle)` 軌道を予測する。

---

## 2. ファイル一覧と実装状況

| ファイル | 操作 | 状態 | 説明 |
|----------|------|------|------|
| `managers/gru_model.py` | 新規作成 | **実装済** | GRUモデル定義 (timm/MobileNetV3-Small) |
| `managers/gru_dataset.py` | 新規作成 | **実装済** | 時系列シーケンスDataset |
| `managers/gru_training_manager.py` | 新規作成 | **実装済** | 学習ループ・推論・MLflow連携 |
| `managers/__init__.py` | 追記 | **実装済** | GRUクラスのexport |
| `managers/mlflow_manager.py` | 追記 | **実装済** | `ModelType.GRU_TRAJECTORY`, `log_gru_trajectory_model()` |
| `config.py` | 追記 | **実装済** | `GRU_DEFAULT_*` パラメータ群 |
| `translations.py` | 追記 | **実装済** | GRU学習・推論の日英翻訳キー |
| `main.py` | 追記 | **実装済** | UIセクション・学習ダイアログ・推論・結果表示 |
| `requirements.txt` | 追記 | **実装済** | `timm>=1.0.14` |

---

## 3. 入力仕様

### 3.1 画像入力 (Image Sources)

- **1〜5ソース必須**、ユーザーがUIから使用するソースを選択（**最低1個、最大5個**）
- 画像ソースが1つも選択されていない場合は学習開始ボタンを非活性化し、エラーメッセージ「画像ソースを1つ以上選択してください」を表示
- 各ソースは既存データ構造のバリアント名で指定（例: `cam`, `cam2`, `lidar`）
- 画像は `.jpg` (カメラ) または BEV変換済み `.jpg`/`.png` (LiDAR)
- 前処理: `Resize(128,128)` → `ToTensor()` → `Normalize(ImageNet mean/std)` → `(3, 128, 128)`
- LiDARのBEV画像も同様にRGB 3chとして扱う（`Image.open().convert('RGB')`）

### 3.2 ego_state 入力

| フィールド | アノテーションキー | 必須 | 型 | 範囲 |
|-----------|-------------------|------|-----|------|
| steering | `angle` | **必須** | float | -1.0 〜 1.0 |
| throttle | `throttle` | **必須** | float | -1.0 〜 1.0 |
| vx | `speed` | オプション | float | m/s |
| vy | (未使用) | - | float | 常に0.0 |
| omega | (未使用) | - | float | 常に0.0 |

- ego_stateベクトル次元: 常に5次元 `[steering, throttle, vx, vy, omega]`
- 存在しないキーは `0.0` でパディング

### 3.3 時系列設定

| パラメータ | デフォルト | config.py定数 | 説明 |
|-----------|---------|--------------|------|
| `seq_len` | 8 | `GRU_DEFAULT_SEQ_LEN` | 入力シーケンス長 (過去Tフレーム) |
| `pred_horizon` | 10 | `GRU_DEFAULT_PRED_HORIZON` | 予測ステップ数 (将来Nフレーム) |
| `stride` | 1 | `GRU_DEFAULT_STRIDE` | シーケンス抽出のストライド |

---

## 4. モデルアーキテクチャ (`managers/gru_model.py`) [実装済]

```
GRUTrajectoryModel
├── ImageEncoder (shared weights)  ← timm MobileNetV3-Small (num_classes=0)
│   └── forward_features → AdaptiveAvgPool2d(1) → Linear(576→128) → ReLU
├── EgoStateEncoder               ← Linear(5→32) → ReLU
├── ModalFusion                   ← cat(img_feats * num_sources, ego_feat) → Linear → ReLU
├── GRU                           ← 1層, hidden=256, batch_first=True
│   └── 勾配クリッピング max_norm=1.0
└── TrajectoryHead                ← Dropout → Linear(256 → pred_horizon*2) → tanh
```

### 4.1 実装の注意点

- バックボーンは仕様書では `torchvision MobileNetV3-Small` だったが、実装では **`timm`** を使用
  - `timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=0)`
- `forward_features()` は空間特徴マップ `(B, C, H, W)` を返すため、`AdaptiveAvgPool2d(1)` で集約
- 出力は `tanh` で `[-1, 1]` に制約

### 4.2 シグネチャ

```python
class GRUTrajectoryModel(nn.Module):
    def __init__(self, num_image_sources, ego_dim=5, img_feat_dim=128,
                 ego_feat_dim=32, gru_hidden=256, gru_layers=1,
                 pred_horizon=10, dropout=0.1)

    def forward(self, images, ego_states):
        # images:     (B, T, num_sources, 3, H, W)
        # ego_states: (B, T, 5)
        # return:     (B, pred_horizon, 2)  — [steering, throttle] per step
```

---

## 5. データセット (`managers/gru_dataset.py`) [実装済]

### 5.1 GRUSequenceDataset

```python
class GRUSequenceDataset(Dataset):
    def __init__(self, valid_indexes, annotations, images,
                 source_images_map, selected_sources,
                 seq_len=8, pred_horizon=10, stride=1,
                 img_size=(128, 128), augment=False)
```

**引数の対応（仕様→実装）:**
- `records` → `valid_indexes` + `annotations` + `images` + `source_images_map`
- `image_keys` → `selected_sources` （バリアント名のリスト）
- `data_folder` → 不要（`images`リスト・`source_images_map`が絶対パスを保持）

### 5.2 セッション境界検出

```python
def _detect_session_boundaries(self) -> List[List[int]]:
    # valid_indexes内のインデックスギャップ（差分>1）でセッション区切りを判定
    # 連続するインデックスを同一セッションとしてグループ化
```

### 5.3 シーケンス構築ルール

- インデックス `i` の場合: 入力 = `session[i : i+seq_len]`, ターゲット = `session[i+seq_len : i+seq_len+pred_horizon]`
- セッションをまたぐシーケンスは生成しない（インデックスギャップ>1で境界検出）
- 削除インデックスは `valid_indexes` 構築時に除外済み（`main.py`側で処理）

### 5.4 データ拡張

- **水平フリップ**: 50%の確率で画像を水平反転 + steering符号反転
- augment=True は学習時のみ。Validation / Prediction 時は常にFalse

### 5.5 画像前処理

```python
transforms.Compose([
    transforms.Resize(img_size),     # (128, 128)
    transforms.ToTensor(),           # [0, 1] float32
    transforms.Normalize(            # ImageNet正規化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 6. 学習マネージャー (`managers/gru_training_manager.py`) [実装済]

### 6.1 GRUTrainingManager クラス

```python
class GRUTrainingManager:
    def __init__(self, models_dir, mlflow_manager=None)
    def train(self, valid_indexes, annotations, images,
              source_images_map, selected_sources, config,
              progress_callback=None) -> dict
    def predict(self, model_path, valid_indexes, annotations, images,
                source_images_map, progress_callback=None) -> dict
    @staticmethod
    def load_model(model_path, device=None) -> tuple
    def _save_model(...) -> str
```

### 6.2 train() の動作

1. `GRUSequenceDataset` を構築
2. `random_split` で Train/Val 分割（デフォルト 80:20）
3. Val用に `augment=False` の別Datasetを作成し、同じインデックスで `Subset` 構築
4. `GRUTrajectoryModel` をインスタンス化し `device` に転送
5. `Adam` (weight_decay=1e-4) + `ReduceLROnPlateau` (factor=0.5, patience=5)
6. 損失: `nn.MSELoss()`, 勾配クリッピング: `clip_grad_norm_(max_norm=1.0)`
7. Best Val Loss のモデルを追跡・保存
8. キャンセル時もベストモデルを保存
9. MLflow ログ（`mlflow_manager` がある場合）

### 6.3 predict() の動作

1. `load_model()` でチェックポイントからモデル復元
2. チェックポイントに保存された `selected_sources`, `seq_len`, `pred_horizon` を使用
3. `GRUSequenceDataset` を構築（stride=1, augment=False）
4. バッチ推論を実行
5. 各シーケンスの**最終入力フレームインデックス**をキーとして予測を格納
6. 結果: `predictions[frame_index] = [[s0,t0], [s1,t1], ..., [sN,tN]]`

### 6.4 load_model() の動作

1. `torch.load()` でチェックポイント読み込み
2. `model_type == "gru_trajectory"` を検証
3. チェックポイントの `config` から `GRUTrajectoryModel` を再構築
4. `model.eval()` に設定して返却
5. 返り値: `(model, config_dict, selected_sources)`

### 6.5 モデル保存形式

ファイル名: `models/gru_{timestamp}_{seq_len}s_{pred_horizon}h.pth`

```python
{
    "model_state_dict": OrderedDict,
    "model_type": "gru_trajectory",
    "config": {
        "num_image_sources": int,
        "seq_len": int,
        "pred_horizon": int,
        "gru_hidden": int,
        "gru_layers": int,
        "dropout": float,
        "img_size": tuple,
    },
    "selected_sources": List[str],
    "train_losses": List[float],
    "val_losses": List[float],
    "epochs_trained": int,
    "created_at": "ISO8601"
}
```

### 6.6 MLflow ログ項目

`MLflowManager.log_gru_trajectory_model()` で記録:

- params: model_type, seq_len, pred_horizon, gru_hidden, gru_layers, dropout, epochs, lr, batch_size, num_image_sources, selected_sources, augmentation_enabled, stride
- metrics: best_val_loss, final_train_loss, total_training_time, avg_epoch_time, completed_epochs
- tags: model_category=gru_trajectory, task_type=trajectory_prediction, framework=pytorch

---

## 7. UI統合 (`main.py`) [実装済]

> **注意**: 本アプリは **PyQt5 デスクトップアプリ**（Flask/Web ではない）。
> 元の仕様書§7で記載した REST APIエンドポイントは不要。代わりにPyQt5のダイアログ・シグナルで統合。

### 7.1 UIセクション配置

左パネル内、位置推論モデル・ウェイポイントモデルの下に配置:

```
位置推論モデル  (既存)
ウェイポイント推論モデル  [開発中] [▶]   ← 折りたたみ式
GRU軌道予測モデル         [開発中] [▶]   ← 折りたたみ式
```

- 両セクションともヘッダーにオレンジ色の「開発中」バッジ表示
- 展開ボタン `▶`/`▼` で中身を表示/非表示（初期状態: 折りたたみ）

### 7.2 GRUセクション展開時の構成

```
GRU軌道予測モデル  [開発中]  [▼]
├── [GRU軌道学習] [GRU軌道推論]    ← ボタン
└── ☐ GRU予測軌道を表示           ← チェックボックス
```

### 7.3 GRU学習ダイアログ (`train_gru_trajectory_model()`)

`QPushButton` → `QDialog` で設定ダイアログを表示:

```
[GRU軌道予測モデル - 学習設定]
├── 画像ソース選択 (QCheckBox × バリアント数)
│   └── 選択数表示 "X ソース選択中"
├── GRU パラメータ (QFormLayout)
│   ├── 入力シーケンス長: QSpinBox (1-100, default=8)
│   ├── 予測ステップ数:   QSpinBox (1-100, default=10)
│   ├── ストライド:       QSpinBox (1-50, default=1)
│   ├── GRU Hidden Size:  QComboBox [64,128,256,512] (default=256)
│   ├── GRU レイヤー数:   QSpinBox (1-4, default=1)
│   ├── Dropout:          QDoubleSpinBox (0.0-0.5, default=0.1)
│   └── シーケンス情報:   "入力: Xフレーム → 予測: Yフレーム" (動的更新)
├── 学習パラメータ (QFormLayout)
│   ├── Epochs:        QSpinBox (1-500, default=50)
│   ├── Batch Size:    QComboBox [4,8,16,32,64] (default=32)
│   ├── Learning Rate: QComboBox [0.0001,0.0005,0.001,0.005,0.01]
│   ├── Val Split:     QDoubleSpinBox (0.05-0.5, default=0.2)
│   └── Augmentation:  QCheckBox (default=True)
├── データ範囲選択
│   ├── ○ 全アノテーション使用
│   ├── ○ スキップ使用 (N件ごと)
│   └── ○ インデックス範囲指定
└── [学習開始] [キャンセル]
```

学習実行中は `QProgressDialog` でエポック進捗を表示。

### 7.4 GRU推論ダイアログ (`run_gru_prediction()`)

1. `models/` フォルダから `gru_*.pth` ファイルを検索
2. 各モデルのチェックポイントを読み込み、`model_type == "gru_trajectory"` を確認
3. モデル一覧を `QListWidget` で表示（ファイル名 + seq_len/pred_horizon/sources情報）
4. 選択後、`GRUTrainingManager.predict()` を実行
5. 結果を `self.gru_predictions` に格納
6. `gru_prediction_checkbox` を自動ON

### 7.5 予測結果の表示

#### 情報パネル表示 (`_update_gru_prediction_display()`)

`update_inference_display()` の末尾から呼び出し。現在フレームに対応する予測がある場合:

```html
<b>GRU予測(steering) / GRU予測(throttle)</b>
<table>
  <tr><td>t+1:</td><td style="color:#FF6600">+0.123</td><td style="color:#0066FF">+0.456</td></tr>
  <tr><td>t+2:</td><td style="color:#FF6600">+0.134</td><td style="color:#0066FF">+0.445</td></tr>
  ...
</table>
```

#### 画像オーバーレイ (ImageLabel paintEvent)

- 予測の最初のステップ (t+1) の `(steering, throttle)` を座標変換して画像上にオレンジ色の円で描画
- 座標変換: `x = (steering + 1) / 2 * img_width`, `y = (1 - throttle) / 2 * img_height`
- 色: `QColor(255, 102, 0)` (オレンジ), ペン幅4, 円サイズ26px

### 7.6 データ構造

```python
# main.py内の属性
self.gru_predictions = {
    frame_index: [[s0, t0], [s1, t1], ..., [sN, tN]],  # pred_horizon × 2
    ...
}
self.gru_prediction_config = {
    "seq_len": int,
    "pred_horizon": int,
    "gru_hidden": int,
    ...
}
self.show_gru_predictions = bool  # 表示ON/OFF

# ImageLabel内の属性
self.gru_prediction_point = QPoint | None
self.show_gru_prediction = bool
```

---

## 8. 翻訳キー (`translations.py`) [実装済]

### 日本語 (ja)

| キー | 値 |
|------|-----|
| `btn_gru_train` | GRU軌道学習 |
| `btn_gru_predict` | GRU軌道推論 |
| `dlg_gru_training_settings` | GRU軌道予測モデル - 学習設定 |
| `dlg_gru_prediction` | GRU軌道予測 |
| `label_gru_params` | GRU パラメータ |
| `label_seq_len` | 入力シーケンス長 |
| `label_pred_horizon` | 予測ステップ数 |
| `label_stride` | ストライド |
| `label_gru_hidden` | GRU Hidden Size |
| `label_gru_layers` | GRU レイヤー数 |
| `label_dropout` | Dropout |
| `label_gru_seq_info` | 入力: {0}フレーム → 予測: {1}フレーム |
| `msg_gru_training` | GRU軌道予測モデルの学習中... |
| `msg_gru_training_complete` | GRU軌道予測モデルの学習が完了しました |
| `msg_gru_no_sequences` | 有効なシーケンスが生成されませんでした... |
| `msg_gru_predicting` | GRU軌道予測の推論中... |
| `msg_gru_prediction_complete` | GRU軌道予測の推論が完了しました |
| `msg_gru_prediction_count` | {0} フレームの予測を生成しました |
| `msg_no_gru_models` | GRUモデルが見つかりません... |
| `label_select_gru_model` | GRUモデル選択 |
| `label_gru_model_info` | seq_len={0}, pred_horizon={1}, sources={2} |
| `chk_show_gru_prediction` | GRU予測軌道を表示 |
| `label_gru_pred_steering` | GRU予測(steering) |
| `label_gru_pred_throttle` | GRU予測(throttle) |
| `label_gru_section_title` | GRU軌道予測モデル |
| `label_dev_in_progress` | 開発中 |

英語 (en) も同様に対応済み。

---

## 9. config.py定数 [実装済]

```python
GRU_DEFAULT_SEQ_LEN = 8
GRU_DEFAULT_PRED_HORIZON = 10
GRU_DEFAULT_STRIDE = 1
GRU_DEFAULT_GRU_HIDDEN = 256
GRU_DEFAULT_GRU_LAYERS = 1
GRU_DEFAULT_DROPOUT = 0.1
GRU_DEFAULT_IMG_SIZE = (128, 128)
GRU_DEFAULT_EPOCHS = 50
GRU_DEFAULT_BATCH_SIZE = 32
```

---

## 10. 依存ライブラリ [実装済]

`requirements.txt` に追記済み:

```
timm>=1.0.14   # MobileNetV3-Small用 (元仕様のtorchvision→timmに変更)
```

その他は既存の `torch`, `numpy`, `Pillow`, `mlflow`, `torchvision` で対応。

---

## 11. 既存コードとの統合ルール

1. **独立したモジュール構成**: `managers/gru_*.py` の3ファイルで完結。既存クラスの継承なし
2. **データの受け渡し**: `main.py` の `self.annotations`, `self.images`, `self.source_images_map` を直接渡す
3. **モデルファイルの互換性**: 既存MLPモデルと同じ `models/` フォルダに保存。`model_type: "gru_trajectory"` キーで区別
4. **UI配置**: 左パネルの位置推論・ウェイポイントセクションの下に配置。折りたたみ式 + 開発中バッジ

---

## 12. テスト観点

### 学習テスト
- [ ] 画像ソース1個（最小構成）で学習が完走すること
- [ ] 画像ソース2〜5個で学習が完走すること
- [ ] 画像ソース0個選択時に学習開始ボタンが非活性になること
- [ ] `seq_len=1` のエッジケースで動作すること
- [ ] キャンセル時にベストモデルが保存されること
- [ ] セッションをまたぐシーケンスが除外されること（インデックスギャップ>1で境界判定）
- [ ] 削除インデックスを含むシーケンスが除外されること
- [ ] MLflowに正しくログが記録されること

### 推論テスト
- [ ] 保存モデルを読み込んで予測が実行できること
- [ ] 予測結果が情報パネルにテーブル形式で正しく表示されること
- [ ] 画像上にオレンジ色の予測ポイントが正しい位置に表示されること
- [ ] チェックボックスOFF時に表示がクリアされること
- [ ] 推論完了後にチェックボックスが自動ONになること

### UI/UXテスト
- [ ] GRUセクションの展開/折りたたみが正しく動作すること
- [ ] ウェイポイントセクションの展開/折りたたみが正しく動作すること
- [ ] 「開発中」バッジが両セクションに表示されること
- [ ] Jetson Orin Nano (CUDA) / CPU の両方で動作すること

---

## 13. 今後の開発タスク（未実装）

### 13.1 優先度: 高

| タスク | 説明 | 関連ファイル |
|--------|------|-------------|
| 継続学習 | 既存GRUモデルを読み込んで追加学習する機能。`base_model_path` を指定して `load_model()` → optimizerの再構築 | `gru_training_manager.py`, `main.py` |
| 学習曲線表示 | 学習完了後にTrain Loss / Val Loss のグラフをダイアログ内に表示（matplotlib） | `main.py` |
| バッチサイズ自動調整 | シーケンス数が `batch_size` 未満の場合に自動でbatch_sizeを縮小 + 警告表示 | `gru_training_manager.py` |

### 13.2 優先度: 中

| タスク | 説明 | 関連ファイル |
|--------|------|-------------|
| 重み付き損失関数 | 近未来ステップの予測を重視する指数減衰重み `[1.0, 0.9, 0.8, ...]` | `gru_training_manager.py` |
| 予測軌道のフル表示 | 現在は t+1 のみ画像上に表示。t+1〜t+N までの全ステップを軌跡として描画 | `main.py` (ImageLabel paintEvent) |
| EarlyStopping | 既存の `EarlyStopping` クラスを流用してGRU学習にも適用 | `gru_training_manager.py` |
| GRUモデル一覧管理 | モデル管理UIにGRUモデルのフィルタ表示・削除機能 | `main.py` |

### 13.3 優先度: 低

| タスク | 説明 | 関連ファイル |
|--------|------|-------------|
| マルチステップ画像表示 | 予測の各ステップの(steering, throttle)を時系列グラフとして情報パネル内に表示 | `main.py` |
| 推論バッチサイズ設定 | 現在は固定16。VRAM量に応じて調整可能にする | `gru_training_manager.py` |
| ONNX/TensorRTエクスポート | 学習済みモデルをJetson向けに最適化エクスポート | 新規ファイル |
| ego_stateキーのUI設定 | vx/vy/omegaのカタログキー名をUIから指定可能にする | `main.py`, `translations.py` |

---

## 14. ディレクトリ構成（現在）

```
annotation_training_d2j/
├── managers/
│   ├── __init__.py                ← MODIFIED: GRUクラスexport追加
│   ├── gru_model.py               ← NEW: GRUモデル定義
│   ├── gru_training_manager.py    ← NEW: 学習ループ + 推論 + モデルロード
│   ├── gru_dataset.py             ← NEW: シーケンスDataset
│   └── mlflow_manager.py          ← MODIFIED: GRU_TRAJECTORY追加
├── main.py                        ← MODIFIED: UIセクション・ダイアログ・表示
├── config.py                      ← MODIFIED: GRUデフォルト定数
├── translations.py                ← MODIFIED: GRU関連翻訳キー
├── requirements.txt               ← MODIFIED: timm追加
├── dev/
│   └── SPEC_GRU_trajectory.md     ← THIS FILE
└── models/                        ← GRUモデル保存先 (gru_*.pth)
```

---

## 15. 実装時の注意事項（Claude Code向け）

### 15.1 アーキテクチャ理解

- 本アプリは **PyQt5 デスクトップGUI** である（Flask/Web ではない）
- `main.py` は20000行超の大規模ファイル。全体を読む必要はなく、関連メソッドのみ参照すること
- UIは左パネル内にセクションとして動的追加される（`add_*_section()` パターン）
- 学習/推論は **メインスレッドのQProgressDialog** で進捗表示（別スレッドではない）

### 15.2 データアクセスパターン

```python
# main.py内の主要データ属性
self.annotations    # Dict[int, dict]  — キー: フレームインデックス, 値: {"angle": float, "throttle": float, ...}
self.images         # List[str]        — メイン画像の絶対パスリスト
self.source_images_map  # Dict[str, List[str]] — バリアント名→画像パスリスト
self.deleted_indexes    # set[int]     — 削除済みインデックス
self.current_index      # int          — 現在表示中のフレームインデックス
self.mlflow_manager     # MLflowManager
```

### 15.3 既存推論との違い

| 項目 | 既存MLP推論 | GRU推論 |
|------|-----------|---------|
| 入力 | 単一フレーム画像 | T フレームの画像シーケンス + ego_state |
| 出力 | `{angle, throttle, x, y}` | `[[s0,t0], [s1,t1], ..., [sN,tN]]` (軌道) |
| 格納先 | `self.inference_results[idx]` | `self.gru_predictions[idx]` |
| 表示色 | Cyan (0,255,255) | Orange (255,102,0) |
| 表示方法 | 単一点 | t+1の点 + 情報パネルに全ステップテーブル |

### 15.4 翻訳キー追加時の手順

1. `translations.py` の `'ja'` セクション末尾（GRU推論関連の後）に追加
2. `translations.py` の `'en'` セクション末尾にも同キーで英語版を追加
3. フォーマット引数がある場合は `{0}`, `{1}` 形式（`get_text('key', arg0, arg1)` で呼び出し）

### 15.5 スタイル適用

```python
apply_style(button, 'training')  # 学習系ボタン
apply_style(button, 'model')     # モデル操作系ボタン
```
