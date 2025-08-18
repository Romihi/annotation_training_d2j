# AIミニカーアノテーションツール

このツールは、自動運転と物体検知データセットのためのアノテーションを作成・管理するためのグラフィカルインターフェースを提供します。DonkeycarやJetracerなどのAIミニカー向けに設計されており、以下のことが可能です：

1. 画像データセットの読み込み
2. 運転アノテーション（ステアリング角度とスロットル）の作成
3. 物体検知アノテーション（バウンディングボックス）の作成
4. 画像への位置情報の割り当て
5. モデルのトレーニングと評価
6. 様々な形式でのアノテーションのエクスポート

## 機能

### アノテーション機能
- **運転制御**: ステアリング角度とスロットルのアノテーション
- **物体検知**: オブジェクト（車、人、標識、コーンなど）のバウンディングボックス作成
- **位置タグ付け**: 画像に位置識別子を割り当て
- **バッチ操作**: 自動アノテーション、範囲削除など

### トレーニングと推論
- **モデルトレーニング**: 運転制御用のニューラルネットワークをトレーニング
- **YOLO統合**: 物体検知用のYOLOモデルのトレーニングと使用
- **データ拡張**: 拡張によるトレーニングデータの強化
- **推論プレビュー**: 画像上でのモデル予測の表示

### エクスポートオプション
- **Donkeycar形式**: Donkeycar用のアノテーションをエクスポート
- **Jetracer形式**: Jetracer用のアノテーションをエクスポート
- **YOLO形式**: YOLO用のアノテーションをエクスポート
- **アノテーション動画**: アノテーションを可視化した動画の作成

### 高度な機能
- **実験追跡**: モデルパフォーマンス追跡のためのMLflow統合
- **セッション管理**: アノテーションセッションの保存と復元
- **拡張プレビュー**: トレーニング前に拡張効果を確認

## 要件

- Python 3.7以上
- PyQt5
- PyTorch
- Pillow
- NumPy
- MLflow (オプション、実験追跡用)
- Ultralytics (YOLOサポート用)

詳細なバージョン要件については、requirements.txtを参照してください。

## インストール

1. リポジトリをクローンします:
```bash
git clone https://github.com/Romihi/annotation_training_d2j.git
cd minicar-annotation-tool
```

2. 必要なパッケージをインストールします:
```bash
pip install -r requirements.txt
```

3. ツールを実行します:
```bash
python main.py
```

## 操作説明

### 初回起動から学習完了までの詳細手順

#### 1. 準備とツールの起動

1. **事前準備**
   - 走行データ（画像）を含むフォルダを用意します
   - フォルダ構造例：
     ```
     your_data_folder/
     └── images/
         ├── 0_abcd123_image_array_.jpg
         ├── 1_abcd124_image_array_.jpg
         └── ...
     ```

2. **ツールの起動**
   ```bash
   python main.py
   ```

#### 2. 画像データの読み込み

1. **フォルダの選択**
   - 左パネルの「データ読込」セクションで「参照...」ボタンをクリック
   - `images`フォルダの**親フォルダ**を選択（重要：imagesフォルダ自体ではなく、その上のフォルダを選択）
   - 複数フォルダの場合：Ctrlキーを押しながら複数選択可能

2. **画像の読み込み**
   - 「画像を読込」ボタンをクリック
   - 読み込み成功すると：
     - 中央に最初の画像が表示されます
     - 下部にサムネイルギャラリーが表示されます
     - 左パネルに統計情報（総画像数など）が表示されます

3. **既存アノテーションの読み込み（オプション）**
   - 以前のアノテーションがある場合：「アノテーションデータを読込」ボタンをクリック
   - 自動的に同階層のcatalogファイルを検索して読み込みます

#### 3. アノテーションの作成

##### A. 自動運転アノテーション（ステアリング・スロットル）

1. **モードの確認**
   - 左下の「Annotation Mode」が「自動運転モード」になっていることを確認
   - 違う場合は「Bキー」で切り替え

2. **アノテーション方法**
   - 画像上の任意の点をクリック
   - クリック位置に基づいて自動的に値が計算されます：
     - 横軸（X）：ステアリング角度（-1.0〜1.0）
     - 縦軸（Y）：スロットル値（-1.0〜1.0）
   - 赤い点が表示され、アノテーションが記録されます

3. **効率的な作業のコツ**
   - 「設定」→「アノテーション設定」で「クリック後自動次へ」にチェックを入れると自動的に次の画像に移動
   - スキップ枚数を設定して、一定間隔でアノテーション可能

##### B. 物体検知アノテーション（バウンディングボックス）

1. **モードの切り替え**
   - 「Bキー」を押して「物体検知モード」に切り替え
   - または左パネルでクラス設定を行うと自動的に切り替わります

2. **クラスの設定**
   - 「検知クラス設定」で検知したいクラスを入力
   - 例：`car,person,sign,cone`（カンマ区切り）
   - 「プリセット」ボタンで一般的なクラスセットを選択可能

3. **バウンディングボックスの描画**
   - マウスでドラッグして矩形を描画
   - リリース時にクラス選択ダイアログが表示
   - クラスを選択して確定

4. **編集操作**
   - 選択：既存のボックスをクリック
   - 移動：選択したボックスをドラッグ
   - リサイズ：ボックスの角をドラッグ
   - 削除：選択してDeleteキー

##### C. 位置情報アノテーション

1. **位置ボタンの使用**
   - 右パネルの数字ボタン（0-7）をクリック
   - 現在の画像に位置IDが割り当てられます
   - コース上の特定の場所を識別するのに使用

#### 4. データの保存とエクスポート

1. **Donkeycar形式でエクスポート**
   - 「Donkey」ボタンをクリック
   - 保存先フォルダを選択
   - catalog.jsonとmyconfig.pyが生成されます

2. **YOLO形式でエクスポート**（物体検知の場合）
   - 「YOLO」ボタンをクリック
   - タスクタイプ（検知/セグメンテーション）を選択
   - train/valの分割比率を設定（デフォルト80:20）

3. **Jetracer形式でエクスポート**
   - 「Jetracer」ボタンをクリック
   - 位置情報付きのデータセットが生成されます

#### 5. モデルの学習

##### A. 自動運転モデルの学習

1. **モデル選択**
   - 「走行モデル選択」ドロップダウンからモデルを選択
   - 推奨：初心者は「linear」、精度重視なら「vit_b_16」

2. **学習設定**
   - 「モデル学習・保存」ボタンをクリック
   - 学習パラメータを設定：
     - エポック数：30（デフォルト）
     - バッチサイズ：16
     - 学習率：0.001
     - Early Stopping：有効（推奨）

3. **データ拡張設定**
   - 「データオーグメンテーションを有効にする」にチェック
   - 各種拡張パラメータを調整（プレビュー機能で確認可能）

4. **学習開始**
   - 「学習開始」ボタンをクリック
   - プログレスバーで進捗を確認
   - 完了後、modelsフォルダに保存されます

##### B. YOLOモデルの学習（物体検知）

1. **YOLOモデル選択**
   - YOLOモデルタイプを選択（yolov8n推奨）
   - タスクに応じて適切なモデルを選択

2. **学習設定**
   - 「YOLO学習」ボタンをクリック
   - パラメータ設定：
     - エポック数：30-50
     - 画像サイズ：640（デフォルト）
     - バッチサイズ：GPUメモリに応じて調整

3. **学習実行**
   - データセットの準備が自動的に行われます
   - 学習中はリアルタイムで損失値が表示されます

#### 6. 学習済みモデルの使用

1. **モデルの読み込み**
   - 「モデル読込」ボタンをクリック
   - modelsフォルダから学習済みモデルを選択

2. **推論の実行**
   - 「推論結果表示」にチェック
   - 画像を切り替えると自動的に推論が実行されます
   - 青い点（自動運転）または緑の枠（物体検知）で結果が表示

3. **バッチ推論**
   - 「全画像を推論」ボタンで一括推論
   - 結果はCSVファイルにエクスポート可能

#### 7. 高度な機能

##### オートアノテーション
1. 最初の10-20枚を手動でアノテーション
2. 「オートアノテーション実行」をクリック
3. 設定：
   - 使用モデル：手動アノテーションから学習
   - 信頼度しきい値：0.7以上推奨
4. 実行後、結果を確認して必要に応じて修正

##### MLflow統合
1. 「Mlflowを開く」をクリック
2. ブラウザでMLflow UIが開きます
3. 各学習実行の詳細を確認：
   - 損失グラフ
   - パラメータ
   - メトリクス
   - モデルファイル

##### セッション管理
- 作業は自動的に保存されます
- 次回起動時に前回のセッションを復元可能
- 複数プロジェクトの並行作業に対応

### キーボードショートカット一覧

| キー | 機能 |
|------|------|
| B | アノテーションモード切り替え |
| ← / → | 前/次の画像（自動スキップ設定に連動） |
| Delete / Backspace | 【自動運転モード】現在のアノテーション（angle/throttle/位置）を削除<br>【物体検知/セグメンテーションモード】選択項目の削除 |
| Space | 自動再生/停止 |
| S | 現在の設定を保存 |
| Ctrl + Z | 直前の操作を取り消し |
| 0-7 | 【自動運転モード】位置情報の設定/解除（同じキー再押下で解除） |

### トラブルシューティング

**Q: 画像が読み込まれない**
- A: imagesフォルダの親フォルダを選択しているか確認してください

**Q: アノテーションが保存されない**
- A: エクスポートボタンで明示的に保存する必要があります

**Q: 学習が遅い**
- A: GPUが利用可能か確認。バッチサイズを小さくしてみてください

**Q: メモリエラーが発生する**
- A: 画像サイズを小さくするか、バッチサイズを減らしてください

## カスタムモデルアーキテクチャの作成

このツールでは、独自のニューラルネットワークアーキテクチャを作成して使用できます。

### 1. 基本的なモデルの作成

`model_catalog.py`に新しいモデルクラスを追加します：

```python
class YourCustomModel(BaseModel):
    """あなたのカスタムモデル"""
    
    def __init__(self, input_size=(3, 120, 160), num_outputs=2):
        super().__init__()
        self.name = "your_custom_model"
        
        # カスタムレイヤーを定義
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # 出力サイズを計算
        dummy_input = torch.zeros(1, *input_size)
        with torch.no_grad():
            conv_output = self._forward_conv(dummy_input)
            self.fc_input_size = conv_output.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_outputs)
        self.dropout = nn.Dropout(0.5)
    
    def _forward_conv(self, x):
        """畳み込み層の前向き処理"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
    
    def forward(self, x):
        """前向き処理"""
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 2. TIMMベースのモデルの作成

既存のPre-trainedモデルを使用する場合：

```python
class YourTIMMModel(TIMMBasedModel):
    """TIMMライブラリを使用したカスタムモデル"""
    
    def __init__(self, num_outputs=2):
        # TIMMのモデル名を指定
        super().__init__(
            model_name="efficientnet_b0",  # TIMMのモデル名
            pretrained=True,
            num_outputs=num_outputs
        )
        self.name = "your_timm_model"
```

### 3. マルチ画像入力モデルの作成

複数の画像を入力とするモデル：

```python
class YourMultiImageModel(BaseModel):
    """複数画像入力のカスタムモデル"""
    
    def __init__(self, num_images=2, input_size=(3, 120, 160), num_outputs=2):
        super().__init__()
        self.name = "your_multi_image_model"
        self.num_images = num_images
        
        # 単一画像用の特徴抽出器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 複数画像の特徴を結合
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4 * num_images, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, x):
        # x: (batch_size, num_images, channels, height, width)
        batch_size = x.size(0)
        
        # 各画像の特徴を抽出
        features = []
        for i in range(self.num_images):
            feat = self.feature_extractor(x[:, i])
            features.append(feat.view(batch_size, -1))
        
        # 特徴を結合
        combined = torch.cat(features, dim=1)
        return self.fc(combined)
```

### 4. 位置分類モデルの作成

位置予測用のカスタムモデル：

```python
class YourLocationModel(BaseLocationModel):
    """位置分類用カスタムモデル"""
    
    def __init__(self, num_locations=10, input_size=(3, 120, 160)):
        super().__init__(num_locations=num_locations)
        self.name = "your_location_model"
        
        # 特徴抽出部
        self.backbone = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=True,
            num_classes=0,  # 分類層を除去
            global_pool='avg'
        )
        
        # 位置分類用の出力层
        self.classifier = nn.Linear(
            self.backbone.num_features, 
            num_locations
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```

### 5. モデルカタログへの登録

`MODEL_REGISTRY`に新しいモデルを追加：

```python
# model_catalog.py の最後に追加
MODEL_REGISTRY.update({
    "your_custom_model": YourCustomModel,
    "your_timm_model": YourTIMMModel,
    "your_multi_image_model": YourMultiImageModel,
    "your_location_model": YourLocationModel,
})
```

### 6. モデル情報の追加

`model_info.py`にモデルの詳細情報を追加：

```python
MODEL_INFO.update({
    "your_custom_model": {
        "accuracy_top1": None,  # 未測定の場合はNone
        "accuracy_top5": None,
        "params": 1.2,  # パラメータ数（百万単位）
        "gflops": 0.5,  # 計算量
        "input_size": (3, 120, 160),
        "paper": "Your Paper Title",
        "url": "https://your-paper-url.com"
    }
})
```

### 7. カスタムトレーニング関数

特殊なトレーニングロジックが必要な場合：

```python
def train_custom_model(model, train_loader, val_loader, config):
    """カスタムモデル用の訓練関数"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    criterion = nn.MSELoss()
    
    for epoch in range(config.get('epochs', 10)):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # カスタム前処理
            if isinstance(data, list):  # マルチ画像の場合
                data = torch.stack(data, dim=1)
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')
    
    return model
```

### 8. モデル使用のベストプラクティス

#### 入力データの前処理：
```python
def preprocess_for_your_model(image):
    """カスタムモデル用の前処理"""
    # リサイズ
    image = cv2.resize(image, (160, 120))
    
    # 正規化
    if image.max() > 1:
        image = image / 255.0
    
    # カスタム正規化（必要に応じて）
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    
    return image
```

#### エラーハンドリング：
```python
class YourCustomModel(BaseModel):
    def forward(self, x):
        try:
            # モデルの処理
            return self._forward_impl(x)
        except Exception as e:
            print(f"モデル実行エラー: {e}")
            # デフォルト値を返す
            return torch.zeros(x.size(0), 2)
```

### 9. モデルのテスト

新しいモデルをテストする方法：

```python
def test_custom_model():
    """カスタムモデルのテスト"""
    model = YourCustomModel()
    
    # ダミーデータでテスト
    dummy_input = torch.randn(1, 3, 120, 160)
    
    try:
        output = model(dummy_input)
        print(f"出力形状: {output.shape}")
        print(f"出力値: {output}")
        return True
    except Exception as e:
        print(f"テストエラー: {e}")
        return False

# テスト実行
if __name__ == "__main__":
    test_custom_model()
```

### 使用可能なコンポーネント

- **畳み込み層**: `nn.Conv2d`, `nn.Conv1d`
- **プーリング**: `nn.MaxPool2d`, `nn.AdaptiveAvgPool2d`
- **正規化**: `nn.BatchNorm2d`, `nn.GroupNorm`
- **活性化**: `nn.ReLU`, `nn.GELU`, `nn.Swish`
- **注意機構**: `nn.MultiheadAttention`
- **ドロップアウト**: `nn.Dropout`, `nn.Dropout2d`

このように、様々なタイプのカスタムモデルを作成して、あなたの特定のユースケースに合わせたアーキテクチャを実装できます。

## フォルダ構造

ツールは以下のディレクトリを作成します:
- `annotation/`: すべてのアノテーションデータのメインディレクトリ
- `annotation/data_donkey/`: Donkeycarエクスポートファイル用
- `annotation/data_jetracer/`: Jetracerエクスポートファイル用
- `models/`: 保存されたトレーニング済みモデル用
- `mlruns/`: MLflow実験追跡データ用

## トラブルシューティング

### 一般的な問題

1. **CUDA/GPUエラー**: PyTorch用の互換性のあるCUDAドライバがインストールされていることを確認
2. **画像読み込みの問題**: 画像形式の互換性を確認（JPG、PNG、BMPがサポートされています）
3. **MLflowエラー**: MLflowはオプションです。問題が発生した場合は、その機能を無効にできます

### エラー報告

バグが見つかった場合は、以下の情報を含む問題を開いてください:
- エラーメッセージ
- 再現手順
- オペレーティングシステムとPythonバージョン

## ライセンス（License）
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Third-party Libraries
- PyQt5: Licensed under GPL v3


## 変更履歴

### 2025-01-21 - v1.3.0
#### 🔧 クリック処理の改善
- **連続クリック防止機能を追加**
  - 100ms以内の連続クリックを無視するデバウンス機能を実装
  - 画像描画中のクリックを無効化して意図しないアノテーションを防止
  - クリック無効時は視覚的フィードバック（禁止カーソル）を表示

- **アノテーション表示問題を修復**
  - 既存アノテーション（赤丸）が表示されない問題を解決
  - 画像読み込みの成功/失敗に関係なくアノテーションポイントを適切に設定

#### ⌨️ キーボード操作の改善
- **左右矢印キーの動作を統一**
  - 「クリック時自動スキップ枚数」設定に連動するように変更
  - チェック有効時：設定されたスキップ枚数で移動（デフォルト10枚）
  - チェック無効時：1枚ずつ移動
  - クリック操作とキーボード操作で一貫した動作を実現

## 謝辞

このツールは、Togikaidrive、DonkeycarやJetracerなどのAIミニカープロジェクトをサポートするために開発されました。
