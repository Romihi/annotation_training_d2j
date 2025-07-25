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

## 使用ガイド

### 基本的なワークフロー

1. **画像の読み込み**: 「参照」をクリックして画像フォルダを選択し、「画像を読込」をクリック
2. **アノテーションモードの選択**: 「自動運転」または「物体検知」モードを選択
3. **アノテーションの作成**:
   - 運転モードでは、画像をクリックしてステアリングとスロットル値を設定
   - 物体検知モードでは、クリック＆ドラッグでバウンディングボックスを作成
4. **位置の割り当て**: 右側の位置ボタンを使用して、画像に位置IDをタグ付け
5. **保存とエクスポート**: エクスポートボタンを使用して、希望の形式でアノテーションを保存

### キーボードショートカット

- **Bキー**: 運転モードと物体検知モードの切り替え
- **左/右矢印キー**: 10枚前後に移動
- **Delete/Backspace**: 選択したバウンディングボックスを削除

### モデルのトレーニング

1. ドロップダウンからモデルタイプを選択
2. 「学習・保存」をクリックしてトレーニングパラメータを設定
3. エポック数、学習率、データ拡張などの設定を調整
4. トレーニングを開始し、進捗を監視
5. MLflow（有効な場合）で結果を表示

### 自動アノテーション

1. 数枚の画像を手動でアノテーション
2. ドロップダウンからモデルを選択（またはデフォルトを使用）
3. 「オートアノテーション実行」をクリックして残りの画像を自動的にアノテーション

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


## 謝辞

このツールは、Togikaidrive、DonkeycarやJetracerなどのAIミニカープロジェクトをサポートするために開発されました。
