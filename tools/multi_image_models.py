"""
multi_image_models.py - マルチ画像入力モデルの実装
"""
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# 既存のコードをインポート
from model_catalog import (BaseModel, DonkeyModel, DonkeyModel_FCN, 
                          load_model_weights, MODEL_REGISTRY, get_model)


class MultiInputBaseModel(BaseModel):
    """複数画像入力モデルの基底クラス"""
    def __init__(self, name="multi_base", num_inputs=2, pretrained=False):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        self.pretrained = pretrained
    
    def run(self, *img_arrs):
        """複数の画像アレイに対して推論を実行"""
        # 正しい数の画像が提供されたか確認
        if len(img_arrs) != self.num_inputs:
            raise ValueError(f"期待される入力画像数は{self.num_inputs}枚ですが、{len(img_arrs)}枚が提供されました")
        
        # 前処理パイプラインが初期化されていなければ作成
        if self._preprocess is None:
            self._preprocess = self.get_preprocess()
        
        # 各画像を処理してテンソルに変換
        tensors = []
        for img_arr in img_arrs:
            pil_image = Image.fromarray(img_arr)
            tensor = self._preprocess(pil_image).unsqueeze(0).to(self.device)
            tensors.append(tensor)
        
        # 勾配計算なしで推論を実行
        with torch.no_grad():
            result = self(*tensors)
        
        # CPU上のNumPy配列に変換
        if result.device.type != 'cpu':
            result = result.cpu()
        result = result.numpy().reshape(-1)
        
        # [-1, 1]の範囲に正規化
        result = result * 2 - 1
        
        return result[0], result[1]  # angle, throttle

class MultiImageConcatModel(MultiInputBaseModel):
    """複数画像の特徴量を連結するモデル"""
    def __init__(self, num_inputs=2, input_size=(224, 224), pretrained=False):
        super().__init__(name=f"donkey_multi{num_inputs}_concat", num_inputs=num_inputs, pretrained=pretrained)
        self.input_size = input_size
        
        # 特徴抽出器を作成
        self.feature = DonkeyModel(input_size=input_size, pretrained=pretrained).features
        self.dense = DonkeyModel(input_size=input_size, pretrained=pretrained).dense_layers
        
        # 特徴量のサイズをテスト
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            dummy_output = self.dense(self.feature(dummy_input))
        
        feature_size = dummy_output.shape[1]
        print(f"MultiImageConcat feature size: {feature_size} per image, total: {feature_size * num_inputs}")
        
        # 連結された特徴量用の回帰器
        self.regressor = nn.Linear(feature_size * num_inputs, 2)

    def forward(self, *inputs):
        """複数の入力テンソルを処理して予測を行う"""
        # 各入力から特徴を抽出
        features = []
        for x in inputs:
            # 特徴抽出と密結合層を適用
            f = self.dense(self.feature(x))
            features.append(f)
        
        # 特徴量を特徴次元でコンカチネート
        concat_features = torch.cat(features, dim=1)
        
        # 最終予測
        return self.regressor(concat_features)
    
    def get_preprocess(self):
        """入力画像の前処理パイプラインを返す"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

class MultiImageChannelModel(MultiInputBaseModel):
    """複数画像をチャネル方向に連結するモデル"""
    def __init__(self, num_inputs=2, input_size=(224, 224), pretrained=False):
        super().__init__(name=f"donkey_multi{num_inputs}_channel", num_inputs=num_inputs, pretrained=pretrained)
        self.input_size = input_size
        
        # 入力チャネル数を計算（RGB画像×N枚）
        input_channels = 3 * num_inputs
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=5, stride=2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Conv2d(24, 32, kernel_size=5, stride=2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Flatten()
        )
        
        # 出力次元を計算
        dummy = torch.zeros(1, input_channels, *input_size)
        out_dim = self.conv(dummy).shape[1]
        print(f"MultiImageChannel output dimension: {out_dim}")
        
        # 回帰器
        self.regressor = nn.Sequential(
            nn.Linear(out_dim, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 50), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(50, 2)
        )

    def forward(self, *inputs):
        """複数の入力テンソルを処理して予測を行う"""
        # 入力をチャネル次元で連結
        x = torch.cat(inputs, dim=1)
        
        # 畳み込み層と回帰器を適用
        return self.regressor(self.conv(x))
    
    def get_preprocess(self):
        """入力画像の前処理パイプラインを返す"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

class MultiImageAttentionModel(MultiInputBaseModel):
    """アテンションを使用して複数画像を処理するモデル"""
    def __init__(self, num_inputs=2, input_size=(224, 224), pretrained=False):
        super().__init__(name=f"donkey_multi{num_inputs}_attention", num_inputs=num_inputs, pretrained=pretrained)
        self.input_size = input_size
        
        # 特徴抽出器
        self.feature = DonkeyModel_FCN(input_size=input_size, pretrained=pretrained).features
        
        # 特徴次元を計算
        dummy = self.feature(torch.zeros(1, 3, *input_size))
        dim = dummy.shape[1]
        print(f"Attention model feature dimension: {dim}")
        
        # マルチヘッドアテンション層
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
        # 回帰器
        self.regressor = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, *inputs):
        """複数の入力テンソルをアテンション機構で処理"""
        # 各入力から特徴を抽出してシーケンスとしてスタック
        features = []
        for x in inputs:
            f = self.feature(x).unsqueeze(1)  # シーケンス次元を追加
            features.append(f)
        
        # 特徴をスタックしてシーケンスを形成 [batch, seq_len=num_inputs, features]
        sequence = torch.cat(features, dim=1)
        
        # セルフアテンションをシーケンスに適用
        attn_output, _ = self.attention(sequence, sequence, sequence)
        
        # 残差接続と正規化
        normalized = self.norm(sequence + attn_output)
        
        # シーケンス次元で平均プーリング
        pooled = torch.mean(normalized, dim=1)
        
        # 最終予測
        return self.regressor(pooled)
    
    def get_preprocess(self):
        """入力画像の前処理パイプラインを返す"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

class MultiImageDataset(Dataset):
    """複数の連続画像を入力とするデータセット"""
    def __init__(self, image_paths, annotations, num_inputs=2, transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform
        self.num_inputs = num_inputs
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 各サンプルでN枚の連続画像を処理
        images = []
        
        # image_paths[idx]が既にリストの場合（マルチカメラモード）
        if isinstance(self.image_paths[idx], list):
            # 既にマルチ画像のパスリストがある場合
            image_paths_for_item = self.image_paths[idx]
            
            # デバッグ用出力（最初の数サンプルのみ）
            if idx < 3:  # 最初の3サンプルだけ出力
                print(f"\nサンプル {idx} の画像パス:")
                for i, path in enumerate(image_paths_for_item):
                    print(f"  画像 {i+1}: {os.path.basename(path)}")
        else:
            # 単一画像パスの場合は、連続するインデックスから画像を収集
            image_indices = [max(0, idx - i) for i in range(self.num_inputs - 1, -1, -1)]
            image_paths_for_item = [self.image_paths[i] for i in image_indices]
        
        # すべての画像を読み込み
        for img_path in image_paths_for_item:
            img = Image.open(img_path).convert('RGB')
            
            # 変換を適用
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
        
        # 現在のインデックスのアノテーションを取得
        annotation = self.annotations[idx]
        target = torch.tensor([annotation["angle"], annotation["throttle"]], dtype=torch.float)
        
        return images, target
    
def run_model_with_multi_input_support(model, inputs, device):
    """
    様々な形式の入力に対応したモデル実行ヘルパー
    
    Args:
        model: モデルインスタンス
        inputs: DataLoaderからの入力データ
        device: 実行デバイス
        
    Returns:
        torch.Tensor: モデル出力
    """
    # ケース1: 画像のリスト [img1, img2, ...] (MultiImageDatasetから)
    if isinstance(inputs, list) and all(hasattr(x, 'to') for x in inputs):
        # すべての画像をデバイスに移動
        device_inputs = [img.to(device) for img in inputs]
        return model(*device_inputs)
    
    # ケース2: テンソルのタプル (img1, img2, ...)
    elif isinstance(inputs, tuple) and all(hasattr(x, 'to') for x in inputs):
        device_inputs = [img.to(device) for img in inputs]
        return model(*device_inputs)
    
    # ケース3: DataLoaderからのバッチ（各サンプルが画像のリスト）
    elif isinstance(inputs, list) and isinstance(inputs[0], list):
        # シーケンス内の各位置のバッチをリシェイプ
        batch_size = len(inputs)
        num_inputs = len(inputs[0])
        
        # バッチ内の各位置をスタック
        batched_inputs = []
        for i in range(num_inputs):
            batch_for_position = torch.stack([sample[i] for sample in inputs]).to(device)
            batched_inputs.append(batch_for_position)
        
        return model(*batched_inputs)
    
    # ケース4: 単一テンソル（標準的なケース）
    elif hasattr(inputs, 'to'):
        return model(inputs.to(device))
    
    # エラーケース
    else:
        raise TypeError(f"サポートされていない入力形式: {type(inputs)}. "
                      f"入力の詳細: {str(inputs)[:100]}...")

def is_multi_input_model(model):
    """モデルがマルチ入力モデルかどうかを判定する"""
    return isinstance(model, MultiInputBaseModel)

def get_model_num_inputs(model):
    """モデルが期待する入力画像数を取得する"""
    if isinstance(model, MultiInputBaseModel):
        return model.num_inputs
    elif hasattr(model, 'run') and 'img_arr1' in str(model.run.__code__.co_varnames):
        return 2  # Dual input model (下位互換性)
    else:
        return 1  # デフォルトケース: 単一入力

def update_model_registry():
    """モデルレジストリにマルチ画像モデルを追加"""
    registry_extension = {
        # マルチ画像モデル（デフォルトは2入力）
        "donkey_multi2_concat": MultiImageConcatModel,
        "donkey_multi2_channel": MultiImageChannelModel,
        "donkey_multi2_attention": MultiImageAttentionModel,
        
        # 3画像入力バリエーション
        "donkey_multi3_concat": lambda **kwargs: MultiImageConcatModel(num_inputs=3, **kwargs),
        "donkey_multi3_channel": lambda **kwargs: MultiImageChannelModel(num_inputs=3, **kwargs),
        "donkey_multi3_attention": lambda **kwargs: MultiImageAttentionModel(num_inputs=3, **kwargs),
        
        # 4画像入力バリエーション
        "donkey_multi4_concat": lambda **kwargs: MultiImageConcatModel(num_inputs=4, **kwargs),
        "donkey_multi4_channel": lambda **kwargs: MultiImageChannelModel(num_inputs=4, **kwargs),
        "donkey_multi4_attention": lambda **kwargs: MultiImageAttentionModel(num_inputs=4, **kwargs),
    }
    
    # 既存のレジストリと結合
    return registry_extension

def collate_multi_images(batch):
    """マルチ画像データセット用のカスタムコレート関数"""
    # 画像とターゲットを分離
    images_list, targets = zip(*batch)
    
    # ターゲットを単一テンソルにスタック
    targets_batch = torch.stack(targets)
    
    # 画像は構造を保持する必要がある
    # 各サンプルはN枚の画像のリストで、この構造を維持する
    return list(images_list), targets_batch

def create_multi_image_datasets(
    image_paths, 
    annotations,
    model_name,
    num_inputs=None,
    val_split=0.2,
    batch_size=32,
    num_workers=4,
    use_augmentation=False
):
    """マルチ画像入力モデル用のデータセットを作成する
    
    Args:
        image_paths: 画像パスのリスト
        annotations: アノテーションのリスト
        model_name: 使用するモデル名
        num_inputs: 入力画像数（Noneの場合は自動検出）
        val_split: 検証分割率
        batch_size: バッチサイズ
        num_workers: データローダーのワーカー数
        use_augmentation: データ拡張を使用するかどうか
        
    Returns:
        train_loader, val_loader, dataset_info
    """
    from torch.utils.data import DataLoader, random_split
    import torchvision.transforms as transforms
    from model_catalog import get_model
    
    # モデルを作成して前処理とnum_inputsを取得
    model = get_model(model_name, pretrained=False)
    
    # 入力画像数が指定されていない場合は自動検出
    if num_inputs is None:
        if hasattr(model, 'num_inputs'):
            num_inputs = model.num_inputs
        elif is_multi_input_model(model):
            num_inputs = get_model_num_inputs(model)
        else:
            # モデル名から判断（'multi3'なら3枚など）
            if 'multi' in model_name.lower():
                try:
                    num_part = model_name.lower().split('multi')[1].split('_')[0]
                    num_inputs = int(num_part)
                except:
                    num_inputs = 2  # デフォルト
            else:
                num_inputs = 1  # 単一入力のデフォルト
    
    print(f"モデル '{model_name}' は {num_inputs} 枚の画像入力を使用します")
    
    # モデルの前処理を取得
    base_transform = model.get_preprocess()
    
    # データ拡張を追加（要求された場合）
    if use_augmentation:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
            base_transform,  # ベース変換を最後に適用
        ])
    else:
        transform = base_transform
    
    # 入力画像数に基づいて適切なデータセットを作成
    if num_inputs > 1:
        dataset = MultiImageDataset(
            image_paths=image_paths,
            annotations=annotations,
            num_inputs=num_inputs,
            transform=transform
        )
    else:
        # 単一入力モデル用の標準データセット
        from model_training import AnnotationDataset
        dataset = AnnotationDataset(
            image_paths=image_paths,
            annotations=annotations,
            transform=transform
        )
    
    # トレーニングセットと検証セットに分割
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # データローダーを作成
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_multi_images if num_inputs > 1 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_multi_images if num_inputs > 1 else None
    )
    
    # データセット情報
    dataset_info = {
        'total_samples': len(dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'batch_size': batch_size,
        'num_classes': 2,  # angle and throttle
        'use_augmentation': use_augmentation,
        'num_inputs': num_inputs
    }
    
    return train_loader, val_loader, dataset_info

# MODEL_REGISTRYに追加するためのヘルパー関数
def register_multi_image_models():
    """マルチ画像モデルをMODEL_REGISTRYに登録する"""
    extension = update_model_registry()
    MODEL_REGISTRY.update(extension)
    print(f"マルチ画像モデル ({len(extension)}種類) をレジストリに追加しました")

def train_multi_image_model(
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
    patience: int = 5
) -> Dict[str, Any]:
    """マルチ画像入力モデルのトレーニング関数
    
    モデルトレーニングの処理は通常のtrain_model関数と同じですが、
    画像入力の処理方法をマルチ入力に対応させています。
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    if progress_callback:
        progress_callback(0, num_epochs, "モデルをロード中...")
    
    # まず事前学習済みの重みでモデルを初期化（またはランダム初期化）
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
    
    # 損失関数と最適化アルゴリズム
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # トレーニングループ
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Early Stopping用の変数
    early_stopping_counter = 0
    early_stopped = False
    stopped_epoch = 0
    
    # 保存ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    
    # タイムスタンプを使用してファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'{model_name}_model_{timestamp}.pth')
    best_model_path = os.path.join(save_dir, f'{model_name}_best_{timestamp}.pth')
    
    completed_epochs = 0
    for epoch in range(num_epochs):
        # 進捗コールバック - エポック開始
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs} 開始"
            should_continue = progress_callback(epoch, num_epochs, message)
            if not should_continue:
                break
        
        model.train()
        epoch_loss = 0.0
        
        # トレーニングステップ
        for i, batch_data in enumerate(train_loader):
            inputs, targets = batch_data
            targets = targets.to(device)
            
            optimizer.zero_grad()
            # マルチ画像入力対応の関数を使用
            outputs = run_model_with_multi_input_support(model, inputs, device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # targetsからバッチサイズを取得（常にテンソル）
            batch_size = targets.size(0)
            epoch_loss += loss.item() * batch_size
            
            # バッチごとの進捗コールバック（10%ごと）
            if progress_callback and (i % max(1, len(train_loader) // 10) == 0):
                batch_progress = i / len(train_loader)
                total_progress = (epoch + batch_progress) / num_epochs
                message = f"エポック {epoch+1}/{num_epochs}, バッチ {i}/{len(train_loader)}, 損失: {loss.item():.4f}"
                should_continue = progress_callback(int(total_progress * num_epochs), num_epochs, message)
                if not should_continue:
                    break
        
        # エポック損失の計算
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, targets = batch_data
                targets = targets.to(device)
                
                # マルチ画像入力対応の関数を使用
                outputs = run_model_with_multi_input_support(model, inputs, device)
                loss = criterion(outputs, targets)
                
                # バッチサイズは常にtargetsから取得
                batch_size = targets.size(0)
                val_loss += loss.item() * batch_size
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 学習率の調整
        scheduler.step(val_loss)
        
        # エポックの完了をカウント
        completed_epochs = epoch + 1
        
        # 進捗コールバック - エポック終了
        if progress_callback:
            message = f"エポック {epoch+1}/{num_epochs}, 学習損失: {epoch_loss:.4f}, 検証損失: {val_loss:.4f}"
            should_continue = progress_callback(epoch + 1, num_epochs, message)
            if not should_continue:
                break
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # カウンタをリセット
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, best_model_path)
            
            if progress_callback:
                progress_callback(epoch + 1, num_epochs, 
                                f"エポック {epoch+1}/{num_epochs}: 新しい最良モデルを保存しました（損失: {best_val_loss:.6f}）")
        else:
            # 検証損失が改善しなかった場合
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
    }
    
    # トレーニング結果の可視化
    from model_training import plot_training_results
    plot_training_results(training_results, save_dir, timestamp)
    
    return training_results