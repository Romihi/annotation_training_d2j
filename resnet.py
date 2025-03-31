import pytorch_lightning as pl
import torchvision.models as models
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
from torchvision import transforms


def get_default_transform(for_inference=False):
    """画像変換パイプラインを定義"""
    if for_inference:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])


def load_resnet18(num_classes=2):
    # ImageNetで事前学習されたモデルをロード
    model = models.resnet18(pretrained=True)

    # 特徴抽出層が変更されないようにする
    for layer in model.parameters():
        layer.requires_grad = False

    # 分類器レイヤーを変更
    model.fc = nn.Linear(512, num_classes)

    # 分類器レイヤーのパラメータを学習可能に設定
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


class ResNet18(pl.LightningModule):
    def __init__(self, input_shape=(128, 3, 224, 224), output_size=(2,)):
        super().__init__()

        # PyTorch Lightningでモデルサマリーを表示するために使用
        self.example_input_array = torch.rand(input_shape)

        # メトリクス
        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()

        self.model = load_resnet18(num_classes=output_size[0])

        self.inference_transform = get_default_transform(for_inference=True)

        # 損失履歴の追跡（テスト用）
        self.loss_history = []

    def forward(self, x):
        # 予測/推論アクション
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        loss = F.l1_loss(logits, y)
        self.loss_history.append(loss)
        self.log("train_loss", loss)

        # メトリクスの記録
        self.train_mse(logits, y)
        self.log("train_mse", self.train_mse, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.l1_loss(logits, y)

        self.log("val_loss", loss)

        # メトリクスの記録
        self.valid_mse(logits, y)
        self.log("valid_mse", self.valid_mse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0005)
        return optimizer

    def run(self, img_arr: np.ndarray, other_arr: np.ndarray = None):
        """
        Donkeycarパーツインターフェースでループ内でパーツを実行するためのメソッド。

        :param img_arr:     uint8 [0,255] 画像データを含むnumpy配列
        :param other_arr:   パイロットで使用される追加データのnumpy配列（IMUモデル用のIMU配列や
                            行動モデルの状態ベクトルなど）
        :return:            (angle, throttle)のタプル
        """
        from PIL import Image

        pil_image = Image.fromarray(img_arr)
        tensor_image = self.inference_transform(pil_image)
        tensor_image = tensor_image.unsqueeze(0)

        # 結果は (1, 2)
        result = self.forward(tensor_image)

        # (2,) にリサイズ
        result = result.reshape(-1)

        # [0, 1]から[-1, 1]へ変換
        result = result * 2 - 1
        return result

    def load2(self, model_path1, model_path2):
        """
        二つのモデルを読み込むためのメソッド（manage.pyでの2モデル対応用）
        注：実際には最初のモデルだけを読み込みます
        """
        print(f"model_path1からモデルを読み込み: {model_path1}")
        checkpoint = torch.load(model_path1)
        self.load_state_dict(checkpoint)
        
        # 2つ目のモデルは無視（またはカスタムロジックを実装）
        print(f"注：model_path2は現在無視されています: {model_path2}")