# location_model_manager.py
import os
import torch
import torch.nn as nn
from PIL import Image
import traceback

class LocationModelManager:
    def __init__(self, app_dir_path, models_dir_name):
        self.APP_DIR_PATH = app_dir_path
        self.MODELS_DIR_NAME = models_dir_name
        self.model = None
        self.model_type = None
        self.model_path = None
        self.num_classes = 8  # 固定で8クラス
        
    def get_model_list(self):
        """利用可能な位置モデルのリストを取得"""
        models_dir = os.path.join(self.APP_DIR_PATH, self.MODELS_DIR_NAME)
        os.makedirs(models_dir, exist_ok=True)
        
        # モデルファイルを検索
        all_model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        # 位置モデルでフィルタリング
        model_files = []
        for model_file in all_model_files:
            if any(keyword in model_file.lower() for keyword in ['location', 'loc_model']):
                model_files.append(model_file)
        
        # モデルファイルを日付順にソート（新しいものが上）
        model_files.sort(reverse=True)
        
        return model_files
    
    def load_model(self, model_type, model_path, progress_callback=None):
        """位置モデルを読み込む"""
        try:
            # 進捗表示コールバック
            if progress_callback:
                progress_callback(30, "モデルチェックポイントを読み込み中...")
            
            # モデルチェックポイントをロード
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # クラス数を取得（チェックポイントから）
            num_classes = None
            if 'model_state_dict' in checkpoint:
                # classifierの重みを確認
                for key, value in checkpoint['model_state_dict'].items():
                    if 'classifier.weight' in key:
                        num_classes = value.shape[0]  # 出力層の最初の次元がクラス数
                        break
                    if 'regressor.weight' in key:
                        num_classes = value.shape[0]
                        break
            else:
                # 直接state_dictの場合
                for key, value in checkpoint.items():
                    if 'classifier.weight' in key:
                        num_classes = value.shape[0]
                        break
                    if 'regressor.weight' in key:
                        num_classes = value.shape[0]
                        break
            
            # クラス数がまだ特定できない場合はデフォルト値
            if num_classes is None:
                num_classes = checkpoint.get('num_classes', 8)  # デフォルト8
            
            self.num_classes = num_classes
            
            if progress_callback:
                progress_callback(50, f"モデル '{model_type}' をロード中... (クラス数: {num_classes})")
            
            # モデルを初期化
            if model_type == 'donkey_location':
                from model_catalog import DonkeyLocationModel
                self.model = DonkeyLocationModel(num_classes=num_classes)
            elif model_type == 'resnet18_location':
                from model_catalog import ResNet18LocationModel
                self.model = ResNet18LocationModel(num_classes=num_classes)
            else:
                # その他のモデル対応
                from model_catalog import get_model
                self.model = get_model(model_type, num_classes=num_classes)
            
            if progress_callback:
                progress_callback(70, "モデルの重みをロード中...")
            
            # モデルの重みをロード
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # デバイスを設定
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()
            
            # モデル情報を保存
            self.model_path = model_path
            self.model_type = model_type
            
            return True, num_classes
            
        except Exception as e:
            traceback.print_exc()
            return False, str(e)
    
    def run_inference(self, img_path):
        """指定された画像に対して位置推論を実行"""
        if self.model is None:
            return None
        
        try:
            # 画像を読み込む
            img = Image.open(img_path).convert('RGB')
            
            # モデルの前処理を取得
            if not hasattr(self.model, '_preprocess') or self.model._preprocess is None:
                self.model._preprocess = self.model.get_preprocess()
            
            # 前処理を適用
            tensor_image = self.model._preprocess(img)
            tensor_image = tensor_image.unsqueeze(0)
            
            # デバイスを取得
            device = next(self.model.parameters()).device
            tensor_image = tensor_image.to(device)
            
            # 推論実行
            with torch.no_grad():
                logits = self.model(tensor_image)
                probs = torch.softmax(logits, dim=1)
                
                # クラスインデックスと確率を取得
                max_prob, pred_class = torch.max(probs, dim=1)
                
                # 全クラスの確率をリストとして取得
                all_probs = probs[0].cpu().numpy().tolist()
            
            # 推論結果を返す
            return {
                'pred_class': pred_class.item(),
                'confidence': max_prob.item(),
                'all_probs': all_probs
            }
            
        except Exception as e:
            print(f"位置推論実行エラー: {e}")
            traceback.print_exc()
            return None
    
    def batch_inference(self, img_paths, progress_callback=None):
        """複数の画像に対してバッチ推論を実行"""
        results = {}
        total = len(img_paths)
        
        for i, img_path in enumerate(img_paths):
            if progress_callback:
                progress_callback(i, total, f"画像 {i+1}/{total} を処理中...")
            
            result = self.run_inference(img_path)
            if result:
                results[img_path] = result
        
        return results
    
    # その他の必要なメソッド...