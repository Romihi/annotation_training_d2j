# datasetmanager.py
import os
import json
from datetime import datetime

class YOLODatasetManager:
    """YOLO データセット管理クラス"""
    
    def __init__(self, datasets_dir):
        self.datasets_dir = datasets_dir
        
    def create_dataset_directory(self, task_type, model_type, timestamp=None):
        """タスク別・タイムスタンプ付きデータセットディレクトリを作成"""
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # データセット名の構成: tasktype_modeltype_timestamp
        dataset_name = f"{task_type}_{model_type}_{timestamp}"
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        
        # ディレクトリ構造を作成
        os.makedirs(os.path.join(dataset_path, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "train", "labels"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "val", "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "val", "labels"), exist_ok=True)
        
        return dataset_path, dataset_name, timestamp
    
    def save_dataset_metadata(self, dataset_path, metadata):
        """データセットのメタデータを保存"""
        
        metadata_file = os.path.join(dataset_path, "dataset_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def get_dataset_list(self, task_type=None):
        """データセット一覧を取得"""
        
        if not os.path.exists(self.datasets_dir):
            return []
        
        datasets = []
        for item in os.listdir(self.datasets_dir):
            item_path = os.path.join(self.datasets_dir, item)
            if os.path.isdir(item_path):
                # メタデータファイルを確認
                metadata_file = os.path.join(item_path, "dataset_metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # タスクタイプでフィルタリング
                        if task_type is None or metadata.get('task_type') == task_type:
                            datasets.append({
                                'name': item,
                                'path': item_path,
                                'metadata': metadata
                            })
                    except:
                        pass
        
        # タイムスタンプで降順ソート（新しいものが先頭）
        datasets.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
        return datasets
    
    def cleanup_old_datasets(self, keep_count=5):
        """古いデータセットを削除（指定数だけ保持）"""
        
        for task_type in ['detect', 'segment']:
            datasets = self.get_dataset_list(task_type)
            
            if len(datasets) > keep_count:
                datasets_to_remove = datasets[keep_count:]
                
                for dataset in datasets_to_remove:
                    try:
                        import shutil
                        shutil.rmtree(dataset['path'])
                        print(f"古いデータセットを削除: {dataset['name']}")
                    except Exception as e:
                        print(f"データセット削除エラー: {e}")
