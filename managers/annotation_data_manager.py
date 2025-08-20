# annotation_data_manager.py
import json
import time
from typing import Dict, List, Any, Optional, Tuple

class AnnotationDataManager:
    """アノテーションデータの統合管理クラス"""
    
    def __init__(self):
        # アノテーションデータ
        self.annotations: Dict[int, Dict[str, Any]] = {}
        self.bbox_annotations: Dict[int, List[Dict]] = {}
        self.segmentation_annotations: Dict[int, List[Dict]] = {}
        self.location_annotations: Dict[int, int] = {}
        
        # 推論結果
        self.inference_results: Dict[int, Dict] = {}
        self.detection_inference_results: Dict[int, List] = {}
        self.location_inference_results: Dict[int, Dict] = {}
        self.segmentation_inference_results: Dict[int, Any] = {}
        
        # 管理データ
        self.deleted_indexes: List[int] = []
        self.annotation_timestamps: Dict[int, int] = {}
        self.annotation_history: List[Dict] = []
        
        # 差分ベクトル
        self.inference_diff_vectors: Dict[int, Dict] = {}
        
    def add_driving_annotation(self, index: int, angle: float, throttle: float, 
                              x: int, y: int, location: Optional[int] = None) -> None:
        """自動運転アノテーションを追加"""
        annotation = {
            "angle": angle,
            "throttle": throttle,
            "x": x,
            "y": y
        }
        
        if location is not None:
            annotation["loc"] = location
            self.location_annotations[index] = location
            
        self.annotations[index] = annotation
        self.annotation_timestamps[index] = int(time.time() * 1000)
        
    def add_bbox_annotation(self, index: int, bbox: Dict) -> None:
        """バウンディングボックスアノテーションを追加"""
        if index not in self.bbox_annotations:
            self.bbox_annotations[index] = []
        self.bbox_annotations[index].append(bbox)
        
    def add_segmentation_annotation(self, index: int, segmentation: Dict) -> None:
        """セグメンテーションアノテーションを追加"""
        if index not in self.segmentation_annotations:
            self.segmentation_annotations[index] = []
        self.segmentation_annotations[index].append(segmentation)
        
    def add_location_annotation(self, index: int, location: int) -> None:
        """位置アノテーションを独立して追加（運転アノテーションがない場合でも）"""
        # 位置情報専用辞書に保存
        self.location_annotations[index] = location
        
        # メインアノテーションに既にエントリがある場合は位置情報を追加
        if index in self.annotations:
            self.annotations[index]["loc"] = location
        else:
            # 運転アノテーションがない場合でも位置アノテーション専用エントリを作成
            self.annotations[index] = {"loc": location}
            
        # タイムスタンプを更新
        self.annotation_timestamps[index] = int(time.time() * 1000)
        
    def get_annotation_by_index(self, index: int) -> Optional[Dict]:
        """インデックスでアノテーションを取得"""
        return self.annotations.get(index)
        
    def is_deleted(self, index: int) -> bool:
        """削除済みかチェック"""
        return index in self.deleted_indexes
        
    def mark_as_deleted(self, index: int) -> None:
        """削除済みとしてマーク"""
        if index not in self.deleted_indexes:
            self.deleted_indexes.append(index)
            self.deleted_indexes.sort()
            
    def restore_deleted(self, index: int) -> None:
        """削除済みを復元"""
        if index in self.deleted_indexes:
            self.deleted_indexes.remove(index)
            
    def get_statistics(self) -> Dict[str, int]:
        """統計情報を取得"""
        # 位置アノテーションのみ（運転アノテーションがない）の数をカウント
        location_only_count = 0
        driving_annotations_count = 0
        
        for index, annotation in self.annotations.items():
            if "loc" in annotation:
                if len(annotation) == 1:  # "loc"のみの場合
                    location_only_count += 1
                else:  # 運転アノテーションも含む場合
                    driving_annotations_count += 1
        
        return {
            "total_annotations": len(self.annotations),
            "deleted_count": len(self.deleted_indexes),
            "bbox_count": sum(len(bboxes) for bboxes in self.bbox_annotations.values()),
            "segmentation_count": sum(len(segs) for segs in self.segmentation_annotations.values()),
            "location_count": len(self.location_annotations),
            "location_only_count": location_only_count,
            "driving_annotations_count": driving_annotations_count
        }
        
    def clear_all(self) -> None:
        """全データをクリア"""
        self.annotations.clear()
        self.bbox_annotations.clear()
        self.segmentation_annotations.clear()
        self.location_annotations.clear()
        self.inference_results.clear()
        self.deleted_indexes.clear()
        self.annotation_timestamps.clear()