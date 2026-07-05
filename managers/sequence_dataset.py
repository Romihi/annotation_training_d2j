"""
時系列モデル用シーケンスDataset

valid_indexesからセッション境界を検出し、
各セッション内でスライディングウィンドウでシーケンスを生成する。

全モデルアーキテクチャ (GRU, TCN, CausalCNN) で共通使用。
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class SequenceDataset(Dataset):
    """時系列モデル用のシーケンスデータセット"""

    def __init__(self, valid_indexes, annotations, images,
                 source_images_map, selected_sources,
                 seq_len=8, pred_horizon=10, stride=1,
                 img_size=(128, 128), augment=False):
        """
        Args:
            valid_indexes: 削除済みを除外した有効インデックスリスト（ソート済み）
            annotations: Dict[int, dict]
            images: List[str] — メイン画像パスリスト
            source_images_map: Dict[variant_name, List[str]]
            selected_sources: List[str] — 使用する画像ソース名リスト
            seq_len: 入力シーケンス長
            pred_horizon: 予測ステップ数
            stride: スライディングウィンドウのストライド
            img_size: 画像リサイズサイズ (H, W)
            augment: データ拡張を行うか
        """
        self.valid_indexes = sorted(valid_indexes)
        self.annotations = annotations
        self.images = images
        self.source_images_map = source_images_map if source_images_map else {}
        self.selected_sources = selected_sources
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.stride = stride
        self.img_size = img_size
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.sequences = []
        self._build_sequences()

    def _detect_session_boundaries(self):
        """インデックスのギャップからセッション境界を検出

        Returns:
            List[List[int]] — セッションごとのインデックスリスト
        """
        if not self.valid_indexes:
            return []

        sessions = []
        current_session = [self.valid_indexes[0]]

        for i in range(1, len(self.valid_indexes)):
            if self.valid_indexes[i] - self.valid_indexes[i - 1] > 1:
                sessions.append(current_session)
                current_session = [self.valid_indexes[i]]
            else:
                current_session.append(self.valid_indexes[i])

        if current_session:
            sessions.append(current_session)

        return sessions

    def _build_sequences(self):
        """各セッション内でスライディングウィンドウでシーケンスペアを構築"""
        sessions = self._detect_session_boundaries()
        total_len = self.seq_len + self.pred_horizon

        for session in sessions:
            if len(session) < total_len:
                continue
            for i in range(0, len(session) - total_len + 1, self.stride):
                input_indexes = session[i:i + self.seq_len]
                target_indexes = session[i + self.seq_len:i + total_len]
                self.sequences.append((input_indexes, target_indexes))

    def _get_ego_state(self, index):
        """アノテーションから自車状態ベクトルを取得

        Returns:
            List[float] — [steering, throttle, vx, vy, omega]

        TODO: vx, vy, omega は現在アノテーションに含まれていないため常に0。
              IMU/オドメトリデータが利用可能になった場合に拡張する。
              ego_dim を動的に変更する仕組み（EgoStateEncoder対応含む）も要検討。
        """
        ann = self.annotations.get(index, {})
        steering = float(ann.get("angle", 0.0))
        throttle = float(ann.get("throttle", 0.0))
        vx = float(ann.get("speed", 0.0))
        vy = 0.0   # TODO: オドメトリ/IMUデータから取得
        omega = 0.0  # TODO: ヨーレートセンサーデータから取得
        return [steering, throttle, vx, vy, omega]

    def _load_image(self, index, source_name):
        """指定ソースから画像を読み込み

        Args:
            index: 画像インデックス
            source_name: ソース名 ('cam', 'cam2', etc.)

        Returns:
            Tensor (3, H, W)
        """
        img_path = None
        if source_name in self.source_images_map:
            source_images = self.source_images_map[source_name]
            if index < len(source_images):
                img_path = source_images[index]
        elif source_name == 'cam' and self.images and index < len(self.images):
            img_path = self.images[index]

        if img_path:
            try:
                img = Image.open(img_path).convert('RGB')
                return self.transform(img)
            except Exception:
                pass

        # Fallback: black image
        return torch.zeros(3, self.img_size[0], self.img_size[1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                images: (T, S, 3, H, W)
                ego_states: (T, 5)
                targets: (pred_horizon, 2)
        """
        input_indexes, target_indexes = self.sequences[idx]

        do_flip = self.augment and (torch.rand(1).item() < 0.5)

        images_list = []
        ego_states_list = []

        for idx_frame in input_indexes:
            frame_images = []
            for source in self.selected_sources:
                img = self._load_image(idx_frame, source)
                if do_flip:
                    img = torch.flip(img, [-1])
                frame_images.append(img)
            images_list.append(torch.stack(frame_images))

            ego = self._get_ego_state(idx_frame)
            if do_flip:
                ego[0] = -ego[0]
            ego_states_list.append(torch.tensor(ego, dtype=torch.float32))

        images = torch.stack(images_list)
        ego_states = torch.stack(ego_states_list)

        targets_list = []
        for idx_frame in target_indexes:
            ann = self.annotations.get(idx_frame, {})
            steering = float(ann.get("angle", 0.0))
            throttle = float(ann.get("throttle", 0.0))
            if do_flip:
                steering = -steering
            targets_list.append([steering, throttle])

        targets = torch.tensor(targets_list, dtype=torch.float32)

        return {
            "images": images,
            "ego_states": ego_states,
            "targets": targets
        }
