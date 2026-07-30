"""
TogiVAD-Nano（軌道語彙分類 E2E モデル）の学習マネージャ

時系列モデル(SequenceTrainingManager)と同じコールバック/結果辞書の契約に
準拠し、train_sequence_model ダイアログから "togivad" アーキテクチャとして
呼び出される。progress_callback: (current, total, message) -> bool。

時系列モデルとの違い:
  - モデル本体はリポジトリ直下の togivad パッケージ（TogiVADNano）を使う
  - ターゲットは将来の操作量ではなく **ego座標系の将来軌道**（1秒/20点）で、
    K本の軌道語彙への分類として学習する（VADv2式）
  - ラベルは pose_manager の実測自己位置系列から生成する。ソースは
    config['pose_source']（既定 "pose"。"slam" も選択可。平滑化slamは
    実験後に検討）
  - 入力は単一フレーム（シーケンスではない）× 選択画像ソース（1/2/4/5台）

チェックポイント形式は togivad/train.py と互換（state_dict/config/vocab）に
加え、アプリ用メタデータ（model_type="togivad" 等）を持つ。
"""

import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_APP_DIR)


def _import_togivad():
    """リポジトリ直下の togivad パッケージを import する（遅延・パス自動解決）"""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from togivad.config import make_config
    from togivad.model.net import TogiVADNano
    from togivad.vocabulary import (build_vocabulary_from_logs,
                                    nearest_vocab_index, synthetic_vocabulary)
    return make_config, TogiVADNano, build_vocabulary_from_logs, \
        nearest_vocab_index, synthetic_vocabulary


class TogivadDataset(Dataset):
    """単一フレーム（画像×ソース, ego, 将来軌道）のデータセット。

    ラベル軌道・ego入力は構築時に pose_manager から一括計算してキャッシュする
    （pose_manager の探索は O(N) のため毎エポック呼ばない）。
    語彙インデックスは vocab 差し替え後に確定するため __getitem__ で計算する。
    """

    def __init__(self, valid_indexes, annotations, images, source_images_map,
                 selected_sources, pose_manager, pose_source, cfg,
                 exclude=None, return_prev=False, return_lidar=False,
                 return_control=False):
        self.cfg = cfg
        self.selected_sources = selected_sources
        self.images = images
        self.source_images_map = source_images_map or {}
        self.vocab = None                     # train() が語彙構築後に設定する
        self._nearest = None                  # nearest_vocab_index（同上）
        # T1-a: True のとき __getitem__ が前フレーム画像 prev_images と
        # ego 相対運動 ego_dpose を追加で返す（時系列 2 フレーム展開学習用）。
        self.return_prev = return_prev
        # Fusion: True のとき item["lidar_bev"] (1,bev,bev) を追加で返す
        # （セッションの lidar/{idx}_lidar_distance_array_.npy を
        # occupancy_from_scan でラスタ化。欠損はゼロ=無情報）。
        self.return_lidar = return_lidar
        # Pilot: True のとき item["ctrl"] (2,)=[angle,throttle]（現フレーム t の
        # 運転アノテーション）を追加で返す。cfg.use_control のとき ego 末尾には
        # 直前フレーム (t−1) の指令が付く（リーク回避の入出力分離）。
        self.return_control = return_control
        self._lidar_meta_cache = {}           # session_dir -> meta dict

        self.transform = transforms.Compose([
            transforms.Resize((cfg.image_h, cfg.image_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[cfg.norm_mean] * 3,
                                 std=[cfg.norm_std] * 3),
        ])

        exclude = exclude or set()
        self.samples = []                     # [(index, traj(H,2), ego(ego_dim,))]
        # T1-a: サンプルと同順の前フレーム情報。prev_idx=None は先頭/欠落
        # （現フレームを複製し dpose=0 で warp 恒等にフォールバック）。
        self.prev_indexes = []                # [int | None]
        self.dposes = []                      # [np.ndarray(3,)]
        self.ctrls = []                       # Pilot: [np.ndarray(2,)] (t の指令)
        for idx in sorted(valid_indexes):
            if not all(self._image_path(idx, s) for s in selected_sources):
                continue
            traj = pose_manager.compute_future_trajectory(
                idx, horizon=cfg.horizon, dt=cfg.dt,
                exclude=exclude, prefer=pose_source)
            if traj is None:
                continue
            ann = annotations.get(idx, {})
            # 符号付き車速を優先（enc/speed 由来の "speed" は符号なし）
            speed = float(ann.get("pose/speed", ann.get("speed", 0.0)) or 0.0)
            yaw_rate = pose_manager.yaw_rate(idx, prefer=pose_source)
            ego = np.concatenate([
                [speed, yaw_rate],
                np.zeros(2 * cfg.raceline_points)]).astype(np.float32)
            if getattr(cfg, "use_control", False):
                # Pilot: 直前フレーム (t−1) の指令を ego 末尾に追記
                pa = annotations.get(idx - 1, {})
                ctrl_prev = np.array(
                    [float(pa.get("angle", 0.0) or 0.0),
                     float(pa.get("throttle", 0.0) or 0.0)], np.float32)
                ego = np.concatenate([ego, ctrl_prev]).astype(np.float32)
            self.samples.append((idx, traj.astype(np.float32), ego))
            if return_prev:
                prev_idx, dpose = self._prev_info(
                    idx, selected_sources, pose_manager, pose_source, exclude)
                self.prev_indexes.append(prev_idx)
                self.dposes.append(dpose)
            if return_control:
                self.ctrls.append(np.array(
                    [float(ann.get("angle", 0.0) or 0.0),
                     float(ann.get("throttle", 0.0) or 0.0)], np.float32))

    def _prev_info(self, idx, selected_sources, pose_manager, pose_source,
                   exclude):
        """前フレーム（直前の記録フレーム idx-1）と ego 相対運動 dpose を返す。

        dpose は pose_manager.relative_dpose（実測 pose 差分・_timestamp_ms 由来の
        実 dt）で計算し、走行軌道表示・学習ラベルと同一情報源に揃える。前フレーム
        画像が無い / dt ギャップ超過 / pose 欠落なら (None, zeros) で warp 恒等。
        """
        prev_idx = idx - 1
        ok = (prev_idx not in exclude
              and all(self._image_path(prev_idx, s) for s in selected_sources))
        dpose = pose_manager.relative_dpose(
            prev_idx, idx, prefer=pose_source) if ok else None
        if dpose is None:
            return None, np.zeros(3, np.float32)
        return prev_idx, dpose.astype(np.float32)

    def _image_path(self, index, source_name):
        if source_name in self.source_images_map:
            lst = self.source_images_map[source_name]
            if index < len(lst):
                return lst[index]
        elif source_name == 'cam' and self.images and index < len(self.images):
            return self.images[index]
        return None

    def set_vocab(self, vocab, nearest_fn):
        self.vocab = np.asarray(vocab, dtype=np.float32)
        self._nearest = nearest_fn

    def trajectories(self):
        """語彙構築(k-means)用の将来軌道 (N, H, 2)"""
        return np.asarray([t for _, t, _ in self.samples], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _load_images(self, index):
        imgs = []
        for source in self.selected_sources:
            path = self._image_path(index, source)
            try:
                img = Image.open(path).convert('RGB')
                imgs.append(self.transform(img))
            except Exception:
                imgs.append(torch.zeros(
                    3, self.cfg.image_h, self.cfg.image_w))
        return torch.stack(imgs)                               # (S, 3, H, W)

    def _lidar_occ(self, index):
        """Fusion: セッションの LiDAR npy → BEV 占有 (1,bev,bev)。欠損はゼロ。

        画像パス（<session>/images/{idx}_..jpg）からセッションを引き、togivad と
        同一コード（occupancy_from_scan / manifest の角度規約）でラスタ化する。
        """
        cfg = self.cfg
        try:
            from togivad.dataset import _load_lidar_meta, occupancy_from_scan
            img = (self.images[index]
                   if self.images and index < len(self.images)
                   else self._image_path(index, self.selected_sources[0]))
            session = os.path.dirname(os.path.dirname(img))
            prefix = os.path.basename(img).split('_')[0]
            npy = os.path.join(session, 'lidar',
                               f'{prefix}_lidar_distance_array_.npy')
            if not os.path.exists(npy):
                return np.zeros((1, cfg.bev_size, cfg.bev_size), np.float32)
            meta = self._lidar_meta_cache.get(session)
            if meta is None:
                meta = _load_lidar_meta(session)
                self._lidar_meta_cache[session] = meta
            occ, _ = occupancy_from_scan(np.load(npy), meta, cfg)
            return occ
        except Exception:
            return np.zeros((1, cfg.bev_size, cfg.bev_size), np.float32)

    def __getitem__(self, i):
        index, traj, ego = self.samples[i]
        label = self._nearest(self.vocab, traj)
        item = {
            "images": self._load_images(index),                # (S, 3, H, W)
            "ego": torch.from_numpy(ego),
            "label": torch.tensor(label, dtype=torch.long),
            "traj": torch.from_numpy(traj),                    # (H, 2) 実測GT
            "index": torch.tensor(index),
        }
        if self.return_prev:                                   # T1-a: 前フレーム
            prev_idx = self.prev_indexes[i]
            src = prev_idx if prev_idx is not None else index  # 無ければ現を複製
            item["prev_images"] = self._load_images(src)
            item["ego_dpose"] = torch.from_numpy(self.dposes[i])
        if self.return_lidar:                                  # Fusion: LiDAR占有
            item["lidar_bev"] = torch.from_numpy(self._lidar_occ(index))
        if self.return_control:                                # Pilot: 制御教師(t)
            item["ctrl"] = torch.from_numpy(self.ctrls[i])
        return item


class TogivadTrainingManager:
    """TogiVAD-Nano の学習マネージャ（アプリ内学習）"""

    # サポートするカメラ台数（togivad.make_config のプリセット）
    SUPPORTED_CAMERA_COUNTS = (1, 2, 4, 5)
    MODEL_PREFIX = "togivad_"

    def __init__(self, models_dir, mlflow_manager=None):
        self.models_dir = models_dir
        self.mlflow_manager = mlflow_manager
        os.makedirs(models_dir, exist_ok=True)

    def train(self, valid_indexes, annotations, images, source_images_map,
              selected_sources, pose_manager, config, progress_callback=None):
        """学習を実行

        Args:
            valid_indexes / annotations / images / source_images_map /
            selected_sources: SequenceTrainingManager.train と同じ
            pose_manager: PoseSourceManager（ラベル軌道・ego入力の情報源）
            config: dict —
                pose_source: "pose"（既定）| "slam"
                pred_seconds: 予測先の時間 [s]（既定 1.0。x秒先を指定）
                pred_points:  予測点数（既定 20。この点数を等時間間隔で補間生成）
                              → dt = pred_seconds / pred_points, horizon = pred_points
                              （未指定時は togivad 既定の 1.0s / 20点 = MPPI互換）
                vocab_k, vocab_from_logs, ego_dropout,
                epochs, batch_size, learning_rate, val_split,
                use_early_stopping, patience
            progress_callback: (current, total, message) -> bool

        Returns:
            dict — SequenceTrainingManager.train と同じキー構成
        """
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_cams = len(selected_sources)
        if n_cams not in self.SUPPORTED_CAMERA_COUNTS:
            return {"status": "error", "message": "togivad_bad_source_count"}
        if pose_manager is None or not pose_manager.has_any_pose():
            return {"status": "error", "message": "togivad_no_pose"}

        (make_config, TogiVADNano, build_vocabulary_from_logs,
         nearest_vocab_index, synthetic_vocabulary) = _import_togivad()

        pose_source = config.get('pose_source', 'pose')
        vocab_k = int(config.get('vocab_k', 128))
        vocab_from_logs = bool(config.get('vocab_from_logs', True))
        ego_dropout = float(config.get('ego_dropout', 0.3))
        # T1-b: 語彙残差回帰（脱量子化）。既定 False で従来の分類のみ挙動。
        use_residual = bool(config.get('use_residual', False))
        lambda_residual = float(config.get('lambda_residual', 1.0))
        # T1-a: pose-warp 時系列 BEV 融合。前フレーム＋実測 dpose で 2 フレーム展開。
        use_temporal = bool(config.get('use_temporal', False))
        # Fusion: LiDAR 占有 BEV を追加入力し 1×1 conv で融合。
        use_lidar = bool(config.get('use_lidar', False))
        # Pilot: 制御入出力（模倣 L1 + pure-pursuit 整合 + 平滑の 3 損失）。
        use_control = bool(config.get('use_control', False))
        lambda_control = float(config.get('lambda_control', 1.0))
        lambda_consist = float(config.get('lambda_consist', 0.2))
        lambda_smooth = float(config.get('lambda_smooth', 0.05))
        epochs = int(config.get('epochs', 20))
        batch_size = int(config.get('batch_size', 16))
        learning_rate = float(config.get('learning_rate', 3e-4))
        val_split = float(config.get('val_split', 0.2))
        use_early_stopping = config.get('use_early_stopping', True)
        patience = int(config.get('patience', 10))

        # 予測点の生成: x秒先(pred_seconds)を pred_points 点で等時間間隔サンプル。
        # TogiVADConfig の既定(1.0s / 20点)を上書きする。dataset は cfg.horizon/
        # cfg.dt をそのまま使い、compute_future_trajectory が時刻補間する。
        pred_points = int(config.get('pred_points', 20))
        pred_seconds = float(config.get('pred_seconds', pred_points * 0.05))
        if pred_points < 1 or pred_seconds <= 0:
            return {"status": "error", "message": "togivad_bad_horizon"}
        dt = pred_seconds / pred_points
        cfg = make_config(n_cams, vocab_k=vocab_k, horizon=pred_points, dt=dt,
                          use_residual=use_residual, use_temporal=use_temporal,
                          use_lidar=use_lidar, use_control=use_control)

        # 1. Dataset構築（品質不良フレームは書き戻しと同様に除外）
        if progress_callback:
            if progress_callback(0, epochs, "データセットを構築中...") is False:
                return {"status": "cancelled"}
        try:
            quality_excluded = pose_manager.flag_quality_issues()
        except Exception:
            quality_excluded = set()

        dataset = TogivadDataset(
            valid_indexes=valid_indexes, annotations=annotations,
            images=images, source_images_map=source_images_map,
            selected_sources=selected_sources, pose_manager=pose_manager,
            pose_source=pose_source, cfg=cfg, exclude=quality_excluded,
            return_prev=use_temporal, return_lidar=use_lidar,
            return_control=use_control)

        if len(dataset) == 0:
            return {"status": "error", "message": "no_sequences"}

        # 2. 語彙構築（既定: 実走行ログの k-means。不足時は合成）
        vocab = synthetic_vocabulary(cfg)
        if vocab_from_logs:
            segs = dataset.trajectories()
            if len(segs) >= 32:
                if progress_callback:
                    progress_callback(0, epochs, "軌道語彙を構築中 (k-means)...")
                vocab = build_vocabulary_from_logs(segs, cfg)
        dataset.set_vocab(vocab, nearest_vocab_index)

        # 3. Train/Val split（拡張なしのためインデックス分割のみ）
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        use_pin = (device.type == 'cuda')
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0, pin_memory=use_pin)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0, pin_memory=use_pin)

        # 4. Model構築
        model = TogiVADNano(cfg, vocab=vocab).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)

        model_params_total = sum(p.numel() for p in model.parameters())
        vocab_t = torch.as_tensor(vocab, dtype=torch.float32, device=device)
        # Pilot: 整合正則化の教師（語彙ごとの pure-pursuit 舵。静的）と
        # control 出力のインデックス（output_spec 単一情報源）
        steer_vocab = ctl_idx = None
        if use_control:
            from togivad.config import output_spec
            from togivad.train import vocab_pursuit_steering
            steer_vocab = vocab_pursuit_steering(vocab, cfg).to(device)
            ctl_idx = output_spec(cfg).index("control")

        # 5. Training loop
        train_losses, val_losses = [], []
        val_top1s, val_ades = [], []
        best_val_loss = float('inf')
        best_metrics = {}
        best_model_state = None
        epochs_no_improve = 0
        early_stopped = False
        epoch_times = []

        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            running_loss, num_batches = 0.0, 0
            total_train_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader):
                imgs = batch['images'].to(device)
                ego = batch['ego'].to(device)
                labels = batch['label'].to(device)

                # Pilot 平滑損失用の直前舵（ego 末尾 2 次元。dropout 前に保持）
                ang_prev = ego[:, -2].clone() if use_control else None
                # ego status ショートカット対策（togivad/train.py と同じ）
                if ego_dropout > 0:
                    drop = (torch.rand(ego.shape[0], 1, device=device)
                            < ego_dropout).float()
                    ego = ego * (1.0 - drop)

                # Fusion: LiDAR 占有ラスタを追加入力（欠損フレームはゼロ）
                kw = ({"lidar_bev": batch['lidar_bev'].to(device)}
                      if use_lidar else {})
                optimizer.zero_grad()
                if use_temporal:
                    # 前フレームを先に通して生 BEV 状態を得る（勾配は切る）。
                    # bev_state は常に最終出力。ego は生 BEV に影響しない。
                    prev_imgs = batch['prev_images'].to(device)
                    dpose = batch['ego_dpose'].to(device)
                    with torch.no_grad():
                        prev_bev = model(prev_imgs, ego)[-1]
                    out = model(imgs, ego, prev_bev=prev_bev, ego_dpose=dpose,
                                **kw)
                else:
                    out = model(imgs, ego, **kw)
                traj_logits = out[2]
                loss = F.cross_entropy(traj_logits, labels)
                # T1-b: 選択語彙(GT近傍 anchor)＋残差 を実測GTへ回帰
                if use_residual:
                    traj_gt = batch['traj'].to(device)          # (B, T, 2)
                    anchor = vocab_t[labels]                    # (B, T, 2)
                    pred_traj = anchor + out[3]                 # anchor + residual
                    loss = loss + lambda_residual * F.smooth_l1_loss(
                        pred_traj, traj_gt)
                # Pilot: 模倣 L1 + 選択語彙軌道の pure-pursuit 舵整合 + 平滑
                if use_control:
                    ctrl_pred = out[ctl_idx]                    # (B, 2)
                    ctrl_gt = batch['ctrl'].to(device)
                    with torch.no_grad():                       # 選択は勾配停止
                        tgt_steer = steer_vocab[traj_logits.argmax(dim=1)]
                    loss = loss \
                        + lambda_control * F.l1_loss(ctrl_pred, ctrl_gt) \
                        + lambda_consist * F.l1_loss(ctrl_pred[:, 0],
                                                     tgt_steer) \
                        + lambda_smooth * (ctrl_pred[:, 0]
                                           - ang_prev).abs().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

                if progress_callback and (batch_idx % 5 == 0
                                          or batch_idx == total_train_batches - 1):
                    msg = (f"[TOGIVAD/{pose_source}] エポック {epoch + 1}/{epochs} 学習中...\n"
                           f"バッチ {batch_idx + 1}/{total_train_batches}")
                    if progress_callback(epoch, epochs, msg) is False:
                        return {"status": "cancelled"}

            train_loss = running_loss / max(num_batches, 1)
            train_losses.append(train_loss)

            # 検証: CE損失 + top1/top5 + ADE（argmax語彙軌道 vs 実測GT）
            model.eval()
            val_running, val_batches = 0.0, 0
            top1 = top5 = cnt = 0
            ade_sum = 0.0
            ctl_mae_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch['images'].to(device)
                    ego = batch['ego'].to(device)
                    labels = batch['label'].to(device)
                    traj_gt = batch['traj'].to(device)

                    kw = ({"lidar_bev": batch['lidar_bev'].to(device)}
                          if use_lidar else {})
                    if use_temporal:
                        prev_imgs = batch['prev_images'].to(device)
                        dpose = batch['ego_dpose'].to(device)
                        prev_bev = model(prev_imgs, ego)[-1]
                        out = model(imgs, ego, prev_bev=prev_bev,
                                    ego_dpose=dpose, **kw)
                    else:
                        out = model(imgs, ego, **kw)
                    traj_logits = out[2]
                    val_running += F.cross_entropy(traj_logits, labels).item()
                    val_batches += 1
                    if use_control:                        # Pilot: angle MAE
                        ctl_mae_sum += (out[ctl_idx][:, 0]
                                        - batch['ctrl'].to(device)[:, 0]) \
                            .abs().sum().item()

                    pred_idx = traj_logits.argmax(1)
                    top1 += (pred_idx == labels).sum().item()
                    k = min(5, traj_logits.shape[1])
                    top5 += (traj_logits.topk(k, dim=1).indices
                             == labels[:, None]).any(1).sum().item()
                    pred_traj = vocab_t[pred_idx]                  # (B, H, 2)
                    if use_residual:                               # 実出力=語彙+残差
                        pred_traj = pred_traj + out[3]
                    ade_sum += torch.linalg.norm(
                        pred_traj - traj_gt, dim=2).mean(1).sum().item()
                    cnt += labels.shape[0]

            val_loss = val_running / max(val_batches, 1)
            val_losses.append(val_loss)
            val_top1 = top1 / max(cnt, 1)
            val_ade = ade_sum / max(cnt, 1)
            val_top1s.append(val_top1)
            val_ades.append(val_ade)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
                best_metrics = {"val_top1": val_top1,
                                "val_top5": top5 / max(cnt, 1),
                                "val_ade_m": val_ade}
                if use_control:
                    best_metrics["val_ctl_angle_mae"] = \
                        ctl_mae_sum / max(cnt, 1)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            elapsed = time.time() - start_time

            if progress_callback:
                avg_epoch = sum(epoch_times) / len(epoch_times)
                remaining = avg_epoch * (epochs - epoch - 1)
                time_line = f"経過時間: {self._format_time(elapsed)}"
                if remaining > 0:
                    eta = (datetime.now()
                           + timedelta(seconds=remaining)).strftime("%H:%M:%S")
                    time_line += (f" | 残り時間: {self._format_time(remaining)}"
                                  f" | 終了予定: {eta}")
                msg = (
                    f"[TOGIVAD/{pose_source}] エポック {epoch + 1}/{epochs} 完了\n"
                    f"学習CE: {train_loss:.4f} / 検証CE: {val_loss:.4f} "
                    f"(Best: {best_val_loss:.4f})\n"
                    f"top1: {val_top1:.1%} | ADE: {val_ade:.3f}m\n"
                    f"{time_line}"
                )
                if progress_callback(epoch + 1, epochs, msg) is False:
                    if best_model_state is not None:
                        model_path = self._save_model(
                            best_model_state, cfg, vocab, config,
                            selected_sources, pose_source,
                            train_losses, val_losses, epoch + 1)
                        return {
                            "status": "cancelled", "model_path": model_path,
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "best_val_loss": best_val_loss,
                            "epochs_trained": epoch + 1,
                            "total_time": time.time() - start_time,
                        }
                    return {"status": "cancelled"}

            if use_early_stopping and epochs_no_improve >= patience:
                early_stopped = True
                if progress_callback:
                    progress_callback(
                        epoch + 1, epochs,
                        f"[TOGIVAD] 早期終了: {patience}エポック改善なし "
                        f"(epoch {epoch + 1})")
                break

        # 6. Save best model
        total_time = time.time() - start_time
        if best_model_state is None:
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

        model_path = self._save_model(
            best_model_state, cfg, vocab, config, selected_sources,
            pose_source, train_losses, val_losses, len(train_losses))

        result = {
            "status": "completed",
            "model_path": model_path,
            "model_arch": "togivad",
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "epochs_trained": len(train_losses),
            "total_time": total_time,
            "avg_epoch_time": (sum(epoch_times) / len(epoch_times)
                               if epoch_times else 0.0),
            "train_samples": train_size,
            "val_samples": val_size,
            "total_sequences": len(dataset),
            "early_stopped": early_stopped,
            "val_top1": best_metrics.get("val_top1", 0.0),
            "val_ade_m": best_metrics.get("val_ade_m", 0.0),
        }

        best_epoch = (val_losses.index(best_val_loss) + 1) if val_losses else 0
        curve_path = self._save_training_curve(model_path, train_losses,
                                               val_losses, val_ades)

        # MLflow logging
        if self.mlflow_manager:
            try:
                training_params = {
                    "model_type": "togivad",
                    "model_arch": "togivad",
                    "data_folder": os.path.basename(self.models_dir),
                    "pose_source": pose_source,
                    "vocab_k": vocab_k,
                    "vocab_from_logs": vocab_from_logs,
                    "horizon": cfg.horizon,
                    "dt": cfg.dt,
                    "pred_seconds": pred_seconds,
                    "pred_points": pred_points,
                    "ego_dropout": ego_dropout,
                    "use_residual": use_residual,
                    "lambda_residual": lambda_residual if use_residual else 0.0,
                    "use_temporal": use_temporal,
                    "use_lidar": use_lidar,
                    "use_control": use_control,
                    "lambda_control": lambda_control if use_control else 0.0,
                    "lambda_consist": lambda_consist if use_control else 0.0,
                    "lambda_smooth": lambda_smooth if use_control else 0.0,
                    "num_epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_image_sources": n_cams,
                    "selected_sources": ",".join(selected_sources),
                    "img_size": (cfg.image_h, cfg.image_w),
                    "val_split": val_split,
                    "weight_decay": 1e-4,
                    "use_early_stopping": use_early_stopping,
                    "patience": patience,
                    "quality_excluded_frames": len(quality_excluded),
                    "model_params_total": model_params_total,
                    "device": str(device),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "comment": config.get('comment'),
                    "model_name": config.get('model_name'),
                }
                metrics = {
                    "best_val_loss": best_val_loss,
                    "final_train_loss": result["final_train_loss"],
                    "final_val_loss": val_losses[-1] if val_losses else 0.0,
                    "best_epoch": best_epoch,
                    "best_val_top1": best_metrics.get("val_top1", 0.0),
                    "best_val_top5": best_metrics.get("val_top5", 0.0),
                    "best_val_ade_m": best_metrics.get("val_ade_m", 0.0),
                    "total_training_time": total_time,
                    "avg_epoch_time": result["avg_epoch_time"],
                    "completed_epochs": len(train_losses),
                    "status": "completed",
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                }
                dataset_info = {
                    "train_samples": train_size,
                    "val_samples": val_size,
                    "total_sequences": len(dataset),
                }
                self.mlflow_manager.log_togivad_model(
                    model_path, training_params, metrics, dataset_info,
                    extra_artifacts=[curve_path] if curve_path else None)
            except Exception as e:
                print(f"MLflow logging failed: {e}")

        return result

    @staticmethod
    def _format_time(seconds):
        if seconds < 0:
            return "計算中..."
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}時間{minutes:02d}分{secs:02d}秒"
        if minutes > 0:
            return f"{minutes}分{secs:02d}秒"
        return f"{secs}秒"

    def _save_training_curve(self, model_path, train_losses, val_losses,
                             val_ades):
        """学習曲線(CE)とADEの2軸グラフをPNG保存（失敗時None）"""
        if not train_losses:
            return None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            curve_path = os.path.splitext(model_path)[0] + "_training_curve.png"
            fig, ax = plt.subplots(figsize=(8, 5))
            ep = range(1, len(train_losses) + 1)
            ax.plot(ep, train_losses, label='Train CE', color='#1f77b4')
            if val_losses:
                ax.plot(ep, val_losses, label='Val CE', color='#ff7f0e')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Cross Entropy')
            ax.grid(True, alpha=0.3)
            if val_ades:
                ax2 = ax.twinx()
                ax2.plot(ep, val_ades, label='Val ADE [m]',
                         color='#2ca02c', linestyle='--')
                ax2.set_ylabel('ADE [m]')
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2)
            else:
                ax.legend()
            ax.set_title('TogiVAD Training Curve')
            plt.tight_layout()
            plt.savefig(curve_path, dpi=100)
            plt.close()
            return curve_path
        except Exception as e:
            print(f"学習曲線の保存に失敗: {e}")
            return None

    def _save_model(self, model_state_dict, cfg, vocab, config,
                    selected_sources, pose_source,
                    train_losses, val_losses, epochs_trained):
        """学習済みモデルを保存（togivad/train.py 互換 + アプリ用メタデータ）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = (f"{self.MODEL_PREFIX}{timestamp}"
                    f"_{len(selected_sources)}cam_k{cfg.vocab_k}.pth")
        model_path = os.path.join(self.models_dir, filename)

        save_dict = {
            # togivad/export_onnx.py・infer.py が読む互換キー
            "state_dict": model_state_dict,
            "config": cfg.to_dict(),
            "vocab": np.asarray(vocab),
            # アプリ用メタデータ
            "model_type": "togivad",
            "model_arch": "togivad",
            "pose_source": pose_source,
            "selected_sources": selected_sources,
            "training_config": {k: v for k, v in config.items()
                                if isinstance(v, (int, float, str, bool,
                                                  list, tuple, type(None)))},
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epochs_trained": epochs_trained,
            "created_at": datetime.now().isoformat(),
        }
        torch.save(save_dict, model_path)
        print(f"TogiVAD model saved to: {model_path}")
        return model_path

    @staticmethod
    def load_model(model_path, device=None):
        """保存済みTogiVADモデルをロード

        Returns:
            tuple — (model, cfg, vocab(np.ndarray), meta(dict))
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        (make_config, TogiVADNano, _bv, _nv, _sv) = _import_togivad()
        from togivad.config import TogiVADConfig

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if ckpt.get("model_type", "togivad") != "togivad" \
                and "state_dict" not in ckpt:
            raise ValueError(f"togivadモデルではありません: {model_path}")
        cfg = TogiVADConfig.from_dict(ckpt["config"])
        vocab = np.asarray(ckpt["vocab"], dtype=np.float32)
        model = TogiVADNano(cfg, vocab=vocab).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        meta = {k: ckpt.get(k) for k in
                ("pose_source", "selected_sources", "training_config",
                 "epochs_trained", "created_at")}
        return model, cfg, vocab, meta

    def predict_current(self, model, cfg, vocab, selected_sources,
                        target_index, images, source_images_map,
                        annotations, pose_manager=None, pose_source="pose",
                        device=None, topk=5):
        """単一フレームの軌道予測（マップビュー重畳などの逐次推論用）

        Returns:
            dict | None — {"trajectories": [(H,2)...topk], "probs": [...],
                           "best": (H,2)} すべて ego 座標系 [m]
        """
        if device is None:
            device = next(model.parameters()).device
        ds = TogivadDataset.__new__(TogivadDataset)
        ds.cfg = cfg
        ds.selected_sources = selected_sources
        ds.images = images
        ds.source_images_map = source_images_map or {}
        ds.transform = transforms.Compose([
            transforms.Resize((cfg.image_h, cfg.image_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[cfg.norm_mean] * 3,
                                 std=[cfg.norm_std] * 3),
        ])
        imgs = []
        for source in selected_sources:
            path = ds._image_path(target_index, source)
            if path is None:
                return None
            try:
                imgs.append(ds.transform(Image.open(path).convert('RGB')))
            except Exception:
                return None

        ann = annotations.get(target_index, {}) if annotations else {}
        speed = float(ann.get("pose/speed", ann.get("speed", 0.0)) or 0.0)
        yaw_rate = (pose_manager.yaw_rate(target_index, prefer=pose_source)
                    if pose_manager is not None else 0.0)
        ego = np.concatenate([[speed, yaw_rate],
                              np.zeros(2 * cfg.raceline_points)]
                             ).astype(np.float32)
        if getattr(cfg, "use_control", False):
            # Pilot: 直前フレームの運転アノテーションを直前指令として与える
            pa = annotations.get(target_index - 1, {}) if annotations else {}
            ego = np.concatenate([ego, np.array(
                [float(pa.get("angle", 0.0) or 0.0),
                 float(pa.get("throttle", 0.0) or 0.0)],
                np.float32)]).astype(np.float32)

        with torch.no_grad():
            kw = {}
            if getattr(cfg, "use_lidar", False):
                # Fusion: セッションの LiDAR をラスタ化して入力（欠損はゼロ）
                ds.selected_sources = selected_sources
                ds._lidar_meta_cache = {}
                kw["lidar_bev"] = torch.from_numpy(
                    TogivadDataset._lidar_occ(ds, target_index)
                ).unsqueeze(0).to(device)
            out = model(
                torch.stack(imgs).unsqueeze(0).to(device),
                torch.from_numpy(ego).unsqueeze(0).to(device), **kw)
            logits = out[2]
            probs = torch.softmax(logits[0], dim=0).cpu().numpy()
            # T1-b: 残差があれば表示軌道にも反映（脱量子化後の実軌道）
            residual = (out[3][0].cpu().numpy()
                        if getattr(cfg, "use_residual", False) and len(out) > 3
                        else None)
            control = None
            if getattr(cfg, "use_control", False):
                from togivad.config import output_spec
                c = out[output_spec(cfg).index("control")][0].cpu().numpy()
                control = (float(c[0]), float(c[1]))       # Pilot 生出力
        top = np.argsort(probs)[::-1][:topk]
        vocab = np.asarray(vocab, dtype=np.float32)
        trajs = [vocab[k] + residual if residual is not None else vocab[k]
                 for k in top]
        # pure pursuit（togivad.infer.track と同一幾何）: 非 Pilot モデルでも
        # 運転推論表示（angle/throttle）を出せるようにする
        import math
        bt = trajs[0]
        dist = np.linalg.norm(bt, axis=1)
        k = int(np.searchsorted(dist, cfg.lookahead_m))
        k = min(max(k, 1), len(bt) - 1)
        tx, ty = float(bt[k, 0]), float(bt[k, 1])
        ld = max(math.hypot(tx, ty), 1e-3)
        steer = math.atan2(2.0 * cfg.wheelbase_m
                           * math.sin(math.atan2(ty, tx)), ld)
        tv = float(np.linalg.norm(np.diff(
            np.vstack([[0, 0], bt]), axis=0), axis=1).sum()) / \
            (len(bt) * cfg.dt)
        pursuit = (float(np.clip(steer / cfg.max_steer_rad, -1.0, 1.0)),
                   float(np.clip(tv / cfg.v_max_mps, -1.0, 1.0)))
        return {
            "trajectories": trajs,
            "probs": [float(probs[k]) for k in top],
            "best": trajs[0],
            "control": control,        # Pilot: (angle, throttle) / 無効時 None
            "pursuit": pursuit,        # 選択軌道の pure pursuit (angle, throttle)
        }
