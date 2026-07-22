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
                 exclude=None):
        self.cfg = cfg
        self.selected_sources = selected_sources
        self.images = images
        self.source_images_map = source_images_map or {}
        self.vocab = None                     # train() が語彙構築後に設定する
        self._nearest = None                  # nearest_vocab_index（同上）

        self.transform = transforms.Compose([
            transforms.Resize((cfg.image_h, cfg.image_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[cfg.norm_mean] * 3,
                                 std=[cfg.norm_std] * 3),
        ])

        exclude = exclude or set()
        self.samples = []                     # [(index, traj(H,2), ego(ego_dim,))]
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
            self.samples.append((idx, traj.astype(np.float32), ego))

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

    def __getitem__(self, i):
        index, traj, ego = self.samples[i]
        imgs = []
        for source in self.selected_sources:
            path = self._image_path(index, source)
            try:
                img = Image.open(path).convert('RGB')
                imgs.append(self.transform(img))
            except Exception:
                imgs.append(torch.zeros(
                    3, self.cfg.image_h, self.cfg.image_w))
        label = self._nearest(self.vocab, traj)
        return {
            "images": torch.stack(imgs),                       # (S, 3, H, W)
            "ego": torch.from_numpy(ego),
            "label": torch.tensor(label, dtype=torch.long),
            "traj": torch.from_numpy(traj),                    # (H, 2) 実測GT
            "index": torch.tensor(index),
        }


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
        cfg = make_config(n_cams, vocab_k=vocab_k, horizon=pred_points, dt=dt)

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
            pose_source=pose_source, cfg=cfg, exclude=quality_excluded)

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

                # ego status ショートカット対策（togivad/train.py と同じ）
                if ego_dropout > 0:
                    drop = (torch.rand(ego.shape[0], 1, device=device)
                            < ego_dropout).float()
                    ego = ego * (1.0 - drop)

                optimizer.zero_grad()
                _occ, _agents, traj_logits = model(imgs, ego)
                loss = F.cross_entropy(traj_logits, labels)
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
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch['images'].to(device)
                    ego = batch['ego'].to(device)
                    labels = batch['label'].to(device)
                    traj_gt = batch['traj'].to(device)

                    _occ, _agents, traj_logits = model(imgs, ego)
                    val_running += F.cross_entropy(traj_logits, labels).item()
                    val_batches += 1

                    pred_idx = traj_logits.argmax(1)
                    top1 += (pred_idx == labels).sum().item()
                    k = min(5, traj_logits.shape[1])
                    top5 += (traj_logits.topk(k, dim=1).indices
                             == labels[:, None]).any(1).sum().item()
                    pred_traj = vocab_t[pred_idx]                  # (B, H, 2)
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

        with torch.no_grad():
            _occ, _agents, logits = model(
                torch.stack(imgs).unsqueeze(0).to(device),
                torch.from_numpy(ego).unsqueeze(0).to(device))
            probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        top = np.argsort(probs)[::-1][:topk]
        vocab = np.asarray(vocab, dtype=np.float32)
        return {
            "trajectories": [vocab[k] for k in top],
            "probs": [float(probs[k]) for k in top],
            "best": vocab[top[0]],
        }
