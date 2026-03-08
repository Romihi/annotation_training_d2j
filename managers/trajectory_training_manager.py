"""
時系列モデルの学習マネージャ

GRU / TCN / CausalCNN の全アーキテクチャに対応。
既存のmodel_training.train_model()と同じコールバックパターンに準拠。
progress_callback: (current, total, message) -> bool（Falseでキャンセル）
"""

import os
import time
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, random_split

from model_catalog import create_trajectory_model, TRAJECTORY_ARCHITECTURES
from .trajectory_dataset import TrajectorySequenceDataset


class TrajectoryTrainingManager:
    """時系列モデルの学習マネージャ"""

    # モデルファイル名プレフィックス → model_type識別に使用
    MODEL_FILE_PREFIX = "traj_"
    # 後方互換: 旧GRUモデルのプレフィックス
    LEGACY_GRU_PREFIX = "gru_"

    def __init__(self, models_dir, mlflow_manager=None):
        self.models_dir = models_dir
        self.mlflow_manager = mlflow_manager
        os.makedirs(models_dir, exist_ok=True)

    def train(self, valid_indexes, annotations, images,
              source_images_map, selected_sources, config,
              progress_callback=None):
        """時系列モデルの学習を実行

        Args:
            valid_indexes: 有効なインデックスリスト
            annotations: Dict[int, dict]
            images: List[str]
            source_images_map: Dict[variant_name, List[str]]
            selected_sources: List[str]
            config: dict — 学習設定
                model_arch: "gru" | "tcn" | "causal_cnn"
                seq_len, pred_horizon, stride, hidden_dim, num_layers,
                dropout, img_size, epochs, batch_size, learning_rate,
                val_split, augment,
                (TCN固有) tcn_channels, kernel_size
                (CausalCNN固有) cnn_channels, kernel_size
            progress_callback: (current, total, message) -> bool

        Returns:
            dict — 学習結果
        """
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_arch = config.get('model_arch', 'gru')
        seq_len = config.get('seq_len', 8)
        pred_horizon = config.get('pred_horizon', 10)
        stride = config.get('stride', 1)
        hidden_dim = config.get('hidden_dim', 256)
        dropout = config.get('dropout', 0.1)
        img_size = config.get('img_size', (128, 128))
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.001)
        val_split = config.get('val_split', 0.2)
        augment = config.get('augment', True)

        # 1. Dataset作成
        if progress_callback:
            cont = progress_callback(0, epochs, "データセットを構築中...")
            if cont is False:
                return {"status": "cancelled"}

        dataset = TrajectorySequenceDataset(
            valid_indexes=valid_indexes,
            annotations=annotations,
            images=images,
            source_images_map=source_images_map,
            selected_sources=selected_sources,
            seq_len=seq_len,
            pred_horizon=pred_horizon,
            stride=stride,
            img_size=img_size,
            augment=augment
        )

        if len(dataset) == 0:
            return {"status": "error", "message": "no_sequences"}

        # 2. Train/Val split
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        val_dataset_no_aug = TrajectorySequenceDataset(
            valid_indexes=valid_indexes,
            annotations=annotations,
            images=images,
            source_images_map=source_images_map,
            selected_sources=selected_sources,
            seq_len=seq_len,
            pred_horizon=pred_horizon,
            stride=stride,
            img_size=img_size,
            augment=False
        )
        val_indices = val_dataset.indices
        val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)

        # 3. DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )

        # 4. Model構築
        num_image_sources = len(selected_sources)
        model = create_trajectory_model(model_arch, num_image_sources, config).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        # 5. Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        epoch_times = []

        for epoch in range(epochs):
            epoch_start = time.time()

            model.train()
            running_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                images_batch = batch['images'].to(device)
                ego_states = batch['ego_states'].to(device)
                targets = batch['targets'].to(device)

                optimizer.zero_grad()
                predictions = model(images_batch, ego_states)
                loss = criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            train_loss = running_loss / max(num_batches, 1)
            train_losses.append(train_loss)

            model.eval()
            val_running_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    images_batch = batch['images'].to(device)
                    ego_states = batch['ego_states'].to(device)
                    targets = batch['targets'].to(device)

                    predictions = model(images_batch, ego_states)
                    loss = criterion(predictions, targets)
                    val_running_loss += loss.item()
                    val_batches += 1

            val_loss = val_running_loss / max(val_batches, 1)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            elapsed = time.time() - start_time

            if progress_callback:
                avg_epoch = sum(epoch_times) / len(epoch_times)
                remaining_epochs = epochs - epoch - 1
                remaining = avg_epoch * remaining_epochs
                arch_label = model_arch.upper()

                # 経過時間フォーマット
                elapsed_str = self._format_time(elapsed)
                time_line = f"経過時間: {elapsed_str}"
                if remaining > 0:
                    remaining_str = self._format_time(remaining)
                    eta = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M:%S")
                    time_line += f" | 残り時間: {remaining_str} | 終了予定: {eta}"

                msg = (
                    f"[{arch_label}] エポック {epoch + 1}/{epochs} 完了\n"
                    f"学習損失: {train_loss:.6f}\n"
                    f"検証損失: {val_loss:.6f} (Best: {best_val_loss:.6f})\n"
                    f"{time_line}"
                )
                cont = progress_callback(epoch + 1, epochs, msg)
                if cont is False:
                    if best_model_state is not None:
                        model_path = self._save_model(
                            best_model_state, config, selected_sources,
                            train_losses, val_losses, epoch + 1, num_image_sources
                        )
                        return {
                            "status": "cancelled",
                            "model_path": model_path,
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "best_val_loss": best_val_loss,
                            "epochs_trained": epoch + 1,
                            "total_time": time.time() - start_time
                        }
                    return {"status": "cancelled"}

        # 6. Save best model
        total_time = time.time() - start_time

        if best_model_state is None:
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model_path = self._save_model(
            best_model_state, config, selected_sources,
            train_losses, val_losses, epochs, num_image_sources
        )

        result = {
            "status": "completed",
            "model_path": model_path,
            "model_arch": model_arch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "epochs_trained": epochs,
            "total_time": total_time,
            "avg_epoch_time": sum(epoch_times) / len(epoch_times) if epoch_times else 0.0,
            "train_samples": train_size,
            "val_samples": val_size,
            "total_sequences": len(dataset)
        }

        # MLflow logging
        if self.mlflow_manager:
            try:
                training_params = {
                    "model_type": "trajectory",
                    "model_arch": model_arch,
                    "data_folder": os.path.basename(self.models_dir),
                    "seq_len": seq_len,
                    "pred_horizon": pred_horizon,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "num_epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_image_sources": num_image_sources,
                    "selected_sources": ",".join(selected_sources),
                    "augmentation_enabled": augment,
                    "stride": stride
                }
                metrics = {
                    "best_val_loss": best_val_loss,
                    "final_train_loss": result["final_train_loss"],
                    "total_training_time": total_time,
                    "avg_epoch_time": result["avg_epoch_time"],
                    "completed_epochs": epochs,
                    "status": "completed"
                }
                dataset_info = {
                    "train_samples": train_size,
                    "val_samples": val_size,
                    "total_sequences": len(dataset)
                }
                self.mlflow_manager.log_gru_trajectory_model(
                    model_path, training_params, metrics, dataset_info
                )
            except Exception as e:
                print(f"MLflow logging failed: {e}")

        return result

    @staticmethod
    def _format_time(seconds):
        """秒数を時:分:秒の形式にフォーマット"""
        if seconds < 0:
            return "計算中..."
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}時間{minutes:02d}分{secs:02d}秒"
        elif minutes > 0:
            return f"{minutes}分{secs:02d}秒"
        else:
            return f"{secs}秒"

    def _save_model(self, model_state_dict, config, selected_sources,
                    train_losses, val_losses, epochs_trained, num_image_sources):
        """学習済みモデルを保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_arch = config.get('model_arch', 'gru')
        seq_len = config.get('seq_len', 8)
        pred_horizon = config.get('pred_horizon', 10)
        filename = f"traj_{model_arch}_{timestamp}_{seq_len}s_{pred_horizon}h.pth"
        model_path = os.path.join(self.models_dir, filename)

        save_dict = {
            "model_state_dict": model_state_dict,
            "model_type": "trajectory",
            "model_arch": model_arch,
            "config": {
                "num_image_sources": num_image_sources,
                "model_arch": model_arch,
                "seq_len": seq_len,
                "pred_horizon": pred_horizon,
                "hidden_dim": config.get('hidden_dim', 256),
                "dropout": config.get('dropout', 0.1),
                "img_size": config.get('img_size', (128, 128)),
            },
            "selected_sources": selected_sources,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epochs_trained": epochs_trained,
            "created_at": datetime.now().isoformat()
        }

        # アーキテクチャ固有パラメータを保存
        if model_arch == "gru":
            save_dict["config"]["num_layers"] = config.get('num_layers', 1)
        elif model_arch == "tcn":
            save_dict["config"]["tcn_channels"] = config.get('tcn_channels', [128, 128, 256])
            save_dict["config"]["kernel_size"] = config.get('kernel_size', 3)
        elif model_arch == "causal_cnn":
            save_dict["config"]["cnn_channels"] = config.get('cnn_channels', [64, 128, 256])
            save_dict["config"]["kernel_size"] = config.get('kernel_size', 3)

        torch.save(save_dict, model_path)
        print(f"Trajectory model ({model_arch}) saved to: {model_path}")
        return model_path

    @staticmethod
    def load_model(model_path, device=None):
        """保存済みモデルをロード（後方互換対応）

        Returns:
            tuple — (model, config_dict, selected_sources)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # 後方互換: 旧GRUフォーマット
        model_type = checkpoint.get('model_type', '')
        if model_type == 'gru_trajectory':
            model_arch = 'gru'
            cfg = checkpoint['config']
            # 旧フォーマットのキーマッピング
            cfg.setdefault('model_arch', 'gru')
            cfg.setdefault('hidden_dim', cfg.pop('gru_hidden', 256))
            cfg.setdefault('num_layers', cfg.pop('gru_layers', 1))
        elif model_type == 'trajectory':
            model_arch = checkpoint.get('model_arch', 'gru')
            cfg = checkpoint['config']
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")

        model = create_trajectory_model(model_arch, cfg['num_image_sources'], cfg).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, cfg, checkpoint.get('selected_sources', [])

    def predict(self, model_path, valid_indexes, annotations, images,
                source_images_map, progress_callback=None):
        """学習済みモデルで予測を実行

        Returns:
            dict — {"status", "predictions", "config", "total_predictions"}
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if progress_callback:
            cont = progress_callback(0, 100, "モデルをロード中...")
            if cont is False:
                return {"status": "cancelled"}

        model, cfg, selected_sources = self.load_model(model_path, device)
        seq_len = cfg.get('seq_len', 8)
        pred_horizon = cfg.get('pred_horizon', 10)
        img_size = cfg.get('img_size', (128, 128))

        dataset = TrajectorySequenceDataset(
            valid_indexes=valid_indexes,
            annotations=annotations,
            images=images,
            source_images_map=source_images_map,
            selected_sources=selected_sources,
            seq_len=seq_len,
            pred_horizon=pred_horizon,
            stride=1,
            img_size=img_size,
            augment=False
        )

        if len(dataset) == 0:
            return {"status": "error", "message": "no_sequences"}

        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        predictions = {}
        total_batches = len(loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                images_batch = batch['images'].to(device)
                ego_states = batch['ego_states'].to(device)

                trajectories = model(images_batch, ego_states)
                trajectories = trajectories.cpu().numpy()

                batch_start = batch_idx * loader.batch_size
                for i in range(trajectories.shape[0]):
                    seq_idx = batch_start + i
                    if seq_idx < len(dataset.sequences):
                        input_indexes, _ = dataset.sequences[seq_idx]
                        key_index = input_indexes[-1]
                        predictions[key_index] = trajectories[i].tolist()

                if progress_callback:
                    pct = int((batch_idx + 1) * 100 / total_batches)
                    cont = progress_callback(
                        pct, 100,
                        f"予測中... {batch_idx + 1}/{total_batches} バッチ"
                    )
                    if cont is False:
                        return {
                            "status": "cancelled",
                            "predictions": predictions,
                            "config": cfg
                        }

        return {
            "status": "completed",
            "predictions": predictions,
            "config": cfg,
            "total_predictions": len(predictions)
        }
