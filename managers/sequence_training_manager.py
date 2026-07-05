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

from model_catalog import create_sequence_model, SEQUENCE_ARCHITECTURES
from .sequence_dataset import SequenceDataset


class SequenceTrainingManager:
    """時系列モデルの学習マネージャ"""

    # 時系列モデルファイル名プレフィックス (アーキテクチャ名がそのままプレフィックスになる)
    SEQUENCE_PREFIXES = ("gru_", "tcn_", "causal_cnn_")
    # 後方互換: 旧形式のプレフィックス
    LEGACY_PREFIXES = ("traj_",)

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

        dataset = SequenceDataset(
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

        val_dataset_no_aug = SequenceDataset(
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
        # pin_memory は CUDA 利用時のみ有効（CPU環境での警告と無駄を回避）
        use_pin_memory = (device.type == 'cuda')
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=use_pin_memory
        )

        # 4. Model構築
        num_image_sources = len(selected_sources)
        model = create_sequence_model(model_arch, num_image_sources, config).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        # モデルのパラメータ数（記録用）
        model_params_total = sum(p.numel() for p in model.parameters())
        model_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 早期終了の設定
        use_early_stopping = config.get('use_early_stopping', True)
        patience = config.get('patience', 10)
        epochs_no_improve = 0
        early_stopped = False

        # 入力/出力テンソル形状（signature相当の記録用。最初のバッチで取得）
        input_image_shape = None
        ego_state_dim = None
        output_shape = None

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
            total_train_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader):
                images_batch = batch['images'].to(device)
                ego_states = batch['ego_states'].to(device)
                targets = batch['targets'].to(device)

                # 入力/出力形状を最初のバッチで記録（signature相当）
                if input_image_shape is None:
                    input_image_shape = tuple(int(x) for x in images_batch.shape[1:])
                    ego_state_dim = int(ego_states.shape[-1])
                    output_shape = tuple(int(x) for x in targets.shape[1:])

                optimizer.zero_grad()
                predictions = model(images_batch, ego_states)
                loss = criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

                # エポック内でも定期的にUIを更新してフリーズを防ぐ（CPU学習時は特に重要）
                # キャンセルも即時反映できるようにする
                if progress_callback and (batch_idx % 5 == 0 or batch_idx == total_train_batches - 1):
                    batch_msg = (
                        f"[{model_arch.upper()}] エポック {epoch + 1}/{epochs} 学習中...\n"
                        f"バッチ {batch_idx + 1}/{total_train_batches}"
                    )
                    cont = progress_callback(epoch, epochs, batch_msg)
                    if cont is False:
                        return {"status": "cancelled"}

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
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

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

            # 早期終了判定（patienceエポック改善なしで打ち切り）
            if use_early_stopping and epochs_no_improve >= patience:
                early_stopped = True
                if progress_callback:
                    progress_callback(
                        epoch + 1, epochs,
                        f"[{model_arch.upper()}] 早期終了: {patience}エポック改善なし (epoch {epoch + 1})"
                    )
                break

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
            "epochs_trained": len(train_losses),
            "total_time": total_time,
            "avg_epoch_time": sum(epoch_times) / len(epoch_times) if epoch_times else 0.0,
            "train_samples": train_size,
            "val_samples": val_size,
            "total_sequences": len(dataset),
            "early_stopped": early_stopped
        }

        # ベスト検証損失のエポックを特定（記録用）
        best_epoch = (val_losses.index(best_val_loss) + 1) if val_losses else 0
        completed_epochs = len(train_losses)

        # 学習曲線グラフを生成（MLflowアーティファクトとして添付）
        curve_path = self._save_training_curve(model_path, train_losses, val_losses, model_arch)

        # MLflow logging
        if self.mlflow_manager:
            try:
                training_params = {
                    "model_type": "sequence",
                    "model_arch": model_arch,
                    "data_folder": os.path.basename(self.models_dir),
                    "seq_len": seq_len,
                    "pred_horizon": pred_horizon,
                    "hidden_dim": hidden_dim,
                    "num_layers": config.get('num_layers', 1),
                    "dropout": dropout,
                    "num_epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_image_sources": num_image_sources,
                    "selected_sources": ",".join(selected_sources),
                    "augmentation_enabled": augment,
                    "stride": stride,
                    # アーキテクチャ/学習設定（他モデルと同等の情報量に揃える）
                    "img_size": img_size,
                    "val_split": val_split,
                    "weight_decay": 1e-4,
                    "fusion_method": config.get('fusion_method'),
                    "attn_heads": config.get('attn_heads'),
                    "kernel_size": config.get('kernel_size'),
                    "tcn_channels": config.get('tcn_channels'),
                    "cnn_channels": config.get('cnn_channels'),
                    "comment": config.get('comment'),
                    "model_name": config.get('model_name'),
                    # 早期終了
                    "use_early_stopping": use_early_stopping,
                    "patience": patience,
                    # モデルI/Oスキーマ（signature相当）
                    "input_image_shape": input_image_shape,
                    "ego_state_dim": ego_state_dim,
                    "output_shape": output_shape,
                    # モデル・実行環境メタデータ
                    "model_params_total": model_params_total,
                    "model_params_trainable": model_params_trainable,
                    "device": str(device),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                }
                metrics = {
                    "best_val_loss": best_val_loss,
                    "final_train_loss": result["final_train_loss"],
                    "final_val_loss": val_losses[-1] if val_losses else 0.0,
                    "best_epoch": best_epoch,
                    "total_training_time": total_time,
                    "avg_epoch_time": result["avg_epoch_time"],
                    "completed_epochs": completed_epochs,
                    "status": "completed",
                    # 学習曲線（エポック毎の損失）— 他モデルと同様にMLflowに記録
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                }
                dataset_info = {
                    "train_samples": train_size,
                    "val_samples": val_size,
                    "total_sequences": len(dataset)
                }
                self.mlflow_manager.log_sequence_model(
                    model_path, training_params, metrics, dataset_info,
                    extra_artifacts=[curve_path] if curve_path else None
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

    def _save_training_curve(self, model_path, train_losses, val_losses, model_arch):
        """学習曲線(train/val loss)をPNGとして保存し、パスを返す（失敗時None）"""
        if not train_losses:
            return None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            curve_path = os.path.splitext(model_path)[0] + "_training_curve.png"
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(train_losses) + 1), train_losses,
                     label='Train Loss', color='#1f77b4')
            if val_losses:
                plt.plot(range(1, len(val_losses) + 1), val_losses,
                         label='Val Loss', color='#ff7f0e')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.title(f'{model_arch.upper()} Training Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(curve_path, dpi=100)
            plt.close()
            return curve_path
        except Exception as e:
            print(f"学習曲線の保存に失敗: {e}")
            return None

    def _save_model(self, model_state_dict, config, selected_sources,
                    train_losses, val_losses, epochs_trained, num_image_sources):
        """学習済みモデルを保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_arch = config.get('model_arch', 'gru')
        seq_len = config.get('seq_len', 8)
        pred_horizon = config.get('pred_horizon', 10)
        filename = f"{model_arch}_{timestamp}_{seq_len}s_{pred_horizon}h.pth"
        model_path = os.path.join(self.models_dir, filename)

        save_dict = {
            "model_state_dict": model_state_dict,
            "model_type": "sequence",
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

        # マルチカメラ融合パラメータを保存
        save_dict["config"]["fusion_method"] = config.get('fusion_method', 'concat')
        save_dict["config"]["attn_heads"] = config.get('attn_heads', 4)

        torch.save(save_dict, model_path)
        print(f"Sequence model ({model_arch}) saved to: {model_path}")
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
        elif model_type in ('sequence', 'trajectory'):
            model_arch = checkpoint.get('model_arch', 'gru')
            cfg = checkpoint['config']
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")

        # 後方互換: 旧モデルに fusion_method / attn_heads が無い場合のデフォルト補完
        cfg.setdefault('fusion_method', 'concat')
        cfg.setdefault('attn_heads', 4)

        model = create_sequence_model(model_arch, cfg['num_image_sources'], cfg).to(device)
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

        dataset = SequenceDataset(
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

    def predict_current(self, model, cfg, selected_sources, target_index,
                        annotations, images, source_images_map, device,
                        deleted_indexes=None):
        """ロード済みモデルで単一フレーム(target_index)の予測を返す（逐次推論用）。

        target_index を入力シーケンスの末尾フレームとして予測する。
        必要な履歴(seq_len)・未来(pred_horizon)フレームが揃わない場合はNoneを返す。

        Returns:
            list | None — [[steering, throttle], ...] (pred_horizon個) または None
        """
        seq_len = cfg.get('seq_len', 8)
        pred_horizon = cfg.get('pred_horizon', 10)
        img_size = cfg.get('img_size', (128, 128))

        start = target_index - seq_len + 1
        end = target_index + pred_horizon  # 含む
        if start < 0 or end >= len(images):
            return None

        deleted = deleted_indexes or set()
        window = list(range(start, end + 1))
        # 連続窓に削除フレームが含まれるとシーケンスを作れない
        for idx in window:
            if idx in deleted:
                return None

        dataset = SequenceDataset(
            valid_indexes=window,
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
            return None

        # target_index を末尾入力に持つシーケンスを探して推論
        for si, (input_indexes, _) in enumerate(dataset.sequences):
            if input_indexes[-1] == target_index:
                sample = dataset[si]
                with torch.no_grad():
                    imgs = sample['images'].unsqueeze(0).to(device)
                    ego = sample['ego_states'].unsqueeze(0).to(device)
                    traj = model(imgs, ego).cpu().numpy()[0]
                return traj.tolist()
        return None
