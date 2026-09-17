"""
LiDAR Policy（2D LiDAR → angle/throttle）の学習マネージャ。

SequenceTrainingManager / TogivadTrainingManager と同じコールバック・結果辞書の
契約に準拠し、train_sequence_model ダイアログから "lidar_policy" アーキとして
呼び出される。progress_callback: (current, total, message) -> bool。

- 入力: セッション lidar/*.npy（生 [mm]）を K フレーム積層 + 車速
- 教師: 運転アノテーション (angle, throttle)。任意で pose_manager の将来軌道を
  補助ヘッドで回帰（表現学習の正則化。推論では使わない）
- 分割: 既定は連続ブロック分割（フレームのランダム分割は時系列相関で val が
  過大評価されるため）
- 保存: lidar_policy_{timestamp}_{preset}_b{bins}_k{K}.pth
  （model_state_dict / config(LidarPolicyConfig) / アプリ用メタ）
- ONNX: export_onnx() で実機 planner plan=="lidar_bc" 用に書き出し
  （入力 scan_mm (1,K,N_raw) と speed (1,1)、前処理はグラフ内）
"""

import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .lidar_policy_dataset import (LidarPolicyDataset, block_split_indices,
                                   load_lidar_meta, scan_path_for_image)
from .lidar_policy_models import (LIDAR_POLICY_PRESETS, FlatObsWrapper,
                                  LidarPolicyConfig, build_lidar_policy)


def _onnx_export(net, args, onnx_path, opset, **kw):
    """TorchScript ベースの従来エクスポータを優先する（Jetson の古い onnxruntime 向けに
    opset を確実に守る。dynamo 版は opset 18 未満へ落とせず警告を大量に出す）。"""
    try:
        torch.onnx.export(net, args, onnx_path, opset_version=opset, dynamo=False, **kw)
    except TypeError:                       # 古い torch（dynamo 引数なし）
        torch.onnx.export(net, args, onnx_path, opset_version=opset, **kw)


class LidarPolicyTrainingManager:
    MODEL_PREFIX = "lidar_policy_"
    ARCH_NAME = "lidar_policy"

    def __init__(self, models_dir, mlflow_manager=None):
        self.models_dir = models_dir
        self.mlflow_manager = mlflow_manager
        os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------ train
    def train(self, valid_indexes, annotations, images, pose_manager, config,
              progress_callback=None):
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        preset = str(config.get('preset', 'base'))
        if preset not in LIDAR_POLICY_PRESETS:
            return {"status": "error", "message": f"unknown preset: {preset}"}
        epochs = int(config.get('epochs', 30))
        batch_size = int(config.get('batch_size', 64))
        learning_rate = float(config.get('learning_rate', 1e-3))
        val_split = float(config.get('val_split', 0.2))
        use_early_stopping = bool(config.get('use_early_stopping', True))
        patience = int(config.get('patience', 10))
        augment = bool(config.get('augment', True))
        mirror = bool(config.get('mirror', False))
        mode_filter = str(config.get('mode_filter', 'all'))
        split_mode = str(config.get('split_mode', 'block'))
        pose_source = str(config.get('pose_source', 'pose'))
        w_steer = float(config.get('w_steer', 1.0))
        w_speed = float(config.get('w_speed', 0.5))
        w_traj = float(config.get('w_traj', 0.2))
        steer_balance = bool(config.get('steer_balance', True))

        use_traj = bool(config.get('use_traj', True))
        if use_traj and (pose_manager is None or not pose_manager.has_any_pose()):
            use_traj = False                                  # 自己位置なし → 補助ヘッド OFF

        # LiDAR 角度規約・点数はセッションの manifest から取る（先頭の有効フレーム）
        meta = {"angle_start": -135.0, "angle_end": 135.0, "clockwise": False,
                "data_points": None}
        for idx in sorted(valid_indexes):
            if idx < len(images):
                session, npy = scan_path_for_image(images[idx])
                if npy is not None:
                    meta = load_lidar_meta(session)
                    if meta.get("data_points") is None:
                        meta["data_points"] = int(np.load(npy).reshape(-1).shape[0])
                    break
        if meta.get("data_points") is None:
            return {"status": "error", "message": "lidar_no_scans"}

        pred_points = int(config.get('pred_points', 10))
        pred_seconds = float(config.get('pred_seconds', 1.0))
        if pred_points < 1 or pred_seconds <= 0:
            return {"status": "error", "message": "togivad_bad_horizon"}

        cfg = LidarPolicyConfig(
            preset=preset,
            num_beams_raw=int(meta["data_points"]),
            num_bins=int(config.get('num_bins', 541)),
            downsample_mode=str(config.get('downsample_mode', 'minpool')),
            stack_frames=int(config.get('stack_frames', 3)),
            use_valid_ch=bool(config.get('use_valid_ch', True)),
            max_range_mm=float(config.get('max_range_mm', 10000.0)),
            angle_start_deg=float(meta["angle_start"]),
            angle_end_deg=float(meta["angle_end"]),
            clockwise=bool(meta["clockwise"]),
            max_speed=float(config.get('max_speed', 5.0)),
            hidden_dim=int(config.get('hidden_dim',
                                      LIDAR_POLICY_PRESETS[preset]["hidden_dim"])),
            dropout=float(config.get('dropout', 0.1)),
            use_traj=use_traj,
            horizon=pred_points,
            dt=pred_seconds / pred_points,
        )
        if cfg.num_bins > cfg.num_beams_raw:
            cfg.num_bins = cfg.num_beams_raw

        # 1. Dataset
        if progress_callback:
            if progress_callback(0, epochs, "LiDAR データセットを構築中...") is False:
                return {"status": "cancelled"}
        try:
            quality_excluded = (pose_manager.flag_quality_issues()
                                if (use_traj and pose_manager is not None) else set())
        except Exception:
            quality_excluded = set()

        dataset = LidarPolicyDataset(
            valid_indexes=valid_indexes, annotations=annotations, images=images,
            cfg=cfg, pose_manager=pose_manager if use_traj else None,
            pose_source=pose_source, exclude=quality_excluded,
            mode_filter=mode_filter, augment=False, mirror=mirror)
        if len(dataset) == 0:
            return {"status": "error",
                    "message": "lidar_no_scans" if dataset.n_no_scan else "no_sequences"}
        if use_traj and not dataset.has_traj():
            use_traj = cfg.use_traj = False                   # 軌道が1つも作れない
            w_traj = 0.0

        # 2. Split（block: 連続ブロック / random: 従来のランダム）
        n = len(dataset)
        if split_mode == 'random':
            perm = np.random.permutation(n).tolist()
            val_size = max(1, int(n * val_split))
            val_idx, train_idx = perm[:val_size], perm[val_size:]
        else:
            train_idx, val_idx = block_split_indices(n, val_split)
        if not train_idx or not val_idx:
            return {"status": "error", "message": "no_sequences"}
        train_size, val_size = len(train_idx), len(val_idx)

        # 学習側だけ拡張を有効にするため、同じ samples を共有する別インスタンスを作る
        train_ds = LidarPolicyDataset.__new__(LidarPolicyDataset)
        train_ds.__dict__.update(dataset.__dict__)
        train_ds.augment = augment
        train_ds._scan_cache = dataset._scan_cache             # キャッシュ共有

        use_pin = (device.type == 'cuda')
        train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                                  shuffle=True, num_workers=0, pin_memory=use_pin)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                                shuffle=False, num_workers=0, pin_memory=use_pin)

        # 操舵の逆頻度重み（|steer| 5ビン。大舵角サンプルが少ない偏りを補正）
        bin_edges = np.linspace(0.0, 1.0, 6)
        bin_w = np.ones(5, np.float32)
        if steer_balance:
            st = np.abs(dataset.steer_values()[train_idx])
            hist, _ = np.histogram(np.clip(st, 0, 1), bins=bin_edges)
            freq = hist / max(hist.sum(), 1)
            w = 1.0 / np.maximum(freq, 1e-3)
            w = w / (w * freq).sum()                           # 期待値 1 に正規化
            bin_w = np.clip(w, 0.5, 4.0).astype(np.float32)
        bin_w_t = torch.as_tensor(bin_w, device=device)

        def steer_weight(target_steer):
            b = torch.clamp((target_steer.abs() * 5).long(), 0, 4)
            return bin_w_t[b]

        # 3. Model
        model = build_lidar_policy(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        model_params_total = sum(p.numel() for p in model.parameters())

        def compute_loss(out, batch):
            action = batch['action'].to(device)
            if cfg.use_traj:
                pred_a, pred_t = out
            else:
                pred_a, pred_t = out, None
            l_steer = (F.smooth_l1_loss(pred_a[:, 0], action[:, 0], reduction='none',
                                        beta=0.1) * steer_weight(action[:, 0])).mean()
            l_speed = F.smooth_l1_loss(pred_a[:, 1], action[:, 1], beta=0.1)
            loss = w_steer * l_steer + w_speed * l_speed
            l_traj = torch.zeros((), device=device)
            if pred_t is not None and w_traj > 0:
                tgt = batch['traj'].to(device)
                m = batch['traj_mask'].to(device)
                per = (pred_t - tgt).abs().mean(dim=(1, 2))
                l_traj = (per * m).sum() / m.sum().clamp(min=1.0)
                loss = loss + w_traj * l_traj
            return loss, pred_a, pred_t

        # 4. Loop
        train_losses, val_losses, val_steer_maes = [], [], []
        best_val_loss = float('inf')
        best_metrics = {}
        best_model_state = None
        epochs_no_improve = 0
        early_stopped = False
        epoch_times = []
        tag = f"[LIDAR/{preset}]"

        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            running, nb = 0.0, 0
            total_batches = len(train_loader)
            for bi, batch in enumerate(train_loader):
                optimizer.zero_grad()
                out = model(batch['scan'].to(device), batch['state'].to(device))
                loss, _, _ = compute_loss(out, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running += loss.item()
                nb += 1
                if progress_callback and bi % 20 == 0:
                    msg = (f"{tag} エポック {epoch + 1}/{epochs} "
                           f"バッチ {bi + 1}/{total_batches} loss={loss.item():.4f}")
                    if progress_callback(epoch, epochs, msg) is False:
                        return {"status": "cancelled"}
            train_loss = running / max(nb, 1)
            train_losses.append(train_loss)

            # validation
            model.eval()
            v_run, v_nb = 0.0, 0
            mae_s = mae_t = ade = 0.0
            cnt = traj_cnt = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch['scan'].to(device), batch['state'].to(device))
                    loss, pred_a, pred_t = compute_loss(out, batch)
                    v_run += loss.item()
                    v_nb += 1
                    action = batch['action'].to(device)
                    mae_s += (pred_a[:, 0] - action[:, 0]).abs().sum().item()
                    mae_t += (pred_a[:, 1] - action[:, 1]).abs().sum().item()
                    cnt += action.shape[0]
                    if pred_t is not None:
                        m = batch['traj_mask'].to(device)
                        d = torch.linalg.norm(pred_t - batch['traj'].to(device), dim=2).mean(1)
                        ade += (d * m).sum().item()
                        traj_cnt += int(m.sum().item())
            val_loss = v_run / max(v_nb, 1)
            val_losses.append(val_loss)
            val_steer_mae = mae_s / max(cnt, 1)
            val_speed_mae = mae_t / max(cnt, 1)
            val_ade = ade / max(traj_cnt, 1) if traj_cnt else 0.0
            val_steer_maes.append(val_steer_mae)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = {"val_steer_mae": val_steer_mae,
                                "val_speed_mae": val_speed_mae,
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
                    eta = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M:%S")
                    time_line += (f" | 残り時間: {self._format_time(remaining)}"
                                  f" | 終了予定: {eta}")
                ade_line = f" | ADE: {val_ade:.3f}m" if cfg.use_traj else ""
                msg = (f"{tag} エポック {epoch + 1}/{epochs} 完了\n"
                       f"学習loss: {train_loss:.4f} / 検証loss: {val_loss:.4f} "
                       f"(Best: {best_val_loss:.4f})\n"
                       f"steer MAE: {val_steer_mae:.4f} | throttle MAE: {val_speed_mae:.4f}"
                       f"{ade_line}\n{time_line}")
                if progress_callback(epoch + 1, epochs, msg) is False:
                    if best_model_state is not None:
                        model_path = self._save_model(best_model_state, cfg, config, meta,
                                                      pose_source, train_losses, val_losses,
                                                      best_metrics, bin_w, epoch + 1)
                        return {"status": "cancelled", "model_path": model_path,
                                "train_losses": train_losses, "val_losses": val_losses,
                                "best_val_loss": best_val_loss,
                                "epochs_trained": epoch + 1,
                                "total_time": time.time() - start_time}
                    return {"status": "cancelled"}

            if use_early_stopping and epochs_no_improve >= patience:
                early_stopped = True
                if progress_callback:
                    progress_callback(epoch + 1, epochs,
                                      f"{tag} 早期終了: {patience}エポック改善なし "
                                      f"(epoch {epoch + 1})")
                break

        # 5. Save
        total_time = time.time() - start_time
        if best_model_state is None:
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model_path = self._save_model(best_model_state, cfg, config, meta, pose_source,
                                      train_losses, val_losses, best_metrics, bin_w,
                                      len(train_losses))
        curve_path = self._save_training_curve(model_path, train_losses, val_losses,
                                               val_steer_maes)
        onnx_path = None
        try:
            onnx_path = self.export_onnx(model_path)
        except Exception as e:
            print(f"ONNX export failed: {e}")

        result = {
            "status": "completed",
            "model_path": model_path,
            "onnx_path": onnx_path,
            "model_arch": self.ARCH_NAME,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "epochs_trained": len(train_losses),
            "total_time": total_time,
            "avg_epoch_time": (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0,
            "train_samples": train_size,
            "val_samples": val_size,
            "total_sequences": n,
            "early_stopped": early_stopped,
            "val_steer_mae": best_metrics.get("val_steer_mae", 0.0),
            "val_speed_mae": best_metrics.get("val_speed_mae", 0.0),
            "val_ade_m": best_metrics.get("val_ade_m", 0.0),
            "use_traj": cfg.use_traj,
            "n_no_scan": dataset.n_no_scan,
            "n_mode_skipped": dataset.n_mode_skipped,
        }

        if self.mlflow_manager:
            try:
                best_epoch = (val_losses.index(best_val_loss) + 1) if val_losses else 0
                training_params = {
                    "model_type": self.ARCH_NAME, "model_arch": self.ARCH_NAME,
                    "data_folder": os.path.basename(self.models_dir),
                    "preset": preset, "num_bins": cfg.num_bins,
                    "num_beams_raw": cfg.num_beams_raw,
                    "downsample_mode": cfg.downsample_mode,
                    "stack_frames": cfg.stack_frames, "use_valid_ch": cfg.use_valid_ch,
                    "max_range_mm": cfg.max_range_mm, "max_speed": cfg.max_speed,
                    "hidden_dim": cfg.hidden_dim, "dropout": cfg.dropout,
                    "use_traj": cfg.use_traj, "horizon": cfg.horizon, "dt": cfg.dt,
                    "pose_source": pose_source if cfg.use_traj else "",
                    "w_steer": w_steer, "w_speed": w_speed,
                    "w_traj": w_traj if cfg.use_traj else 0.0,
                    "steer_balance": steer_balance,
                    "mode_filter": mode_filter, "split_mode": split_mode,
                    "augment": augment, "mirror": mirror,
                    "num_epochs": epochs, "learning_rate": learning_rate,
                    "batch_size": batch_size, "val_split": val_split,
                    "weight_decay": 1e-4,
                    "use_early_stopping": use_early_stopping, "patience": patience,
                    "quality_excluded_frames": len(quality_excluded),
                    "model_params_total": model_params_total,
                    "device": str(device), "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "comment": config.get('comment'),
                    "model_name": config.get('model_name'),
                }
                metrics = {
                    "best_val_loss": best_val_loss,
                    "final_train_loss": result["final_train_loss"],
                    "final_val_loss": val_losses[-1] if val_losses else 0.0,
                    "best_epoch": best_epoch,
                    "best_val_steer_mae": best_metrics.get("val_steer_mae", 0.0),
                    "best_val_speed_mae": best_metrics.get("val_speed_mae", 0.0),
                    "best_val_ade_m": best_metrics.get("val_ade_m", 0.0),
                    "total_training_time": total_time,
                    "avg_epoch_time": result["avg_epoch_time"],
                    "completed_epochs": len(train_losses),
                    "status": "completed",
                    "train_losses": train_losses, "val_losses": val_losses,
                }
                dataset_info = {"train_samples": train_size, "val_samples": val_size,
                                "total_sequences": n}
                extra = [p for p in (curve_path, onnx_path) if p]
                self.mlflow_manager.log_lidar_policy_model(
                    model_path, training_params, metrics, dataset_info,
                    extra_artifacts=extra or None)
            except Exception as e:
                print(f"MLflow logging failed: {e}")

        return result

    # ------------------------------------------------------------ persistence
    def _save_model(self, state_dict, cfg, config, meta, pose_source, train_losses,
                    val_losses, best_metrics, steer_bin_weights, epochs_trained):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        custom = str(config.get('model_name') or '').strip()
        if custom:
            # 自動運転ダイアログのモデル名（prefix はファイル一覧のフィルタに必要）
            base = custom if custom.startswith(self.MODEL_PREFIX) else self.MODEL_PREFIX + custom
            filename = base[:-4] + ".pth" if base.endswith(".pth") else base + ".pth"
        else:
            filename = (f"{self.MODEL_PREFIX}{timestamp}_{cfg.preset}"
                        f"_b{cfg.num_bins}_k{cfg.stack_frames}.pth")
        model_path = os.path.join(self.models_dir, filename)
        torch.save({
            "model_state_dict": state_dict,
            "config": cfg.to_dict(),
            "model_type": self.ARCH_NAME,
            "model_arch": self.ARCH_NAME,
            "pose_source": pose_source,
            "lidar_meta": meta,
            "steer_bin_weights": np.asarray(steer_bin_weights).tolist(),
            "training_config": {k: v for k, v in config.items()
                                if isinstance(v, (int, float, str, bool, list, tuple,
                                                  type(None)))},
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_metrics": best_metrics,
            "epochs_trained": epochs_trained,
            "created_at": datetime.now().isoformat(),
        }, model_path)
        print(f"LiDAR policy model saved to: {model_path}")
        return model_path

    @staticmethod
    def load_model(model_path, device=None):
        """Returns (model, cfg(LidarPolicyConfig), meta(dict))"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if ckpt.get("model_type") != LidarPolicyTrainingManager.ARCH_NAME:
            raise ValueError(f"lidar_policy モデルではありません: {model_path}")
        cfg = LidarPolicyConfig.from_dict(ckpt["config"])
        model = build_lidar_policy(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        meta = {k: ckpt.get(k) for k in ("pose_source", "lidar_meta", "training_config",
                                         "best_metrics", "epochs_trained", "created_at")}
        return model, cfg, meta

    def _save_training_curve(self, model_path, train_losses, val_losses, val_maes):
        if not train_losses:
            return None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            curve_path = os.path.splitext(model_path)[0] + "_training_curve.png"
            fig, ax = plt.subplots(figsize=(8, 5))
            ep = range(1, len(train_losses) + 1)
            ax.plot(ep, train_losses, label='Train loss', color='#1f77b4')
            if val_losses:
                ax.plot(ep, val_losses, label='Val loss', color='#ff7f0e')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            if val_maes:
                ax2 = ax.twinx()
                ax2.plot(ep, val_maes, label='Val steer MAE', color='#2ca02c', linestyle='--')
                ax2.set_ylabel('steer MAE')
                l1, n1 = ax.get_legend_handles_labels()
                l2, n2 = ax2.get_legend_handles_labels()
                ax.legend(l1 + l2, n1 + n2)
            else:
                ax.legend()
            ax.set_title('LiDAR Policy Training Curve')
            plt.tight_layout()
            plt.savefig(curve_path, dpi=100)
            plt.close()
            return curve_path
        except Exception as e:
            print(f"学習曲線の保存に失敗: {e}")
            return None

    # ------------------------------------------------------------------ onnx
    @staticmethod
    def export_onnx(model_path, onnx_path=None, flat_obs=False, opset=17):
        """学習済み .pth → ONNX。

        既定: 入力 scan_mm (1,K,N_raw) float32 [mm] / speed (1,1) [m/s]、
              出力 action (1,2) [angle, throttle]（+ traj (1,H,2)）。前処理はグラフ内。
        flat_obs=True: RL 互換 obs (1, num_bins+1) 単一入力（rl_policy.RLPolicy 用。
              downsample_mode="index"・K=1・valid ch なしで学習したモデル向け）。
        メタデータ（config）は ONNX metadata_props と sidecar .json に書く。
        """
        model, cfg, meta = LidarPolicyTrainingManager.load_model(model_path, 'cpu')
        if onnx_path is None:
            suffix = "_flat.onnx" if flat_obs else ".onnx"
            onnx_path = os.path.splitext(model_path)[0] + suffix

        if flat_obs:
            if cfg.stack_frames != 1 or cfg.use_valid_ch:
                raise ValueError("flat_obs は stack_frames=1 かつ valid ch なしのモデルのみ")
            net = FlatObsWrapper(model).eval()
            obs = torch.zeros(1, cfg.num_bins + cfg.state_dim)
            _onnx_export(net, (obs,), onnx_path, opset,
                         input_names=["obs"], output_names=["action"],
                         dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}})
        else:
            class _Export(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, scan_mm, speed):
                    state = torch.clamp(speed.abs() / cfg.max_speed, 0.0, 1.0)
                    return self.m(scan_mm, state)

            net = _Export(model).eval()
            scan = torch.zeros(1, cfg.stack_frames, cfg.num_beams_raw)
            speed = torch.zeros(1, 1)
            out_names = ["action"] + (["traj"] if cfg.use_traj else [])
            dyn = {"scan_mm": {0: "batch"}, "speed": {0: "batch"}, "action": {0: "batch"}}
            if cfg.use_traj:
                dyn["traj"] = {0: "batch"}
            _onnx_export(net, (scan, speed), onnx_path, opset,
                         input_names=["scan_mm", "speed"], output_names=out_names,
                         dynamic_axes=dyn)

        sidecar = {"model_type": LidarPolicyTrainingManager.ARCH_NAME,
                   "flat_obs": flat_obs, "config": cfg.to_dict(),
                   "lidar_meta": meta.get("lidar_meta"),
                   "source_checkpoint": os.path.basename(model_path)}
        try:
            import onnx
            m = onnx.load(onnx_path)
            for k, v in (("model_type", sidecar["model_type"]),
                         ("config", json.dumps(cfg.to_dict())),
                         ("flat_obs", str(flat_obs))):
                p = m.metadata_props.add()
                p.key, p.value = k, v
            onnx.save(m, onnx_path)
        except Exception as e:
            print(f"ONNX metadata write skipped: {e}")
        with open(os.path.splitext(onnx_path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
        print(f"LiDAR policy ONNX exported: {onnx_path}")
        return onnx_path

    # ------------------------------------------------------------- inference
    def predict_current(self, model, cfg, vocab, selected_sources, target_index,
                        images, source_images_map, annotations, pose_manager=None,
                        pose_source="pose", device=None, topk=None):
        """単一フレーム推論（GUI 逐次推論用。TogivadTrainingManager.predict_current と
        同じ引数順で呼べるよう vocab/sources を受け流す）。

        Returns:
            dict | None — {"control": (angle, throttle), "best": (H,2)|None}
        """
        if device is None:
            device = next(model.parameters()).device
        ds = LidarPolicyDataset.__new__(LidarPolicyDataset)
        ds.__dict__.update(dict(cfg=cfg, images=images, augment=False, mirror=False,
                                beam_dropout=0.0, range_noise_mm=0.0, roll_beams=0,
                                _scan_cache={}, _mode_cache={}, samples=[],
                                n_no_scan=0, n_mode_skipped=0))
        if target_index >= len(images):
            return None
        session, npy = scan_path_for_image(images[target_index])
        if npy is None:
            return None
        paths = [npy]
        for k in range(1, cfg.stack_frames):
            j = target_index - k
            prev = None
            if 0 <= j < len(images):
                s2, p2 = scan_path_for_image(images[j])
                if s2 == session:
                    prev = p2
            paths.insert(0, prev if prev is not None else paths[0])
        ann = annotations.get(target_index, {}) if annotations else {}
        speed = float(ann.get("speed", ann.get("pose/speed", 0.0)) or 0.0)
        state = torch.tensor([[np.clip(abs(speed) / cfg.max_speed, 0.0, 1.0)]],
                             dtype=torch.float32, device=device)
        scan = torch.from_numpy(np.stack([ds._load_scan(p) for p in paths], 0)
                                ).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(scan, state)
        if cfg.use_traj:
            action, traj = out
            best = traj[0].cpu().numpy()
        else:
            action, best = out, None
        a = action[0].cpu().numpy()
        return {"control": (float(a[0]), float(a[1])), "best": best}

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
