# -*- coding: utf-8 -*-
"""オフライン重み付け BC（Phase 1）: 走行ログから報酬→リターン→アドバンテージ→サンプル重みを計算する。

設計書: dev/SPEC_offline_rl_throttle.md §4.1〜4.3

入力はツールが読み込んだアノテーション辞書 ``{idx: {...}}`` とタイムスタンプ辞書
``{idx: ms}``。センサーキーは main.py の読込時にそのまま数値で保存されている
（例: 'lidar/FrFR', 'pose/slip'）。速度は 'speed'（m/s, 生値）に入っている。

出力は ``{idx: weight}``（平均 1.0 に正規化済み）と統計辞書。
Qt / torch には依存しない（単体テスト・CLI からも使える）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# 設定
# ============================================================================
@dataclass
class RewardConfig:
    """報酬・重み計算の設定（UI の既定値は config.RL_WEIGHT_DEFAULTS で上書き可）"""
    method: str = 'awr'              # 'awr' | 'filtered'
    target_head: str = 'speed'       # 'speed' | 'throttle' | 'both'
    gamma: float = 0.97              # 割引率
    beta: float = 0.7                # AWR 温度（標準化アドバンテージに対して）
    w_min: float = 0.1               # 重み下限（filtered の非採用フレームにも適用）
    w_max: float = 5.0               # 重み上限
    top_k: float = 0.4               # filtered: 上位割合 (0-1)
    baseline_mode: str = 'speed_bin' # 'global' | 'speed_bin'
    speed_bins: int = 8
    episode_gap_s: float = 1.0       # この秒数以上の間隔で別エピソードとみなす

    # 報酬係数
    c_speed: float = 1.0
    c_wall: float = 1.0
    c_side: float = 0.5
    c_slip: float = 2.0
    c_stuck: float = 5.0
    c_yaw: float = 0.2

    # 正規化・閾値
    speed_norm: float = 5.0          # 速度正規化 [m/s]（速度報酬は v/speed_norm を 1 でクリップ）
    wall_mm: float = 1000.0          # 前方壁ペナルティが効き始める距離 [mm]
    side_mm: float = 500.0           # 斜め前壁ペナルティが効き始める距離 [mm]
    yaw_norm: float = 100.0          # ヨーレート変化の正規化 [deg/s]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'RewardConfig':
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ============================================================================
# フレーム報酬
# ============================================================================
_SPEED_KEYS = ('speed', 'enc/speed', 'user/speed', 'pose/speed')


def _get_speed(ann: dict) -> float:
    for k in _SPEED_KEYS:
        v = ann.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def _get(ann: dict, key: str, default: float = 0.0) -> float:
    v = ann.get(key)
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f):
        return default
    return f


def reward_terms(ann: dict, prev_ann: Optional[dict], cfg: RewardConfig) -> Dict[str, float]:
    """フレーム報酬を項ごとに返す（合計が r_t）。欠損キーの項は 0。"""
    terms: Dict[str, float] = {}

    v = _get_speed(ann)
    v_n = max(0.0, min(1.0, v / cfg.speed_norm)) if cfg.speed_norm > 0 else 0.0
    terms['speed'] = cfg.c_speed * v_n

    # 前方壁: FrFR が wall_mm 以内で線形にペナルティ（0 mm で -c_wall）
    fr = _get(ann, 'lidar/FrFR', default=float('inf'))
    if math.isfinite(fr) and cfg.wall_mm > 0:
        terms['wall'] = -cfg.c_wall * max(0.0, (cfg.wall_mm - fr) / cfg.wall_mm)
    else:
        terms['wall'] = 0.0

    # 斜め前壁: FrLH / FrRH の小さい方
    sides = [s for s in (_get(ann, 'lidar/FrLH', float('inf')),
                         _get(ann, 'lidar/FrRH', float('inf'))) if math.isfinite(s)]
    if sides and cfg.side_mm > 0:
        d = min(sides)
        terms['side'] = -cfg.c_side * max(0.0, (cfg.side_mm - d) / cfg.side_mm)
    else:
        terms['side'] = 0.0

    terms['slip'] = -cfg.c_slip * (1.0 if _get(ann, 'pose/slip') > 0.5 else 0.0)
    terms['stuck'] = -cfg.c_stuck * (1.0 if _get(ann, 'pose/stuck') > 0.5 else 0.0)

    # ヨーレート変化（乱暴な操作）
    if prev_ann is not None and 'pose/gyro_z' in ann and 'pose/gyro_z' in prev_ann and cfg.yaw_norm > 0:
        dyaw = abs(_get(ann, 'pose/gyro_z') - _get(prev_ann, 'pose/gyro_z'))
        terms['yaw'] = -cfg.c_yaw * min(1.0, dyaw / cfg.yaw_norm)
    else:
        terms['yaw'] = 0.0

    return terms


def frame_reward(ann: dict, prev_ann: Optional[dict], cfg: RewardConfig) -> float:
    return float(sum(reward_terms(ann, prev_ann, cfg).values()))


# ============================================================================
# エピソード分割・リターン・ベースライン
# ============================================================================
def split_episodes(idxs: List[int], timestamps: Optional[Dict[int, float]],
                   gap_s: float) -> List[List[int]]:
    """タイムスタンプ間隔（ms）が gap_s 秒を超える箇所、または連番が途切れる箇所で分割。"""
    if not idxs:
        return []
    episodes: List[List[int]] = [[idxs[0]]]
    for prev, cur in zip(idxs[:-1], idxs[1:]):
        new_ep = False
        if timestamps is not None and prev in timestamps and cur in timestamps:
            try:
                if (float(timestamps[cur]) - float(timestamps[prev])) / 1000.0 > gap_s:
                    new_ep = True
            except (TypeError, ValueError):
                pass
        elif cur - prev > 1:
            new_ep = True
        if new_ep:
            episodes.append([cur])
        else:
            episodes[-1].append(cur)
    return episodes


def returns_to_go(rewards: Dict[int, float], episodes: List[List[int]], gamma: float) -> Dict[int, float]:
    G: Dict[int, float] = {}
    for ep in episodes:
        g = 0.0
        for i in reversed(ep):
            g = rewards[i] + gamma * g
            G[i] = g
    return G


def baseline_values(G: Dict[int, float], annotations: Dict[int, dict],
                    cfg: RewardConfig) -> Dict[int, float]:
    idxs = list(G.keys())
    if not idxs:
        return {}
    if cfg.baseline_mode == 'speed_bin' and cfg.speed_bins > 1:
        speeds = np.array([_get_speed(annotations[i]) for i in idxs], dtype=float)
        edges = np.quantile(speeds, np.linspace(0.0, 1.0, cfg.speed_bins + 1))
        # 重複エッジ（速度が一定のとき）は global にフォールバック
        if len(np.unique(edges)) < 3:
            mean = float(np.mean([G[i] for i in idxs]))
            return {i: mean for i in idxs}
        bins = np.clip(np.searchsorted(edges, speeds, side='right') - 1, 0, cfg.speed_bins - 1)
        Gs = np.array([G[i] for i in idxs], dtype=float)
        out: Dict[int, float] = {}
        for b in range(cfg.speed_bins):
            mask = bins == b
            if mask.any():
                m = float(Gs[mask].mean())
                for i, use in zip(idxs, mask):
                    if use:
                        out[i] = m
        return out
    mean = float(np.mean([G[i] for i in idxs]))
    return {i: mean for i in idxs}


# ============================================================================
# メイン: 重み計算
# ============================================================================
def compute_offline_weights(annotations: Dict[int, dict],
                            timestamps: Optional[Dict[int, float]],
                            cfg: RewardConfig) -> Tuple[Dict[int, float], dict]:
    """全フレームの重みと統計を返す。

    Returns:
        weights: {idx: w}  平均 1.0 に正規化済み
        stats:   分布統計・報酬項寄与・エピソード数・上位/下位フレーム等
    """
    idxs = sorted(i for i in annotations.keys() if isinstance(i, int))
    if not idxs:
        return {}, {'n_frames': 0}

    # フレーム報酬（項別も保持）
    rewards: Dict[int, float] = {}
    term_sums: Dict[str, float] = {}
    prev = None
    prev_idx = None
    for i in idxs:
        ann = annotations[i]
        # 連番でなければ prev を使わない（yaw 変化の誤計算防止）
        p = prev if (prev_idx is not None and i - prev_idx == 1) else None
        terms = reward_terms(ann, p, cfg)
        rewards[i] = float(sum(terms.values()))
        for k, v in terms.items():
            term_sums[k] = term_sums.get(k, 0.0) + v
        prev, prev_idx = ann, i

    episodes = split_episodes(idxs, timestamps, cfg.episode_gap_s)
    G = returns_to_go(rewards, episodes, cfg.gamma)
    b = baseline_values(G, annotations, cfg)
    A = np.array([G[i] - b[i] for i in idxs], dtype=float)

    # 標準化
    std = float(A.std())
    if std > 1e-8:
        A_std = (A - A.mean()) / std
    else:
        A_std = np.zeros_like(A)

    if cfg.method == 'filtered':
        k = min(max(cfg.top_k, 0.0), 1.0)
        if k <= 0.0:
            w = np.full_like(A_std, cfg.w_min)
        else:
            thr = np.quantile(A_std, 1.0 - k) if k < 1.0 else -np.inf
            w = np.where(A_std >= thr, 1.0, cfg.w_min)
    else:
        beta = cfg.beta if cfg.beta > 1e-6 else 1e-6
        w = np.clip(np.exp(A_std / beta), cfg.w_min, cfg.w_max)

    mean_w = float(w.mean())
    if mean_w > 0:
        w = w / mean_w

    weights = {i: float(x) for i, x in zip(idxs, w)}

    order = np.argsort(A_std)
    n_show = min(8, len(idxs))
    stats = {
        'n_frames': len(idxs),
        'n_episodes': len(episodes),
        'method': cfg.method,
        'target_head': cfg.target_head,
        'weight_mean': float(np.mean(w)),
        'weight_median': float(np.median(w)),
        'weight_p05': float(np.percentile(w, 5)),
        'weight_p95': float(np.percentile(w, 95)),
        'weight_max': float(np.max(w)),
        'weight_min': float(np.min(w)),
        'reward_mean': float(np.mean(list(rewards.values()))),
        'return_mean': float(np.mean([G[i] for i in idxs])),
        'term_contrib': {k: float(v / len(idxs)) for k, v in term_sums.items()},
        'top_frames': [int(idxs[j]) for j in order[::-1][:n_show]],
        'bottom_frames': [int(idxs[j]) for j in order[:n_show]],
        # プレビュー用の系列
        'series_idx': [int(i) for i in idxs],
        'series_weight': [float(x) for x in w],
        'series_advantage': [float(x) for x in A_std],
        'series_speed': [float(_get_speed(annotations[i])) for i in idxs],
    }
    return weights, stats


def summarize_stats(stats: dict) -> str:
    """UI 表示用の 1 行サマリ"""
    if not stats or stats.get('n_frames', 0) == 0:
        return '対象フレームなし'
    return (f"frames={stats['n_frames']} episodes={stats['n_episodes']} "
            f"mean={stats['weight_mean']:.2f} median={stats['weight_median']:.2f} "
            f"p05={stats['weight_p05']:.2f} p95={stats['weight_p95']:.2f} max={stats['weight_max']:.2f}")


# ============================================================================
# 出力列の対応（設計書 §4.3）
# ============================================================================
def rl_weight_column_indices(num_outputs: int, use_speed: bool, target_head: str) -> List[int]:
    """サンプル重みを掛ける出力列のインデックスを返す。

    出力並び: 各フレーム [angle, throttle(, speed)] × (1 + 将来フレーム数)
    angle 列は常に対象外。speed 列が無いモデルで target_head='speed' のときは throttle に
    フォールバックする。
    """
    group = 3 if use_speed else 2
    n_frames = max(1, num_outputs // group)
    head = target_head
    if head == 'speed' and not use_speed:
        head = 'throttle'
    cols: List[int] = []
    for f in range(n_frames):
        base = f * group
        if head in ('throttle', 'both'):
            cols.append(base + 1)
        if head in ('speed', 'both') and use_speed:
            cols.append(base + 2)
    return [c for c in cols if c < num_outputs]
