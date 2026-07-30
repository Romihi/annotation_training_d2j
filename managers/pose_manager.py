# pose_manager.py
"""
pose/slam/vslam/aruco の統合管理

記録カタログの各フレームに含まれる複数の自己位置ソース（pose/slam/vslam/aruco）を
共通スキーマ（x[m], y[m], theta[rad]）に正規化し、優先順位に基づくソース選択、
セッション内での縮退ソース（値が全く動かない＝未稼働）の検出を行う。

Phase 2: 品質フィルタ（テレポート・status異常検出）、区間ごとのソース上書き、
欠損/低品質区間の補間、togivad互換の将来軌道(ego座標系)計算を提供する。

catalogキー名の契約は docs/RECORD_KEY_NAMING.md を参照。
togivad側の実装は togivad/dataset.py::future_trajectory を参照（本モジュールは
speed+yawレートの積分ではなく、実測pose系列から直接ego座標変換する点が異なる）。
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ソースごとの優先順位（デフォルト）
# aruco: 俯瞰カメラによる絶対位置（最も信頼できるが、対応区間のみ）
# slam: 2D LiDAR SLAM（地図座標系、通常は安定）
# vslam: Visual/Inertial SLAM（連続的だがドリフトしうる）
# pose: 車載デッドレコニング＋IMU融合（常時稼働だが長時間でドリフト）
DEFAULT_PRIORITY = ["aruco", "slam", "vslam", "pose"]

OK_STATUSES = {"ok"}
INTERP_SOURCE = "interp"
INTERP_STATUS = "interp"


@dataclass
class PoseSample:
    """1フレーム・1ソース分の正規化済み自己位置"""
    index: int
    source: str
    x: float          # meters
    y: float           # meters
    theta: float        # radians, (-pi, pi]
    status: str = "unknown"
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.status in OK_STATUSES


class PoseSourceManager:
    """catalogエントリからpose/slam/vslam/arucoを読み取り、統合的に扱うためのマネージャ"""

    def __init__(self, priority: Optional[List[str]] = None, min_ok_ratio: float = 0.05):
        self.priority = list(priority) if priority else list(DEFAULT_PRIORITY)
        self.min_ok_ratio = min_ok_ratio

        self._raw: Dict[int, Dict[str, PoseSample]] = {}
        self._timestamps_ms: Dict[int, int] = {}
        self._source_total_counts: Dict[str, int] = {s: 0 for s in self.priority}
        self._source_ok_counts: Dict[str, int] = {s: 0 for s in self.priority}
        self._available_sources_cache: Optional[List[str]] = None

        # Phase 2: 区間ごとのソース上書き [(start_index, end_index, source), ...]
        self._range_overrides: List[Tuple[int, int, str]] = []
        # Phase 2: 欠損/低品質区間を補間した結果のインデックス集合
        self._interpolated_indexes: Set[int] = set()
        # 学習用軌道ラベル（togivad/future_traj）が保存されているフレーム集合
        self._future_traj_indexes: Set[int] = set()

    def reset(self) -> None:
        self._raw.clear()
        self._timestamps_ms.clear()
        self._source_total_counts = {s: 0 for s in self.priority}
        self._source_ok_counts = {s: 0 for s in self.priority}
        self._available_sources_cache = None
        self._range_overrides.clear()
        self._interpolated_indexes.clear()
        self._future_traj_indexes.clear()

    def ingest_entry(self, index: int, entry: Dict) -> None:
        """1つのcatalog行（辞書）からpose系フィールドを抽出して取り込む"""
        timestamp_ms = entry.get("_timestamp_ms")
        if timestamp_ms is not None:
            self._timestamps_ms[index] = int(timestamp_ms)

        # 保存済みの学習用軌道ラベルを検出（マップビューでの可視化に使用）
        if "togivad/future_traj" in entry:
            self._future_traj_indexes.add(index)

        samples: Dict[str, PoseSample] = {}
        for source in self.priority:
            sample = self._extract_source(source, index, entry)
            if sample is None:
                continue
            samples[source] = sample
            self._source_total_counts[source] += 1
            if sample.is_ok:
                self._source_ok_counts[source] += 1

        if samples:
            self._raw[index] = samples
            self._available_sources_cache = None

    def _extract_source(self, source: str, index: int, entry: Dict) -> Optional[PoseSample]:
        if source == "pose":
            return self._extract_pose_sensor(index, entry)

        x = entry.get(f"{source}/x")
        y = entry.get(f"{source}/y")
        theta = entry.get(f"{source}/theta")
        if x is None or y is None or theta is None:
            return None

        status = entry.get(f"{source}/status", "unknown")
        extra = {}
        if source == "aruco":
            for key in ("n_markers", "reproj_err"):
                full_key = f"aruco/{key}"
                if full_key in entry:
                    extra[key] = entry[full_key]

        return PoseSample(index=index, source=source, x=float(x), y=float(y),
                           theta=float(theta), status=str(status), extra=extra)

    def _extract_pose_sensor(self, index: int, entry: Dict) -> Optional[PoseSample]:
        # 新スキーマ（m/rad、slam/vslam/arucoと同一単位系）
        if "pose/x" in entry and "pose/y" in entry and "pose/theta" in entry:
            x = float(entry["pose/x"])
            y = float(entry["pose/y"])
            theta = float(entry["pose/theta"])
        # 旧スキーマ（mm/deg）へのフォールバック - 既存記録セッションとの互換のため
        elif "pose/x_mm" in entry and "pose/y_mm" in entry and "pose/yaw_deg" in entry:
            x = float(entry["pose/x_mm"]) / 1000.0
            y = float(entry["pose/y_mm"]) / 1000.0
            theta = math.radians(float(entry["pose/yaw_deg"]))
        else:
            return None

        # poseセンサーは常時稼働のデッドレコニングのため、明示的なstatusフィールドを持たない
        status = str(entry.get("pose/status", "ok"))

        extra = {}
        # road_condition: 0=smooth / 1=rough（悪路検知、FW追加予定フィールド）
        for key in ("speed", "slip", "corr", "v_imu", "gyro_z", "accel", "pitch", "roll",
                     "road_condition"):
            full_key = f"pose/{key}"
            if full_key in entry:
                extra[key] = entry[full_key]

        return PoseSample(index=index, source="pose", x=x, y=y, theta=theta,
                           status=status, extra=extra)

    def available_sources(self) -> List[str]:
        """セッション内で実際に有効なソース（優先順）を返す（縮退ソースは除外）"""
        if self._available_sources_cache is None:
            self._available_sources_cache = [
                s for s in self.priority
                if self._source_total_counts[s] > 0
                and not self._is_degenerate(s)
                and (self._source_ok_counts[s] / max(1, self._source_total_counts[s])) >= self.min_ok_ratio
            ]
        return self._available_sources_cache

    def heading_varies(self, source: str, min_range_rad: float = 0.087) -> bool:
        """source の方位(theta)が実際に変化するか（凍結ヨーの検出）。

        pose の BNO055 が未校正等でヨーが凍結すると theta が全フレーム一定になり、
        デッドレコニング位置も凍結方位に沿って直進する。その結果 ego 座標系の
        将来軌道が常に真っ直ぐ（横方向=0）になってしまう。theta の範囲が
        min_range_rad(既定 ~5°)未満なら凍結とみなし False を返す。
        """
        thetas = [s[source].theta for s in self._raw.values() if source in s]
        if len(thetas) < 2:
            return False
        return (max(thetas) - min(thetas)) > min_range_rad

    def best_trajectory_source(self, priority=("pose", "slam", "vslam")) -> Optional[str]:
        """走行軌道に使える（方位が凍結していない）ソースを優先順で返す。

        pose を優先しつつ、pose のヨーが凍結しているセッションでは slam 等へ
        フォールバックする。どれも使えなければ available_sources の先頭。
        """
        available = set(self.available_sources())
        for src in priority:
            if src in available and self.heading_varies(src):
                return src
        avail = self.available_sources()
        return avail[0] if avail else None

    def _is_degenerate(self, source: str) -> bool:
        """値が全く変化しない（＝未稼働・センサー未接続）ソースを検出"""
        xs: List[float] = []
        ys: List[float] = []
        for samples in self._raw.values():
            sample = samples.get(source)
            if sample is not None:
                xs.append(sample.x)
                ys.append(sample.y)
        if len(xs) < 2:
            return True
        return (max(xs) - min(xs) < 1e-6) and (max(ys) - min(ys) < 1e-6)

    def get_pose(self, index: int, prefer: Optional[str] = None) -> Optional[PoseSample]:
        """指定フレームの自己位置を優先順位に従って取得（statusがokのものを優先）

        区間上書き（set_range_override）や補間結果（interpolate_gaps）は
        通常の優先順位より常に優先される。
        """
        if index in self._interpolated_indexes:
            samples = self._raw.get(index)
            if samples and INTERP_SOURCE in samples:
                return samples[INTERP_SOURCE]

        samples = self._raw.get(index)
        if not samples:
            return None

        range_source = self._range_override_source(index)
        order = self._resolve_order(range_source or prefer)

        for source in order:
            sample = samples.get(source)
            if sample is not None and sample.is_ok:
                return sample
        # okなソースが無ければ、statusを問わず優先順位に従って返す
        for source in order:
            sample = samples.get(source)
            if sample is not None:
                return sample
        return None

    def _resolve_order(self, prefer: Optional[str]) -> List[str]:
        available = self.available_sources()
        if prefer is None:
            return available
        if prefer in available:
            return [prefer] + [s for s in available if s != prefer]
        return [prefer] + available

    def _range_override_source(self, index: int) -> Optional[str]:
        # 後から設定した上書きほど優先（リスト末尾を優先して検索）
        for start, end, source in reversed(self._range_overrides):
            if start <= index <= end:
                return source
        return None

    def get_trajectory(self, source: Optional[str] = None,
                        indexes: Optional[List[int]] = None) -> List[PoseSample]:
        """セッション全体（または指定インデックス群）の軌跡を返す

        source=None の場合は各フレームでの優先順位選択（get_pose相当）の結果を返す。
        source を指定した場合はそのソースのサンプルのみを返す（無い場合はスキップ）。
        """
        idxs = indexes if indexes is not None else sorted(self._raw.keys())
        result: List[PoseSample] = []
        for idx in idxs:
            if source is None:
                sample = self.get_pose(idx)
            else:
                sample = self._raw.get(idx, {}).get(source)
            if sample is not None:
                result.append(sample)
        return result

    def flag_jumps(self, poses: List[PoseSample], max_jump_m: float = 1.0) -> Set[int]:
        """隣接フレーム間で不自然な位置飛び（テレポート）を検出し、インデックス集合を返す"""
        flagged: Set[int] = set()
        prev: Optional[PoseSample] = None
        for pose in poses:
            if prev is not None:
                dist = math.hypot(pose.x - prev.x, pose.y - prev.y)
                if dist > max_jump_m:
                    flagged.add(pose.index)
            prev = pose
        return flagged

    def has_any_pose(self) -> bool:
        return bool(self._raw)

    def known_indexes(self) -> List[int]:
        """自己位置データが存在するフレームインデックス一覧（昇順）"""
        return sorted(self._raw.keys())

    def future_traj_indexes(self) -> Set[int]:
        """学習用軌道ラベル（togivad/future_traj）が保存されているフレーム集合"""
        return set(self._future_traj_indexes)

    def mark_future_traj(self, indexes) -> None:
        """軌道ラベルを保存したフレームを記録する（書き戻し完了後に呼ぶ）"""
        self._future_traj_indexes.update(indexes)

    def rough_road_indexes(self) -> Set[int]:
        """悪路（pose/road_condition == 1）が検知されたフレーム集合"""
        result: Set[int] = set()
        for idx, samples in self._raw.items():
            pose = samples.get("pose")
            if pose is not None and pose.extra.get("road_condition", 0) == 1:
                result.add(idx)
        return result

    def slip_indexes(self, min_slip: float = 1.0) -> Set[int]:
        """スリップが検知されたフレーム集合（pose/slip >= min_slip）

        pose/slip は状態値で、1以上がスリップ検知状態。
        （現行ファームウェアは検知が過敏で大半のフレームが1になる既知の問題が
        あるが、それはFW側で改善予定のためここでは閾値1.0で忠実に拾う）
        """
        result: Set[int] = set()
        for idx, samples in self._raw.items():
            pose = samples.get("pose")
            if pose is not None and float(pose.extra.get("slip", 0.0)) >= min_slip:
                result.add(idx)
        return result

    # --- Phase 2: 品質フィルタ ------------------------------------------------

    def flag_quality_issues(self, max_jump_m: float = 1.0) -> Set[int]:
        """statusがok以外、または位置飛び(テレポート)のあるフレームを検出する

        get_pose()（区間上書き・補間済みを含む）の結果に基づくため、既に
        上書き/補間で解消済みの区間は再フラグされない。
        """
        poses = self.get_trajectory()
        flagged = self.flag_jumps(poses, max_jump_m=max_jump_m)
        for pose in poses:
            if not pose.is_ok:
                flagged.add(pose.index)
        return flagged

    # --- Phase 2: 区間編集 ----------------------------------------------------

    def set_range_override(self, start_index: int, end_index: int, source: str) -> None:
        """[start_index, end_index] の区間で使用するソースを強制指定する"""
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        self._range_overrides.append((start_index, end_index, source))

    def clear_range_overrides(self) -> None:
        self._range_overrides.clear()

    def interpolate_gaps(self, max_gap: int = 10) -> Set[int]:
        """okな自己位置が得られないフレームを、前後のokな点から補間して埋める

        インデックス差がmax_gap以下の範囲のみ対象とする。補間結果は
        source="interp", status="interp" として扱われ、get_pose()で最優先される。
        戻り値は新たに補間で埋めたインデックス集合。
        """
        known_indexes = sorted(self._raw.keys())
        filled: Set[int] = set()

        anchors: List[Tuple[int, PoseSample]] = []
        for idx in known_indexes:
            pose = self.get_pose(idx)
            if pose is not None and pose.is_ok:
                anchors.append((idx, pose))

        for (idx_a, pose_a), (idx_b, pose_b) in zip(anchors, anchors[1:]):
            gap = idx_b - idx_a
            if gap <= 1 or gap - 1 > max_gap:
                continue
            for idx in range(idx_a + 1, idx_b):
                if idx not in self._raw:
                    continue
                existing = self.get_pose(idx)
                if existing is not None and existing.is_ok:
                    continue
                ratio = (idx - idx_a) / gap
                x = pose_a.x + (pose_b.x - pose_a.x) * ratio
                y = pose_a.y + (pose_b.y - pose_a.y) * ratio
                theta = _interpolate_angle(pose_a.theta, pose_b.theta, ratio)
                sample = PoseSample(index=idx, source=INTERP_SOURCE, x=x, y=y,
                                     theta=theta, status=INTERP_STATUS)
                self._raw.setdefault(idx, {})[INTERP_SOURCE] = sample
                self._interpolated_indexes.add(idx)
                filled.add(idx)

        return filled

    def clear_interpolation(self) -> None:
        for idx in self._interpolated_indexes:
            samples = self._raw.get(idx)
            if samples:
                samples.pop(INTERP_SOURCE, None)
        self._interpolated_indexes.clear()

    # --- Phase 2: togivad互換 将来軌道(ego座標系)計算 --------------------------

    def compute_future_trajectory(self, index: int, horizon: int = 20, dt: float = 0.05,
                                   max_dt_gap_s: float = 0.5,
                                   exclude: Optional[Set[int]] = None,
                                   prefer: Optional[str] = None) -> Optional[np.ndarray]:
        """フレームindexから horizon*dt 秒先までの ego座標系将来軌道 (horizon, 2)。

        togivad/dataset.py::future_trajectory と同じ出力仕様（+X前方/+Y左、
        t=dt..horizon*dtの等間隔点）だが、speed+yawレートの積分ではなく
        実測pose系列（get_pose、上書き/補間を反映）から直接ego座標変換する。
        作成できなければNoneを返す。

        prefer でソースを指定できる（例: togivad学習の "pose"（既定）/"slam"。
        get_pose と同じ意味論で、指定ソースが無いフレームは優先順位に
        フォールバックする）。
        """
        origin = self.get_pose(index, prefer=prefer)
        if origin is None:
            return None
        t0 = self._timestamps_ms.get(index)
        if t0 is None:
            return None

        exclude = exclude or set()
        need_s = horizon * dt

        known_indexes = sorted(k for k in self._timestamps_ms.keys() if k >= index)
        ts: List[float] = [0.0]
        xs: List[float] = [origin.x]
        ys: List[float] = [origin.y]

        k_pos = known_indexes.index(index)
        reached = ts[-1] >= need_s + dt
        for i in range(k_pos, len(known_indexes) - 1):
            idx_a = known_indexes[i]
            idx_b = known_indexes[i + 1]
            ta = self._timestamps_ms[idx_a]
            tb = self._timestamps_ms[idx_b]
            dtau = (tb - ta) / 1000.0
            if dtau <= 0 or dtau > max_dt_gap_s:
                break

            if idx_b not in exclude:
                pose_b = self.get_pose(idx_b, prefer=prefer)
                if pose_b is not None:
                    ts.append((tb - t0) / 1000.0)
                    xs.append(pose_b.x)
                    ys.append(pose_b.y)

            if ts[-1] >= need_s + dt:
                reached = True
                break

        if not reached or len(ts) < 2:
            return None

        tq = (np.arange(horizon) + 1) * dt
        x_world = np.interp(tq, ts, xs)
        y_world = np.interp(tq, ts, ys)

        dx = x_world - origin.x
        dy = y_world - origin.y
        cos_t, sin_t = math.cos(origin.theta), math.sin(origin.theta)
        x_ego = dx * cos_t + dy * sin_t
        y_ego = -dx * sin_t + dy * cos_t
        return np.stack([x_ego, y_ego], axis=1)


    def yaw_rate(self, index: int, prefer: Optional[str] = None,
                 max_dt_gap_s: float = 0.5) -> float:
        """フレームindexのヨーレート [rad/s]（次の既知フレームとのtheta差分）。

        togivad学習の ego 入力用。学習ラベル（compute_future_trajectory）と
        同じソース選択(prefer)・同じ実測系列から作ることで情報源を揃える。
        次フレームが無い/時間ギャップが大きい場合は 0.0。
        """
        t0 = self._timestamps_ms.get(index)
        p0 = self.get_pose(index, prefer=prefer)
        if t0 is None or p0 is None:
            return 0.0
        later = [k for k in self._timestamps_ms.keys() if k > index]
        if not later:
            return 0.0
        idx_b = min(later)
        t1 = self._timestamps_ms[idx_b]
        dtau = (t1 - t0) / 1000.0
        if dtau <= 1e-6 or dtau > max_dt_gap_s:
            return 0.0
        p1 = self.get_pose(idx_b, prefer=prefer)
        if p1 is None:
            return 0.0
        d = (p1.theta - p0.theta + math.pi) % (2 * math.pi) - math.pi
        return d / dtau

    def relative_dpose(self, prev_index: int, index: int,
                       prefer: Optional[str] = None,
                       max_dt_gap_s: float = 0.5) -> Optional[np.ndarray]:
        """前フレーム prev_index → フレーム index の ego 相対運動 [dx, dy, dθ]。

        TogiVAD 時系列融合（T1-a）の warp 入力。**実測 pose 差分**から作るので
        走行軌道表示・学習ラベル（compute_future_trajectory）と同一情報源・同一
        座標規約になる（速度×dt の近似は使わない）。世界→ego 変換は
        compute_future_trajectory と同じ式。

        座標: 返り値は **prev フレーム系**（+X 前方 / +Y 左）での並進 (dx, dy) と
        方位差 dθ（(-π, π]）。dt ギャップ超過・pose 欠落・時刻欠落では None。
        """
        t0 = self._timestamps_ms.get(prev_index)
        t1 = self._timestamps_ms.get(index)
        if t0 is None or t1 is None:
            return None
        dtau = (t1 - t0) / 1000.0
        if dtau <= 1e-6 or dtau > max_dt_gap_s:
            return None
        p0 = self.get_pose(prev_index, prefer=prefer)
        p1 = self.get_pose(index, prefer=prefer)
        if p0 is None or p1 is None:
            return None
        dx_w, dy_w = p1.x - p0.x, p1.y - p0.y
        cos_t, sin_t = math.cos(p0.theta), math.sin(p0.theta)
        dx = dx_w * cos_t + dy_w * sin_t            # 前 ego 系 前方
        dy = -dx_w * sin_t + dy_w * cos_t           # 前 ego 系 左
        dth = (p1.theta - p0.theta + math.pi) % (2 * math.pi) - math.pi
        return np.array([dx, dy, dth], dtype=np.float32)


def _interpolate_angle(a: float, b: float, ratio: float) -> float:
    """最短経路での角度補間（radian, 折り返し対応）"""
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + diff * ratio
