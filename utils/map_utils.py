# utils/map_utils.py
# coding:utf-8
"""走行データフォルダに紐づく地図（占有格子）の自動読み込み。

togikaidrive-dev 側の run.py が走行終了時に記録フォルダへ書き込む
`map_ref.json`（走行→地図の参照メタ）と `map/`（ラスタのみの軽量
スナップショット）を読む。仕様: togikaidrive-dev docs/RECORD_KEY_NAMING.md §7。

解決順（find_map_for_run）:
  1. <data_dir>/map/ のスナップショット … zip 転送後の学習PCでも自己完結
  2. map_ref.json の map_dir … togikaidrive-dev リポジトリと同居している場合
     （data/data_<TS> の2階層上をリポジトリルートとみなして解決）
  3. TS ヒューリスティック … メタの無い旧データ用。data_<TS> と同じ
     タイムスタンプの地図フォルダを data/maps/<backend>/ から探す
     （mapping 走行は SESSION_TIMESTAMP を共有するため一致する）
  どれも無ければ None（地図なし表示）。

使い方:
    from utils.map_utils import find_map_for_run, load_map
    ref = find_map_for_run(data_dir)          # None なら地図なし
    if ref:
        m = load_map(ref["map_yaml"])          # MapData(image, resolution, origin)
        col, row = m.world_to_px(x, y)         # 記録の slam/x,y (map座標[m]) → 画素

確認用 CLI:
    python -m utils.map_utils <data_dir> [--preview out.png]
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

_TS_RE = re.compile(r"(\d{8}_\d{6})")


# ---- 解決（どの地図か） -----------------------------------------------------
def load_map_ref(data_dir: str) -> Optional[dict]:
    """<data_dir>/map_ref.json を読む。無ければ None（推定はしない）。"""
    path = os.path.join(data_dir, "map_ref.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _yaml_in(map_dir: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(map_dir, "*_map.yaml")))
    return hits[0] if hits else None


def find_map_for_run(data_dir: str,
                     maps_root: Optional[str] = None) -> Optional[dict]:
    """データフォルダに対応する地図を解決して {map_yaml, map_dir, source, ref} を返す。

    maps_root: data/maps 相当のルートを明示する場合に指定
    （省略時は data_dir の親構成 <root>/data/data_<TS> から <root>/data/maps を推定）。
    source は "snapshot" / "map_ref" / "inferred" のいずれか。
    """
    data_dir = os.path.abspath(data_dir)
    ref = load_map_ref(data_dir)

    # 1. 同梱スナップショット（可搬・最優先）
    snap = _yaml_in(os.path.join(data_dir, "map"))
    if snap:
        return {"map_yaml": snap, "map_dir": os.path.dirname(snap),
                "source": "snapshot", "ref": ref}

    if maps_root is None:
        # <root>/data/data_<TS> → <root>/data/maps を推定
        maps_root = os.path.join(os.path.dirname(data_dir), "maps")

    # 2. map_ref.json の map_dir（リポジトリ相対）を解決
    if ref and ref.get("map_dir"):
        root = os.path.dirname(os.path.dirname(data_dir))  # リポジトリルート想定
        for cand in (os.path.join(root, ref["map_dir"]),
                     os.path.join(os.path.dirname(maps_root), "..",
                                  ref["map_dir"])):
            y = _yaml_in(cand) if os.path.isdir(cand) else None
            if y:
                return {"map_yaml": y, "map_dir": os.path.abspath(cand),
                        "source": "map_ref", "ref": ref}

    # 3. TS ヒューリスティック（旧データ: 同TSの地図 = 同セッションの mapping）
    m = _TS_RE.search(os.path.basename(data_dir))
    if m and os.path.isdir(maps_root):
        ts = m.group(1)
        for d in sorted(glob.glob(os.path.join(maps_root, "*", f"*_{ts}"))):
            y = _yaml_in(d)
            if y:
                return {"map_yaml": y, "map_dir": d,
                        "source": "inferred", "ref": ref}
    return None


def find_latest_map(maps_root: str) -> Optional[dict]:
    """maps_root 直下の全バックエンドから最も新しい <name>_<TS>/ 地図を返す。

    走行との対応が取れないときの最終フォールバック
    （「直近に保存された地図」を表示する）。無ければ None。
    """
    best = None  # (ts, yaml, dir)
    for d in glob.glob(os.path.join(maps_root, "*", "*_*")):
        if not os.path.isdir(d) or "_archive" in d:
            continue
        m = _TS_RE.search(os.path.basename(d))
        y = _yaml_in(d)
        if m and y and (best is None or m.group(1) > best[0]):
            best = (m.group(1), y, d)
    if best:
        return {"map_yaml": best[1], "map_dir": best[2],
                "source": "latest", "ref": None}
    return None


def resolve_background_map(data_dir: str,
                           maps_root: Optional[str] = None) -> Optional[dict]:
    """マップビュー背景用の地図を自動解決する。

    優先順: 走行に紐づく地図（スナップショット→map_ref→同TS推定）
    → 直近に保存された地図（find_latest_map）。無ければ None。
    """
    data_dir = os.path.abspath(data_dir)
    hit = find_map_for_run(data_dir, maps_root)
    if hit:
        return hit
    if maps_root is None:
        maps_root = os.path.join(os.path.dirname(data_dir), "maps")
    return find_latest_map(maps_root) if os.path.isdir(maps_root) else None


# ---- 走行時に追従した経路（centerline / raceline CSV） --------------------------
# 走行データの manifest.json 3行目（メタ辞書）に記録される path_csv（追従した
# ウェイポイント CSV の走行機側フルパス）を、学習PC側の地図フォルダで解決する。
# CSV 形式は togikaidrive-dev localization/path_follow.py のローダーと同じ:
#   centerline : x_m, y_m, [w_tr_right_m, w_tr_left_m, [v_mps]]
#   raceline   : s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2（7列以上）
#   （'#' 行・ヘッダ行はスキップ）
PATH_CSV_GLOBS = ("*_centerline*.csv", "*_raceline*.csv")


def load_manifest_meta(data_dir: str) -> Optional[dict]:
    """<data_dir>/manifest.json の3行目（記録メタ辞書）を返す。無ければ None。"""
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 3:
            return None
        meta = json.loads(lines[2])
        return meta if isinstance(meta, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _map_dir_candidates(data_dir: str, ref: Optional[dict]) -> list:
    """map_ref の map_dir（リポジトリ相対 "data/maps/<backend>/<name>_<TS>"）を
    学習PC上で探す候補フォルダ列。data_dir の各祖先について
    <祖先>/<map_dir> と <祖先>/maps/<backend>/<name>_<TS> を試す
    （データが data/ 直下でなく data/<任意名>/data_<TS> に置かれていても届く）。
    """
    if not ref or not ref.get("map_dir"):
        return []
    rel = ref["map_dir"].replace("\\", "/").strip("/")
    parts = rel.split("/")
    tails = [rel]
    if "maps" in parts:
        tails.append("/".join(parts[parts.index("maps"):]))   # maps/<backend>/<name>
    cands = []
    cur = os.path.abspath(data_dir)
    for _ in range(6):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
        for t in tails:
            c = os.path.normpath(os.path.join(cur, t))
            if c not in cands:
                cands.append(c)
    return cands


def list_path_csvs(map_dir: str) -> list:
    """地図フォルダ内の追従経路 CSV（centerline/raceline）をソート済みで返す。
    course_editor の制御点ファイル（*_wp.csv）とバックアップ（.bak）は除く。"""
    if not map_dir or not os.path.isdir(map_dir):
        return []
    hits = set()
    for g in PATH_CSV_GLOBS:
        for p in glob.glob(os.path.join(map_dir, g)):
            if p.endswith(".csv") and not p.endswith("_wp.csv"):
                hits.add(os.path.normpath(p))
    return sorted(hits)


def resolve_path_csv(data_dir: str, map_dir: Optional[str] = None) -> Optional[dict]:
    """走行時に追従した経路 CSV を解決して
    {csv, map_dir, candidates, source, variant} を返す（無ければ None）。

    解決順:
      1. <data_dir>/map/ 同梱スナップショット内の同名 CSV
      2. map_ref.json の map_dir を学習PC上で解決（_map_dir_candidates）した
         フォルダ内の、manifest path_csv と同名の CSV
      3. 同フォルダの <map_name>_centerline_<variant>.csv → <map_name>_raceline_<variant>.csv
         → <map_name>_centerline.csv → <map_name>_raceline.csv
    source は "snapshot" / "manifest" / "variant" のいずれか。
    candidates はそのフォルダ内で切り替え可能な経路 CSV 一覧。
    """
    data_dir = os.path.abspath(data_dir)
    meta = load_manifest_meta(data_dir) or {}
    ref = load_map_ref(data_dir) or {}
    path_csv = meta.get("path_csv")
    basename = os.path.basename(path_csv) if path_csv else None
    variant = meta.get("path_variant")
    map_name = ref.get("map_name") or meta.get("map_name") or "map"

    dirs = []
    snap = os.path.join(data_dir, "map")
    if os.path.isdir(snap):
        dirs.append((snap, "snapshot"))
    if map_dir and os.path.isdir(map_dir):
        dirs.append((os.path.abspath(map_dir), "manifest"))
    for c in _map_dir_candidates(data_dir, ref):
        if os.path.isdir(c):
            dirs.append((c, "manifest"))

    for d, src in dirs:
        cands = list_path_csvs(d)
        if not cands:
            continue
        if basename:
            for c in cands:
                if os.path.basename(c) == basename:
                    return {"csv": c, "map_dir": d, "candidates": cands,
                            "source": src, "variant": variant}
        names = []
        if variant:
            names += [f"{map_name}_centerline_{variant}.csv",
                      f"{map_name}_raceline_{variant}.csv"]
        names += [f"{map_name}_centerline.csv", f"{map_name}_raceline.csv"]
        for n in names:
            p = os.path.normpath(os.path.join(d, n))
            if p in cands:
                return {"csv": p, "map_dir": d, "candidates": cands,
                        "source": "variant", "variant": variant}
        # 名前が一致しなくても候補があれば一覧だけ返す（手動切替用）
        return {"csv": None, "map_dir": d, "candidates": cands,
                "source": src, "variant": variant}
    return None


def load_path_csv(path: str) -> dict:
    """追従経路 CSV を読み {xy: [(x,y),...], speed: [v,...]|None, kind} で返す。
    kind は "raceline"（7列以上）/ "manual"（centerline + v_mps）/ "centerline"。
    形式判定は path_follow.PurePursuitFollower._load_csv と同じ。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p for p in s.replace(";", ",").split(",") if p != ""]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue   # ヘッダ行
            if len(vals) >= 2:
                rows.append(vals)
    if not rows:
        raise ValueError(f"no numeric rows in {path}")
    ncol = max(len(r) for r in rows)
    if ncol >= 7:
        return {"xy": [(r[1], r[2]) for r in rows],
                "speed": [r[5] for r in rows], "kind": "raceline"}
    xy = [(r[0], r[1]) for r in rows]
    if ncol >= 5 and all(len(r) >= 5 for r in rows):
        return {"xy": xy, "speed": [r[4] for r in rows], "kind": "manual"}
    return {"xy": xy, "speed": None, "kind": "centerline"}


# ---- 位置領域（location_regions.json） ---------------------------------------
# 軌跡マップ上で定義した「位置クラス領域」（閉ポリゴン）の永続化。地図（マップ
# フォルダ）に紐づけて保存することで、同じコースの別セッションでも再利用できる。
# スキーマ:
#   {"version": 2, "map_yaml": "map_map.yaml",
#    "regions": [{"loc": 0, "polygon": [[x, y], ...]}, ...]}
# polygon は map 座標系 [m]（記録の pose/slam x,y と同じ座標系）の頂点列
# （3点以上・閉路は暗黙。最終点と先頭点は自動で結ばれる）。
LOCATION_REGIONS_FILENAME = "location_regions.json"


def location_regions_path(dir_path: str) -> str:
    return os.path.join(dir_path, LOCATION_REGIONS_FILENAME)


def load_location_regions(dir_path: str) -> Optional[list]:
    """<dir_path>/location_regions.json を読み、regions リストを返す。

    無い・壊れている場合は None。各領域は {"loc": int, "polygon": [(x,y),...]}
    に正規化し、形式不正（頂点3点未満を含む）のエントリは読み飛ばす。
    """
    path = location_regions_path(dir_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    regions = []
    for r in (data.get("regions") or []):
        try:
            loc = int(r["loc"])
            polygon = [(float(p[0]), float(p[1])) for p in r["polygon"]]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if len(polygon) >= 3:
            regions.append({"loc": loc, "polygon": polygon})
    return regions


def save_location_regions(dir_path: str, regions: list,
                          map_yaml: Optional[str] = None) -> str:
    """位置領域を <dir_path>/location_regions.json へ保存してパスを返す。

    座標は mm 精度（小数3桁）へ丸めてファイルを小さく保つ。
    """
    data = {
        "version": 2,
        "map_yaml": os.path.basename(map_yaml) if map_yaml else None,
        "regions": [
            {"loc": int(r["loc"]),
             "polygon": [[round(float(x), 3), round(float(y), 3)]
                         for x, y in r["polygon"]]}
            for r in regions
        ],
    }
    path = location_regions_path(dir_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return path


# ---- 読み込み（ラスタ + メタ） ----------------------------------------------
@dataclass
class MapData:
    image: "np.ndarray"        # HxW (grayscale) or HxWx3
    resolution: float          # [m/px]
    origin_x: float            # 地図左下の world 座標 [m]
    origin_y: float
    yaml_path: str

    @property
    def extent(self):
        """matplotlib imshow 用 (xmin, xmax, ymin, ymax) [m]。"""
        h, w = self.image.shape[:2]
        return (self.origin_x, self.origin_x + w * self.resolution,
                self.origin_y, self.origin_y + h * self.resolution)

    def world_to_px(self, x: float, y: float):
        """map 座標 [m]（記録の slam/x,y）→ 画像画素 (col, row)。

        row は画像座標（上原点）。origin は地図の左下セルに対応する。
        """
        col = (x - self.origin_x) / self.resolution
        row = self.image.shape[0] - 1 - (y - self.origin_y) / self.resolution
        return col, row


def _read_map_yaml(yaml_path: str) -> dict:
    """ROS map_server 形式 yaml の必要フィールドだけを読む簡易パーサ
    （image / resolution / origin。pyyaml 非依存）。"""
    out = {}
    with open(yaml_path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "image":
                out["image"] = v
            elif k == "resolution":
                out["resolution"] = float(v)
            elif k == "origin":
                nums = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", v)
                out["origin"] = [float(n) for n in nums[:3]]
    return out


def load_map(yaml_path: str) -> Optional[MapData]:
    """地図 yaml（+参照ラスタ）を読み込む。画像が読めなければ None。

    yaml の image が pgm を指しスナップショットに png しか無い等の場合は
    同名の .png / .pgm を自動で試す。
    """
    import numpy as np
    meta = _read_map_yaml(yaml_path)
    if "resolution" not in meta:
        return None
    base_dir = os.path.dirname(os.path.abspath(yaml_path))
    cands = []
    if meta.get("image"):
        p = meta["image"]
        p = p if os.path.isabs(p) else os.path.join(base_dir, p)
        cands += [p, os.path.splitext(p)[0] + ".png",
                  os.path.splitext(p)[0] + ".pgm"]
    cands += sorted(glob.glob(os.path.join(base_dir, "*_map.png")))
    img = None
    for p in cands:
        if not os.path.exists(p):
            continue
        try:
            import cv2
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        except ImportError:
            from PIL import Image
            img = np.asarray(Image.open(p))
        if img is not None:
            break
    if img is None:
        return None
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    return MapData(image=img, resolution=float(meta["resolution"]),
                   origin_x=float(origin[0]), origin_y=float(origin[1]),
                   yaml_path=os.path.abspath(yaml_path))


# ---- 確認用 CLI --------------------------------------------------------------
def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="データフォルダに紐づく地図の自動解決・読み込み確認")
    ap.add_argument("data_dir", help="data/data_<TS> フォルダ")
    ap.add_argument("--maps-root", default=None,
                    help="data/maps 相当のルート（省略時は自動推定）")
    ap.add_argument("--preview", default=None,
                    help="地図PNGプレビューの保存先（matplotlib）")
    args = ap.parse_args()

    hit = find_map_for_run(args.data_dir, args.maps_root)
    if not hit:
        print("地図なし（map_ref.json / スナップショット / 同TS地図 いずれも見つからず）")
        return
    print(f"解決: {hit['source']}  yaml={hit['map_yaml']}")
    if hit.get("ref"):
        r = hit["ref"]
        print(f"  map_ref: mode={r.get('mode')} backend={r.get('backend')} "
              f"map_dir={r.get('map_dir')}")
    m = load_map(hit["map_yaml"])
    if m is None:
        print("ラスタの読み込みに失敗")
        return
    h, w = m.image.shape[:2]
    print(f"  {w}x{h}px  resolution={m.resolution}m/px  "
          f"origin=({m.origin_x}, {m.origin_y})  extent={tuple(round(v,2) for v in m.extent)}")
    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(m.image, cmap="gray", extent=m.extent, origin="upper")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(os.path.basename(hit["map_yaml"]))
        fig.savefig(args.preview, dpi=110, bbox_inches="tight")
        print(f"  プレビュー保存: {args.preview}")


if __name__ == "__main__":
    _main()
