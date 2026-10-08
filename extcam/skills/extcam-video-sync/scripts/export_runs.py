#!/usr/bin/env python3
"""走行フォルダ群を、動画の時刻合わせに要る最小限の束（Claude.ai にアップロードできる大きさ）にする。

    python3 export_runs.py data/data_20261007_* --out vsync_bundle [--map data/maps/<backend>/<map>] [--hz 20] [--zip]

出力:
  <out>/<走行名>.traj.csv   t（Jetson の UNIX 秒）, x, y, theta（自己位置、--hz に間引き）
  <out>/map_bounds.json      コース外形（走行の map_ref.json から自動で探す。--map で指定も可）
  <out>/runs.json            走行ごとの時間範囲・自己位置の種類・地図
画像・LiDAR・catalog の他の列は含めない（1 走行あたり数十 KB）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

POSE_SOURCES = ("fused", "slam", "aruco", "vslam", "pose")


def read_catalog(run):
    rows = []
    cats = sorted(glob.glob(os.path.join(run, "catalog_*.catalog")),
                  key=lambda s: int(re.findall(r"(\d+)\.catalog$", s)[0]))
    for p in cats:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def find_map_dir(run):
    ref = os.path.join(run, "map_ref.json")
    if os.path.isfile(ref):
        try:
            md = json.load(open(ref, encoding="utf-8")).get("map_dir")
        except (OSError, ValueError):
            md = None
        if md:
            root = os.path.dirname(os.path.dirname(os.path.abspath(run)))
            p = md if os.path.isabs(md) else os.path.join(root, md)
            if os.path.isfile(os.path.join(p, "map_bounds.json")):
                return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=None, help="地図フォルダ（map_bounds.json のある所）")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--zip", action="store_true", help="<out>.zip も作る")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    runs = [r for p in a.runs for r in (sorted(glob.glob(p)) or [p])
            if os.path.isdir(r) and glob.glob(os.path.join(r, "catalog_*.catalog"))]
    index, maps = [], set()
    for run in runs:
        rows = read_catalog(run)
        src = next((s for s in POSE_SOURCES if any(f"{s}/x" in r for r in rows[:200])), None)
        if src is None:
            print(f"skip {run}: 自己位置なし", file=sys.stderr)
            continue
        pts, last = [], -1e9
        for r in sorted((r for r in rows if r.get("_timestamp_ms") is not None), key=lambda r: r["_timestamp_ms"]):
            if f"{src}/x" not in r or r.get(f"{src}/status", "ok") != "ok":
                continue
            t = r["_timestamp_ms"] / 1000.0
            if t - last < 1.0 / a.hz:
                continue
            last = t
            pts.append((t, r[f"{src}/x"], r[f"{src}/y"], r[f"{src}/theta"]))
        if len(pts) < 20:
            continue
        name = os.path.basename(os.path.normpath(run))
        with open(os.path.join(a.out, name + ".traj.csv"), "w", encoding="utf-8") as f:
            f.write("t,x,y,theta\n")
            for t, x, y, th in pts:
                f.write(f"{t:.3f},{x:.4f},{y:.4f},{th:.4f}\n")
        md = a.map or find_map_dir(run)
        if md:
            maps.add(os.path.abspath(md))
        index.append({"name": name, "t_start": pts[0][0], "t_end": pts[-1][0], "n": len(pts), "pose_source": src,
                      "map_dir": md})
        print(f"{name}: {len(pts)} 点 {pts[-1][0] - pts[0][0]:.0f} s（{src}）")
    if len(maps) > 1:
        print(f"注意: 地図が複数あります {sorted(maps)}。map_bounds.json は最初のものだけ入れる", file=sys.stderr)
    if maps:
        shutil.copy(os.path.join(sorted(maps)[0], "map_bounds.json"), os.path.join(a.out, "map_bounds.json"))
    with open(os.path.join(a.out, "runs.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": index, "maps": sorted(maps)}, f, ensure_ascii=False, indent=1)
    if a.zip:
        z = shutil.make_archive(a.out.rstrip("/"), "zip", a.out)
        print(f"→ {z}（{os.path.getsize(z) / 1e6:.1f} MB）を動画と一緒に Claude.ai へアップロードする")
    return 0


if __name__ == "__main__":
    sys.exit(main())
