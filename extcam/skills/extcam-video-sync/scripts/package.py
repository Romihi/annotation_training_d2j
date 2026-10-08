#!/usr/bin/env python3
"""このスキルを Claude.ai 用の zip にする / Claude Code へ入れる。

    python3 package.py zip [--out DIR]                 # DIR/extcam-video-sync.zip（Claude.ai の Settings → Capabilities → Skills でアップロード）
    python3 package.py install --user                   # ~/.claude/skills/extcam-video-sync へコピー（どのプロジェクトでも使える）
    python3 package.py install --project PATH           # PATH/.claude/skills/extcam-video-sync へコピー

シンボリックリンクではなくコピーにするのは、Windows（リンクが無効な clone）でも同じ手順で入るようにするため。
スキルを更新したら同じコマンドで入れ直す。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = os.path.basename(SKILL_DIR)
FILES = ["SKILL.md", "reference.md", "scripts/vsync.py", "scripts/export_runs.py", "scripts/package.py"]


def build_zip(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{NAME}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in FILES:
            z.write(os.path.join(SKILL_DIR, rel), f"{NAME}/{rel}")
    return path


def install(dest_root: str) -> str:
    dest = os.path.join(dest_root, NAME)
    if os.path.islink(dest):
        print(f"{dest} はリンクなのでそのまま（リンク先が正本）")
        return dest
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    for rel in FILES:
        os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
        shutil.copy2(os.path.join(SKILL_DIR, rel), os.path.join(dest, rel))
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zip")
    z.add_argument("--out", default=os.path.join(SKILL_DIR, "dist"))
    i = sub.add_parser("install")
    g = i.add_mutually_exclusive_group(required=True)
    g.add_argument("--user", action="store_true")
    g.add_argument("--project", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "zip":
        p = build_zip(a.out)
        print(f"→ {p}（{os.path.getsize(p) / 1024:.0f} KB）を Claude.ai の Settings → Capabilities → Skills からアップロード")
    else:
        root = os.path.join(os.path.expanduser("~"), ".claude", "skills") if a.user \
            else os.path.join(os.path.abspath(a.project), ".claude", "skills")
        print(f"→ {install(root)}（Claude Code で /{NAME} または「動画と走行の時刻を合わせて」で使える）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
