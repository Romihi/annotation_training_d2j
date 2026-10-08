"""外部カメラ動画 × 走行記録の時刻合わせ（スキル extcam-video-sync の CLI をそのまま呼ぶ入口）。

    python -m extcam.vsync prepare VIDEO --work W
    python -m extcam.vsync export-runs data/data_YYYYMMDD_* --out bundle --zip
    python -m extcam.vsync package zip | install --user

手順と判断基準は extcam/skills/extcam-video-sync/SKILL.md。
"""
from __future__ import annotations

import os
import runpy
import sys

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills", "extcam-video-sync", "scripts")


def main():
    argv = sys.argv[1:]
    script = "vsync.py"
    if argv and argv[0] in ("export-runs", "package"):
        script = {"export-runs": "export_runs.py", "package": "package.py"}[argv[0]]
        argv = argv[1:]
    sys.argv = [script] + argv
    runpy.run_path(os.path.join(_SCRIPTS, script), run_name="__main__")


if __name__ == "__main__":
    main()
