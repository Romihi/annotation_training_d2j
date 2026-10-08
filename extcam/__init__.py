"""コース脇カメラ（sidecam）の走行動画から、車両の位置・姿勢を追跡する。

走行フォルダの extcam/（togikaidrive の runtime/ext_camera.py が取り込む clip_NN.mkv と clip_NN.frames.csv）を読み、
較正 → キーポイントのラベル付け → YOLO11n-pose の学習 → BoT-SORT による追跡 → 地図座標での平滑化 を行う。

Qt に依存しない中核（geometry / clip / runlog / labels / train / tracker）と、別ウィンドウの GUI（window）に分かれる。
中核は CLI でも動く: python -m extcam.train ... / python -m extcam.tracker ...
"""
