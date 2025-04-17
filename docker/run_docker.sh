#!/bin/bash

# カレントスクリプトの絶対パスから、アプリのルート（docker の 1 つ上）を取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(realpath "$SCRIPT_DIR/..")"

# Docker ビルド
docker build -t annotation-tool "$APP_ROOT" -f "$SCRIPT_DIR/Dockerfile"

# X11 許可（Jetson GUI用）
xhost +local:root

# Docker 実行（アプリコード全体を共有、dataも含まれる）
docker run -it --rm \
  --runtime=nvidia \
  --shm-size=2g \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$APP_ROOT":/app \
  annotation-tool

