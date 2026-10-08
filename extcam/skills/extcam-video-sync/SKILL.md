---
name: extcam-video-sync
description: スマホや手持ちカメラ等で撮ったミニカーのコース走行動画を、togikaidrive の走行記録（自己位置）と時刻合わせする。どの走行の何時何分何秒の映像かを特定し、カメラ較正と時刻差を求めて、通過時刻と重ね描きで検証する。動画のメタデータ時刻が信用できないとき（書き出し・編集した動画）に使う。
---

# 外部カメラ動画 × 走行記録の時刻合わせ

**ゴール**: 動画の t=0 が車両 Jetson の何時何分何秒か（`offset`、UNIX 秒）と、それがどの走行か、を ±0.1 s で決める。
副産物としてカメラ較正（地図 ⇔ 画像）が得られる。

**方針**（[reference.md](reference.md) に理由と失敗例）:
1. メタデータの時刻: **iPhone のオリジナル（.MOV）の `quicktime_creationdate` は撮影開始として信用できる**（秒単位）→ `sync --around` に使う。
   共有・書き出しした mp4 の作成時刻やファイル名は書き出し時刻のことが多く、候補を絞るヒント止まり。
2. 車の軌跡だけで画像⇔地図の変換を当てはめない（ほぼ一直線の点に潰れた解が当てはまる）。
3. **先に動かない柵でカメラを較正 → 時刻差は 1 次元で探す → 通過時刻で確かめる**。
4. 自動化できない判断（カメラがコースのどちら側か、マスクに何が混ざっているか、重ね描きが合っているか）は
   **画像を見て Claude が決める**。各段の出力画像を必ず Read で見る。

## 準備

- スクリプト: このスキルの `scripts/vsync.py`（numpy・opencv・scipy のみ）。
  - Claude Code（togikaidrive リポジトリ内）: `python3 annotation_training_d2j/extcam/skills/extcam-video-sync/scripts/vsync.py …`
    または `cd annotation_training_d2j && python3 -m extcam.vsync …`
  - Claude.ai: アップロードされたスキルの `scripts/vsync.py`。cv2 が無ければ `pip install opencv-python-headless`。
- 入力:
  - 動画。**できれば iPhone のオリジナル（.MOV、HEVC・HDR のままでよい）**。撮影時刻が入っていて、解像度も高い。
    OpenCV で開けなければ（HEVC のデコーダが無い環境）ffmpeg で H.264 に変換してもらう
  - 走行: リポジトリなら `data/data_<TS>` フォルダ。Claude.ai なら利用者が手元で
    `python3 scripts/export_runs.py data/data_YYYYMMDD_* --out bundle --zip` を実行して作った `bundle.zip`
    （`*.traj.csv` と `map_bounds.json`、数百 KB）
  - コース外形 `map_bounds.json`（リポジトリなら走行の `map_ref.json` → `map_dir` にある）
- 作業フォルダ `W`（例: リポジトリなら `/tmp/vsync_<動画名>`、Claude.ai なら `/tmp/w`）

## 手順

### 1. 動画を下ごしらえ（prepare）
```
python3 vsync.py prepare VIDEO --work W
```
- 出力の `time_hints` を控える。`quicktime_creationdate` があればそれが撮影開始（秒単位、手順 6 の `--around`）。
  無ければ mp4 の作成時刻・ファイル名の時刻を見て、**走行の時間帯と合わなければ書き出し時刻とみなし、使わない**。
- 1080p で 780 フレームの下ごしらえに Jetson で約 100 s。
- `stab_fail` が多い（>10%）なら手ぶれ補正が効いていない → 画角が大きく動く区間を切って撮り直し・別区間を検討。
- `W/contact.jpg` と `W/ref.jpg` を見る: コースが写っているか、車が見えるか、手持ちか三脚か。

### 2. 柵のマスク（mask）
```
python3 vsync.py mask --work W --ignore-top 0.4 [--ignore-rect x0,y0,x1,y1 ...]
```
- `W/barrier_vis.jpg` を見る。柵（赤・白）が緑になり、天井・カーテン・机・モニターが緑になっていないこと。
  混ざったら `--ignore-top`（上から何割を捨てるか）と `--ignore-rect`（元の解像度の矩形）で除いてやり直す。
  床の白線が少し混ざるのは許容。柵の色が違うコースなら `--no-red` / `--no-white` を使う。
- **iPhone の HDR 動画**（ffmpeg / メタデータに HLG・Dolby Vision、画が白っぽく彩度が浅い）は `--hdr` を付ける。
  付けないと白がカーテンや床に広がり、赤が薄れて較正が崩れる。

### 3. カメラの側を決める（mapplot → 判断）
```
python3 vsync.py mapplot --bounds map_bounds.json --work W --runs <走行 1 本>
```
- `W/map.png`（地図、x 右・y 上、外形に頂点番号）と `W/ref.jpg` を見比べ、**カメラがコースのどちら側から撮っているか**を決める。
  手がかり: 内側の仕切りの形（U 字の閉じた側・開いた側が画像の左右どちらか）、手前に大きく写る構造物、
  奥の壁に沿う外形の辺。画像の「右」は、カメラの視線を時計回りに 90° 回した地図上の向き。
- その側の外側に、カメラ位置の探索範囲 `--cam-box x0,x1,y0,y1`（地図 m）を決める。迷ったら候補の側ごとに次の段を回し、`cost` が最小のものを採る。

### 4. 自動較正（calibrate）
```
python3 vsync.py calibrate --work W --bounds map_bounds.json --cam-box x0,x1,y0,y1
```
- 数分かかる（Jetson で約 2 分）。`W/calib_overlay.jpg` を見て、**緑の点（外形を描き戻したもの）が柵に沿っているか**を確認する。
  内側の仕切りの形まで合っていれば成功。合っていなければ `--cam-box` を見直す（`--h-min/--h-max` 高さ、`--f-min/--f-max` 焦点距離も絞れる）。
- 目安: `outline_to_barrier_px` < 6、`barrier_to_outline_px` < 16、`extent_iou` > 0.95。
- **同じ場所から撮った別の動画・同じ動画の別解像度の較正があれば `--init その calib.json`**（焦点距離は解像度比で換算される）。
  ゼロからの探索より速く、合いやすい（2026-10-08: ゼロから IoU 0.885 → init で 0.98）。

### 5. 車の検出（detect）
```
python3 vsync.py detect --work W
```
- `W/det_track.jpg`（色=時刻、青→赤）を見る。点がコース上の走行ラインに沿って並んでいればよい。
  手前の柵の上や画面の隅に点の塊があれば誤検出（手順 2 の除外か `--min-area` / `--diff-thr` で調整）。
  奥のレーンは仕切りに隠れて検出が抜けることがある（問題ない）。

### 6. 時刻差の探索（sync）
```
python3 vsync.py sync --work W --runs "data/data_20261007_*"      # または bundle フォルダ
```
- 出力の `best`（走行・`offset`・`video_t0_local`）、`margin_F_vs_2nd`、`offset_curve` を読む。
- 判定: `margin_F_vs_2nd` ≥ 0.1、`offset_curve` が 0 で最小かつ ±0.1 s で `med_px` が 1.5 倍以上に増える、`best.med_px` < 30。
  margin が小さいとき（周回の位相が似た走行が複数ある）は、候補ごとに手順 7 を回して比べる。
- 時刻の見当があれば `--around <UNIX秒> --window 30`（`quicktime_creationdate` なら ±30 s で足りる。20 走行の全探索 15 s → 3 s）。
  見当が無ければ全走行を探す。見当があっても、一度は全探索で 1 位が変わらないことを確かめると安心。

### 7. 検証（verify）
```
python3 vsync.py verify --work W --run <best の run> --offset <best の offset>
```
- `passes`: 検出が続いた区間（車が見えた通過）と、自己位置の投影がそこを通った区間の開始時刻の差 `d_start_s`。
  主な通過（長い区間）で ±0.3 s 以内なら一致。
- `W/verify_overlay.jpg`: ●=検出、×=自己位置の投影、同じ色=同じ時刻。走行ラインに沿って ● と × が同じ色で重なること。
- `ground_err_m` の p50 は位置の一致の目安（手持ち・広角・遠景で 0.2 m 前後）。真値として使えるのは三脚・近距離・歪み較正済みのとき。

## 報告

利用者には次を返す: 走行フォルダ名、動画 t=0 の Jetson 時刻（`video_t0_local` と UNIX 秒）、2 位との差、
offset 曲線の鋭さ、通過時刻の一致、位置誤差 p50、使えない点（メタデータが書き出し時刻だった等）。
`W/verify_overlay.jpg` と `W/calib_overlay.jpg` を見せる。

## 結果を extcam に取り込む（リポジトリ内）

時刻合わせが合格したら、アノテーションツールの外部カメラ機能で使えるよう走行フォルダへ書き出す:
```
cd annotation_training_d2j
python -m extcam.importer analyze VIDEO --data-root ../data [--hdr] [--cam-box …] [--init …]   # 手順 1〜7 をまとめて＋合否の目安
python -m extcam.importer write --work ../data/extcam_imports/<動画名>                           # 走行の extcam/ へ
```
GUI ではアノテーションツールの「🎥 外部カメラ」→「0. 取り込み」タブ（段ごとのボタン、各段の画像、地図でカメラの範囲、判定、書き出し）。
書き出すもの: `clip_NN.mkv`（ffmpeg が無ければ .avi）・`clip_NN.frames.csv`（t_jetson = t_cam + offset）・
`clip_NN.calib.json`（このクリップ専用の較正、基準フレームの画素）・`clip_NN.stab.npy`（手持ちのときの手ぶれ補正）。
較正・ラベル・学習・追跡の各タブは、クリップ専用の較正と手ぶれ補正を自動で使う。
**書き出し先の走行フォルダは必ず確認する**（GUI は書き出す前に表示して確認する）。
