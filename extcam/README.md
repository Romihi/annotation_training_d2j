# extcam — 外部カメラ動画で車両の位置・姿勢を追跡する

コース脇の据え置きカメラ（togikaidrive の `sidecam/`）が撮った走行動画を、走行フォルダの `extcam/` から読み、
**較正 → ラベル付け → YOLO11n-pose の学習 → BoT-SORT で追跡 → 地図座標で平滑化** を行う別ウィンドウ。

起動: アノテーションツールのツールバー「🎥 外部カメラ」、または単体で `python -m extcam.window data/data_<TS>`。

状態（2026-10-08）: 合成データ（実コースの外形上を走る車を既知のカメラで描画）で、較正・仮ラベル・学習・追跡まで通しで確認。
**実カメラ・実走行の動画では未検証。**

## 入力（走行フォルダ）

| ファイル | 中身 |
|---|---|
| `extcam/clip_NN.mkv` `clip_NN.frames.csv` | sidecam の切り出し動画と、フレームごとの時刻（`t_jetson` は catalog の `_timestamp_ms` と同じ時計） |
| `map_ref.json` → `map_dir` | course_editor の地図（`*.yaml` + 画像）とコース外形 `map_bounds.json`（outer / holes）。較正の対応点に使う |
| `catalog_*.catalog` `images/` | 車両の自己位置（fused > slam > aruco > vslam > pose の順で最初にあるもの）と車載画像。仮ラベルと比較に使う |
| リポジトリ直下の `vehicle.json` | 車体寸法（キーポイントの既定値） |

## 出力（走行フォルダの `extcam/`）

| ファイル | 中身 |
|---|---|
| `calib.json` | カメラ（K・歪み・R・t・再投影誤差）、車両キーポイントの寸法、対応点、地図フォルダ |
| `frames/clip_NN/*.jpg` | ラベル付け用に書き出したフレーム |
| `labels/clip_NN.json` | 箱＋4 点（前左・前右・後左・後右、vis 0/1/2）。`confirmed` が人の確認済み |
| `track_NN.raw.csv` | 検出ごと（ID・conf・箱・点・地図座標の生の姿勢） |
| `track_NN.csv` | 平滑化後（base_link＝後軸中心の x, y, yaw, v, yaw_rate, 外れ値フラグ） |
| `track_NN.edits.json` | 手直し（除外 ID・ID の統合・除外フレーム） |
| `track_NN.summary.json` | ID ごとの件数と、自己位置との差（位置 RMSE・前後／横の偏り・向き誤差） |

## 手順

1. **較正**: 画像で床の点（コース外形の角）→ 地図で同じ点（頂点に吸着）を 6 組以上、画面全体に散らして取る。
   焦点距離は床の平面から推定する（`内部パラメータを読み込む` でチェッカーボード較正の K・dist を固定にもできる）。
   緑の線（外形を画像へ投影したもの）が床の白線に重なれば OK。「1 px の距離」で遠い側の粗さを確認する。
   キーポイントはルーフ（高さ h の平面）にあるので、**h と 4 隅の位置を実車で測って入れる**（姿勢の精度に直結）。
2. **ラベル**: 「候補を作る」で間隔おきにフレームを書き出す → 「自己位置から仮ラベル」で自己位置を投影した箱と点を付ける →
   1 枚ずつ直して Enter（確認済みにして次へ）。車がいなければ N。
   2 回目以降は学習済みモデルで仮ラベル → 直す、を繰り返す。学習に使うのは確認済みだけ（オプションで未確認も）。
3. **学習**: `python -m extcam.train` を別プロセスで実行（yolo11n-pose、既定 imgsz 1280）。走行フォルダは複数まとめられる。
4. **追跡**: `python -m extcam.tracker run` を別プロセスで実行。BoT-SORT は ID 付けだけ（固定カメラなので GMC なし、ReID なし）。
   ルーフの点を高さ h の平面へ逆投影 → 4 点の剛体当てはめで後軸の位置と向き → ID ごとに等速モデルのカルマンフィルタと
   前後両方向の平滑化（RTS、4σ を超える観測は外れ値として捨てる、0.5 s 以上の空白で区切る）。
   表で ID の除外・統合を直して「平滑化し直す」（推論はやり直さない）。

## 保存と再開

- ラベルは編集のたびに 0.8 秒待って `labels/clip_NN.json` へ自動保存する（フレームの移動・Enter・N・クリップ切替・閉じるときにも書く）。
  ツールが落ちても失うのは最後の 0.8 秒の操作だけ。手で触ったフレームは自動で確認済みになる。
- 開き直すと、前回最後に開いていたフレーム（`last_frame`）へ戻る。記録が無ければ最初の未確認フレーム。
- 較正の対応点の追加・削除、「較正する」、車両寸法の変更は、そのたびに `calib.json` へ自動保存する（「保存」ボタンは確認用）。

## 他のカメラの動画を取り込む（「0. 取り込み」タブ / extcam.importer）

スマホ等で撮った動画（手持ち可）を、走行と時刻合わせしてその走行の `extcam/` へ書き出す。sidecam の動画と同じように
較正・ラベル・学習・追跡で使える（クリップ専用の較正 `clip_NN.calib.json` と手ぶれ補正 `clip_NN.stab.npy` を自動で使う）。

1. 「0. 取り込み」で動画とデータ（data/）を選ぶ → ① 下ごしらえ（撮影時刻・手ぶれ補正）
2. ② 柵のマスク（天井・机が混ざれば「上を除く割合」「除く矩形」、iPhone の HDR は「HDR 動画」）
3. ③ 地図 → 右の「地図」タブでカメラのいる側を 2 クリック → ④ 較正（同じ場所の前回の calib.json があれば「較正の初期値」）
4. ⑤ 車の検出 → ⑥ 時刻探索 → ⑦ 検証。「判定」タブで合否の目安を確認
5. ⑧ 走行フォルダへ書き出す（書き出し先を確認するダイアログが出る）→ そのまま較正タブで開く

CLI: `python -m extcam.importer analyze VIDEO --data-root ../data …` と `python -m extcam.importer write --work …`。
作業フォルダは既定で `<data>/extcam_imports/<動画名>/`（途中から再開できる。「作業フォルダ」で前回のものを選ぶ）。
ffmpeg が無いと MJPG の .avi になり容量が数倍（`EXTCAM_FFMPEG` か PATH の ffmpeg を使う）。

検証（2026-10-08）: iPhone 12 mini の手持ち 26 s（HEVC・HDR、IMG_3320.MOV）→ data_20261007_214644 へ取り込み、
手ぶれ 94 px のフレームでも自己位置の仮ラベルと外形の重ね描きが車・柵に乗ることを確認（実際の走行フォルダへの書き出しは未実施）。

## 他のカメラの動画を走行と時刻合わせする（スキル extcam-video-sync）

スマホ等で撮った動画（メタデータの時刻が書き出し時刻で使えないもの）を、どの走行の何時何分何秒かに合わせる。
手順・判断基準は [skills/extcam-video-sync/SKILL.md](skills/extcam-video-sync/SKILL.md)、失敗例は同じフォルダの reference.md。

```bash
python -m extcam.vsync prepare VIDEO --work W          # 以下 mask → mapplot → calibrate --cam-box … → detect → sync → verify
python -m extcam.vsync export-runs data/data_YYYYMMDD_* --out bundle --zip   # Claude.ai に渡す軌跡の束（数百 KB）
python -m extcam.vsync package zip                     # Claude.ai 用のスキル zip（skills/extcam-video-sync/dist/）
python -m extcam.vsync package install --user          # Claude Code（~/.claude/skills）へ入れる
```

検証（2026-10-08）: iPhone 手持ち 26 s の動画 → data_20261007_214644 の 21:48:34.34 から（2 位との差 0.13、±0.1 s で誤差 3 倍、
通過時刻 ±0.2 s で一致、位置 p50 0.18 m）。カメラの側（`--cam-box`）は画像と地図を見て決める必要がある（全自動では誤った解に落ちた）。

## 合成データでの確認（2026-10-08、Jetson Orin）

- 較正: 外形の頂点 10 組（0.7 px のノイズ）から焦点距離 761（真値 760）、カメラ位置の誤差 5 mm、再投影 0.51 px。
- 自己位置（σ 2 cm＋横 3 cm の偏り）からの仮ラベル: 点のずれ 中央値 2.4 px。
- 学習: Jetson（メモリ 7.6 GB）では imgsz 1280 は batch 4・workers 0 でないとメモリ不足で落ちる。PC の GPU なら既定の batch 8 でよい。
- 学習: 確認済み 149 枚（自己位置の仮ラベルをそのまま確認したもの）、yolo11n-pose・imgsz 1280・計 47 エポック → pose mAP50 0.995。
- 追跡（20 s・1200 フレーム、Jetson で 23 fps）: ID は 1 本で途切れなし。真値に対して
  位置 RMSE 2.9 cm（生 3.5 cm）・p95 5.8 cm、横の偏り +1.9 cm（仮ラベルの元にした自己位置の偏り 3 cm を学習が引き継いだ分）、
  向き p50 4.9°・p95 17°、速度の中央値 2.50 m/s（真値 2.5）。向きは車が小さく写る（約 60 px）と点 1 px のずれで数度動くので、
  実機では**仮ラベルを手で直す**ことと、ルーフの 4 隅を目立たせる（色テープ）ことが効く見込み。

## CLI

```bash
python -m extcam.tracker prelabel data/data_<TS> --clip 0 --from pose --step 30
python -m extcam.train --runs data/data_A data/data_B --out models/extcam/run1 --epochs 100
python -m extcam.tracker run data/data_<TS> --clip 0 --model models/extcam/run1/train/weights/best.pt
python -m extcam.tracker smooth data/data_<TS> --clip 0      # edits を反映して平滑化だけ
python -m pytest dev/test_extcam.py -q -p no:anyio -p no:cacheprovider
```

## 注意

- 投影は車体の傾き（ロール・ピッチ）を無視する。旋回中の大きなロールでは、高い位置の点ほど横にずれる。
- 動画は concat で時刻が振り直された可変フレームレートなので、フレーム番号は先頭から順に読んで数える
  （OpenCV のシークは使わない）。後ろへ戻る操作は開き直しになり、長いクリップでは数秒かかる。
- ultralytics の追跡は、そのフレームで検出された車だけを出す。見えなかったフレームは track_NN.csv に行が無い。
