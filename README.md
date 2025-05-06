# Raspberry Pi での人流カウントシステム

## 概要

本システムは、Raspberry Pi とIMX500 AIカメラモジュールを使用した人流カウントシステムです。ここでは人の移動方向を検出してカウントします。cronによるスケジュール運用機能についても記載します。

## 機能

- IMX500 AIカメラモジュールによる人物検出
- 指定ラインを横切る人の方向別カウント
- JSONファイルへのカウントデータ保存
- ヘッドレス環境でも実行可能

## 必要条件

- Raspberry Pi（3B+以上推奨）
- IMX500 AIカメラモジュール
- Raspberry Pi OS（Bullseye以降）

## セットアップ手順

### 1. システム準備

```bash
# システムアップデート
sudo apt update
sudo apt upgrade -y
```

# 必要なシステムパッケージのインストール
1. カメラが付いている側

```bash
sudo apt install -y python3-pip python3-venv git cmake build-essential \
    libatlas-base-dev libhdf5-dev libhdf5-serial-dev libjpeg-dev \
    libopenjp2-7-dev python3-picamera2 imx500-all
```

2. 解析専用・カメラを使わない

```bash
sudo apt install -y python3-pip python3-venv git cmake build-essential \
    libatlas-base-dev libhdf5-dev libhdf5-serial-dev libjpeg-dev \
    libopenjp2-7-dev
```

### 2. Python環境構築

```bash
# 仮想環境の作成と有効化
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 基本パッケージのインストール
pip install --upgrade pip wheel setuptools

# 必要なライブラリのインストール
pip install numpy opencv-python-headless
```

### 3. IMX500モデルの確認

```bash
# モデルファイルの確認
ls -la /usr/share/imx500-models/
```

### 4. 設定ファイルの準備

```bash
# 設定ファイルを作成する
vim config.json

{
  "DETECTION_THRESHOLD": 0.55,
  "IOU_THRESHOLD": 0.45,
  "MAX_DETECTIONS": 50,
  "CENTER_LINE_MARGIN_PX": 100,
  "RECOVERY_DISTANCE_PX": 150,
  "TRACKING_TIMEOUT": 0.25,
  "COUNTING_INTERVAL": 60,
  "ACTIVE_TIMEOUT_SEC": 1.5,
  "DIRECTION_MISMATCH_PENALTY": 1.0,
  "MAX_ACCEPTABLE_COST": 1.0,
  "MIN_BOX_HEIGHT": 0,
  "OUTPUT_DIR": "people-count-data",
  "DEBUG_MODE": false,
  "DEBUG_IMAGES_SUBDIR_NAME": "debug_images",
  "RTSP_SERVER_IP": "None"
}
```

```bash

# 設定メモ
# DETECTION_THRESHOLD 
# 検出器が出力する「検出信頼度スコア」の下限値。これ未満は無視する。
# - 値を上げると誤検出（偽陽性）は減るが、見落とし（偽陰性）が増えやすい。
# - 値を下げると検出感度は増すが、誤検出リスクが高まる。
# - 適切な値は検出器（モデル）の特性、ターゲットとなる画面のノイズの多寡による。
#   通常0.4～0.7程度を試行して決定。推奨: バリデーション動画でF1スコア最大化する値
# ----------------------------------------------

# IOU_THRESHOLD  
# マッチング時、追跡対象と検出結果の「バウンディングボックスの重なり（IoU）」の下限値。
# この値より大きい場合のみ同一人物候補とする。
# - 値を大きくすると、ほぼ完全な重なりでのみマッチし、誤追跡減だが途切れやすい。
# - 値を小さくすると、多少のズレやサイズ変動も許容し、追跡の継続性は増すものの、近距離他人の誤マッチリスク増。
# - 通常0.2～0.5あたりで調整（高フレームレート＆精度が良いカメラなら大きくできる）。
#   人物サイズ/動きの激しさ/カメラの安定度で最適値が変わる。
# ----------------------------------------------

# MAX_DETECTIONS 
# 1フレームで扱う検出結果の最大数。これ以上は間引きされるか無視される。
# - 混雑状況（同時に写る人数）や計算リソースに応じて適宜調整。
# - 多すぎると計算負荷・誤追跡リスク増、少なすぎると本来追跡すべき人を取りこぼす。
# - 現場映像の最大混雑人数よりやや余裕を持たせると安定。
# ----------------------------------------------

# CENTER_LINE_MARGIN_PX 
# ライン中心から±何ピクセルを「ライン近傍」とみなすかの閾値（ピクセル数）。
# - 中心からこの範囲内にいれば「ライン付近」と判定される。
# - 広すぎると誤判定が増えるが、狭すぎると本来ライン近傍の人を逃しやすくなる。
# - 実際のカメラ画角やライン検出精度、利用目的に応じて調整のこと。
# ----------------------------------------------------------

# RECOVERY_DISTANCE_PX
# 過去の人物と新しい検出の中心座標（x）の距離が 何ピクセル以内なら「同一人物が復帰した」とみなすかの閾値。
# - この距離以内であれば、lost_peopleリストから追跡を再開する。
# - 値が大きいと誤復帰（他人を繋ぐ）リスク、値が小さいと復帰し損なうリスクがある。
# - 映像解像度や1フレームあたりの人の移動量に応じて適宜調整。
# ----------------------------------------------------------

# TRACKING_TIMEOUT 人物を追跡し続ける最大時間（秒）
# COUNTING_INTERVAL カウントデータを保存する間隔（秒）
# ACTIVE_TIMEOUT_SEC  lost_people保持猶予（秒）
# DIRECTION_MISMATCH_PENALTY   逆方向へのマッチに与える追加コスト
# MAX_ACCEPTABLE_COST   最大許容コスト
# MIN_BOX_HEIGHT 人物ボックスの高さフィルタ。これより小さいBoxは排除(ピクセル)
# OUTPUT_DIR データ保存ディレクトリ
# OUTPUT_PREFIX　出力ファイル名のプレフィックス(カメラ名はcamera_name.jsonから取得)
# DEBUG_MODE デバッグモードのオン/オフ
```


- RTSPでストリーム配信したいときは、RTSPサーバのURLを記載する。
- 例 : `rtsp://192.168.10.11`

#### AWSに設定ファイルをバックアップしたい場合
```python
# カメラ名の設定
python setup.py
```

- camera で始めてほしい
- OK例 : `camera1_2024-06-01_001`
- OK例 : `cameraABC_2024-05-15_999`
- NG例 : `cam1_2024-06-01_001`
- NG例 : `camera-1_2024-06-01_001`


#### AWSに設定ファイルをバックアップする必要がない場合
```bash
# カメラ名ファイルを作成する
vim camera_name.json

{
  "CAMERA_NAME": "cameraTestA"
}
```

### 5. アプリケーションの実行

```bash
# プログラム実行(RTSP配信したい場合はRTSP_SERVER_IPを記載したら動作します)
python people_counter.py

# ラズパイ側で解析しないで、データを流してサーバ側で解析したい場合
python people_detection_stream.py

# その場合のサーバ側で実行するプログラム
python people_count_service.py
```

## 時間制限運用設定

### システムサービスの設定

1. サービスファイルの作成

```bash
sudo vim /etc/systemd/system/people-counter-non-gui.service
```

以下の内容を追加（ユーザー名とパスは環境に合わせて変更）:

```ini
[Unit]
Description=People Counter Service
After=network.target

[Service]
User=change_here_user_name
WorkingDirectory=/home/change_here_user_name/people_counter_imx500
ExecStart=/home/change_here_user_name/people_counter_imx500/venv/bin/python /home/change_here_user_name/people_counter_imx500/people_counter_non_gui.py
Restart=on-failure
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=people-counter-non-gui

[Install]
WantedBy=multi-user.target
```

2. サービスの設定

```bash
sudo systemctl daemon-reload
sudo systemctl disable people-counter-non-gui  # 自動起動を無効化
```

3. cronによるスケジュール設定

```bash
crontab -e
```

以下の例を追加（平日8:00-22:00、土曜9:00-21:00運用の場合）:

```
# 平日（月〜金）の朝8時に開始、夜10時に停止
0 8 * * 1-5 sudo systemctl start people-counter-non-gui
0 22 * * 1-5 sudo systemctl stop people-counter-non-gui

# 土曜日は朝9時から夜9時まで
0 9 * * 6 sudo systemctl start people-counter-non-gui
0 21 * * 6 sudo systemctl stop people-counter-non-gui
```

## カスタマイズ

`people_counter_non_gui.py`内の以下のパラメータを調整できます:

- `DETECTION_THRESHOLD`: 検出信頼度の閾値（デフォルト: 0.5）
- `MAX_TRACKING_DISTANCE`: 同一人物と判定する最大距離（ピクセル単位）
- `COUNTING_INTERVAL`: データ保存間隔（秒）


## トラブルシューティング

### カメラ認識の問題

```bash
# カメラデバイスの確認
ls -la /dev/video*

# IMX500ドライバの状態確認
lsmod | grep imx500
```

### モジュールエラー

```
ImportError: No module named 'picamera2.devices.imx500'
```

対処法:
```bash
# IMX500サポートの確認
dpkg -l | grep picamera2
dpkg -l | grep imx500

# 必要に応じてパッケージ再インストール
sudo apt install --reinstall python3-picamera2 imx500-all
```

### サービス起動の問題

```bash
# エラーログの確認
sudo journalctl -u people-counter-non-gui -n 50

# cronログの確認
grep CRON /var/log/syslog
```

### ライブラリパスの問題

```bash
# IMX500ライブラリの検索
find /usr -name "*imx500*.so*"

# 必要に応じてパスを設定
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/imx500/lib
```

## 動作確認

```bash
# IMX500アクセステスト
source venv/bin/activate
python test_imx500.py

# サービス状態確認
sudo systemctl status people-counter-non-gui

# サービスログの確認
journalctl -u people-counter-non-gui.service -n 20
```

## Note

### セキュリティ対策

```bash
# デフォルトパスワードの変更
passwd

# SSHアクセス制限の設定
sudo vim /etc/ssh/sshd_config
```

推奨SSH設定:
```
PermitRootLogin no
PasswordAuthentication no
```

公開鍵認証の設定:
```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
vim ~/.ssh/authorized_keys
# ここに公開鍵を貼り付け
chmod 600 ~/.ssh/authorized_keys

# SSHサービス再起動
sudo systemctl restart ssh
```

### jsonファイルのアップロード

#### S3へのアップロード処理(upload_directory_to_s3.py)をシステムサービス登録して定期的に実行する設定

0. 実行前の設定(環境変数を設定)
```text
AWS_ACCESS_KEY_ID       = XXXXXXXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY   = xyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzz
AWS_REGION              = ap-northeast-1
S3_BUCKET_NAME          = plese-change-here-bucket-name
S3_PREFIX               = this is option
DELETE_AFTER_UPLOAD     = true(アップロード後削除) or false(アップロード後削除しない)
```
   

1. サービスファイルの作成

```bash
sudo vim /etc/systemd/system/upload-directory-to-s3.service
```

以下の内容を追加（ユーザー名とパスは環境に合わせて変更）:

```ini
[Unit]
Description=Upload Directory to S3
After=network.target

[Service]
User=change_here_user_name
WorkingDirectory=/home/change_here_user_name/people_counter_imx500
ExecStart=/home/change_here_user_name/people_counter_imx500/venv/bin/python /home/change_here_user_name/people_counter_imx500/upload_directory_to_s3.py
Restart=on-failure
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=upload-directory-to-s3

[Install]
WantedBy=multi-user.target
```

2. サービスの設定

```bash
sudo systemctl daemon-reload
sudo systemctl disable upload-directory-to-s3  # 自動起動を無効化
```

3. cronによるスケジュール設定

```bash
crontab -e
```

以下の例を追加（5分おきに実行の場合）:

```
*/5 * * * * sudo systemctl start upload-directory-to-s3
```

3. サービスの設定

```bash
# サービス状態確認
sudo systemctl status upload-directory-to-s3

# サービスログの確認
journalctl -u upload-directory-to-s3.service -n 20
```