# Raspberry Pi での人流カウントシステム

## 概要

本システムは、GUIを準備してないRaspberry Pi とIMX500 AIカメラモジュールを使用した人流カウントシステムです。ここでは人の移動方向を検出してカウントします。cronによるスケジュール運用機能についても記載します。

## 機能

- IMX500 AIカメラモジュールによる人物検出
- 指定ラインを横切る人の方向別カウント
- JSONファイルへのカウントデータ保存
- GUIを不要で実行可能

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

### 4. アプリケーションの実行

```bash
# プログラム実行
python people_counter_non_gui.py
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