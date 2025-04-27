#!/bin/bash
set -e

# ======= 設定ここから =======
SERVICE_NAME="upload-directory-to-s3"
TIMER_INTERVAL="*:0/5"   # ← タイマー間隔
WORK_DIR="$(pwd)"
VENV_PATH="$WORK_DIR/venv"
PYTHON_PATH="$VENV_PATH/bin/python"
SCRIPT_PATH="$WORK_DIR/upload_directory_to_s3.py"
SYSTEMD_DIR="/etc/systemd/system"
ENV_FILE="$WORK_DIR/.env"
# ======= 設定ここまで =======

YELLOW='\033[0;33m'
NC='\033[0m'

# .envチェック
if [ ! -f "$ENV_FILE" ]; then
  echo -e "⚠️  ${YELLOW}.env ファイルが見つかりません。\nAWS認証情報（AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_REGION）を環境変数や.envに設定してください。${NC}"
fi

# systemdサービスファイル
sudo tee "${SYSTEMD_DIR}/${SERVICE_NAME}.service" > /dev/null << EOF
[Unit]
Description=Upload Directory to S3

[Service]
Type=oneshot
WorkingDirectory=${WORK_DIR}
ExecStart=${PYTHON_PATH} ${SCRIPT_PATH}
EOF

# systemdタイマーファイル
sudo tee "${SYSTEMD_DIR}/${SERVICE_NAME}.timer" > /dev/null << EOF
[Unit]
Description=Timer for uploading directory to S3

[Timer]
OnCalendar=${TIMER_INTERVAL}
Persistent=true

[Install]
WantedBy=timers.target
EOF

echo -e ""
echo -e "1. systemd サービス・タイマーファイルの作成が完了しました。"
echo -e "2. 以下のコマンドでサービスを有効化・開始してください："
echo -e "   ${YELLOW}sudo systemctl daemon-reload${NC}"
echo -e "   ${YELLOW}sudo systemctl enable ${SERVICE_NAME}.timer${NC}"
echo -e "   ${YELLOW}sudo systemctl start ${SERVICE_NAME}.timer${NC}"
echo -e ""
echo -e "3. ステータス確認："
echo -e "   ${YELLOW}sudo systemctl status ${SERVICE_NAME}.timer${NC}"