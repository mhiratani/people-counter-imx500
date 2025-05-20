#!/bin/bash
set -e

# ======= 設定ここから =======
SERVICE_NAME="post-ip-report"
WORK_DIR="$(pwd)"
VENV_PATH="$WORK_DIR/venv"
PYTHON_PATH="$VENV_PATH/bin/python"
SCRIPT_PATH="$WORK_DIR/post_ip_report.py"
SYSTEMD_DIR="/etc/systemd/system"
ENV_FILE="$WORK_DIR/.env"
# ======= 設定ここまで =======

YELLOW='\033[0;33m'
NC='\033[0m'

# .envチェック
if [ ! -f "$ENV_FILE" ]; then
  echo -e "⚠️  ${YELLOW}.env ファイルが見つかりません。\nAPI_ENDPOINTなどを環境変数や.envに設定してください。${NC}"
fi

# systemdサービスファイルのみ （起動時に一発だけ）
sudo tee "${SYSTEMD_DIR}/${SERVICE_NAME}.service" > /dev/null << EOF
[Unit]
Description=Report Pi IP to API (post_ip_report.py)
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=${WORK_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${PYTHON_PATH} ${SCRIPT_PATH}
EOF

echo -e ""
echo -e "1. systemd サービスファイルの作成が完了しました。"
echo -e "2. 以下のコマンドでサービスを有効化・起動時実行してください："
echo -e "   ${YELLOW}sudo systemctl daemon-reload${NC}"
echo -e "   ${YELLOW}sudo systemctl enable ${SERVICE_NAME}.service${NC}"
echo -e "   ${YELLOW}sudo systemctl start ${SERVICE_NAME}.service${NC}"
echo -e ""
echo -e "3. ステータス確認："
echo -e "   ${YELLOW}sudo systemctl status ${SERVICE_NAME}.service${NC}"