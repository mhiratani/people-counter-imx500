#!/bin/bash
set -e

# ======= 設定ここから =======
SERVICE_NAME="server-metrics-to-influx"
TIMER_NAME="${SERVICE_NAME}.timer"
WORK_DIR="$(pwd)"
VENV_PATH="$WORK_DIR/venv"
PYTHON_PATH="$VENV_PATH/bin/python"
SCRIPT_PATH="$WORK_DIR/server_metrics_to_influx.py"
SYSTEMD_DIR="/etc/systemd/system"
ENV_FILE="$WORK_DIR/.env"
INTERVAL="5min"   # 例: 5分ごと
# ======= 設定ここまで =======

YELLOW='\033[0;33m'
NC='\033[0m'

# .envチェック
if [ ! -f "$ENV_FILE" ]; then
  echo -e "⚠️  ${YELLOW}.env ファイルが見つかりません。\nINFLUX_URL などを環境変数や .env に設定してください。${NC}"
fi

# systemdサービスファイル作成
sudo tee "${SYSTEMD_DIR}/${SERVICE_NAME}.service" > /dev/null << EOF
[Unit]
Description=Send server metrics to InfluxDB (server_metrics_to_influx.py)
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=${WORK_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${PYTHON_PATH} ${SCRIPT_PATH}
EOF

# systemdタイマーファイル作成
sudo tee "${SYSTEMD_DIR}/${TIMER_NAME}" > /dev/null << EOF
[Unit]
Description=Periodic send server metrics to InfluxDB

[Timer]
OnBootSec=2min
OnUnitActiveSec=${INTERVAL}
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=timers.target
EOF

echo -e ""
echo -e "1. systemd サービス＆タイマー用ファイルの作成が完了しました。"
echo -e "2. 以下のコマンドで有効化・起動してください："
echo -e "   ${YELLOW}sudo systemctl daemon-reload${NC}"
echo -e "   ${YELLOW}sudo systemctl enable ${TIMER_NAME}${NC}"
echo -e "   ${YELLOW}sudo systemctl start ${TIMER_NAME}${NC}"
echo -e ""
echo -e "3. ステータス確認："
echo -e "   ${YELLOW}sudo systemctl status ${TIMER_NAME}${NC}"
echo -e "   ${YELLOW}sudo systemctl list-timers${NC}"