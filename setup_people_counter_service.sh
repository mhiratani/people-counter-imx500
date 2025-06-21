#!/bin/bash

#------------ 設定ここから ------------#
CURRENT_USER=${SUDO_USER:-$(whoami)}
SERVICE_NAME="people-counter-web-rtc"
HOME_DIR="/home/${CURRENT_USER}"
PROJECT_DIR="${HOME_DIR}/people-counter-imx500"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON_SCRIPT="${PROJECT_DIR}/people_counter_stream_webRTC.py"
#------------ 設定ここまで ------------#

# 色
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# systemdサービス内容
read -r -d '' SERVICE_CONTENT <<EOL
[Unit]
Description=People Counter Web-RTC Service
After=network.target

[Service]
ExecStart=${VENV_DIR}/bin/python ${PYTHON_SCRIPT}
Restart=always
RestartSec=5
User=${CURRENT_USER}
Group=${CURRENT_USER}
Environment=PYTHONUNBUFFERED=1
WorkingDirectory=${PROJECT_DIR}

[Install]
WantedBy=multi-user.target
EOL

# 実行権限付与
chmod +x "${PYTHON_SCRIPT}"
echo -e "${GREEN}${PYTHON_SCRIPT} に実行権限付与${NC}"

# 仮想環境がなければ作成
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo -e "${GREEN}仮想環境を作成しました${NC}"
else
    echo -e "${YELLOW}仮想環境は既に存在します${NC}"
fi

# パッケージインストール（requirements.txtがあれば）
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    echo -e "${GREEN}必要なパッケージをインストール中...${NC}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r "${PROJECT_DIR}/requirements.txt"
    deactivate
fi

# サービスファイル配置
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
if [ -f "${SERVICE_FILE}" ]; then
    echo -e "${YELLOW}${SERVICE_NAME}.service は既に存在します${NC}"
    read -p "上書きしますか？ (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo bash -c "cat > '${SERVICE_FILE}'" <<< "${SERVICE_CONTENT}"
        echo -e "${GREEN}サービスファイルを更新しました${NC}"
    else
        echo -e "${YELLOW}サービスファイルは変更されませんでした${NC}"
    fi
else
    sudo bash -c "cat > '${SERVICE_FILE}'" <<< "${SERVICE_CONTENT}"
    echo -e "${GREEN}サービスファイルを作成しました${NC}"
fi

echo -e "${GREEN}セットアップが完了しました${NC}"
echo -e "1. ${YELLOW}sudo systemctl daemon-reload${NC} を実行してください"
echo -e "2. ${YELLOW}sudo systemctl enable ${SERVICE_NAME}.service${NC} で自動起動設定"
echo -e "3. ${YELLOW}sudo systemctl start ${SERVICE_NAME}.service${NC} でサービス開始"
echo -e "4. ${YELLOW}sudo systemctl status ${SERVICE_NAME}.service${NC} で状態確認"