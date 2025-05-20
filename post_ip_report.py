import subprocess
import re
import socket
import requests
import urllib3
import time
import os
import json
from dotenv import load_dotenv
load_dotenv()

urllib3.disable_warnings()

def get_wlan0_ip():
    try:
        res = subprocess.check_output(['ip', 'a', 'show', 'wlan0'], encoding='utf-8')
        match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)/\d+', res)
        if match:
            return match.group(1)
        else:
            return "未接続"
    except Exception:
        return "未接続"

def get_camera_name():
    try:
        with open('camera_name.json', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('CAMERA_NAME', None)
    except Exception as e:
        print(f"camera_name.jsonの読み込みに失敗しました: {e}")
        return None

def main():
    hostname = socket.gethostname()
    url = os.getenv('API_ENDPOINT')
    if url is None:
        print("API_ENDPOINT環境変数が設定されていません。")
        return

    camera_name = get_camera_name()
    if not camera_name:
        print("camera_nameが取得できませんでした。")
        return

    while True:
        ip = get_wlan0_ip()
        if ip == "未接続":
            print("wlan0が未接続です。30秒後に再試行します。")
            time.sleep(30)
            continue

        payload = {
            "hostname": hostname,
            "ip": ip,
            "camera_name": camera_name
        }
        try:
            response = requests.post(url, json=payload, verify=False)
            print(f"Status code: {response.status_code}")
            print(f"Response text: {response.text}")
        except Exception as e:
            print(f"POSTエラー: {e}")
        break  # 必要あればループ構造を調整

if __name__ == "__main__":
    main()