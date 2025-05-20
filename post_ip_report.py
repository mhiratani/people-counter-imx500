import subprocess
import re
import socket
import requests
import urllib3
import time
import os
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

def main():
    hostname = socket.gethostname()
    url = os.getenv('API_ENDPOINT')
    if url is None:
        print("API_ENDPOINT環境変数が設定されていません。")
        return

    while True:
        ip = get_wlan0_ip()
        if ip == "未接続":
            print("wlan0が未接続です。30秒後に再試行します。")
            time.sleep(30)
            continue

        payload = {
            "hostname": hostname,
            "ip": ip
        }
        try:
            response = requests.post(url, json=payload, verify=False)
            print(f"Status code: {response.status_code}")
            print(f"Response text: {response.text}")
        except Exception as e:
            print(f"POSTエラー: {e}")
        break  # 成功またはエラー後に終了、必要あればここを調整

if __name__ == "__main__":
    main()