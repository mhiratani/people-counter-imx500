import os
from dotenv import load_dotenv
import psutil
import socket
import subprocess
import requests
import time

# .envファイルの読み込み
load_dotenv()

# InfluxDB設定を.envから取得
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

hostname = socket.gethostname()

def send_to_influx(lines):
    headers = {
        "Authorization": f"Token {INFLUX_TOKEN}",
        "Content-Type": "text/plain; charset=utf-8"
    }
    params = {
        "org": INFLUX_ORG,
        "bucket": INFLUX_BUCKET,
        "precision": "s"
    }
    data = "\n".join(lines)
    response = requests.post(INFLUX_URL, headers=headers, params=params, data=data)
    if not response.ok:
        print(f"InfluxDB write error: {response.text}")

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    mem = psutil.virtual_memory()
    return mem.percent, mem.used, mem.total

def get_disk_usage():
    disk = psutil.disk_usage('/')
    return disk.percent, disk.used, disk.total

def get_active_services():
    try:
        result = subprocess.run(
            ["systemctl", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        services = []
        for line in result.stdout.strip().splitlines():
            if line:
                service_name = line.split()[0]
                services.append(service_name)
        return services
    except Exception as e:
        print(f"Error getting active services: {e}")
        return []

def main():
    timestamp = int(time.time())
    cpu = get_cpu_usage()
    mem_percent, mem_used, mem_total = get_memory_usage()
    disk_percent, disk_used, disk_total = get_disk_usage()
    
    lines = []
    lines.append(
        f"system_metrics,host={hostname} cpu={cpu},memory_percent={mem_percent},memory_used={mem_used},disk_percent={disk_percent},disk_used={disk_used} {timestamp}"
    )

    for svc in get_active_services():
        lines.append(
            f"running_service,host={hostname},service={svc} running=1 {timestamp}"
        )

    send_to_influx(lines)

if __name__ == "__main__":
    main()