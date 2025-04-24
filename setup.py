import os
import json
import boto3
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

def input_camera_name():
    while True:
        name = input("カメラ名を英数字・_・-で入力してください: ").strip()
        if name and all(c.isalnum() or c in "_-" for c in name):
            return name
        print("⚠️ 無効な名前です。")

def save_camera_name(camera_name, path):
    with open(path, "w") as f:
        json.dump({"CAMERA_NAME": camera_name}, f, indent=2)
    print(f"{path} に保存しました。")

def upload_config_to_s3(local_config_path, bucket, camera_name):
    s3 = boto3.client('s3')
    s3_key = f"settings/{camera_name}.json"
    if os.path.exists(local_config_path):
        s3.upload_file(local_config_path, bucket, s3_key)
        print(f"{local_config_path} を s3://{bucket}/{s3_key} にアップロードしました")
    else:
        print("config.jsonがありません")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    camera_name_config_path = os.path.join(script_dir, "camera_name.json")
    local_config_path = os.path.join(script_dir, "config.json")
    S3_BUCKET = os.getenv('S3_BUCKET_NAME')  # .envや環境変数に登録しておく

    camera_name = input_camera_name()
    save_camera_name(camera_name, camera_name_config_path)
    if S3_BUCKET:
        upload_config_to_s3(local_config_path, S3_BUCKET, camera_name)
    else:
        print("S3_BUCKET_NAME環境変数が設定されていません。")