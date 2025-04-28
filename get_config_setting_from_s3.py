import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def download_config_from_s3(bucket, key, filename):
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, filename)

if __name__ == "__main__":
    BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

    # camera_name.config(JSON)を同ディレクトリから読む
    script_dir = os.path.dirname(os.path.abspath(__file__))
    camera_name_path = os.path.join(script_dir, "camera_name.json")
    camera_name = load_config(camera_name_path)['camera_name']

    # S3キー（ファイル名）をカメラ名に基づき決定
    s3_key = f"settings/{camera_name}.json"
    config_local_path = os.path.join(script_dir, "config.json")
    download_config_from_s3(BUCKET_NAME, s3_key, config_local_path)
    print(f"S3({s3_key}) → {config_local_path} にダウンロード完了")