import os
import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
load_dotenv()

def check_and_get_env_vars(required_vars):
    """
    必要な環境変数が設定されているかチェックし、取得する関数
    """
    env_vars = {}
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            env_vars[var] = value
    
    if missing_vars:
        print("エラー: 以下の環境変数が設定されていません:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\n.envファイルまたはシステム環境変数を確認してください。")
        return None
    
    return env_vars

def validate_bucket_access(s3_client, bucket_name):
    """
    バケットアクセス権限の検証
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"エラー: バケット '{bucket_name}' が見つかりません")
        elif error_code == '403':
            print(f"エラー: バケット '{bucket_name}' へのアクセス権限がありません")
        else:
            print(f"エラー: バケットアクセスエラー - {e}")
        return False
    
def create_s3_client(env_vars):
    """
    S3クライアントを作成し、認証をテストする関数
    """
    try:
        client = boto3.client(
            's3',
            aws_access_key_id=env_vars['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=env_vars['AWS_SECRET_ACCESS_KEY'],
            region_name=env_vars['AWS_REGION']
        )
        # 認証テスト
        client.list_buckets()
        return client
        
    except NoCredentialsError:
        print("エラー: AWS認証情報が無効です")
        return None
    except ClientError as e:
        print(f"エラー: AWS接続エラー - {e}")
        return None
    
def load_config(path):
    """
    設定ファイルを読み込む
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"設定ファイルの形式が不正です: {e}")
