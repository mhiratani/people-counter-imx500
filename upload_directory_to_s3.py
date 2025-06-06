import os
import boto3
import json
import sys
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

def check_and_get_env_vars():
    """
    必要な環境変数が設定されているかチェックし、取得する関数
    """
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY', 
        'AWS_REGION',
        'S3_BUCKET_NAME'
    ]
    
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"設定ファイルの形式が不正です: {e}")

def upload_directory_to_s3(s3_client, local_directory, bucket_name, s3_prefix='', delete_after_upload=False):
    """
    ローカルディレクトリ全体をS3バケットにアップロードする関数
    アップロード成功後にファイルを削除するオプションあり（ディレクトリ構造は維持）

    Parameters:
        s3_client: S3クライアントオブジェクト
        local_directory (str): アップロードするローカルディレクトリのパス
        bucket_name (str): アップロード先のS3バケット名
        s3_prefix (str): S3内のプレフィックス（フォルダパス）
        delete_after_upload (bool): アップロード後にファイルを削除するかどうか
    Returns:
        bool: 成功した場合はTrue、失敗した場合はFalse
    """
    # ディレクトリが存在するか確認
    if not os.path.isdir(local_directory):
        print(f"ディレクトリが見つかりません: {local_directory}")
        return False
    
    success = True
    uploaded_files = []
    upload_targets = []
    
    # ディレクトリ内のすべてのファイルを走査
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            # 実際のアップロードファイルの絶対パス
            local_file_path = os.path.join(root, file)
            upload_targets.append((root, file))

    # アップロード対象がなかった場合
    if len(upload_targets) == 0:
        print(f"アップロード対象のファイルがありません: {local_directory}")
        return False

    for root, file in upload_targets:
        local_file_path = os.path.join(root, file)
        # S3パスの決定
        relative_path = os.path.relpath(local_file_path, local_directory)
        if s3_prefix:
            s3_file_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")
        else:
            s3_file_path = relative_path.replace("\\", "/")

        try:
            print(f"アップロード中: {local_file_path} -> s3://{bucket_name}/{s3_file_path}")
            # ファイルをS3にアップロード
            s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
            # アップロード成功したファイルを記録
            uploaded_files.append(local_file_path)
        except ClientError as e:
            print(f"アップロード失敗: {local_file_path}, エラー: {e}")
            success = False

    # アップロード後削除オプション
    # アップロード成功かつ削除オプションが有効な場合、ファイルのみを削除（ディレクトリ構造は維持）
    if success and delete_after_upload:
        for file_path in uploaded_files:
            try:
                os.remove(file_path)
                print(f"削除しました: {file_path}")
            except OSError as e:
                print(f"ファイル削除失敗: {file_path}, エラー: {e}")

    # サマリメッセージ
    if success:
        print(f"ディレクトリ {local_directory} のアップロードが完了しました。")
        if delete_after_upload:
            print("アップロードしたファイルを削除しました。")

    return success


if __name__ == "__main__":
    # 環境変数チェック・取得
    env_vars = check_and_get_env_vars()
    if env_vars is None:
        sys.exit(1)
    
    # S3クライアントを作成
    s3_client = create_s3_client(env_vars)
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) # カレントディレクトリの取得
    config_path = os.path.join(script_dir, "config.json")   # カレントに"config.json"がある前提
    upload_dir = load_config(config_path)['OUTPUT_DIR']    # S3にアップロードするディレクトリの特定
    
    # 環境変数から取得した値を使用
    BUCKET_NAME = env_vars['S3_BUCKET_NAME']
    DELETE_AFTER_UPLOAD = os.getenv('DELETE_AFTER_UPLOAD', 'false').lower() == 'true'
    S3_PREFIX = os.getenv('S3_PREFIX', '')  # デフォルトは空文字

    # ディレクトリをアップロード
    upload_directory_to_s3(s3_client, upload_dir, BUCKET_NAME, S3_PREFIX, DELETE_AFTER_UPLOAD)