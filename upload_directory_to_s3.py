import os
import boto3
from botocore.exceptions import ClientError
import logging
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

def upload_directory_to_s3(local_directory, bucket_name, s3_prefix='', delete_after_upload=False):
    """
    ローカルディレクトリ全体をS3バケットにアップロードする関数
    アップロード成功後にファイルを削除するオプションあり（ディレクトリ構造は維持）
    
    Parameters:
    local_directory (str): アップロードするローカルディレクトリのパス
    bucket_name (str): アップロード先のS3バケット名
    s3_prefix (str): S3内のプレフィックス（フォルダパス）
    delete_after_upload (bool): アップロード後にファイルを削除するかどうか
    
    Returns:
    bool: 成功した場合はTrue、失敗した場合はFalse
    """
    # .envから読み込んだ環境変数を使用してS3クライアントを作成
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )
    
    # ディレクトリが存在するか確認
    if not os.path.isdir(local_directory):
        logging.error(f"ディレクトリが見つかりません: {local_directory}")
        return False
    
    success = True
    uploaded_files = []
    
    # ディレクトリ内のすべてのファイルを走査
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            # ローカルファイルの完全パス
            local_file_path = os.path.join(root, file)
            
            # S3内のパスを計算（相対パスを維持）
            relative_path = os.path.relpath(local_file_path, local_directory)
            s3_file_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            
            try:
                print(f"アップロード中: {local_file_path} -> s3://{bucket_name}/{s3_file_path}")
                # ファイルをS3にアップロード
                s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
                # アップロード成功したファイルを記録
                uploaded_files.append(local_file_path)
            except ClientError as e:
                logging.error(f"アップロード失敗: {local_file_path}, エラー: {e}")
                success = False
    
    # アップロード成功かつ削除オプションが有効な場合、ファイルのみを削除（ディレクトリ構造は維持）
    if success and delete_after_upload:
        for file_path in uploaded_files:
            try:
                os.remove(file_path)
                print(f"削除しました: {file_path}")
            except OSError as e:
                logging.error(f"ファイル削除失敗: {file_path}, エラー: {e}")
                # ファイル削除の失敗は全体の成功判定には影響させない
    
    if success:
        print(f"ディレクトリ {local_directory} のアップロードが完了。")
        if delete_after_upload:
            print(f"アップロードしたファイルを削除")
    
    return success

if __name__ == "__main__":
    # .envから設定を読み込む）
    LOCAL_DIRECTORY = os.getenv('LOCAL_DIRECTORY')  # アップロードするローカルディレクトリのパス
    BUCKET_NAME = os.getenv('S3_BUCKET_NAME')       # S3バケット名
    S3_PREFIX = os.getenv('S3_PREFIX','')           # プレフィックスが指定されていない場合は空文字列を使用
    DELETE_AFTER_UPLOAD = os.getenv('DELETE_AFTER_UPLOAD', 'false').lower() == 'true'  # アップロード後に削除するかどうか
    
    # ディレクトリをアップロード
    upload_directory_to_s3(LOCAL_DIRECTORY, BUCKET_NAME, S3_PREFIX, DELETE_AFTER_UPLOAD)