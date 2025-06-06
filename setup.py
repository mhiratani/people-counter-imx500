import os
import json
import sys
import aws_access_tools # 自作関数群

from botocore.exceptions import ClientError
from pathlib import Path
from datetime import datetime

def input_camera_name():
    """
    カメラ名の入力と検証
    """
    import re
    pattern = re.compile(r'^camera[a-zA-Z0-9_]+$')
    
    print("=" * 50)
    print("カメラ名設定")
    print("=" * 50)
    print("ルール:")
    print("- 'camera'で始まる")
    print("- その後に英数字または_を1文字以上")
    print("- 例: cameraA1, camera_01, cameraMain")
    print("-" * 50)
    
    while True:
        try:
            name = input("カメラ名を入力してください: ").strip()
            
            if not name:
                print("カメラ名を入力してください。")
                continue
                
            if pattern.match(name):
                # 確認
                confirm = input(f"カメラ名 '{name}' で確定しますか？ (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    print(f"カメラ名 '{name}' が設定されました")
                    return name
                else:
                    print("再入力してください。")
                    continue
            else:
                print("無効な形式です。'camera'で始まり、その後に英数字または_を続けてください。")
                
        except KeyboardInterrupt:
            print("\n\n処理が中断されました。")
            sys.exit(1)
        except Exception as e:
            print(f"エラー: 入力エラー - {e}")

def save_camera_name(camera_name, path):
    """
    カメラ名をJSONファイルに保存
    """
    try:
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        camera_config = {
            "CAMERA_NAME": camera_name,
            "created_at": str(datetime.now()),
            "version": "1.0"
        }
        
        with open(path, "w", encoding='utf-8') as f:
            json.dump(camera_config, f, indent=2, ensure_ascii=False)
        
        print(f"カメラ名設定を {path} に保存しました")
        return True
        
    except Exception as e:
        print(f"エラー: ファイル保存エラー - {e}")
        return False

def upload_config_to_s3(s3_client, local_config_path, bucket, camera_name):
    """
    設定ファイルをS3にアップロード
    """
    if not os.path.exists(local_config_path):
        print(f"エラー: アップロード対象ファイルが見つかりません - {local_config_path}")
        return False
    
    s3_key = f"settings/{camera_name}.json"
    
    try:
        print(f"S3アップロード中: {local_config_path} -> s3://{bucket}/{s3_key}")
        s3_client.upload_file(local_config_path, bucket, s3_key)
        
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            print(f"アップロード成功: s3://{bucket}/{s3_key}")
            return True
        except ClientError:
            print("アップロード後の確認に失敗しました")
            return False
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"エラー: S3アップロード失敗 - {e}")
        if error_code == 'AccessDenied':
            print("エラー: アップロード権限がありません")
        return False
        
    except Exception as e:
        print(f"エラー: 予期しないエラー - {e}")
        return False

if __name__ == "__main__":
    # 参照したい環境変数を指定
    required_vars = [
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY', 
    'AWS_REGION',
    'S3_BUCKET_NAME'
    ]

    # 環境変数チェック・取得
    env_vars = aws_access_tools.check_and_get_env_vars(required_vars)
    if env_vars is None:
        sys.exit(1)
    
    # S3クライアントを作成
    s3_client = aws_access_tools.create_s3_client(env_vars)
    # バケットアクセス検証
    bucket_name = env_vars['S3_BUCKET_NAME']
    if not aws_access_tools.validate_bucket_access(s3_client, bucket_name):
        sys.exit(1)
    
    # パス設定
    script_dir = Path(__file__).parent
    camera_name_config_path = script_dir / "camera_name.json"
    local_config_path = script_dir / "config.json"
    
    try:
        # カメラ名入力
        camera_name = input_camera_name()
        
        # カメラ名保存
        if not save_camera_name(camera_name, camera_name_config_path):
            print("エラー: カメラ名の保存に失敗しました")
            sys.exit(1)
        
        # 設定ファイル確認
        config = aws_access_tools.load_config(local_config_path)
        if config is None:
            print("エラー: 設定ファイルの読み込みに失敗しました")
            sys.exit(1)
        
        # S3アップロード
        if upload_config_to_s3(s3_client, local_config_path, bucket_name, camera_name):
            print("処理が正常に完了しました")
        else:
            print("エラー: S3アップロードに失敗しました")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n処理が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 予期しないエラーが発生しました - {e}")
        sys.exit(1)