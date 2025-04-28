import json
import os
import sys
import time
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# python-dotenv を使用して .env ファイルから環境変数を読み込む
from dotenv import load_dotenv

from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

import modules

# モデル設定
# IMX500を利用するために必要
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

# S3設定
S3_PREFIX = 'shooting_test' # S3バケット内のプレフィックス

# ======= .env ファイルから環境変数を読み込む =======
# この関数を呼び出すことで、.env ファイルの内容がos.environにロードされる
load_dotenv()
print(".env ファイルから環境変数を読み込みました。")

# ======= 設定パラメータ =======
# 設定は config.json および camera_name.json に定義しそこから読み込む
def load_config(path):
    """指定されたパスからJSON設定ファイルを読み込む"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"設定ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"設定ファイル '{path}' の形式が不正です。", file=sys.stderr)
        sys.exit(1)

# 設定ファイルの読み込み
config = load_config('config.json')
camera_name_config = load_config('camera_name.json') # camera_name.jsonという名前に変更

# 出力設定
OUTPUT_DIR = config.get('OUTPUT_DIR', 'people-count-data')
OUTPUT_PREFIX = camera_name_config.get('CAMERA_NAME', 'cameraA')

def main():
    """
    スクリプトのメイン処理:
    カメラを初期化し、AIモデル入力サイズの画像を1枚保存し、S3にアップロードして終了する。
    """
    picam2 = None
    imx500 = None
    s3_client = None
    local_image_path = None # 保存されたローカル画像のパスを保持する変数

    try:
        # 環境変数からS3設定を読み込み (load_dotenv() により既にロードされている前提)
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION')
        bucket_name = os.getenv('S3_BUCKET_NAME')

        if not all([aws_access_key_id, aws_secret_access_key, aws_region, bucket_name]):
            print("エラー: S3アップロードに必要な環境変数 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME) が設定されていません。", file=sys.stderr)
            # 環境変数が不足している場合はカメラ初期化前に終了
            sys.exit(1)

        # .envから読み込んだ環境変数を使用してS3クライアントを作成
        print("S3クライアントを作成中...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        print("S3クライアント作成完了。")

        # 出力ディレクトリの作成
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"ローカル出力ディレクトリ: {os.path.abspath(OUTPUT_DIR)}")

        # IMX500 AIカメラモジュールの初期化
        print("IMX500 AIカメラモジュールを初期化中...")
        imx500 = IMX500(MODEL_PATH)

        # AIモデルの入力サイズを取得
        intrinsics = imx500.network_intrinsics
        if not intrinsics or intrinsics.task != "object detection":
             print("警告: 指定されたモデルはオブジェクト検出タスク用ではない可能性があります。", file=sys.stderr)
             # AI入力サイズが取得できない可能性があるので確認
             ai_input_size = (320, 320) # デフォルト値またはエラー処理が必要
             print(f"AIモデル入力サイズが検出できませんでした。デフォルト値 {ai_input_size} を使用します。")
        else:
             ai_input_size = intrinsics.network_input_size
             if not ai_input_size:
                  ai_input_size = (320, 320) # 取得できない場合のフォールバック
                  print(f"AIモデル入力サイズが Intrinsics から取得できませんでした。デフォルト値 {ai_input_size} を使用します。")
             else:
                print(f"AIモデル入力サイズ: {ai_input_size}")

        # Picamera2の初期化
        print("カメラを初期化中...")
        picam2 = Picamera2(imx500.camera_num)

        # AIモデル入力サイズの画像をキャプチャするために、loresストリームを設定
        # mainストリームも必要なので、小さめに設定
        # create_preview_configuration を使用して lores ストリームのサイズを指定
        config = picam2.create_preview_configuration(
             main={"size": (640, 480)}, # mainストリームは小さめに
             lores={"size": ai_input_size, "format": "XRGB8888"} # loresストリームをAI入力サイズに設定
             # controls={"FrameRate": intrinsics.inference_rate} # 必要に応じてフレームレート設定を追加
             )

        # カメラの設定と起動
        picam2.configure(config)
        picam2.start()

        # カメラセンサーと自動設定(AWB, AE等)が落ち着くまで少し待機
        time.sleep(1)
        print(f"カメラ起動完了。AI入力サイズ ({ai_input_size}) のフレームを取得します。")

        # AI入力サイズのフレームを取得 (loresストリームからキャプチャ)
        image_array = picam2.capture_array('lores') # <-- loresストリームからキャプチャ

        # 取得した画像の情報を基に中心線を計算 (AI入力サイズ基準)
        frame_height, frame_width = image_array.shape[:2]
        center_line_x = frame_width // 2
        print(f"キャプチャフレームサイズ (AI入力サイズ): {frame_width}x{frame_height}, 中心線X座標: {center_line_x}")


        # 取得した画像をローカルに保存
        print("起動時画像をローカルに保存しています...")
        image_filename = modules.save_image_at_startup(image_array, center_line_x, OUTPUT_DIR, OUTPUT_PREFIX)
        print(f"起動時画像をローカルに保存しました: {image_filename}")

        # ローカルに保存した画像をS3にアップロード
        print(f"画像をS3バケット '{bucket_name}' にアップロードしています...")
        s3_object_key = f"{S3_PREFIX}/{image_filename}" # S3上のオブジェクトキー
        s3_client.upload_file(image_filename, bucket_name, s3_object_key)
        print(f"画像をS3にアップロードしました: s3://{bucket_name}/{s3_object_key}")

    except (NoCredentialsError, PartialCredentialsError):
        print("エラー: AWS認証情報が見つからないか不完全です。環境変数を確認してください。", file=sys.stderr)
        sys.exit(1)
    except ClientError as e:
        # Boto3関連のクライアントエラー
        print(f"S3アップロード中にエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # その他のエラー (カメラ初期化、ファイル保存など)
        print(f"処理中にエラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # リソースの解放
        print("リソースを解放しています...")
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
                print("カメラを停止し、閉じました。")
            except Exception as e:
                print(f"カメラ停止/解放エラー: {e}", file=sys.stderr)

        # ローカルに保存した画像を削除する
        if local_image_path and os.path.exists(local_image_path):
            try:
                os.remove(local_image_path)
                print(f"ローカル画像ファイルを削除しました: {local_image_path}")
            except OSError as e:
                print(f"ローカル画像ファイルの削除に失敗しました: {e}", file=sys.stderr)

        print("プログラムを終了します。")
        sys.exit(0) # 正常終了

if __name__ == "__main__":
    main()