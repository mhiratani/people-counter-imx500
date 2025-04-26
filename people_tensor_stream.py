import json
import os
import sys
import time
import websocket
from datetime import datetime, timezone, timedelta

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)
from picamera2.devices.imx500.postprocess import scale_boxes


# モデル設定
# https://www.raspberrypi.com/documentation/accessories/ai-camera.html の
# "Run the following script from the repository to run YOLOv8 object detection:"を参照して選んだモデル
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

# ======= 設定パラメータ =======
def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

config = load_config('config.json')
camera_name_dict = load_config('camera_name.json')
CAMERA_NAME = camera_name_dict.get('CAMERA_NAME', 'cameraA')

WEBSOCKET_URL = config.get("websocket_url", None)

# 人流カウント設定
PERSON_CLASS_ID = 0
# -------------------------------------
# 人物クラスのID（通常COCOデータセットでは0）
# -------------------------------------

DETECTION_THRESHOLD = config.get('DETECTION_THRESHOLD', 0.55)
# ----------------------------------------------
# 検出器が出力する「検出信頼度スコア」の下限値。これ未満は無視する。
# - 値を上げると誤検出（偽陽性）は減るが、見落とし（偽陰性）が増えやすい。
# - 値を下げると検出感度は増すが、誤検出リスクが高まる。
# - 適切な値は検出器（モデル）の特性、ターゲットとなる画面のノイズの多寡による。
#   通常0.4～0.7程度を試行して決定。推奨: バリデーション動画でF1スコア最大化する値
# ----------------------------------------------

IOU_THRESHOLD = config.get('IOU_THRESHOLD', 0.3)
# ----------------------------------------------
# マッチング時、追跡対象と検出結果の「バウンディングボックスの重なり（IoU）」の下限値。
# この値より大きい場合のみ同一人物候補とする。
# - 値を大きくすると、ほぼ完全な重なりでのみマッチし、誤追跡減だが途切れやすい。
# - 値を小さくすると、多少のズレやサイズ変動も許容し、追跡の継続性は増すものの、近距離他人の誤マッチリスク増。
# - 通常0.2～0.5あたりで調整（高フレームレート＆精度が良いカメラなら大きくできる）。
#   人物サイズ/動きの激しさ/カメラの安定度で最適値が変わる。
# ----------------------------------------------

MAX_DETECTIONS = config.get('MAX_DETECTIONS', 30)
# ----------------------------------------------
# 1フレームで扱う検出結果の最大数。これ以上は間引きされるか無視される。
# - 混雑状況（同時に写る人数）や計算リソースに応じて適宜調整。
# - 多すぎると計算負荷・誤追跡リスク増、少なすぎると本来追跡すべき人を取りこぼす。
# - 現場映像の最大混雑人数よりやや余裕を持たせると安定。
# ----------------------------------------------

# ======= クラス定義 =======
class Detection:
    def __init__(self, coords, category, conf, metadata):
        """検出オブジェクトを作成し、バウンディングボックス、カテゴリ、信頼度を記録"""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2) # [x, y, w, h] 形式

# WebSocket準備
ws = None
if WEBSOCKET_URL and websocket:
    try:
        ws = websocket.WebSocket()
        ws.connect(WEBSOCKET_URL)
        print(f"WebSocket接続成功: {WEBSOCKET_URL}")
    except Exception as e:
        print(f"WebSocket接続失敗: {e}", file=sys.stderr)
        ws = None
elif WEBSOCKET_URL and not websocket:
    print("websocket-client未インストール: pip install websocket-client を実行してください", file=sys.stderr)

def parse_detections(metadata: dict):
    """
    AIモデルの出力テンソルを解析し、検出された人物のリストを返す

    :param metadata: モデル出力のメタデータ
    :param intrinsics: モデル・カメラの内部パラメータ設定
    :param picam2: カメラデバイスオブジェクト
    :return: 検出された人物のDetectionオブジェクトリスト
    """
    try:
        # intrinsicsはmain関数で初期化されるグローバル変数と仮定
        # picam2もmain関数で初期化されるグローバル変数と仮定
        global intrinsics, picam2

        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return [] # 検出がない場合は空リストを返す

        input_w, input_h = imx500.get_input_size()

        if intrinsics.postprocess == "nanodet":
            boxes, scores, classes = \
                postprocess_nanodet_detection(outputs=np_outputs[0], conf=DETECTION_THRESHOLD,
                                             iou_thres=IOU_THRESHOLD, max_out_dets=MAX_DETECTIONS)[0]
            # NanoDetの出力ボックスは[x1, y1, x2, y2]形式なので[x, y, w, h]に変換
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes] # [x1, y1, x2, y2] -> [x, y, w, h]

        else: # SSDなどの場合
            # デフォルトはSSD系と仮定 [y_min, x_min, y_max, x_max] 正規化されている可能性あり
            # imx500.convert_inference_coords がこれらの変換と正規化解除を行う
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            # boxesの形状が (N, 4) であればそのまま使用、そうでなければzip等で整形
            if boxes.shape[1] != 4:
                boxes = np.array(list(zip(*np.array_split(boxes, 4, axis=1))))

        # 信頼度と人物クラスのみをフィルタリングし、Detectionオブジェクトを作成
        detections = [
            Detection(box_coords, category, score, metadata)
            for box_coords, score, category in zip(boxes, scores, classes)
            if score > DETECTION_THRESHOLD and int(category) == PERSON_CLASS_ID
        ]

        # Detectionオブジェクト作成時に convert_inference_coords が呼ばれるため、
        # ここで取得した box は既にフレームサイズに合わせた [x, y, w, h] 形式になっているはず

        return detections
    except Exception as e:
        print(f"検出処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return [] # エラー時は空リストを返す

def process_frame_callback(request):
    """フレームごとの処理を行うコールバック関数"""

    try:
        with MappedArray(request, 'main') as m:
            frame_height, frame_width = m.array.shape[:2]
            center_line_x = frame_width // 2

        # メタデータを取得
        metadata = request.get_metadata()
        if metadata is None:
            return
        else:
            detections = parse_detections(metadata)
            # DetectionオブジェクトをJSONシリアライズ可能な辞書のリストに変換
            json_serializable_detections = []
            for d in detections:
                # dir(d) の出力に基づいて、正しい属性名を使用します
                json_serializable_detections.append({
                    "box": d.box,      # 'box' 属性を使用
                    "score": float(d.conf),   # 'conf' 属性を使用 (これがスコア)
                    "class_id": int(d.category) # 'category' 属性を使用
                    # 他に必要な属性があればここに追加
                })
            
            JST = timezone(timedelta(hours=9))
            # 送信処理
            packet = {
                "center_line_x": center_line_x,
                "camera_id": CAMERA_NAME,
                "timestamp": datetime.now(JST).isoformat(),
                "detections": json_serializable_detections
            }
            msg = json.dumps(packet)
            if ws:
                ws.send(msg)
            else:
                print(f"[DummySend] {msg}")

    except Exception as e:
        print(f"コールバックエラー: {e}")
        import traceback
        traceback.print_exc()
    return


# ======= メイン処理 =======
if __name__ == "__main__":

    # IMX500の初期化
    print("IMX500 AIカメラモジュールを初期化中...")
    try:
        imx500 = IMX500(MODEL_PATH)
        intrinsics = imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            print("ネットワークはオブジェクト検出タスクではありません", file=sys.stderr)
            sys.exit(1)

        # デフォルトラベル
        if intrinsics.labels is None:
            try:
                # コードが実行されるディレクトリからの相対パス
                label_path = os.path.join(os.path.dirname(__file__), "assets/coco_labels.txt")
                with open(label_path, "r") as f:
                    intrinsics.labels = f.read().splitlines()
            except FileNotFoundError:
                print("assets/coco_labels.txt が見つかりません。デフォルトのCOCOラベルを使用します。", file=sys.stderr)
                # COCOデータセットの一般的なラベルの一部
                intrinsics.labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        intrinsics.update_with_defaults()

        # Picamera2の初期化
        print("カメラを初期化中...")
        picam2 = Picamera2(imx500.camera_num)
        main = {'format': 'XRGB8888'} # AI検出結果の描画は行わないのでプレビューは不要だが、コールバックには必要

        # ヘッドレス環境用の設定
        config = picam2.create_preview_configuration(main, controls={"FrameRate": intrinsics.inference_rate}, buffer_count=6)

        imx500.show_network_fw_progress_bar()

        # カメラの設定と起動
        picam2.configure(config)
        time.sleep(0.5)  # 少し待機
        picam2.start()  # ヘッドレスモードでスタート
        
        if getattr(intrinsics, 'preserve_aspect_ratio', False):
            imx500.set_auto_aspect_ratio()
        print("カメラ起動完了")
    except Exception as e:
        print(f"カメラ初期化エラーまたはIMX500初期化エラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # コールバックを設定
    picam2.pre_callback = process_frame_callback
    print("Ctrl+Cで終了します")
    
    try:
        # メインループ - コールバックが処理を行うので、ここでは待機するだけ
        while True:
            time.sleep(0.01)  # CPUの負荷を減らすために短い時間待機
    except KeyboardInterrupt:
        print("終了中...")

    finally:
        # リソースの解放
        try:
            if 'picam2' in locals() and picam2: # picam2が初期化されているか確認
                picam2.stop()
                picam2.close() # カメラを閉じる
                print("カメラを停止しました")
            if 'imx500' in locals() and imx500: # imx500が初期化されているか確認
                imx500.close() # AIモジュールを閉じる
                print("IMX500モジュールを閉じました")
            if ws:
                ws.close()
                print("WebSocketを閉じました")
            print("プログラムを終了します")
        except Exception as e:
            print(f"終了処理エラー: {e}", file=sys.stderr)