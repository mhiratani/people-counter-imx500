import json
import os
import sys
import time
import asyncio
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
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

WEBSOCKET_URL = config.get("WEBSOCKET_URL", None)

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

# WebSocket送信用のキューと接続オブジェクト
# キューの最大サイズを設定してメモリ溢れを防ぐ (例: 10フレーム分のデータ)
data_queue = asyncio.Queue(maxsize=60)
# WebSocket接続オブジェクトを共有するための変数
# これを sender_task が参照します
ws_connection = None
# 接続管理タスクから sender_task に接続状態を通知するためのイベント（任意）
# connection_ready = asyncio.Event()

# ====== ヘルパー関数・クラス定義 ======
# Detectionクラスや parse_detections 関数は元のコードと同じとします
# 例として、検出データ構造を示す parse_detections を簡単な形で再掲します
class Detection:
    def __init__(self, coords, category, conf, metadata):
        """検出オブジェクトを作成し、バウンディングボックス、カテゴリ、信頼度を記録"""
        self.category = category
        self.conf = conf
        # imx500とpicam2インスタンスはメイン関数内で初期化され、ここで利用できるようにする必要があります
        # parse_detections関数に引数として渡すか、グローバルまたはクラス変数としてアクセス可能にする必要があります
        # 簡潔のため、ここでは仮にグローバル変数としてアクセスするとします (設計によっては改善の余地あり)
        try:
            self.box = imx500.convert_inference_coords(coords, metadata, picam2) # [x, y, w, h] 形式
        except NameError:
             print("Warning: imx500 or picam2 not initialized when creating Detection", file=sys.stderr)
             self.box = coords # fallback

# ====== 非同期タスク ======

async def websocket_manager():
    """WebSocket接続を管理し、切断されたら再接続を試みるタスク"""
    global ws_connection
    reconnect_delay = 5  # 再接続待ち時間 (秒)

    print("WebSocket接続管理タスクを開始")
    while True:
        if ws_connection is None or ws_connection.closed:
            print(f"WebSocketに接続を試行: {WEBSOCKET_URL}")
            try:
                # 非同期接続
                ws_connection = await connect(WEBSOCKET_URL)
                print("WebSocket接続成功")
                # connection_ready.set() # 接続準備完了を通知 (任意)
            except Exception as e:
                print(f"WebSocket接続失敗: {e}. {reconnect_delay}秒後に再試行...", file=sys.stderr)
                ws_connection = None # 接続失敗時は None に戻す
                # connection_ready.clear() # 接続が利用不可になったことを通知 (任意)
                await asyncio.sleep(reconnect_delay)
        else:
            # 接続が確立されている場合は、切断されるのを待つか、軽いping/pongなどを送る
            # websocketsライブラリは通常、接続が閉じられたら例外を発生させる
            try:
                # 接続が生きているか確認するためのパッシブな待機
                # 相手からの切断やエラーが発生するまでここで待機状態になることが多い
                await ws_connection.wait_closed()
                print("WebSocket接続が閉じられました。再接続を試行します。")
                ws_connection = None
                # connection_ready.clear()
            except Exception as e:
                print(f"WebSocketエラー: {e}. 接続管理を再開します。", file=sys.stderr)
                ws_connection = None
                # connection_ready.clear()
            await asyncio.sleep(1)

async def sender_task(queue: asyncio.Queue):
    """キューからデータを取り出し、WebSocketで送信するタスク"""
    print("データ送信タスクを開始")
    while True:
        # キューからデータを取り出す (データが入るまで待機)
        packet = await queue.get()
        print(f"送信したいデータ：{packet}")
        # WebSocket接続が確立されているか確認
        if ws_connection and not ws_connection.closed:
            try:
                msg = json.dumps(packet)
                # 非同期送信
                await ws_connection.send(msg)
                print(f"WebSocket送信成功: {msg[:100]}...") # 送信確認ログ (量が多いと邪魔かも)
            except Exception as e:
                print(f"WebSocket送信エラー: {e}", file=sys.stderr)
                # 送信に失敗した場合は、データを再キューイングするか破棄するか決める
                # 今回はシンプルに破棄します（キュー溢れリスクを避けるため）
                await queue.put(packet) # 再キューイングする場合
        else:
            # 接続がない場合はデータを破棄 (または再キューイング)
            print("WebSocketが未接続または閉じられているためデータを破棄")
            pass # ログは量が多いと邪魔なのでコメントアウト

        # queue.task_done() # get()したアイテムの処理が完了したことを通知 (join用, 今回は必須ではない)


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

# ====== カメラフレーム処理コールバック ======
# Picamera2 から同期的に呼び出されるため、async def にはできない
# ここでは await を使わず、キューにデータを置くだけ
def process_frame_callback(request):
    """フレームごとの処理を行うコールバック関数 (カメラライブラリから呼ばれる)"""

    try:
        with MappedArray(request, 'main') as m:
            frame_height, frame_width = m.array.shape[:2]
            center_line_x = frame_width // 2

        # メタデータを取得
        metadata = request.get_metadata()
        if metadata is None:
            return # メタデータがない場合は処理しない

        detections = parse_detections(metadata) # Detectionオブジェクトのリスト

        # DetectionオブジェクトをJSONシリアライズ可能な辞書のリストに変換
        json_serializable_detections = []
        # フレームサイズはMappedArrayから取得するか、初期設定で保持しておく
        # ここではrequestから取得する元のロジックを模倣
        try:
             with MappedArray(request, 'main') as m:
                 frame_height, frame_width = m.array.shape[:2]
                 center_line_x = frame_width // 2
        except Exception as map_e:
             print(f"MappedArrayエラー: {map_e}", file=sys.stderr)
             frame_width, frame_height = 0, 0 # fallback
             center_line_x = 0


        for d in detections:
            json_serializable_detections.append({
                "box": d.box,
                "score": float(d.conf),
                "class_id": int(d.category)
                # 他に必要な属性があればここに追加
            })

        JST = timezone(timedelta(hours=9))
        # 送信用パケットを作成
        packet = {
            "center_line_x": center_line_x,
            "camera_id": CAMERA_NAME,
            "timestamp": datetime.now(JST).isoformat(),
            "detections": json_serializable_detections
        }

        # データを非同期キューに入れる
        # キューが満杯の場合はエラーになる可能性があります (QueueFull)
        # 非同期タスクがキューから取り出す速度が遅い場合に発生します。
        # エラー処理として、ここでは警告を出力し、そのフレームのデータは破棄します。
        try:
            data_queue.put_nowait(packet)
            # print("データをキューに追加") # キュー追加確認ログ (量が多いと邪魔)
        except asyncio.QueueFull:
            print("警告: データキューが満杯です。データをスキップします。", file=sys.stderr)

    except Exception as e:
        print(f"コールバック処理エラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

# ====== メイン非同期処理 ======

async def main():
    """非同期アプリケーションのエントリーポイント"""
    global imx500, picam2, intrinsics # インスタンスをグローバルとしてアクセス可能にする（parse_detectionsのため）

    # IMX500の初期化 (元のコードと同じ)
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
        # ヘッドレス環境用の設定
        # Picamera2の設定部分は、コールバックが呼ばれるように適切に行う
        # main_config = {'size': (intrinsics.width, intrinsics.height), 'format': 'XRGB8888'} # 推論サイズに合わせる
        main_config = {'format': 'XRGB8888'}
        config = picam2.create_preview_configuration(main_config, controls={"FrameRate": intrinsics.inference_rate}, buffer_count=6)

        imx500.show_network_fw_progress_bar()

        # カメラの設定と起動
        picam2.configure(config)
        time.sleep(0.5) # 少し待機
        if getattr(intrinsics, 'preserve_aspect_ratio', False):
            imx500.set_auto_aspect_ratio()

        # ====== 非同期タスクの開始 ======

        # WebSocket接続管理タスクを開始
        ws_manager_task = asyncio.create_task(websocket_manager())

        # データ送信タスクを開始 (キューを渡す)
        sender_task_obj = asyncio.create_task(sender_task(data_queue))

        # カメラの起動
        # start() してから callback が設定されると最初のフレームを取りこぼす可能性があるので、
        # callback 設定後に start() する方が安全かもしれません。
        # ただし、Picamera2のドキュメントやサンプルに従うのがベストです。
        # 元コードでは設定後にstart()しているのでそれに従います。
        picam2.start()
        print("カメラ起動完了")
        # コールバックを設定
        picam2.pre_callback = process_frame_callback
        print("カメラコールバック設定完了。Ctrl+Cで終了します。")


        # ====== メインループ ======
        # ここでは非同期タスクが動いているので、await でタスクの完了を待つか、
        # シグナルを待つなどしてプログラムを維持します。
        # 簡単には、終了シグナルを待つか、タスクが完了するまで待機します。
        # 通常、常時実行されるプログラムなので、KeyboardInterrupt を待つ形になります。
        try:
            # ここで sender_task や ws_manager_task がキャンセルされるまで待機する
            # あるいは、単に無限ループで KeyboardInterrupt を待つ asyncio の run に任せる
            # asyncio.run(main()) が KeyboardInterrupt を捕捉してクリーンアップに進む
            await asyncio.Future() # 何もせず無限に待機し、キャンセルされるのを待つ
        except asyncio.CancelledError:
            print("メイン処理がキャンセルされました。")


    except Exception as e:
        print(f"初期化エラーまたはメイン処理エラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        # ====== リソースの解放 ======
        print("終了処理を開始...")
        # 非同期タスクをキャンセルして完了を待つ
        if 'sender_task_obj' in locals() and sender_task_obj:
            sender_task_obj.cancel()
        if 'ws_manager_task' in locals() and ws_manager_task:
             ws_manager_task.cancel()

        # タスクがキャンセルされ、完了するのを待つ（エラーは無視する）
        try:
            await asyncio.gather(sender_task_obj, ws_manager_task, return_exceptions=True)
        except Exception: # gather 自体が例外を出す場合もあるので広めに捕捉
             pass
        print("非同期タスク停止完了")

        # WebSocket接続を閉じる
        if ws_connection and not ws_connection.closed:
            print("WebSocketを閉じています...")
            try:
                await ws_connection.close()
                print("WebSocketを閉じました")
            except Exception as e:
                 print(f"WebSocketクローズエラー: {e}", file=sys.stderr)


        # カメラとIMX500モジュールを閉じる (元のコードと同じ)
        try:
            if 'picam2' in locals() and picam2:
                picam2.stop()
                picam2.close() # カメラを閉じる
                print("カメラを停止しました")
            if 'imx500' in locals() and imx500: # imx500が初期化されているか確認
                imx500.close() # AIモジュールを閉じる
                print("IMX500モジュールを閉じました")
        except Exception as e:
            print(f"カメラ/IMX500クローズエラー: {e}", file=sys.stderr)

        print("プログラムを終了します")


# ====== プログラム実行開始 ======
if __name__ == "__main__":
    # asyncioアプリケーションとして実行
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Ctrl+Cを受信しました。終了処理中...")
