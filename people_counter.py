import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment    # scipyの線形割当アルゴリズム

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)
from picamera2.devices.imx500.postprocess import scale_boxes


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
mobilenet = MobileNetV2(include_top=False, pooling='avg', input_shape=(96,96,3), weights='imagenet')

import modules

# ffmpegによるRTSP配信プロセス用
import queue
import threading
import subprocess
frame_queue = queue.Queue(maxsize=3)

# 描画設定
import cv2
LINE_COLOR = (0, 255, 0) # Green (BGR format for OpenCV)
BOX_COLOR = (0, 0, 255)  # Red
TRAJECTORY_COLOR = (255, 0, 0) # Blue
TEXT_COLOR = (0, 0, 255) # Red
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 2


# モデル設定
# https://www.raspberrypi.com/documentation/accessories/ai-camera.html の
# "Run the following script from the repository to run YOLOv8 object detection:"を参照して選んだモデル
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

# ======= 設定パラメータ ======= 
# 設定は config.json に定義してそこから読み込む
def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

config = load_config('config.json')
camera_name = load_config('camera_name.json')

# 人流カウント設定
PERSON_CLASS_ID = 0
# 人物クラスのID（通常COCOデータセットでは0）

MAX_TRACKING_DISTANCE = config.get('MAX_TRACKING_DISTANCE', 60)
# ----------------------------------------------
# 追跡対象と新しい検出結果の「中心点間距離」の最大許容値（ピクセル単位）。
# この値以下ならマッチング候補と見なす。
# - 値を大きくすると、急な移動や検出ずれにも追従しやすくなるが、近くの他人を誤ってマッチさせやすくなる。
# - 値を小さくすると誤追跡は減るが、カメラ揺れ・一時ロスト・急な移動で追跡が切れやすくなる。
# - 映像解像度、人物サイズ、フレームレート、移動速度に応じて現場で要チューニング。
#   目安: 検出ボックスの幅の半分～1倍程度や、1フレームで起きうる最大移動距離
# ----------------------------------------------

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

TRACKING_TIMEOUT = config.get('TRACKING_TIMEOUT', 5.0)      # 人物を追跡し続ける最大時間（秒）
COUNTING_INTERVAL = config.get('COUNTING_INTERVAL', 60)     # カウントデータを保存する間隔（秒）

# 出力設定
OUTPUT_DIR = config.get('OUTPUT_DIR', 'people_count_data')  # データ保存ディレクトリ
OUTPUT_PREFIX = camera_name.get('CAMERA_NAME', 'cameraA')   # 出力ファイル名のプレフィックス(カメラ名はcamera_name.jsonから取得)
DATE_DIR = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y-%m-%d"))

DEBUG_MODE = str(config.get('DEBUG_MODE', 'False')).lower() == 'true'   # デバッグモードのオン/オフ
DEBUG_IMAGES_SUBDIR_NAME = config.get('DEBUG_IMAGES_SUBDIR_NAME', 'debug_images')
                                                                        # デバッグディレクトリの名前
# デバッグディレクトリを出力ディレクトリの配下に定義ディレクトリの名前
DEBUG_IMAGES_DIR = os.path.join(OUTPUT_DIR, DEBUG_IMAGES_SUBDIR_NAME)

# ログ設定
LOG_INTERVAL = 5  # ログ出力間隔（秒）

# グローバル変数
last_log_time = 0

# RTSP配信先URL
RTSP_SERVER_IP = config.get('RTSP_SERVER_IP','None')
RTSP_SERVER_PORT = 8554

# ============ RTSPスレッドセットアップ ============
def rtsp_writer_thread(ffmpeg_proc):
    while True:
        frame = frame_queue.get()
        try:
            ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())
        except Exception as e:
            print(f"[RTSP配信エラー]: {e}")
        finally:
            frame_queue.task_done()

def start_rtsp_thread(ffmpeg_proc):
    t = threading.Thread(target=rtsp_writer_thread, args=(ffmpeg_proc,), daemon=True)
    t.start()
    return t

def send_frame_for_rtsp(frame):
    try:
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put_nowait(frame)
    except queue.Full:
        pass  # 複数同時発生時もスルー

def extract_appearance_feature(image, box):
    """box=(x,y,w,h)の部分を96x96で切り抜き特徴ベクトルに変換"""
    x, y, w, h = map(int, box)
    crop = image[max(0,y):max(0,y+h), max(0,x):max(0,x+w)]
    if crop.size == 0:
        return np.zeros((1280,), dtype=np.float32)
    crop = cv2.resize(crop, (96,96))

    if crop.shape[2] == 4:       # 4ch(XRGB/BGRAなど)→3ch(RGB)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
    elif crop.shape[2] == 3:     # 3chの場合はBGR→RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    crop = preprocess_input(crop.astype(np.float32))
    feature = mobilenet.predict(crop[None], verbose=0)[0]
    # L2正規化して比較しやすく
    feature = feature / (np.linalg.norm(feature) + 1e-9)
    return feature

def appearance_distance(f1, f2):
    """コサイン類似度"""
    if f1 is None or f2 is None:
        return 1.0  # 極端に違う扱い
    return 1.0 - np.d

# ============ コールバック関数の属性を初期化 ============
def init_process_frame_callback():
    process_frame_callback.image_saved = False
    # グローバル変数をここで初期化 (mainでも行うが、コールバックが先に呼ばれる可能性も考慮)
    global active_people, counter
    active_people = []
    # counterはmainで初期化されるはずだが、念のためNoneチェック
    if counter is None:
        counter = PeopleCounter(time.time(), OUTPUT_DIR, OUTPUT_PREFIX, DATE_DIR, DEBUG_IMAGES_DIR)


# ======= クラス定義 =======
class Detection:
    def __init__(self, coords, category, conf, metadata):
        """検出オブジェクトを作成し、バウンディングボックス、カテゴリ、信頼度を記録"""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2) # [x, y, w, h] 形式


class Person:
    next_id = 0

    def __init__(self, box, appearance=None):
        self.id = Person.next_id
        Person.next_id += 1
        self.box = box # [x, y, w, h] 形式
        self.trajectory = [self.get_center()]
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.crossed_direction = None
        self.appearance = appearance

    def get_center(self):
        """バウンディングボックスの中心座標を取得"""
        x, y, w, h = self.box
        return (x + w//2, y + h//2)

    def update(self, box, appearance=None):
        """新しい検出結果で人物の情報を更新"""
        self.box = box # [x, y, w, h] 形式
        self.trajectory.append(self.get_center())
        if len(self.trajectory) > 30:  # 軌跡は最大30ポイントまで保持
            self.trajectory.pop(0)
        self.last_seen = time.time()
        if appearance is not None:
            self.appearance = appearance

class PeopleCounter:
    def __init__(self, start_time, output_dir=OUTPUT_DIR, output_prefix=OUTPUT_PREFIX, date_dir=DATE_DIR, debug_images_dir=DEBUG_IMAGES_DIR):
        self.right_to_left = 0  # 右から左へ移動（期間カウント）
        self.left_to_right = 0  # 左から右へ移動（期間カウント）
        self.total_right_to_left = 0  # 累積カウント
        self.total_left_to_right = 0  # 累積カウント
        self.start_time = start_time
        self.last_save_time = start_time
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.date_dir = date_dir
        self.debug_images_dir = debug_images_dir

    def update(self, direction):
        """方向に基づいてカウンターを更新"""
        if direction == "right_to_left":
            self.right_to_left += 1
            self.total_right_to_left += 1
        elif direction == "left_to_right":
            self.left_to_right += 1
            self.total_left_to_right += 1

    def get_counts(self):
        """現在のカウント状況を取得"""
        return {
            "right_to_left": self.right_to_left,
            "left_to_right": self.left_to_right,
        }

    def get_total_counts(self):
        """累積カウント状況を取得"""
        return {
            "right_to_left": self.total_right_to_left,
            "left_to_right": self.total_left_to_right,
        }

    def save_to_json(self):
        """カウントデータをJSONファイルに保存"""
        current_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        data = {
            "timestamp": timestamp,
            "duration_seconds": int(current_time - self.last_save_time),
            "period_counts": self.get_counts(),
            "total_counts": self.get_total_counts()
        }

        # ファイルパスを正しく構築
        filename = os.path.join(self.date_dir, f"{self.output_prefix}_{timestamp}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)

            print(f"Data saved to {filename}")

            # 期間カウンターのみリセット
            self.right_to_left = 0
            self.left_to_right = 0
            self.last_save_time = current_time
            return True
        except Exception as e:
            print(f"Failed to save data to {filename}: {e}")
            return False # 保存失敗


# ======= 検出と追跡の関数 =======
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


@lru_cache
def get_labels():
    """モデルのラベルを取得"""
    global intrinsics
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def calculate_iou(box1, box2):
    """
    IOU（交差率）を計算する関数
    「左上隅・幅・高さ（[x, y, w, h]）」形式の矩形2つからIoUを算出します。
    IoU（Intersection over Union）とは、物体検出モデルが予測したバウンディングボックスと
    正解バウンディングボックス（アノテーション）との重なり具合を評価する指標です。
    戻り値は「2つの矩形がどれくらい重なっているかの割合（0〜1）」
    +-----------+
    | box1      |
    |   +-------|-------+
    |   |   overlap     |
    +---+-------+-------+
        |      box2     |
        +---------------+
    上図のように、重なる部分がIoUの「intersecion」
    """
    # box1, box2 のフォーマット: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 幅・高さの健全性チェック
    if w1 < 0 or h1 < 0 or w2 < 0 or h2 < 0:
        print(f"[Warning]:不正な矩形サイズ検出: box1={box1}, box2={box2}")
        return 0.0

    # 計算しやすいように [x1, y1, x2, y2] 形式に変換する
    box1_tlbr = [x1, y1, x1 + w1, y1 + h1]
    box2_tlbr = [x2, y2, x2 + w2, y2 + h2]

    # 2つの矩形の共通部分（交差領域）の座標を求める
    x_intersect_min = max(box1_tlbr[0], box2_tlbr[0])
    y_intersect_min = max(box1_tlbr[1], box2_tlbr[1])
    x_intersect_max = min(box1_tlbr[2], box2_tlbr[2])
    y_intersect_max = min(box1_tlbr[3], box2_tlbr[3])

    # 交差領域の幅と高さを計算（重なりがなければ0となる）
    intersect_w = max(0, x_intersect_max - x_intersect_min)
    intersect_h = max(0, y_intersect_max - y_intersect_min)
    intersection_area = intersect_w * intersect_h

    # それぞれの矩形の面積を計算
    box1_area = w1 * h1
    box2_area = w2 * h2

    # IoU（交差率）を計算
    # IoU = 共通領域（intersection）/ 全領域（union）
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou

def track_people(detections, active_people, frame_id=None):
    """
    物体検出で得られた人物候補（detections）と、既存の追跡対象（active_people）を
    効率的かつ精度良くマッチングし、追跡リストを更新します。
    ※ IOUと距離、ハンガリアンアルゴリズムを使用
    """
    num_people = len(active_people)
    num_detections = len(detections)

    # 検出結果も追跡対象もいない場合はそのまま返す
    if num_detections == 0 and num_people == 0:
        return []

    # 新しい検出結果がない場合、既存の追跡対象は維持（ただし後にタイムアウトで削除される）
    if num_detections == 0:
        return active_people

    # 追跡対象がいない場合、全ての検出を新しい人物とする
    if num_people == 0:
        return [Person(det.box, det.appearance) for det in detections]

    # コスト行列を作成
    # 行: active_people, 列: detections
    # コストは小さいほど良い。マッチング不可能なペアには大きな値 (inf) を設定
    alpha = 1.0
    beta = 2.0
    gamma = 1.0  # appearance重み
    cost_matrix = np.full((len(active_people), len(detections)), np.inf)

    # コスト行列を計算
    for i, person in enumerate(active_people):
        # 追跡対象の予測位置（ここではシンプルに最新の位置を使用）
        person_box = person.box
        person_center = person.get_center()

        for j, detection in enumerate(detections):
            detection_box = detection.box
            detection_center = (detection_box[0] + detection_box[2]//2, detection_box[1] + detection_box[3]//2)

            # 距離とIOUを計算
            distance = np.sqrt((person_center[0] - detection_center[0])**2 + (person_center[1] - detection_center[1])**2)
            iou = calculate_iou(person_box, detection_box)

            app_dist = appearance_distance(person.appearance, detection.appearance)
            cost = alpha * (distance/200.0) + beta*(1-iou) + gamma*app_dist
            cost_matrix[i,j] = cost

    # コスト行列の全要素がinf or どの行or列も全てinfならreturn
    if (
        np.all(np.isinf(cost_matrix)) or 
        np.any(np.all(np.isinf(cost_matrix), axis=0)) or 
        np.any(np.all(np.isinf(cost_matrix), axis=1)) or
        np.sum(np.isfinite(cost_matrix)) < max(cost_matrix.shape)   # 有限値の要素数 < 行or列の大きいほう（マッチングに必要な最小数）なら諦める
    ):
        # print("Assignment infeasible: some row or column is all inf.")
        return active_people
    else:
        # ハンガリアンアルゴリズムを実行し、最適なマッチングを見つける
        # matched_person_indices: active_peopleのインデックスの配列
        # matched_detection_indices: detectionsのインデックスの配列
        # print("Will run linear_sum_assignment")
        try:
            matched_person_indices, matched_detection_indices = linear_sum_assignment(cost_matrix)

            # マッチング結果を処理
            new_people = []
            # マッチした検出結果のインデックスを記録
            used_detections = set(matched_detection_indices)
            used_person = set(matched_person_indices)
            # コストが高すぎる場合は不一致とみなす（例: 1.2以上。適宜調整）
            MAX_COST = 1.2
            # マッチした人物を更新して新しいリストに追加
            for i, j in zip(matched_person_indices, matched_detection_indices):
                # コストがinfの場合は有効なマッチではないのでスキップ (linear_sum_assignmentはinfも考慮する)
                if cost_matrix[i,j] < MAX_COST:
                    person = active_people[i]
                    detection = detections[j]
                    person.update(detection.box, detection.appearance)
                    new_people.append(person)
                else:
                    pass # 不一致

            # マッチしなかった新しい検出結果を新しい人物としてリストに追加
            for j, detection in enumerate(detections):
                if j not in used_detections:
                    new_people.append(Person(detection.box, detection.appearance))  # 新しい人物を作成
            # 失踪した人物も追加
            for i, person in enumerate(active_people):
                if i not in used_person:
                    new_people.append(person)

            print("distance=", distance, "iou=", iou)
            return new_people

        except Exception as e:
            print("【Error】ハンガリアン法(マッチング)で例外発生。")
            print(f"例外内容：{e}")
            print(f"フレームID: {frame_id}")
            print(f"コスト行列 shape: {cost_matrix.shape}")
            print(f"コスト行列の一部: {cost_matrix}")
            print(f"検出結果一覧: {detections}")
            print(f"追跡対象一覧: {active_people}")
            return active_people

def check_line_crossing(person, center_line_x, frame=None):
    """中央ラインを横切ったかチェック"""
    if len(person.trajectory) < 2:
        return None

    prev_x = person.trajectory[-2][0]
    curr_x = person.trajectory[-1][0]

    # 左→右: 中央線を未満→以上で通過
    if (prev_x < center_line_x and curr_x >= center_line_x):
        person.crossed_direction = "left_to_right"

        # デバッグモードで画像を保存
        if DEBUG_MODE and frame is not None:
            modules.save_debug_image(frame, person, center_line_x, "left_to_right", counter.debug_images_dir, counter.output_prefix)

        return "left_to_right"
    # 右→左: 中央線を以上→未満で通過
    elif (prev_x >= center_line_x and curr_x < center_line_x):
        person.crossed_direction = "right_to_left"

        # デバッグモードで画像を保存
        if DEBUG_MODE and frame is not None:
            modules.save_debug_image(frame, person, center_line_x, "right_to_left", counter.debug_images_dir, counter.output_prefix)

        return "right_to_left"

    return None

def process_frame_callback(request):
    """フレームごとの処理を行うコールバック関数"""
    global active_people, counter, last_log_time

    # 関数の属性が初期化されていない場合は初期化
    if not hasattr(process_frame_callback, 'image_saved'):
        init_process_frame_callback() # ここでactive_peopleとcounterも初期化される

    with MappedArray(request, 'main') as m:
        frame = m.array.copy()
    try:
        # メタデータを取得
        metadata = request.get_metadata()
        # SensorTimestampをframe_idに利用
        frame_id = metadata.get('SensorTimestamp')
        if metadata is None:
            # print("メタデータがNoneです") # デバッグ用
            # メタデータがない場合でも、既存のactive_peopleはタイムアウトで削除する必要があるため処理を進める
            # ただし検出処理はスキップ
            detections = []
        else:
            # 検出処理
            detections = parse_detections(metadata)
            for det in detections:
                det.appearance = extract_appearance_feature(frame, det.box)

        # 人物追跡を更新
        active_people = track_people(detections, active_people, frame_id)
        if not isinstance(active_people, list):
            print(f"track_people returned : {type(active_people)}")

        # フレームサイズを取得 (デバッグ画像保存やライン描画で使用)
        if frame is not None:
            frame_height, frame_width = frame.shape[:2]
            center_line_x = frame_width // 2

            # 起動時の画像を一度だけ保存
            if not process_frame_callback.image_saved:
                modules.save_image_at_startup(frame, center_line_x, counter.date_dir, counter.output_prefix)
                process_frame_callback.image_saved = True

            # 中央ラインを描画
            cv2.line(frame, (center_line_x, 0), (center_line_x, frame_height), 
                    (255, 255, 0), 2)
            
            # 人物の検出ボックスと軌跡を描画
            for person in active_people:
                x, y, w, h = person.box
                
                # 人物の方向によって色を変える
                if person.crossed_direction == "left_to_right":
                    color = (0, 255, 0)  # 緑: 左から右
                elif person.crossed_direction == "right_to_left":
                    color = (0, 0, 255)  # 赤: 右から左
                else:
                    color = (255, 255, 255)  # 白: まだカウントされていない
                
                # 検出ボックスを描画
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # ID表示
                cv2.putText(frame, f"ID: {person.id}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 軌跡を描画
                if len(person.trajectory) > 1:
                    for i in range(1, len(person.trajectory)):
                        cv2.line(frame, person.trajectory[i-1], person.trajectory[i], color, 2)
            
            # カウント情報を表示
            total_counts = counter.get_total_counts()
            cv2.putText(frame, f"right_to_left: {total_counts['right_to_left']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"left_to_right: {total_counts['left_to_right']}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 時刻とフレームIDを表示
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            text_str = f"FrameID: {frame_id} / {timestamp}"
            cv2.putText(frame, text_str,
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # ========== RTSP非同期配信 ==========
            if RTSP_SERVER_IP != 'None' and ffmpeg_proc and ffmpeg_proc.stdin:
                frame_for_rtsp = frame
                # BGRA→BGR変換
                if frame_for_rtsp.shape[2] == 4:
                    frame_for_rtsp = cv2.cvtColor(frame_for_rtsp, cv2.COLOR_BGRA2BGR)
                send_frame_for_rtsp(frame_for_rtsp)
            # ===================================

        # ラインを横切った人をカウント
        for person in active_people:
            # 少なくとも2フレーム以上の軌跡がある人物が対象
            if len(person.trajectory) >= 2:
                direction = check_line_crossing(person, center_line_x, frame)
                # print(f"[DEBUG] 人物ID {person.id} のライン判定")
                # print(f"[DEBUG] 軌跡: {person.trajectory[-2:]} (最後の2点を表示)")
                # distances = [abs(xy[0] - center_line_x) for xy in person.trajectory[-2:]]
                # print(f"[DEBUG] 直近2点のcenter_line_xまでの距離: {distances}")
                if direction:
                    counter.update(direction)
                    # print(f"[DEBUG] Person ID {person.id} crossed line: {direction}")
                else:
                    # print(f"[DEBUG] {person.id} はまだ横断していません")
                    pass

        # --- アクティブ人物リスト整理 ---
        # 古いトラッキング対象を削除 (last_seen が TRACKING_TIMEOUT を超えたもの)
        current_time = time.time()
        active_people = [p for p in active_people if current_time - p.last_seen < TRACKING_TIMEOUT]

        # 定期的なログ出力
        if current_time - last_log_time >= LOG_INTERVAL:
            total_counts = counter.get_total_counts()
            elapsed = int(current_time - counter.last_save_time)
            remaining = max(0, int(COUNTING_INTERVAL - elapsed)) # 負にならないように
            
            print(f"--- Status Update ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            print(f"Active tracking: {len(active_people)} people")
            print(f"Counts - Period (R->L: {counter.right_to_left}, L->R: {counter.left_to_right})")
            print(f"Counts - Total (R->L: {total_counts['right_to_left']}, L->R: {total_counts['left_to_right']})")
            print(f"Next save in: {remaining} seconds")
            print(f"--------------------------------------------------")


        # 指定間隔ごとにJSONファイルに保存
        if current_time - counter.last_save_time >= COUNTING_INTERVAL:
            counter.save_to_json()

        last_log_time = current_time

    except Exception as e:
        print(f"コールバックエラー: {e}")
        import traceback
        traceback.print_exc()


# ======= メイン処理 =======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMX500 AIカメラモジュール制御")
    parser.add_argument('--preview', action='store_true', help='プレビュー画面を表示する')
    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATE_DIR, exist_ok=True)
    # デバッグディレクトリの作成
    if DEBUG_MODE:
        os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
        print(f"デバッグモード有効: 画像を {DEBUG_IMAGES_DIR} に保存")

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
        config = picam2.create_preview_configuration(main, controls={"FrameRate": intrinsics.inference_rate}, buffer_count=8)

        imx500.show_network_fw_progress_bar()

        # カメラの設定と起動
        # 2段階の初期化
        picam2.configure(config)
        time.sleep(0.5)  # 少し待機

        if args.preview:
            picam2.start(show_preview=True)  # プレビューを表示
        else:
            picam2.start()                   # ヘッドレスモード

        if intrinsics.preserve_aspect_ratio:
            imx500.set_auto_aspect_ratio()
            
        print("カメラ起動完了")

        # RTSP配信用 設定値
        if RTSP_SERVER_IP != 'None':
            FRAME_WIDTH  = 640
            FRAME_HEIGHT = 480
            FRAME_RATE = int(intrinsics.inference_rate) if hasattr(intrinsics, 'inference_rate') else 15
            RTSP_URL = f"rtsp://{RTSP_SERVER_IP}:{RTSP_SERVER_PORT}/stream"

            ffmpeg_cmd = [
                "ffmpeg",
                "-nostats",
                "-loglevel", "error", 
                "-re",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
                "-r", str(FRAME_RATE),
                "-i", "-",              # 入力：標準入力
                "-an",                  # 音声なし
                "-c:v", "libx264",
                "-preset", "ultrafast", # 低遅延
                "-tune", "zerolatency",
                "-f", "rtsp",
                RTSP_URL
            ]
            try:
                ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
                rtsp_thread = start_rtsp_thread(ffmpeg_proc)
                print("ffmpegによるRTSP配信プロセス起動")
            except Exception as e:
                print(f"ffmpeg起動失敗: {e}")
                sys.exit(1)
        else:
            print(f"RTSP_SERVER_IPが未指定のためRTSP配信は行いません")

    except Exception as e:
        print(f"カメラ初期化エラーまたはIMX500初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 人物追跡とカウントの初期化
    active_people = [] # グローバル変数として初期化
    start_time = time.time()
    counter = PeopleCounter(start_time, OUTPUT_DIR, OUTPUT_PREFIX, DATE_DIR, DEBUG_IMAGES_DIR) # グローバル変数として初期化
    last_log_time = start_time

    # コールバックを設定
    # コールバック関数はフレームが準備されるたびに呼ばれる
    picam2.pre_callback = process_frame_callback

    print(f"人流カウント開始 - {COUNTING_INTERVAL}秒ごとにデータを保存します")
    print(f"ログは{LOG_INTERVAL}秒ごとに出力されます")
    print("Ctrl+Cで終了します")
    
    try:
        # メインループ - コールバックが処理を行うので、ここでは待機するだけ
        while True:
            time.sleep(0.01)  # CPUの負荷を減らすために短い時間待機
            # コールバック内でカウントデータが保存される

    except KeyboardInterrupt:
        print("終了中...")
        # 最後のデータを保存
        counter.save_to_json()

    finally:
        # リソースの解放
        try:
            if 'picam2' in locals() and picam2: # picam2が初期化されているか確認
                picam2.stop()
                picam2.close() # カメラを閉じる
                print("カメラを停止しました")

            print("プログラムを終了します")
        except Exception as e:
            print(f"終了処理エラー: {e}")