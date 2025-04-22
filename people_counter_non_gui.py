import json
import os
import sys
import time
from datetime import datetime
from functools import lru_cache

import numpy as np
# scipyの線形割当アルゴリズムをインポート
from scipy.optimize import linear_sum_assignment

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

# ======= 設定パラメータ（必要に応じて変更） =======
# モデル設定
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
DETECTION_THRESHOLD = 0.55  # 検出信頼度の閾値
IOU_THRESHOLD = 0.3  # !!! 新規追加: マッチングのためのIOU閾値
MAX_DETECTIONS = 30  # 1フレームあたりの最大検出数 (増やす可能性も考慮し少し大きく)

# フレーム幅と高さを固定値として定義
FRAME_WIDTH = 1280
FRAME_HEIGHT = 960

# 人流カウント設定
PERSON_CLASS_ID = 0  # 人物クラスのID（通常COCOデータセットでは0）
MAX_TRACKING_DISTANCE = 100 # !!! 閾値を少し大きく - IOUとの組み合わせで判定
TRACKING_TIMEOUT = 5.0  # 人物を追跡し続ける最大時間（秒）
COUNTING_INTERVAL = 60  # カウントデータを保存する間隔（秒）

# 出力設定
OUTPUT_DIR = "people_count_data"    # データ保存ディレクトリ
OUTPUT_PREFIX = "cameraA"           # 出力ファイル名のプレフィックス(カメラ名を入れる)

# ログ設定
LOG_INTERVAL = 5  # ログ出力間隔（秒）

# グローバル変数
# active_people = [] # main関数で初期化する
# counter = None # main関数で初期化する
last_log_time = 0

DEBUG_MODE = False  # デバッグモードのオン/オフ
DEBUG_IMAGES_DIR = "debug_images"  # デバッグ画像の保存ディレクトリ

def init_process_frame_callback():
    # コールバック関数の属性を初期化
    process_frame_callback.image_saved = False
    # グローバル変数をここで初期化 (mainでも行うが、コールバックが先に呼ばれる可能性も考慮)
    global active_people, counter
    active_people = []
    # counterはmainで初期化されるはずだが、念のためNoneチェック
    if counter is None:
        counter = PeopleCounter(time.time(), OUTPUT_DIR, OUTPUT_PREFIX)


# ======= クラス定義 (変更なし) =======
class Detection:
    def __init__(self, coords, category, conf, metadata):
        """検出オブジェクトを作成し、バウンディングボックス、カテゴリ、信頼度を記録"""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2) # [x, y, w, h] 形式


class Person:
    next_id = 0

    def __init__(self, box):
        self.id = Person.next_id
        Person.next_id += 1
        self.box = box # [x, y, w, h] 形式
        self.trajectory = [self.get_center()]
        self.counted = False
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.crossed_direction = None
        # self.output_dir = OUTPUT_DIR # Personクラスで保持する必要はなさそう
        # self.filename_prefix = OUTPUT_PREFIX # Personクラスで保持する必要はなさそう

    def get_center(self):
        """バウンディングボックスの中心座標を取得"""
        x, y, w, h = self.box
        return (x + w//2, y + h//2)

    def update(self, box):
        """新しい検出結果で人物の情報を更新"""
        self.box = box # [x, y, w, h] 形式
        self.trajectory.append(self.get_center())
        if len(self.trajectory) > 30:  # 軌跡は最大30ポイントまで保持
            self.trajectory.pop(0)
        self.last_seen = time.time()


class PeopleCounter:
    def __init__(self, start_time, output_dir=OUTPUT_DIR, output_prefix=OUTPUT_PREFIX):
        self.right_to_left = 0  # 右から左へ移動（期間カウント）
        self.left_to_right = 0  # 左から右へ移動（期間カウント）
        self.total_right_to_left = 0  # 累積カウント
        self.total_left_to_right = 0  # 累積カウント
        self.start_time = start_time
        self.last_save_time = start_time
        self.output_dir = output_dir
        self.output_prefix = output_prefix

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
        """指定間隔でカウントデータをJSONファイルに保存"""
        current_time = time.time()
        # 指定間隔経過したらデータを保存
        if current_time - self.last_save_time >= COUNTING_INTERVAL:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            data = {
                "timestamp": timestamp,
                "duration_seconds": int(current_time - self.last_save_time),
                "period_counts": self.get_counts(),
                "total_counts": self.get_total_counts()
            }

            # ファイルパスを正しく構築
            filename = os.path.join(self.output_dir, f"{self.output_prefix}_{timestamp}.json")
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

        return False

# ======= 検出と追跡の関数 =======
def parse_detections(metadata: dict):
    """AIモデルの出力テンソルを解析し、検出された人物のリストを返す"""
    try:
        # intrinsicsはmain関数で初期化されるグローバル変数と仮定
        # picam2もmain関数で初期化されるグローバル変数と仮定
        global intrinsics, picam2

        bbox_normalization = intrinsics.bbox_normalization

        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return [] # 検出がない場合は空リストを返す

        input_w, input_h = imx500.get_input_size()

        if intrinsics.postprocess == "nanodet":
            boxes, scores, classes = \
                postprocess_nanodet_detection(outputs=np_outputs[0], conf=DETECTION_THRESHOLD,
                                             iou_thres=IOU_THRESHOLD, max_out_dets=MAX_DETECTIONS)[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
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
        # ここで取得した box は既にフレームサイズに合わせた [x, y, w, h] 形式になっているはずです。

        return detections
    except Exception as e:
        print(f"検出処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return [] # エラー時は空リストを返す


@lru_cache
def get_labels():
    """モデルのラベルを取得"""
    # intrinsicsはmain関数で初期化されるグローバル変数と仮定
    global intrinsics
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


# --- IOU計算関数を追加 ---
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    box: [x, y, w, h]
    """
    # box1, box2 format: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2] format for easier calculation
    box1_tlbr = [x1, y1, x1 + w1, y1 + h1]
    box2_tlbr = [x2, y2, x2 + w2, y2 + h2]

    # Determine the coordinates of the intersection rectangle
    x_intersect_min = max(box1_tlbr[0], box2_tlbr[0])
    y_intersect_min = max(box1_tlbr[1], box2_tlbr[1])
    x_intersect_max = min(box1_tlbr[2], box2_tlbr[2])
    y_intersect_max = min(box1_tlbr[3], box2_tlbr[3])

    # Compute the area of intersection rectangle
    intersect_w = max(0, x_intersect_max - x_intersect_min)
    intersect_h = max(0, y_intersect_max - y_intersect_min)
    intersection_area = intersect_w * intersect_h

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the union area
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


# --- track_people 関数をハンガリアンアルゴリズムとIOUを使うように書き換え ---
def track_people(detections, active_people):
    """
    検出された人物と既存の追跡対象をマッチング (IOUと距離、ハンガリアンアルゴリズムを使用)
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
        return [Person(det.box) for det in detections]

    # コスト行列を作成
    # 行: active_people, 列: detections
    # コストは小さいほど良い。マッチング不可能なペアには大きな値 (inf) を設定
    cost_matrix = np.full((num_people, num_detections), np.inf)

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

            # マッチングの条件: 距離が閾値以内 かつ IOUが閾値以上
            if distance < MAX_TRACKING_DISTANCE and iou > IOU_THRESHOLD:
                # コストの定義 (距離が近いほど、IOUが大きいほど良いマッチング -> コスト小)
                # シンプルに距離をコストとする
                cost = distance
                # より高度なコスト例: cost = (1.0 - iou) + (distance / MAX_TRACKING_DISTANCE) * 0.1 # IOUを重視
                cost_matrix[i, j] = cost

    # ハンガリアンアルゴリズムを実行し、最適なマッチングを見つける
    # matched_person_indices: active_peopleのインデックスの配列
    # matched_detection_indices: detectionsのインデックスの配列
    matched_person_indices, matched_detection_indices = linear_sum_assignment(cost_matrix)

    # マッチング結果を処理
    new_people = []
    # マッチした検出結果のインデックスを記録
    used_detections = set(matched_detection_indices)

    # マッチした人物を更新して新しいリストに追加
    for i, j in zip(matched_person_indices, matched_detection_indices):
        # コストがinfの場合は有効なマッチではないのでスキップ (linear_sum_assignmentはinfも考慮する)
        if cost_matrix[i, j] == np.inf:
            continue
        person = active_people[i]
        detection = detections[j]
        person.update(detection.box) # 人物情報を検出結果で更新
        new_people.append(person)

    # マッチしなかった既存の人物を新しいリストに追加 (タイムアウト判定は後で行われる)
    matched_person_ids = {p.id for p in new_people} # 更新された人物のIDセット
    for person in active_people:
        if person.id not in matched_person_ids:
            # この人物は今回のフレームでは検出されなかった
            # 情報を更新しないままリストに追加。last_seenは前回のまま。
            new_people.append(person)

    # マッチしなかった新しい検出結果を新しい人物としてリストに追加
    for j, detection in enumerate(detections):
        if j not in used_detections:
            new_people.append(Person(detection.box)) # 新しい人物を作成

    # ここではタイムアウト判定は行わない。process_frame_callbackでまとめて行う。
    return new_people


def check_line_crossing(person, center_line_x, frame=None):
    """中央ラインを横切ったかチェック (変更なし)"""
    if len(person.trajectory) < 2 or person.counted:
        return None

    prev_x = person.trajectory[-2][0]
    curr_x = person.trajectory[-1][0]

    # 中央ラインを横切った場合
    if (prev_x < center_line_x and curr_x >= center_line_x):
        person.counted = True
        person.crossed_direction = "left_to_right"

        # デバッグモードで画像を保存
        if DEBUG_MODE and frame is not None:
            save_debug_image(frame, person, center_line_x, "left_to_right")

        return "left_to_right"
    elif (prev_x >= center_line_x and curr_x < center_line_x):
        person.counted = True
        person.crossed_direction = "right_to_left"

        # デバッグモードで画像を保存
        if DEBUG_MODE and frame is not None:
            save_debug_image(frame, person, center_line_x, "right_to_left")

    return None

# --- save_debug_image (変更なし) ---
def save_debug_image(frame, person, center_line_x, direction):
    """デバッグ用に画像を保存する関数"""
    try:
        import cv2
        from datetime import datetime

        # デバッグ画像ディレクトリがなければ作成
        os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)

        # 画像にラインと人物のバウンディングボックスを描画
        debug_frame = frame.copy()

        # 中央ラインを描画
        cv2.line(debug_frame, (center_line_x, 0), (center_line_x, debug_frame.shape[0]), (0, 255, 0), 2)

        # 人物のバウンディングボックスを描画
        x, y, w, h = person.box
        # 座標が整数であることを確認 (cv2のrectangleは整数が必要)
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 軌跡を描画
        for i in range(1, len(person.trajectory)):
            # 座標が整数であることを確認
            pt1 = (int(person.trajectory[i-1][0]), int(person.trajectory[i-1][1]))
            pt2 = (int(person.trajectory[i][0]), int(person.trajectory[i][1]))
            cv2.line(debug_frame, pt1, pt2, (255, 0, 0), 2)

        # 情報テキストを追加
        text = f"ID: {person.id}, Dir: {direction}"
        cv2.putText(debug_frame, text, (x, y - 10) if y > 20 else (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # タイムスタンプ付きのファイル名で保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(DEBUG_IMAGES_DIR, f"{timestamp}_crossing_{person.id}_{direction}.jpg")
        cv2.imwrite(filename, debug_frame)
        print(f"デバッグ画像を保存しました: {filename}")
    except Exception as e:
        print(f"デバッグ画像保存エラー: {e}")


# --- save_image_at_startup (変更なし) ---
def save_image_at_startup(frame, center_line_x):
    """起動時に画像を保存する関数"""
    try:
        import cv2
        from datetime import datetime

        # デバッグ画像ディレクトリがなければ作成 (OUTPUT_DIRでも良いがDEBUG_IMAGES_DIRに合わせておく)
        os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)

        # 画像にラインを描画
        debug_frame = frame.copy()

        # 中央ラインを描画
        cv2.line(debug_frame, (center_line_x, 0), (center_line_x, debug_frame.shape[0]), (0, 255, 0), 2)

        # 情報テキストを追加
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        text = f"Start Up Time: {timestamp}, Counting Line X: {center_line_x}"
        cv2.putText(debug_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # タイムスタンプ付きのファイル名で保存
        # ファイル名はOUTPUT_DIRではなくDEBUG_IMAGES_DIRに保存するように変更
        filename = os.path.join(DEBUG_IMAGES_DIR, f"{OUTPUT_PREFIX}_{timestamp}_startupimage.jpg")
        cv2.imwrite(filename, debug_frame)
        print(f"起動時に画像を保存しました: {filename}")
    except Exception as e:
        print(f"起動時に画像を保存する関数の実行エラー: {e}")


# --- process_frame_callback (変更: track_peopleの呼び出しのみ) ---
def process_frame_callback(request):
    """フレームごとの処理を行うコールバック関数"""
    global active_people, counter, last_log_time

    # 関数の属性が初期化されていない場合は初期化
    if not hasattr(process_frame_callback, 'image_saved'):
        init_process_frame_callback() # ここでactive_peopleとcounterも初期化される

    try:
        # メタデータを取得
        metadata = request.get_metadata()
        if metadata is None:
            # print("メタデータがNoneです") # デバッグ用
            # メタデータがない場合でも、既存のactive_peopleはタイムアウトで削除する必要があるため処理を進める
            # ただし検出処理はスキップ
            detections = []
        else:
            # 検出処理
            detections = parse_detections(metadata)

        # フレームサイズを取得 (デバッグ画像保存やライン描画に必要)
        with MappedArray(request, 'main') as m:
            frame_height, frame_width = m.array.shape[:2]
            center_line_x = frame_width // 2

            # 起動時の画像を一度だけ保存
            if not process_frame_callback.image_saved:
                save_image_at_startup(m.array, center_line_x)
                process_frame_callback.image_saved = True
                print("起動時の画像を保存しました")

            # デバッグモードの場合、フレーム画像をコピー (check_line_crossingに渡すため)
            frame_copy = None
            if DEBUG_MODE:
                frame_copy = m.array.copy()

        # 人物追跡を更新 - 書き換えた track_people 関数を呼び出す
        active_people = track_people(detections, active_people)

        # ラインを横切った人をカウント
        # 注意: ここで active_people の中には更新されたものと更新されなかったものが混在
        for person in active_people:
            # 少なくとも2フレーム以上の軌跡がある、かつ、まだカウントされていない人物が対象
            if len(person.trajectory) >= 2 and not person.counted:
                direction = check_line_crossing(person, center_line_x, frame_copy)
                if direction:
                    counter.update(direction)
                    print(f"Person ID {person.id} crossed line: {direction}")

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

            last_log_time = current_time

        # 指定間隔ごとにJSONファイルに保存
        counter.save_to_json()

    except Exception as e:
        print(f"コールバックエラー: {e}")
        import traceback
        traceback.print_exc()


# ======= メイン処理 (変更なし) =======
if __name__ == "__main__":
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
        config = picam2.create_preview_configuration(main, controls={"FrameRate": intrinsics.inference_rate}, buffer_count=6)

        imx500.show_network_fw_progress_bar()

        # カメラの設定と起動
        # 2段階の初期化
        picam2.configure(config)
        time.sleep(0.5)  # 少し待機
        picam2.start()  # ヘッドレスモードでスタート
        
        if intrinsics.preserve_aspect_ratio:
            imx500.set_auto_aspect_ratio()
            
        print("カメラ起動完了")
    except Exception as e:
        print(f"カメラ初期化エラーまたはIMX500初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 人物追跡とカウントの初期化
    active_people = [] # グローバル変数として初期化
    start_time = time.time()
    counter = PeopleCounter(start_time, OUTPUT_DIR, OUTPUT_PREFIX) # グローバル変数として初期化
    global last_log_time # グローバル変数を使用宣言
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
            if 'imx500' in locals() and imx500: # imx500が初期化されているか確認
                imx500.close() # AIモジュールを閉じる
                print("IMX500モジュールを閉じました")

            print("プログラムを終了します")
        except Exception as e:
            print(f"終了処理エラー: {e}")