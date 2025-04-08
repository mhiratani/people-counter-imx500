import json
import os
import sys
import time
from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

# ======= 設定パラメータ（必要に応じて変更） =======
# モデル設定
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
DETECTION_THRESHOLD = 0.55  # 検出信頼度の閾値
IOU_THRESHOLD = 0.65  # 重複検出を除去するためのIOU閾値
MAX_DETECTIONS = 10  # 1フレームあたりの最大検出数

# フレーム幅と高さを固定値として定義
FRAME_WIDTH = 1280
FRAME_HEIGHT = 960


# 人流カウント設定
PERSON_CLASS_ID = 0  # 人物クラスのID（通常COCOデータセットでは0）
MAX_TRACKING_DISTANCE = 50  # 同一人物と判定する最大距離（ピクセル）
TRACKING_TIMEOUT = 5.0  # 人物を追跡し続ける最大時間（秒）
COUNTING_INTERVAL = 60  # カウントデータを保存する間隔（秒）

# 出力設定
OUTPUT_DIR = "people_count_data"  # データ保存ディレクトリ
OUTPUT_PREFIX = "people_count"  # 出力ファイル名のプレフィックス

# グローバル変数
active_people = []
counter = None

# ======= クラス定義 =======
class Detection:
    def __init__(self, coords, category, conf, metadata):
        """検出オブジェクトを作成し、バウンディングボックス、カテゴリ、信頼度を記録"""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


class Person:
    next_id = 0
    
    def __init__(self, box):
        self.id = Person.next_id
        Person.next_id += 1
        self.box = box
        self.trajectory = [self.get_center()]
        self.counted = False
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.crossed_direction = None
    
    def get_center(self):
        """バウンディングボックスの中心座標を取得"""
        x, y, w, h = self.box
        return (x + w//2, y + h//2)
    
    def update(self, box):
        """新しい検出結果で人物の情報を更新"""
        self.box = box
        self.trajectory.append(self.get_center())
        if len(self.trajectory) > 30:  # 軌跡は最大30ポイントまで保持
            self.trajectory.pop(0)
        self.last_seen = time.time()


class PeopleCounter:
    def __init__(self, start_time):
        self.right_to_left = 0  # 右から左へ移動（期間カウント）
        self.left_to_right = 0  # 左から右へ移動（期間カウント）
        self.total_right_to_left = 0  # 累積カウント
        self.total_left_to_right = 0  # 累積カウント
        self.start_time = start_time
        self.last_save_time = start_time
    
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
    
    def save_to_json(self, filename_prefix=OUTPUT_PREFIX):
        """指定間隔でカウントデータをJSONファイルに保存"""
        current_time = time.time()
        # 指定間隔経過したらデータを保存
        if current_time - self.last_save_time >= COUNTING_INTERVAL:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data = {
                "timestamp": timestamp,
                "duration_seconds": int(current_time - self.last_save_time),
                "period_counts": self.get_counts(),
                "total_counts": self.get_total_counts()
            }
            
            # 初回実行時にタイムスタンプ付きの出力ディレクトリを作成
            if not hasattr(self, 'output_dir_with_timestamp'):
                start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir_with_timestamp = f"{OUTPUT_DIR}_{start_timestamp}"
                os.makedirs(self.output_dir_with_timestamp, exist_ok=True)
                print(f"Created output directory: {self.output_dir_with_timestamp}")
            
            # ファイルパスを正しく構築
            filename = os.path.join(self.output_dir_with_timestamp, f"{filename_prefix}_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"Data saved to {filename}")
            
            # 期間カウンターのみリセット
            self.right_to_left = 0
            self.left_to_right = 0
            self.last_save_time = current_time
            return True
        
        return False


# ======= 検出と追跡の関数 =======
def parse_detections(metadata: dict):
    """AIモデルの出力テンソルを解析し、検出された人物のリストを返す"""
    try:
        bbox_normalization = intrinsics.bbox_normalization

        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return None
            
        input_w, input_h = imx500.get_input_size()
        
        if intrinsics.postprocess == "nanodet":
            boxes, scores, classes = \
                postprocess_nanodet_detection(outputs=np_outputs[0], conf=DETECTION_THRESHOLD, 
                                             iou_thres=IOU_THRESHOLD, max_out_dets=MAX_DETECTIONS)[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                boxes = boxes / input_h

            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        # 人物クラスのみをフィルタリング
        detections = [
            Detection(box, category, score, metadata)
            for box, score, category in zip(boxes, scores, classes)
            if score > DETECTION_THRESHOLD and int(category) == PERSON_CLASS_ID
        ]
        return detections
    except Exception as e:
        print(f"検出処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


@lru_cache
def get_labels():
    """モデルのラベルを取得"""
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def track_people(detections, active_people):
    """検出された人物と既存の追跡対象をマッチング"""
    if not detections:
        return active_people
    
    # 現在の検出と既存の人物をマッチング
    new_people = []
    used_detections = set()
    
    # 既存の人物を更新
    for person in active_people:
        best_match = None
        min_distance = float('inf')
        
        person_center = person.get_center()
        
        for i, detection in enumerate(detections):
            if i in used_detections:
                continue
                
            x, y, w, h = detection.box
            detection_center = (x + w//2, y + h//2)
            
            # ユークリッド距離を計算
            distance = np.sqrt((person_center[0] - detection_center[0])**2 + 
                              (person_center[1] - detection_center[1])**2)
            
            if distance < min_distance and distance < MAX_TRACKING_DISTANCE:
                min_distance = distance
                best_match = i
        
        if best_match is not None:
            # 最も近い検出で人物を更新
            person.update(detections[best_match].box)
            used_detections.add(best_match)
            new_people.append(person)
        elif time.time() - person.last_seen < 1.0:  # 1秒以内に見失った人はまだ追跡
            new_people.append(person)
    
    # 未使用の検出から新しい人物を作成
    for i, detection in enumerate(detections):
        if i not in used_detections:
            new_people.append(Person(detection.box))
    
    return new_people


def check_line_crossing(person, center_line_x):
    """中央ラインを横切ったかチェック"""
    if len(person.trajectory) < 2 or person.counted:
        return None
    
    prev_x = person.trajectory[-2][0]
    curr_x = person.trajectory[-1][0]
    
    # 中央ラインを横切った場合
    if (prev_x < center_line_x and curr_x >= center_line_x):
        person.counted = True
        person.crossed_direction = "left_to_right"
        return "left_to_right"
    elif (prev_x >= center_line_x and curr_x < center_line_x):
        person.counted = True
        person.crossed_direction = "right_to_left"
        return "right_to_left"
    
    return None


def process_frame(detections, active_people, counter, frame_width):
    """フレームごとの処理: 人物追跡とカウント"""
    if detections is None:
        return active_people
    
    # 人物追跡を更新
    active_people = track_people(detections, active_people)
    
    # 中央ラインの位置を計算
    center_line_x = frame_width // 2
    
    # ラインを横切った人をカウント
    for person in active_people:
        direction = check_line_crossing(person, center_line_x)
        if direction:
            counter.update(direction)
    
    # 古いトラッキング対象を削除
    active_people = [p for p in active_people if time.time() - p.last_seen < TRACKING_TIMEOUT]
    
    return active_people


# ======= コールバック関数 =======
def process_frame_callback(request):
    """フレームごとの処理を行うコールバック関数"""
    global active_people, counter
    
    try:
        # メタデータを取得
        metadata = request.get_metadata()
        if metadata is None:
            print("警告: メタデータが取得できませんでした")
            return
            
        # フレーム幅を取得
        # frame_width = picam2.stream_configuration["main"]["size"][0]

        # 描画処理
        with MappedArray(request, 'main') as m:
            # 実際のフレームデータから取得（より安全）
            actual_height, actual_width = m.array.shape[:2]
            
            # 実際の値を使用するか、何らかの理由で取得できない場合は固定値を使用
            frame_height = actual_height
            frame_width = actual_width
            center_line_x = frame_width // 2

            # 検出処理
            detections = parse_detections(metadata)
            if detections is not None:
                active_people = process_frame(detections, active_people, counter, frame_width)
            
            # 中央ラインを描画
            cv2.line(m.array, (center_line_x, 0), (center_line_x, frame_height), 
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
                cv2.rectangle(m.array, (x, y), (x + w, y + h), color, 2)
                
                # ID表示
                cv2.putText(m.array, f"ID: {person.id}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 軌跡を描画
                if len(person.trajectory) > 1:
                    for i in range(1, len(person.trajectory)):
                        cv2.line(m.array, person.trajectory[i-1], person.trajectory[i], color, 2)
            
            # カウント情報を表示
            total_counts = counter.get_total_counts()
            cv2.putText(m.array, f"right_to_left: {total_counts['right_to_left']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(m.array, f"left_to_right: {total_counts['left_to_right']}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 経過時間表示
            elapsed = int(time.time() - counter.last_save_time)
            remaining = COUNTING_INTERVAL - elapsed
            cv2.putText(m.array, f"Remaining time: {remaining}sec", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        # 指定間隔ごとにJSONファイルに保存
        if counter.save_to_json(OUTPUT_PREFIX):
            total_counts = counter.get_total_counts()
            print(f"カウント結果: 右→左: {total_counts['right_to_left']}, 左→右: {total_counts['left_to_right']}")
            
    except Exception as e:
        print(f"コールバックエラー: {e}")
        import traceback
        traceback.print_exc()


# ======= メイン処理 =======
if __name__ == "__main__":
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # IMX500の初期化
    imx500 = IMX500(MODEL_PATH)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("ネットワークはオブジェクト検出タスクではありません", file=sys.stderr)
        exit()

    # デフォルトラベル
    if intrinsics.labels is None:
        try:
            with open("assets/coco_labels.txt", "r") as f:
                intrinsics.labels = f.read().splitlines()
        except FileNotFoundError:
            # COCOデータセットの一般的なラベル
            intrinsics.labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
    intrinsics.update_with_defaults()

    # Picamera2の初期化
    picam2 = Picamera2(imx500.camera_num)
    main = {'format': 'XRGB8888'}
    config = picam2.create_preview_configuration(main, controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    print("カメラとAIモデルを初期化中...")
    imx500.show_network_fw_progress_bar()
    
    # カメラの設定と起動
    try:
        # 2段階の初期化
        picam2.configure(config)
        time.sleep(0.5)  # 少し待機
        picam2.start(show_preview=True)  # プレビューを表示
        
        if intrinsics.preserve_aspect_ratio:
            imx500.set_auto_aspect_ratio()
            
        print("カメラ起動完了")
    except Exception as e:
        print(f"カメラ初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 人物追跡とカウントの初期化
    active_people = []
    start_time = time.time()
    counter = PeopleCounter(start_time)

    # コールバックを設定
    picam2.pre_callback = process_frame_callback

    print(f"人流カウント開始 - {COUNTING_INTERVAL}秒ごとにデータを保存します")
    
    try:
        # メインループ - コールバックが処理を行うので、ここでは待機するだけ
        while True:
            time.sleep(1)  # CPUの負荷を減らすために少し待機
            
    except KeyboardInterrupt:
        print("終了中...")
        # 最後のデータを保存
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX)
        counter.save_to_json(output_path)
        
    finally:
        # リソースの解放
        try:
            picam2.stop()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"終了処理エラー: {e}")