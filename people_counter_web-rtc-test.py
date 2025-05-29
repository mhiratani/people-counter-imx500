import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment    # scipyの線形割当アルゴリズム

# NMS(Non-Maximum Suppression)適用
import torch
from torchvision.ops import nms

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)
from picamera2.devices.imx500.postprocess import scale_boxes

import modules

# webRTC配信プロセス用
import ssl
import queue
import threading
import asyncio
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiohttp import web
from av import VideoFrame
# グローバルでQueue用意 
frame_queue = asyncio.Queue(maxsize=5)

# 描画設定
import cv2

from filterpy.kalman import KalmanFilter

# モデル設定
# https://www.raspberrypi.com/documentation/accessories/ai-camera.html の
# "Run the following script from the repository to run YOLOv8 object detection:"を参照して選んだモデル
#MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
MODEL_PATH = "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk"

# ======= クラス定義 =======
class Parameter:
    def __init__(self, model_path=MODEL_PATH):
        self.config = self._load_config('config.json')
        self.camera_name                = self._get_cameraname()
        self.model_path                  = model_path
        self.person_class_id            = 0     # 人物クラスのID（通常COCOデータセットでは0）
        self.detection_threshold        = self.config.get('DETECTION_THRESHOLD')                # 検出器が出力する「検出信頼度スコア」の下限値。これ未満は無視する。
        self.iou_threshold              = self.config.get('IOU_THRESHOLD',)                     # マッチング時、追跡対象と検出結果の「バウンディングボックスの重なり（IoU）」の下限値
        self.max_detections             = self.config.get('MAX_DETECTIONS',)                    # 1フレームで扱う検出結果の最大数。これ以上は間引きされるか無視される
        self.center_line_margin_px      = self.config.get('CENTER_LINE_MARGIN_PX')              # ライン中心から±何ピクセルを「ライン近傍」とみなすかの閾値（ピクセル数）
        self.recovery_distance_px       = self.config.get('RECOVERY_DISTANCE_PX')               # 過去の人物と新しい検出の中心座標（x）の距離が 何ピクセル以内なら「同一人物が復帰した」とみなすかの閾値
        self.tracking_timeout           = self.config.get('TRACKING_TIMEOUT')                   # 人物を追跡し続ける最大時間（秒）
        self.active_timeout_sec         = self.config.get('ACTIVE_TIMEOUT_SEC')                 # lost_people保持猶予(秒)
        self.direction_mismatch_penalty = self.config.get('DIRECTION_MISMATCH_PENALTY')         # 逆方向へのマッチに与える追加コスト
        self.max_acceptable_cost        = self.config.get('MAX_ACCEPTABLE_COST')                # 最大許容コスト
        self.min_box_height             = self.config.get('MIN_BOX_HEIGHT')                     # 人物ボックスの高さフィルタ。これより小さいBoxは排除(ピクセル)
        self.max_box_height             = self.config.get('MAX_BOX_HEIGHT')                     # 人物ボックスの高さフィルタ。これより大きいBoxは排除(ピクセル)
        self.log_output_interval_sec    = self.config.get('LOG_OUTPUT_INTERVAL_SEC')            # ログ出力間隔（秒）
        self.output_dir                 = self.config.get('OUTPUT_DIR', 'people_count_data')    # ログデータを保存するディレクトリ名
        self.debug_mode                 = str(self.config.get('DEBUG_MODE', 'False')).lower() == 'true'
        self.debug_images_subdir_name   = self.config.get('DEBUG_IMAGES_SUBDIR_NAME', 'debug_images')
        self.log_interval               = 10

    def _load_config(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
        return config

    def _get_cameraname(self):
        try:
            # カメラ名を取得するための設定ファイルを読み込む
            camera_config = self._load_config('camera_name.json')
            camera_name = camera_config.get('CAMERA_NAME', 'camera_name_unknown')
        except FileNotFoundError:
            print("camera_name.jsonが見つかりません。デフォルトのカメラ名を使用します。")
            camera_name = "camera_name_unknown"

        return camera_name

class CameraTrack(VideoStreamTrack):
    """
    WebRTC用のカスタムVideoStreamTrack。
    asyncio.QueueからBGR画像(numpy配列)を取り出して、WebRTCのvideoフレームとして送信する。
    """
    kind = "video"  # このトラックが"video"ストリームであることを明示

    def __init__(self, queue):
        super().__init__()
        self.queue = queue  # Producer側から動画フレーム供給されるasyncio.Queue

    async def recv(self):
        """
        WebRTCスタックからフレーム送信を要求された際に呼び出される非同期メソッド。
        キューから次のフレームを取得し、VideoFrameとして返す。
        問題があれば黒画像で代用。
        """
        #print("CameraTrack.recv called!")  # デバッグ用：ちゃんと呼ばれているか確認

        try:
            # 1秒以内にキューからフレームを受け取る。タイムアウトなら例外送出。
            frame = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            # print("CameraTrack got frame from queue!", frame.shape, frame.dtype, frame.max(), frame.min())  # デバッグ用

            # OpenCV画像が偶数高さ・幅でなければ切り詰める（YUV420p変換時に必要）
            if frame.shape[0] % 2 == 1 or frame.shape[1] % 2 == 1:
                frame = frame[:frame.shape[0]//2*2, :frame.shape[1]//2*2, :]

            # BGR画像をaiortc用VideoFrameへ変換（内部でYUV420に自動変換される）
            video_frame = VideoFrame.from_ndarray(frame, format="bgr24")

            # RTP/RTCP用のpts, time_baseをセット（順序保証＆シンク用; VideoStreamTrackに用意されている）
            video_frame.pts, video_frame.time_base = await self.next_timestamp()
            return video_frame  # 正常時：フレーム送信

        except Exception as e:
            print("[ERROR] Exception in CameraTrack.recv:", repr(e))
            # 例外発生時やキューが空の時は全黒画像で代用し、ストリーム切れを防ぐ
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            video_frame.pts, video_frame.time_base = await self.next_timestamp()
            return video_frame

class DirectoryInfo:
    def __init__(self, output_dir, output_prefix, debug_images_subdir_name):
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.date_dir = os.path.join(output_dir, datetime.now().strftime("%Y-%m-%d"))
        self.debug_images_dir = os.path.join(output_dir, debug_images_subdir_name)

    def makedir(self, debug_mode):
    # 出力ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.date_dir, exist_ok=True)
        # デバッグディレクトリの作成
        if debug_mode:
            os.makedirs(self.debug_images_dir, exist_ok=True)
            print(f"デバッグモード有効: 画像を {self.debug_images_dir} に保存")

class Detection:
    """
    オブジェクト検出結果を保持するクラス。
    属性としてバウンディングボックス（[x, y, w, h]）、カテゴリ、信頼度（conf）を持つ。
    """
    def __init__(self, box, category, conf):
        """
        Detectionインスタンスを構築。

        Args:
            box (list or tuple): [x, y, w, h]形式のバウンディングボックス座標
            category (int or str): 検出クラスIDまたはラベル
            conf (float): 信頼度（スコア）
        """
        self.category = category
        self.conf = conf
        self.box = box

    def get_center(self):
        """
        バウンディングボックスの中心座標を取得。

        Returns:
            (int, int): (center_x, center_y)
        """
        x, y, w, h = self.box
        return (x + w//2, y + h//2)

    def get_box_height(self):
        """
        バウンディングボックスの高さを返す
        Returns: int
        """
        return self.box[3]

    @staticmethod
    def parse_detections(metadata, parameters, intrinsics, imx500, picam2):
        """
        AIモデル推論出力からDetectionオブジェクトリストを生成。

        モデル出力に応じて前処理・後処理を分岐（NanoDet系とSSD系など）。
        一定の信頼度を超える「人物」検出のみ抽出し、最終的にバウンディングボックス高さによるフィルタも適用。

        Args:
            metadata: モデル出力メタデータ。
            parameters: 推論パラメータ設定（しきい値、最大検出数、クラスIDなど）。
            intrinsics: モデル・カメラ内部パラメータ（後処理種別も含む）。
            imx500: カメラorAI推論デバイスの抽象インターフェース。
            picam2: カメラデバイスハンドル（実座標変換用）。

        Returns:  list[Detection]: フレーム内の人物を表すDetectionインスタンス配列。
        """
        try:
            np_outputs = imx500.get_outputs(metadata, add_batch=True)
            if np_outputs is None:
                return []  # 検出結果が無い場合は空リスト

            # ネットワーク入力画像サイズ取得
            input_w, input_h = imx500.get_input_size()
            if intrinsics.postprocess == "nanodet":
                # NanoDet系独自の後処理
                boxes, scores, classes = \
                    postprocess_nanodet_detection(np_outputs[0],
                                                  parameters.detection_threshold,
                                                  parameters.iou_threshold,
                                                  parameters.max_detections)[0]
                # 出力ボックス: [x1, y1, x2, y2] → [x, y, w, h]形式に変換
                boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
                boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]

            else:  # SSDなどの汎用系
                # SSD等の出力形式は [ymin, xmin, ymax, xmax]
                # 正規化解除等はimx500.convert_inference_coordsで対応
                boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

                # ▼ Box形状調整（(N,4) でなければreshape）
                if boxes.shape[1] != 4:
                    boxes = np.array(list(zip(*np.array_split(boxes, 4, axis=1))))
                # ▼ スコアしきい値によるフィルタ
                mask = scores > parameters.detection_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                classes = classes[mask]

                # [ymin, xmin, ymax, xmax]→[xmin, ymin, xmax, ymax] へ
                if boxes.shape[0] > 0:
                    boxes_xyxy = boxes.copy()
                    boxes_xyxy = boxes_xyxy[:, [1, 0, 3, 2]]
                    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
                    scores_tensor = torch.tensor(scores, dtype=torch.float32)
                    keep = nms(boxes_tensor, scores_tensor, parameters.iou_threshold)  # NMS適用
                    # NMS適用後のインデックスに合わせてフィルタ
                    boxes = boxes[keep.numpy()]
                    scores = scores[keep.numpy()]
                    classes = classes[keep.numpy()]
                else:
                    # 検出が0件のときは空で返す
                    boxes = np.empty((0,4), dtype=float)
                    scores = np.empty((0,), dtype=float)
                    classes = np.empty((0,), dtype=float)

            # 人物クラスかつ閾値超えの検出のみ残す
            selected = [
                (box, score, category)
                for box, score, category in zip(boxes, scores, classes)
                if score > parameters.detection_threshold and int(category) == parameters.person_class_id
            ]
            # box座標の変換およびDetectionへの格納
            detections = [
                Detection(
                    imx500.convert_inference_coords(box, metadata, picam2),
                    category, score
                )
                for box, score, category in selected
            ]
            # フィルタ: 最小ボックス高さ未満・最大ボックス高さ超過の検出を除去
            detections = [
                det for det in detections
                if parameters.min_box_height <= det.box[3] <= parameters.max_box_height
            ]
            return detections
        
        except Exception as e:
            print(f"検出処理エラー: {e}")
            import traceback
            traceback.print_exc()
            return []  # エラー時は空リストを返却

class Person:
    """
    複数フレームにまたがって人物情報を管理するクラス。
    各Personインスタンスは固有ID、バウンディングボックス、出現・消滅時刻、軌跡情報等、カルマンフィルタ予測位置、移動履歴などを保持。
    """
    next_id = 0  # クラス変数。新規インスタンスごとに自動的にIDを割り当てるカウンタ
    
    # カルマンフィルタ関連パラメータ
    # 初期共分散行列の倍率（初期状態の不確実性）
    KALMAN_INITIAL_COVARIANCE = 10.0    # 初期位置から観測位置まで距離があるため、初期共分散の重要性は低い→調整不要！固定値でOK

    # 観測ノイズ : ピクセル単位、検出精度に依存
    KALMAN_OBSERVATION_NOISE = 15.0     # 使用するモデル依存のため、設置現場ごとに調整の予定はない
    """ 調整指針:
        5.0～15.0: 高品質なYOLO、SSD等
        15.0～30.0: 一般的な検出器（推奨範囲）
    """

    # プロセスノイズ : 小さい値：モデル重視、大きい値：観測重視
    KALMAN_PROCESS_NOISE = 0.3          # 今回用途が通路に対する人流カウントのため、概ね予測可能な動きのハズ。大きく変更の必要性はない
    """ 調整指針:
        0.1～0.5: 予測可能な動き（大人の歩行等）
        0.5～1.0: 一般的な用途（推奨範囲）
        1.0～   : 急な方向転換など非常に予測困難な動き場合
    """

    def __init__(self, box):
        """
        Personインスタンスを初期化し、ユニークIDを付与。
        カルマンフィルタも初期化。

        Args: box (list or tuple): [x, y, w, h]形式のバウンディングボックス
        """
        self.id = Person.next_id      # 一意なIDを割り振り
        Person.next_id += 1

        self.box = box                          # 最新バウンディングボックス
        self.trajectory = [self.get_center()]   # tracking用: 中心座標履歴（初期値は現フレーム）
        self.first_seen = time.time()           # 初回検出時刻
        self.last_seen = time.time()            # 最終検出時刻（trackingロスト検出等に使用）
        self.crossed_direction = None           # 線をまたいだ向き（用途次第・初期値None）
        self.lost_start_time = None             # トラッキングロストが始まった時刻
        self.lost_last_box = None               # ロスト時の最後のバウンディングボックス

        # Kalmanフィルタ初期化（人物の位置・速度を推定するため）
        cx, cy = self.get_center()
        
        # 4次元状態ベクトル（位置x,y + 速度dx,dy）、2次元観測ベクトル（位置x,yのみ観測可能）
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # 初期状態: [x座標, y座標, x方向速度, y方向速度]
        # 速度は初期値0で設定（静止状態から開始と仮定）
        self.kf.x = np.array([cx, cy, 0, 0])  
        
        # 状態遷移行列F: 等速直線運動モデル
        # 次の状態 = 現在位置 + 速度×時間（時間間隔=1フレーム）
        # x(t+1) = x(t) + dx(t), y(t+1) = y(t) + dy(t)
        # dx(t+1) = dx(t), dy(t+1) = dy(t) （速度は一定と仮定）
        self.kf.F = np.array([
            [1,0,1,0],  # x(t+1) = x(t) + dx(t)
            [0,1,0,1],  # y(t+1) = y(t) + dy(t)
            [0,0,1,0],  # dx(t+1) = dx(t)
            [0,0,0,1]   # dy(t+1) = dy(t)
        ])
        
        # 観測行列H: 状態ベクトルから観測可能な値を抽出
        # 位置(x,y)のみ観測可能、速度は直接観測できない
        self.kf.H = np.array([
            [1,0,0,0],  # 観測値1 = x座標
            [0,1,0,0]   # 観測値2 = y座標
        ])
        
        # 初期共分散行列P: 初期状態の不確実性
        self.kf.P *= self.KALMAN_INITIAL_COVARIANCE
        
        # 観測ノイズ共分散行列R: 検出器の精度に依存
        self.kf.R = np.eye(2) * self.KALMAN_OBSERVATION_NOISE
        
        # プロセスノイズ共分散行列Q: モデルの不完全性を表現
        self.kf.Q = np.eye(4) * self.KALMAN_PROCESS_NOISE

    def get_center(self):
        """
        現在のバウンディングボックス中心座標を取得。

        Returns: (int, int): (中心x, 中心y)
        """
        x, y, w, h = self.box
        return (int(x + w//2), int(y + h//2))

    def predict(self):
        """
        カルマンフィルタによる次フレーム位置予測。戻り値は推定中心座標(int, int)。

        Returns: (int, int): 予測中心座標 (center_x, center_y)
        """
        self.kf.predict()
        pred_cx, pred_cy = self.kf.x[:2]
        return (int(pred_cx), int(pred_cy))

    def update(self, box):
        """
        新しい検出ボックス情報で人物情報を更新し、カルマンフィルタにも観測値を反映。

        Args: box (list or tuple): [x, y, w, h]形式のバウンディングボックス
        """
        self.box = box
        obs_cx, obs_cy = self.get_center()
        self.kf.update([obs_cx, obs_cy])
        self.trajectory.append((obs_cx, obs_cy))  # 軌跡に現フレーム中心値を追加

        # 履歴数制限：最大30件のみ保持（古い順にpopで削除）
        if len(self.trajectory) > 30:
            self.trajectory.pop(0)
        self.last_seen = time.time()               # 最終確認時刻を更新

    def get_predicted_box(self):
        """
        カルマンフィルタによる予測中心・直近のサイズでバウンディングボックスを算出。

        Returns: [x, y, w, h] 形式の予測バウンディングボックス
        """
        pred_cx, pred_cy = self.kf.x[:2]
        w, h = self.box[2], self.box[3]
        pred_x = int(pred_cx - w//2)
        pred_y = int(pred_cy - h//2)
        return [pred_x, pred_y, w, h]

    def get_avg_motion(self, window=None):
        """
        軌跡中の移動平均ベクトルを計算。

        Args: window (int or None): 最新何フレーム分で平均算出するか。Noneなら全履歴を対象。
        Returns: (float, float): 移動平均ベクトル (Δx, Δy)
        """
        centers = self.trajectory
        if len(centers) < 2:
            return (0, 0)  # 十分な履歴がない場合はゼロベクトル

        # window指定時は、最新window件だけを対象に平均を計算
        if window is not None and len(centers) > window:
            centers = centers[-window:]

        # 各フレーム間のx, y移動量
        dxs = [centers[i+1][0] - centers[i][0] for i in range(len(centers)-1)]
        dys = [centers[i+1][1] - centers[i][1] for i in range(len(centers)-1)]

        # 移動ベクトルの平均値
        avg_dx = sum(dxs) / len(dxs)
        avg_dy = sum(dys) / len(dys)
        return (avg_dx, avg_dy)

    def get_box_height(self):
        """
        現在のバウンディングボックスの高さを返す
        Returns: int
        """
        return self.box[3]

class PeopleCounter:
    """
    人の通過方向をカウントし、累積/期間ごとの人数カウント管理・データ保存を担当するクラス。
    """
    def __init__(self, directoryInfo):
        self.right_to_left = 0          # 右→左の期間カウンタ
        self.left_to_right = 0          # 左→右の期間カウンタ
        self.total_right_to_left = 0    # 右→左の累積カウンタ
        self.total_left_to_right = 0    # 左→右の累積カウンタ
        self.start_time = time.time()   # カウント開始時刻
        self.last_save_time = time.time()   # 最後に保存した時刻
        self.directoryInfo = directoryInfo  # 保存パス

    def update(self, direction):
        """
        人物の移動方向ごとに各カウンタをインクリメント。

        Args: direction (str): "right_to_left" または "left_to_right"
        """
        if direction == "right_to_left":
            self.right_to_left += 1
            self.total_right_to_left += 1
        elif direction == "left_to_right":
            self.left_to_right += 1
            self.total_left_to_right += 1

    def get_counts(self):
        """
        期間中（最後に保存してから現在まで）の人数カウントを取得。

        Returns: dict: {"right_to_left": n, "left_to_right": m}
        """
        return {
            "right_to_left": self.right_to_left,
            "left_to_right": self.left_to_right,
        }

    def get_total_counts(self):
        """
        累積の人数カウントを取得。

        Returns: dict: {"right_to_left": n_total, "left_to_right": m_total}
        """
        return {
            "right_to_left": self.total_right_to_left,
            "left_to_right": self.total_left_to_right,
        }

    def save_to_json(self):
        """
        カウントデータをJSONファイルに保存し、期間カウンタをリセットする。
        累積値はリセットされない。

        Returns: bool: True=保存成功, False=保存失敗
        """
        current_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        data = {
            "timestamp": timestamp,
            "duration_seconds": int(current_time - self.last_save_time),  # 今回区間の計測時間
            "period_counts": self.get_counts(),                           # 今回区間のカウント
            "total_counts": self.get_total_counts()                       # 累積カウント
        }

        # 保存ファイルパス構築
        filename = os.path.join(
            self.directoryInfo.date_dir,
            f"{self.directoryInfo.output_prefix}_{timestamp}.json"
        )

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)

            print(f"Data saved to {filename}")

            # 期間用カウンタのみリセット。累積値はそのまま。
            self.right_to_left = 0
            self.left_to_right = 0
            self.last_save_time = current_time
            return True
        
        except Exception as e:
            print(f"Failed to save data to {filename}: {e}")
            return False  # 保存失敗時

class Camera:
    def __init__(self, picam2, imx500):
        self.picam2 = picam2
        self.imx500 = imx500

class PeopleFlowManager:
    """
    人物検出・追跡・人数カウント・ライン横断判定・可視化の全体管理クラス。
    1フレームごとに検出〜追跡〜カウント〜描画〜ログ〜定期保存までを一元管理する。
    """

    def __init__(self, config, loop, counter, directoryInfo, intrinsics, camera, parameters):
        """
        クラス各種ハンドル・設定値を初期化。

        Args:
            config: アプリ全体設定
            counter: 人数カウンタ（PeopleCounterインスタンス）
            directoryInfo: 保存ディレクトリ・出力プリフィックス等の設定
            intrinsics: カメラ・モデルの内部パラメータ
            camera: カメラデバイスへのアクセス用インスタンス
            parameters: 推論・トラッキング・カウント関連パラメータ
        """
        self.active_people = []     # 現在追跡中の人物リスト
        self.lost_people = []       # 一時追跡ロスト中の人物リスト
        self.counter = counter
        self.directoryInfo = directoryInfo
        self.last_log_time = time.time()
        self.config = config
        self.loop = loop
        self.intrinsics = intrinsics
        self.camera = camera
        self.parameters = parameters
        self.image_saved = False    # 起動時の一度きりの画像保存用
        self.running = True
        self.render_queue = queue.Queue(maxsize=5)
        self.frame_skip_counter = 0
        self.render_skip_rate = 3

        # スレッドを起動
        self.render_thread = threading.Thread(
            target=self._start_render_worker, 
            daemon=True
            )
        self.render_thread.start()
        print("PeopleFlowManager initialized successfully")
    
    def _start_render_worker(self):
        """レンダーワーカーのエントリーポイント"""
        self._render_worker()
    
    def _render_worker(self):
        """別スレッドでレンダリング処理を実行"""
        print("Render worker started")
        while self.running:
            try:
                render_data = self.render_queue.get(timeout=1.0)
                if render_data is None:  # 終了シグナル
                    break
                self._render_frame(render_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Render worker error: {e}")
                import traceback
                traceback.print_exc()
        print("Render worker stopped")

    @staticmethod
    def _calculate_iou(box1, box2):
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
        Args:
            box1 (list): [x, y, w, h]
            box2 (list): [x, y, w, h]

        Returns: float: IoU値（0.0〜1.0）
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

    def _track_people(self, active_people, lost_people, detections, frame_id, center_line_x):
        """
        検出結果（detections）と既存追跡リスト（active_people）をマッチングし更新。
        ハンガリアン法＋IoU＋中心距離＋方向ペナルティで追跡管理。
        一時的なロスト復帰、完全ロスト削除も管理。

        Args:
            active_people (list): 現在追跡中の人物リスト
            lost_people (list): 一時ロスト中人物リスト
            detections (list[Detection]): 今フレームの検出結果リスト
            frame_id: フレームID
            center_line_x: ラインカウント座標

        Returns: tuple: (new_active_people, new_lost_people)
        """
        num_people = len(active_people)
        num_detections = len(detections)

        # 検出結果も追跡対象もいない場合はそのまま返す
        if num_detections == 0 and num_people == 0:
            return [], lost_people

        # 新しい検出結果がない場合、既存の追跡対象は維持（ただし後にタイムアウトで削除される）
        if num_detections == 0:
            return active_people, lost_people

        # 追跡対象がいない場合、全ての検出を新しい人物とする
        if num_people == 0:
            return [Person(det.box) for det in detections], lost_people

        # コスト行列を作成
        # 行: active_people, 列: detections
        # コストは小さいほど良い。マッチング不可能なペアには大きな値 (inf) を設定
        cost_matrix = np.full((num_people, num_detections), np.inf)

        # コスト行列を計算するループ
        # 既存の追跡ターゲット（active_people）と新たな検出結果（detections）との間で、
        # 各組み合わせペアごとに「重なり具合（IoU）」と「中心間距離」を算出し、総合的なコストを定義。
        for i, person in enumerate(active_people):
            # ------- カルマンフィルタによる次フレーム位置予測を実行 -------
            # この呼び出しで Kalman フィルタの内部状態 (self.kf.x) が次フレームの状態に更新される
            # この戻り値が、次のフレームで人物がいると予測される中心座標
            predicted_center = person.predict()

            # ------- 予測ボックスを取得 -------
            # get_predicted_box は、predict() で更新された self.kf.x を参照して予測ボックスを作成する
            predicted_box = person.get_predicted_box()

            # ------- 検出結果（detections）とコスト計算 -------
            for j, detection in enumerate(detections):
                detection_box = detection.box
                # 検出ボックスの中心座標（x, y）を計算
                detection_center = (
                    detection_box[0] + detection_box[2] // 2,   # 検出boxの中心x
                    detection_box[1] + detection_box[3] // 2    # 検出boxの中心y
                )

                # --- 距離・IoU計算 ---
                # 【距離】予測中心点と検出中心点とのユークリッド距離（ピクセル単位）
                distance = np.sqrt(
                    (predicted_center[0] - detection_center[0]) ** 2 +
                    (predicted_center[1] - detection_center[1]) ** 2
                )
                # 【IoU】予測ボックスと検出boxのIoU（重なり率：0~1）
                iou = self._calculate_iou(predicted_box, detection_box)
                # --- 総合コストの定義 ---
                # 距離が近くIoUが大きい（よく重なっている）ほどコストが小さくなるよう設計
                # → IoU大・距離小の組み合わせほどcostは小さい（良いマッチングと見なされる）
                # 例: 距離200px以上はコスト+1, IoU 1.0→加点ゼロ, IoU 0→+1
                # ------- コスト計算 -------
                cost = (1.0 - iou) + (distance / 200.0)  # 200pxを「最大許容距離」とするスケーリング

                # costにX軸方向の一貫性重視ペナルティ
                avg_motion = person.get_avg_motion()
                avg_motion_x = avg_motion[0]
                intended_dir = np.sign(avg_motion_x)
                actual_dir = np.sign(detection_center[0] - person.trajectory[-1][0])
                if intended_dir != 0 and actual_dir != 0 and intended_dir != actual_dir:
                    cost += self.parameters.direction_mismatch_penalty  # 逆方向へのマッチにペナルティを追加

                cost_matrix[i, j] = cost  # コスト行列の値を格納


        # コスト行列の全要素がinf or どの行or列も全てinfならreturn
        if (
            np.all(np.isinf(cost_matrix)) or 
            np.any(np.all(np.isinf(cost_matrix), axis=0)) or 
            np.any(np.all(np.isinf(cost_matrix), axis=1)) or
            np.sum(np.isfinite(cost_matrix)) < max(cost_matrix.shape)   # 有限値の要素数 < 行or列の大きいほう（マッチングに必要な最小数）なら諦める
        ):
            # print("Assignment infeasible: some row or column is all inf.")
            return active_people, lost_people
        else:
            # ハンガリアンアルゴリズムを実行し、最適なマッチングを見つける
            # matched_person_indices: active_peopleのインデックスの配列
            # matched_detection_indices: detectionsのインデックスの配列
            try:
                matched_person_indices, matched_detection_indices = linear_sum_assignment(cost_matrix)

                # マッチング結果を処理
                new_people = []
                # マッチした検出結果のインデックスを記録 (初期マッチングで使用されたもの)
                used_detections = set()
                used_person = set()

                # コストが高すぎる場合は不一致とみなす
                # マッチした人物を更新して新しいリストに追加
                for i, j in zip(matched_person_indices, matched_detection_indices):
                    # コストがinfの場合は有効なマッチではないのでスキップ (linear_sum_assignmentはinfも考慮する)
                    if cost_matrix[i, j] < self.parameters.max_acceptable_cost:
                        person = active_people[i]
                        detection = detections[j]
                        person.update(detection.box)
                        new_people.append(person)
                        used_person.add(i)       # この人物はマッチに使用された
                        used_detections.add(j)   # *** この検出結果は初期マッチに使用された ***
                    else:
                        # コストが高すぎる場合はマッチとして使用しない
                        pass # 不一致

                # マッチしなかったactive_peopleを一時ロストの候補とし、lost_peopleリストへ追加
                now = time.time()
                for i, person in enumerate(active_people):
                    if i not in used_person:
                        person.lost_start_time = now        # ロスト開始時刻を記録
                        person.lost_last_box = person.box   # ロスト時のボックスを記録
                        lost_people.append(person)          # lost_peopleリストに追加

                # lost_peopleからも「復帰」判定！
                recovered = []
                # ここでは全ての検出結果を見るが、すでにused_detectionsにあるものは復帰には使われないように制御する
                for j, detection in enumerate(detections): # enumerateでインデックスjも取得
                    if j in used_detections:
                        continue # この検出結果は既に初期マッチングで使用されたので、復帰には使わない

                    # 各ロスト人物に対して復帰可能かチェック
                    for lost_person in lost_people:
                        # 一度復帰したlost_personは再度このフレームで復帰させない → recoveredリストで管理
                        if lost_person in recovered:
                            continue

                        if lost_person.crossed_direction is not None:
                            continue # このlost_personは既にカウント済みなのでスキップ

                        # ---- 復帰判定の各種条件 ----
                        # lost_person（見失った人）の中心座標取得
                        lost_cx, _ = lost_person.get_center()
                        det_cx, _ = detection.get_center()
                        # Kalman Filterの推定速度
                        avg_dx, avg_dy = lost_person.kf.x[2], lost_person.kf.x[3]
                        # x方向の検出位置のズレ
                        diff_x = det_cx - lost_cx

                        # lost_personの移動方向と、新たな検出位置の方向が一致しているか？
                        same_direction = (avg_dx * diff_x > 0)

                        # ボックス高さの近似条件
                        lost_height = lost_person.get_box_height()
                        det_height = detection.get_box_height()
                        if lost_height == 0:  # ゼロ除算対策
                            height_ratio = 0
                        else:
                            height_ratio = det_height / lost_height
                        HEIGHT_SIMILARITY_THRESHOLD = 0.95  # 許容する割合
                        height_similar = (1.0 - HEIGHT_SIMILARITY_THRESHOLD) <= height_ratio <= (1.0 + HEIGHT_SIMILARITY_THRESHOLD)

                        # --- 中心線からの位置条件を移動方向に応じて判定 ---
                        is_on_correct_side_of_line_within_margin = False
                        margin = self.parameters.center_line_margin_px

                        if center_line_x is not None: # 中心線が有効な場合のみ判定
                            if avg_dx > 0: # 右方向に移動している場合 (手前側は左側)
                                # lost_cxが [center_line_x - margin, center_line_x] の範囲内にあるか
                                if center_line_x - margin <= lost_cx <= center_line_x:
                                    is_on_correct_side_of_line_within_margin = True
                            elif avg_dx < 0: # 左方向に移動している場合 (手前側は右側)
                                # lost_cxが [center_line_x, center_line_x + margin] の範囲内にあるか
                                if center_line_x <= lost_cx <= center_line_x + margin:
                                    is_on_correct_side_of_line_within_margin = True

                        # ── 以下の条件をすべて満たす場合に復帰可能 ──
                        now_check = time.time() # ここで最新時刻を取得
                        if (
                            center_line_x and   # 中心線が有効か
                            is_on_correct_side_of_line_within_margin and                                        # 中心線から一定範囲（ピクセル）以内か
                            abs(diff_x) < self.parameters.recovery_distance_px and                              # 失った位置と検出位置が近いか
                            now_check - lost_person.lost_start_time < self.parameters.active_timeout_sec and    # 見失ってからの秒数が規定以内か
                            same_direction and  # 進行方向も一致しているか
                            height_similar      # ボックス高さの類似度
                        ):
                            # --- 復帰処理 ---
                            print(f"recovered:{frame_id}/{lost_person.id}")
                            lost_person.update(detection.box)       # lost_personの情報を新しい検出で更新
                            new_people.append(lost_person)          # 新規・復帰リストに追加 (既存のPersonオブジェクト)
                            recovered.append(lost_person)           # 復帰済みリストに追加
                            used_detections.add(j)                  # *** この検出結果は復帰に使用されたので、新規人物作成には使わない！ ***
                            break                                   # この検出結果で1つのロスト人物が復帰したら、他のロスト人物との比較は終了

                # 復帰したlost_peopleをlost_peopleリストから除く
                lost_people = [p for p in lost_people if p not in recovered]

                # 完全ロスト判定
                # ロスト開始時刻から規定時間以上経過した人物をlost_peopleから除く
                now_final = time.time() # 最終的な時刻チェック
                lost_people = [p for p in lost_people if now_final - p.lost_start_time < self.parameters.active_timeout_sec]


                # マッチしなかった (used_detectionsに含まれていない) 新しい検出結果を新しい人物としてリストに追加
                # ここでは、初期マッチングにも復帰にも使われなかった検出結果のみが対象となる
                for j, detection in enumerate(detections):
                    if j not in used_detections:                    #  used_detectionsには初期マッチと復帰に使用された検出結果が含まれている
                        new_people.append(Person(detection.box))    # 新しい人物を作成 (新しいPersonオブジェクト)

                # new_people の中身は
                # 1. 初期マッチングで更新された active_people (既存オブジェクト)
                # 2. ロストから復帰した lost_people (既存オブジェクト)
                # 3. どの既存人物とも対応付けられなかった新規検出 (新しいオブジェクト)
                return new_people, lost_people

            except Exception as e:
                print("【Error】ハンガリアン法(マッチング)で例外発生。")
                print(f"例外内容：{e}")
                print(f"フレームID: {frame_id}")
                print(f"コスト行列 shape: {cost_matrix.shape}")
                print(f"コスト行列の一部: {cost_matrix}")
                print(f"検出結果一覧: {detections}")
                print(f"追跡対象一覧: {active_people}")
                return active_people, lost_people

    def _check_line_crossing(self, person, center_line_x, frame=None):
        """中央ラインを横切ったかチェック"""
        if len(person.trajectory) < 2:
            return None

        if person.crossed_direction is not None:
            return None

        for i in range(1, len(person.trajectory)):
            prev_x = person.trajectory[i-1][0]
            curr_x = person.trajectory[i][0]
            # 左→右: 中央線を未満→以上で通過
            if min(prev_x, curr_x) < center_line_x <= max(prev_x, curr_x):
                # ラインをどちら方向にまたいだか判定
                if prev_x < center_line_x:
                    person.crossed_direction = "left_to_right"
                    # デバッグモードで画像を保存
                    if self.parameters.debug_mode and frame is not None:
                        modules.save_debug_image(frame, person, center_line_x, "left_to_right", self.directoryInfo.debug_images_dir, self.directoryInfo.output_prefix)

                    return "left_to_right"
                # 右→左: 中央線を以上→未満で通過
                else:
                    person.crossed_direction = "right_to_left"
                    # デバッグモードで画像を保存
                    if self.parameters.debug_mode and frame is not None:
                        modules.save_debug_image(frame, person, center_line_x, "right_to_left", self.directoryInfo.debug_images_dir, self.directoryInfo.output_prefix)

                    return "right_to_left"

    def process_frame(self, request):
        """フレームごとの処理を行うコールバック関数"""
        # フレームデータとメタデータの取得
        frame, metadata, frame_id = self._extract_frame_data(request)
        start_time = time.time()
        # 検出処理
        detection_start = time.time()
        detections = self._get_detections(metadata)
        detection_time = time.time() - detection_start
    
        # 人物追跡の更新
        frame_height, frame_width = frame.shape[:2]
        center_line_x = frame_width // 2
        
        tracking_start = time.time()
        self._update_tracking(detections, frame_id, center_line_x)
        tracking_time = time.time() - tracking_start

        # フレーム落ち判定（例：33ms = 30FPS基準）
        frame_threshold = 0.033  # 30FPSの場合
        total_time = time.time() - start_time
        
        if detection_time > frame_threshold:
            print(f"[WARNING] Detection処理でフレーム落ち発生: {detection_time:.3f}s (frame_id: {frame_id})")
        
        if tracking_time > frame_threshold:
            print(f"[WARNING] Tracking処理でフレーム落ち発生: {tracking_time:.3f}s (frame_id: {frame_id})")
        
        if total_time > frame_threshold:
            print(f"[WARNING] 全体処理でフレーム落ち発生: {total_time:.3f}s (frame_id: {frame_id})")
        

        # レンダリング用データをキューに追加（非ブロッキング）
        self._queue_render_data(request, frame_height, frame_width, center_line_x, frame_id)

        # ライン横断チェックとカウント更新
        self._process_line_crossings(center_line_x, frame)

        # 定期処理（ログ出力、データ保存、古いトラッキング削除）は別スレッドで実行
        threading.Thread(target=self._handle_periodic_tasks, daemon=True).start()

    def _queue_render_data(self, request, frame_height, frame_width, center_line_x, frame_id):
        """レンダリング用データをキューに追加"""
        self.frame_skip_counter += 1
        
        # フレームスキップ制御
        if self.frame_skip_counter % self.render_skip_rate == 0:
            # active_peopleのスナップショットを作成
            active_people_snapshot = []
            for person in self.active_people:
                person_copy = type('Person', (), {})()  # オブジェクトコピー
                person_copy.id = person.id
                person_copy.box = person.box
                person_copy.trajectory = person.trajectory.copy() if hasattr(person, 'trajectory') else []
                person_copy.crossed_direction = getattr(person, 'crossed_direction', None)
                active_people_snapshot.append(person_copy)
            
            render_data = {
                'request': request,
                'frame_height': frame_height,
                'frame_width': frame_width,
                'center_line_x': center_line_x,
                'frame_id': frame_id,
                'active_people': active_people_snapshot,
                'counter_snapshot': self.counter.get_total_counts().copy(),
                'image_saved': self.image_saved
            }
            
            try:
                self.render_queue.put_nowait(render_data)
            except queue.Full:
                # キューが満杯の場合は古いデータを破棄
                try:
                    self.render_queue.get_nowait()
                    self.render_queue.put_nowait(render_data)
                except:
                    pass

    def _start_render_worker(self):
        """レンダーワーカーのエントリーポイント"""
        self._render_worker()
    
    def _render_worker(self):
        """別スレッドでレンダリング処理を実行"""
        print("Render worker started")
        while self.running:
            try:
                render_data = self.render_queue.get(timeout=1.0)
                if render_data is None:  # 終了シグナル
                    break
                self._render_frame(render_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Render worker error: {e}")
                import traceback
                traceback.print_exc()
        print("Render worker stopped")


    def _extract_frame_data(self, request):
        """フレームデータとメタデータを抽出"""
        with MappedArray(request, 'main') as m:
            frame = m.array.copy()
        # メタデータを取得
        metadata = request.get_metadata()
        # SensorTimestampをframe_idに利用
        frame_id = metadata.get('SensorTimestamp') if metadata else None
        
        return frame, metadata, frame_id

    def _get_detections(self, metadata):
        """検出処理を実行"""
        if metadata is None:
            # print("メタデータがNoneです") # デバッグ用
            # メタデータがない場合でも、既存のactive_peopleはタイムアウトで削除する必要があるため処理を進める
            # ただし検出処理はスキップ
            return []
        
        # 検出処理
        return Detection.parse_detections(
            metadata, 
            self.parameters, 
            self.intrinsics, 
            self.camera.imx500, 
            self.camera.picam2
        )

    def _update_tracking(self, detections, frame_id, center_line_x):
        """人物追跡を更新"""
        self.active_people, self.lost_people = self._track_people(
            self.active_people, 
            self.lost_people, 
            detections, 
            frame_id, 
            center_line_x
        )
        
        # デバッグ用チェック
        if not isinstance(self.active_people, list):
            print(f"track_people returned : {type(self.active_people)}")

    def _render_frame(self, render_data):
        """非同期でフレームに描画処理を実行"""
        try:
            # _render_frameの処理を非同期化
            request         = render_data['request']
            frame_height    = render_data['frame_height']
            frame_width     = render_data['frame_width']
            center_line_x   = render_data['center_line_x']
            frame_id        = render_data['frame_id']
            
            with MappedArray(request, 'main') as m:
                # 起動時の画像を一度だけ保存
                if not render_data['image_saved'] and not self.image_saved:
                    self._handle_startup_image_save(m.array, center_line_x)
                    self.image_saved = True
                    
                # 描画
                self._draw_center_lines(m.array, center_line_x, frame_height)
                
                # active_peopleのスナップショットを使用して描画
                self._draw_people_tracking(m.array, render_data['active_people'])
                
                # カウント情報の描画（スナップショットを使用）
                self._draw_count_info(m.array, frame_id, render_data['counter_snapshot'])
                
                # WebRTC配信（3フレームに1回配信）
                if self.frame_skip_counter % 3 == 0:
                    self._handle_webrtc_streaming(m.array)
                    
        except Exception as e:
            print(f"Render error: {e}")

    def _handle_startup_image_save(self, array, center_line_x):
        """起動時の画像保存処理"""
        modules.save_image_at_startup(
            array, 
            center_line_x, 
            self.directoryInfo.date_dir, 
            self.directoryInfo.output_prefix
        )

    def _draw_center_lines(self, array, center_line_x, frame_height):
        """中央ラインとマージンラインを描画"""
        # 中央ライン
        cv2.line(array, (center_line_x, 0), (center_line_x, frame_height), (255, 255, 0), 2)
        
        # CENTER_LINE_MARGINを描画
        margin = self.parameters.center_line_margin_px
        left_margin_x = center_line_x - margin
        right_margin_x = center_line_x + margin
        
        cv2.line(array, (left_margin_x, 0), (left_margin_x, frame_height), (0, 128, 255), 2)
        cv2.line(array, (right_margin_x, 0), (right_margin_x, frame_height), (0, 128, 255), 2)

    def _draw_people_tracking(self, array, active_people_snapshot):
        """人物の検出ボックスと軌跡を描画"""
        for person in active_people_snapshot:
            color = self._get_person_color(person)
            self._draw_person_box(array, person, color)
            self._draw_person_trajectory(array, person, color)

    def _get_person_color(self, person):
        """人物の方向に基づいて色を決定"""
        color_map = {
            "left_to_right": (0, 128, 0),    # 緑
            "right_to_left": (0, 0, 255),    # 赤
        }
        return color_map.get(person.crossed_direction, (255, 255, 255))  # デフォルト: 白

    def _draw_person_box(self, array, person, color):
        """人物の検出ボックスと情報を描画"""
        x, y, w, h = person.box
        
        # 検出ボックス
        cv2.rectangle(array, (x, y), (x + w, y + h), color, 2)
        
        # ID表示
        cv2.putText(array, f"ID: {person.id}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # ボックスの高さ表示
        cv2.putText(array, f"H: {int(h)}", (x, y + h + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_person_trajectory(self, array, person, color):
        """人物の軌跡を描画"""
        if len(person.trajectory) > 1:
            for i in range(1, len(person.trajectory)):
                cv2.line(array, person.trajectory[i-1], person.trajectory[i], color, 2)

    def _draw_count_info(self, array, frame_id, counter_snapshot):
        """カウント情報と時刻を描画"""
        # カウント情報
        cv2.putText(array, f"right_to_left: {counter_snapshot['right_to_left']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(array, f"left_to_right: {counter_snapshot['left_to_right']}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        
        # 時刻とフレームID
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text_str = f"FrameID: {frame_id} / {timestamp}"
        cv2.putText(array, text_str, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def _handle_webrtc_streaming(self, array):
        """WebRTC配信用のフレーム処理"""
        frame_to_send = self._prepare_streaming_frame(array)
        
        if not frame_queue.full():
            try:
                frame_queue.put_nowait(frame_to_send)
            except Exception as e:
                print(f"Frame put failed: {e}")

    def _prepare_streaming_frame(self, array):
        """配信用フレームの準備"""
        # カラーフォーマット変換
        if array.shape[2] == 4:
            frame = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        else:
            frame = array.copy()
        
        # 解像度調整（必要に応じて）
        return self._resize_frame_for_streaming(frame)

    def _resize_frame_for_streaming(self, frame):
        """配信用にフレームをリサイズ"""
        original_height, original_width = frame.shape[:2]
        target_width = original_width  # 必要に応じて変更
        target_height = int(original_height * (target_width / original_width))
        
        # 偶数に調整（YUV420p向け）
        target_width = (target_width // 2) * 2
        target_height = (target_height // 2) * 2
        
        if original_width != target_width or original_height != target_height:
            return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return frame

    def _process_line_crossings(self, center_line_x, frame):
        """ライン横断チェックとカウント更新"""
        for person in self.active_people:
            # 少なくとも2フレーム以上の軌跡がある人物が対象
            if len(person.trajectory) >= 2:
                direction = self._check_line_crossing(person, center_line_x, frame)
                if direction:
                    self.counter.update(direction)

    def _handle_periodic_tasks(self):
        """定期処理（ログ出力、データ保存、古いトラッキング削除）"""
        current_time = time.time()
        
        # 古いトラッキング対象を削除 (last_seen が TRACKING_TIMEOUT を超えたもの)
        self._cleanup_old_tracking(current_time)
        
        # 定期ログ出力
        self._handle_periodic_logging(current_time)
        
        # データ保存
        self._handle_periodic_saving(current_time)

    def _cleanup_old_tracking(self, current_time):
        """古いトラッキング対象を削除"""
        self.active_people = [
            p for p in self.active_people 
            if current_time - p.last_seen < self.parameters.tracking_timeout
        ]

    def _handle_periodic_logging(self, current_time):
        """定期的なログ出力"""
        if current_time - self.last_log_time >= self.parameters.log_interval:
            self.last_log_time = current_time
            self._log_status_update(current_time)

    def _log_status_update(self, current_time):
        """ステータス更新ログを出力"""
        total_counts = self.counter.get_total_counts()
        elapsed = int(current_time - self.counter.last_save_time)
        remaining = max(0, int(self.parameters.log_output_interval_sec - elapsed))
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"--- Status Update ({timestamp}) ---")
        print(f"Active tracking: {len(self.active_people)} people")
        print(f"Counts - Period (R->L: {self.counter.right_to_left}, L->R: {self.counter.left_to_right})")
        print(f"Counts - Total (R->L: {total_counts['right_to_left']}, L->R: {total_counts['left_to_right']})")
        print(f"Next save in: {remaining} seconds")
        print("--------------------------------------------------")

    def _handle_periodic_saving(self, current_time):
        """定期的なデータ保存"""
        # 指定間隔ごとにJSONファイルに保存
        if current_time - self.counter.last_save_time >= self.parameters.log_output_interval_sec:
            self.counter.save_to_json()

# ======= メイン処理 =======
def camera_main(stop_event, args, loop):
    # 各種パラメータ設定
    parameters = Parameter(MODEL_PATH)

    print("IMX500 AIカメラモジュールを初期化中...")
    try:
        # モデルファイルパスを指定してIMX500オブジェクトを生成
        imx500 = IMX500(parameters.model_path)
        intrinsics = imx500.network_intrinsics

        # intrinsics（ネットワーク情報）の検証
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"  # デフォルトでオブジェクト検出タスクとする
        elif intrinsics.task != "object detection":
            print("ネットワークはオブジェクト検出タスクではありません", file=sys.stderr)
            sys.exit(1)

        # ラベル未設定時はCOCOデータセット用のラベルをロード
        if intrinsics.labels is None:
            try:
                # assets/coco_labels.txtのパスを決定（実行ディレクトリからの相対パス）
                label_path = os.path.join(os.path.dirname(__file__), "assets/coco_labels.txt")
                with open(label_path, "r") as f:
                    intrinsics.labels = f.read().splitlines()
            except FileNotFoundError:
                print("assets/coco_labels.txt が見つかりません。デフォルトのCOCOラベルを使用します。", file=sys.stderr)
                # 最低限のデフォルトラベル
                intrinsics.labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]

        # 足りない項目はデフォルト値で補う
        intrinsics.update_with_defaults()

        print("Picamera2カメラを初期化中...")
        picam2 = Picamera2(imx500.camera_num)
        main = {'format': 'XRGB8888'}   # カメラ用設定

        # プレビュー構成を作成、ネットワーク進行状況バーを表示
        config = picam2.create_preview_configuration(main, controls={"FrameRate": intrinsics.inference_rate}, buffer_count=8)
        imx500.show_network_fw_progress_bar()
        picam2.configure(config)
        time.sleep(0.5)  # 設定反映のため待機

        # プレビュー表示 or ヘッドレスでカメラ起動
        if args.preview:
            picam2.start(show_preview=True)
        else:
            picam2.start()

        # アスペクト比維持が必要なら自動調整
        if intrinsics.preserve_aspect_ratio:
            imx500.set_auto_aspect_ratio()
            
        print("カメラ起動完了")

    except Exception as e:
        print(f"カメラ初期化エラーまたはIMX500初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 各種パラメータクラス初期化
    camera = Camera(picam2, imx500)
    directoryInfo = DirectoryInfo(parameters.output_dir, parameters.camera_name, parameters.debug_images_subdir_name)
    directoryInfo.makedir(parameters.debug_mode)
    counter = PeopleCounter(directoryInfo)

    # フレーム毎に呼ばれるコールバックをPeopleFlowManagerで設定
    manager = PeopleFlowManager(config, loop, counter, directoryInfo, intrinsics, camera, parameters)
    picam2.pre_callback = manager.process_frame

    print(f"人流カウント開始 - {parameters.log_output_interval_sec}秒ごとにデータを保存します")
    print(f"ログは{parameters.log_interval}秒ごとに出力されます")
    print("Ctrl+Cで終了します")
    
    try:
        while not stop_event.is_set():
            time.sleep(0.01)
    except Exception as e:
        print("カメラメイン例外:", e)
    finally:
        try:
            counter.save_to_json()
            if 'picam2' in locals() and picam2:
                picam2.stop()
                picam2.close()
                print("カメラを停止しました")
            print("プログラムを終了します")
        except Exception as e:
            print(f"終了処理エラー: {e}")

pcs = set()
async def offer(request):
    print("OFFER HANDLER CALLED!")
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    }
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        pcs.add(pc)
        camera_track = CameraTrack(frame_queue)
        pc.addTrack(camera_track)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.Response(
            content_type="application/json",
            headers=headers,
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        # 失敗時もCORSつきエラーを返す
        return web.Response(
            status=500,
            content_type='text/plain',
            headers=headers,
            text="Offer handler error: " + str(e)
        )

async def options_handler(request):
    # CORSプリフライトに応答
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    }
    return web.Response(headers=headers)

async def web_server(stop_event):
    app = web.Application()
    app.router.add_post('/offer', offer)
    app.router.add_options('/offer', options_handler)  # 追加
    runner = web.AppRunner(app)
    await runner.setup()
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    site = web.TCPSite(runner, '0.0.0.0', 8443, ssl_context=ssl_context)
    await site.start()
    print("WebRTC signaling server started")
    try:
        while not stop_event.is_set():
            await asyncio.sleep(1)
        print("[web server] 停止イベントを受信。サーバー停止開始。")
        await runner.cleanup()
    except Exception as e:
        print(f"Webサーバ例外: {e}")

async def main():
    stop_event = threading.Event()
    # コマンドライン引数のパーサを作成
    parser = argparse.ArgumentParser(description="IMX500 AIカメラモジュール制御")
    parser.add_argument('--preview', action='store_true', help='プレビュー画面を表示する')
    args = parser.parse_args()

    loop = asyncio.get_running_loop()  # ここでloopを取得
    # カメラスレッド起動時にloopを渡す
    cam_thread = threading.Thread(target=camera_main, args=(stop_event, args, loop), daemon=True)
    cam_thread.start()
    try:
        await web_server(stop_event)
    except KeyboardInterrupt:
        print("[async main] Ctrl+Cキャッチ、停止イベントセット")
    finally:
        # 確実に停止処理を実行
        stop_event.set()
        print("[async main] カメラスレッドの終了を待機中...")
        cam_thread.join(timeout=5.0)  # タイムアウトを設定
        if cam_thread.is_alive():
            print("[async main] 警告: カメラスレッドがタイムアウト内に終了しませんでした")
        else:
            print("[async main] カメラスレッド終了を確認")
        print("[async main] メインスレッドの終了処理完了")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("プログラムを中断しました")