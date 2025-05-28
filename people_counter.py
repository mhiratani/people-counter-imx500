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

# ffmpegによるRTSP配信プロセス用
import queue
import threading
import subprocess

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
        self.counting_interval          = self.config.get('COUNTING_INTERVAL')                  # カウント間隔（秒）
        self.active_timeout_sec         = self.config.get('ACTIVE_TIMEOUT_SEC')                 # lost_people保持猶予(秒)
        self.direction_mismatch_penalty = self.config.get('DIRECTION_MISMATCH_PENALTY')         # 逆方向へのマッチに与える追加コスト
        self.max_acceptable_cost        = self.config.get('MAX_ACCEPTABLE_COST')                # 最大許容コスト
        self.min_box_height             = self.config.get('MIN_BOX_HEIGHT')                     # 人物ボックスの高さフィルタ。これより小さいBoxは排除(ピクセル)
        self.max_box_height             = self.config.get('MAX_BOX_HEIGHT')                     # 人物ボックスの高さフィルタ。これより大きいBoxは排除(ピクセル)
        self.output_dir                 = self.config.get('OUTPUT_DIR', 'people_count_data')    # ログデータを保存するディレクトリ名
        self.debug_mode                 = str(self.config.get('DEBUG_MODE', 'False')).lower() == 'true'
        self.debug_images_subdir_name   = self.config.get('DEBUG_IMAGES_SUBDIR_NAME', 'debug_images')
        self.rtsp_server_ip             = self.config.get('RTSP_SERVER_IP', 'None')             # RTSP配信先IPアドレス
        self.rtsp_server_port           = 8554
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

class RTSP:
    """
    フレームデータをffmpeg経由でRTSPサーバへ配信するためのクラス。
    フレームは内部キューに蓄積し、別スレッドでffmpegに逐次書き込む。
    """
    def __init__(self, rtsp_server_ip, rtsp_server_port, intrinsics, frame_queue_size=3):
        """
        コンストラクタ

        Args:
            rtsp_server_ip (str): RTSPサーバIPアドレス
            rtsp_server_port (int): RTSPサーバポート
            intrinsics : 画像・動画フレームの内部パラメータ（推論レートなどを保持）
            frame_queue_size (int): 内部フレームキューの最大長（デフォルト: 3）
        """
        self.rtsp_server_ip = rtsp_server_ip
        self.rtsp_server_port = rtsp_server_port
        # ffmpegに送信するフレームの一時保持用キュー。overflow時は古いものから削除。
        self.frame_queue = queue.Queue(maxsize=frame_queue_size)
        # ffmpegプロセスの起動と設定
        self.ffmpeg_proc = self.rtsp_setting(intrinsics)
        # ffmpegプロセスへデータを書き込むスレッドの起動
        self.rtsp_thread = self.start_rtsp_thread(self.ffmpeg_proc)
        self.active = True  # 配信中フラグ
        self.message = False  # 配信停止メッセージ出力フラグ

    def rtsp_writer_thread(self, ffmpeg_proc):
        """
        フレームキューからフレームを取り出してffmpeg stdinへ書き込む無限ループスレッド。
        Args: ffmpeg_proc (subprocess.Popen): 起動済みffmpegプロセス
        """
        while True:
            frame = self.frame_queue.get()
            try:
                # フレームデータをバイト列へ変換してffmpegへ入力
                ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())
            except Exception as e:
                print(f"[RTSP配信エラー]: {e}")
                self.active = False  # 配信停止
            finally:
                self.frame_queue.task_done()

    def start_rtsp_thread(self, ffmpeg_proc):
        """
        ffmpegへのフレーム書き込み専用スレッドを起動
        Args: ffmpeg_proc (subprocess.Popen): 起動済みffmpegプロセス
        Returns: threading.Thread: 起動したデーモンスレッド
        """
        t = threading.Thread(target=self.rtsp_writer_thread, args=(ffmpeg_proc,), daemon=True)
        t.start()
        return t

    def send_frame_for_rtsp(self, frame):
        """
        キューにフレームを追加（キューが満杯なら古いフレームを捨てて新しいフレームを必ず入れる）

        Args:
            frame (np.ndarray): 配信したいフレーム
        """
        if not self.active:
            if not self.message:
                print("RTSP配信は既に停止しています")
                self.message  = True
            return
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # 古いデータを非同期で破棄
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # 想定外の複数同時処理はスキップ

    def rtsp_setting(self, intrinsics):
        """
        RTSP配信用のffmpegプロセスを起動・設定
        Args: intrinsics: 推論時の内部パラメータ（フレームレート取得に利用）
        Returns: subprocess.Popen | str: ffmpegプロセス（または異常時は空文字列）
        """
        # 初期値。失敗時などはこのまま返す。
        ffmpeg_proc = ''  
        if self.rtsp_server_ip != 'None':
            # 解像度やフレームレート等の設定
            FRAME_WIDTH  = 640
            FRAME_HEIGHT = 480
            FRAME_RATE = int(intrinsics.inference_rate) if hasattr(intrinsics, 'inference_rate') else 15
            RTSP_URL = f"rtsp://{self.rtsp_server_ip}:{self.rtsp_server_port}/stream"

            # ffmpegプロセス起動用コマンド
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
                print("ffmpegによるRTSP配信プロセス起動")
            except Exception as e:
                print(f"ffmpeg起動失敗: {e}")
                sys.exit(1)
        else:
            print(f"RTSP_SERVER_IPが未指定のためRTSP配信は行いません")

        return ffmpeg_proc

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
        # 大きな値で初期化（初期状態への信頼度は低い）
        self.kf.P *= 10.
        
        # 観測ノイズ共分散行列R: 検出器の精度に依存
        # ピクセル単位での位置検出誤差を想定（値は実験的に調整が必要）
        self.kf.R = np.eye(2) * 15.  # 対角成分15px^2の観測ノイズ
        
        # プロセスノイズ共分散行列Q: モデルの不完全性を表現
        # 人の動きが完全に等速直線運動ではないことを考慮
        # 小さな値: モデルを信頼、大きな値: 観測を重視
        self.kf.Q = np.eye(4) * 0.5  # 適度なプロセスノイズで予測と観測のバランスを取る

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

    def __init__(self, config, rtsp, counter, directoryInfo, intrinsics, camera, parameters):
        """
        クラス各種ハンドル・設定値を初期化。

        Args:
            config: アプリ全体設定
            rtsp: RTSP配信用オブジェクト
            counter: 人数カウンタ（PeopleCounterインスタンス）
            directoryInfo: 保存ディレクトリ・出力プリフィックス等の設定
            intrinsics: カメラ・モデルの内部パラメータ
            camera: カメラデバイスへのアクセス用インスタンス
            parameters: 推論・トラッキング・カウント関連パラメータ
        """
        self.active_people = []     # 現在追跡中の人物リスト
        self.lost_people = []       # 一時追跡ロスト中の人物リスト
        self.rtsp = rtsp
        self.counter = counter
        self.directoryInfo = directoryInfo
        self.last_log_time = time.time()
        self.config = config
        self.intrinsics = intrinsics
        self.camera = camera
        self.parameters = parameters
        self.image_saved = False    # 起動時の一度きりの画像保存用

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
                        # detection（新検出）の中心座標取得
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
                        HEIGHT_SIMILARITY_THRESHOLD = 0.90  # 許容する割合
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
        with MappedArray(request, 'main') as m:
            frame = m.array.copy()
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
            detections = Detection.parse_detections(metadata, self.parameters, self.intrinsics, self.camera.imx500, self.camera.picam2, )

        # 人物追跡を更新
        frame_height, frame_width = frame.shape[:2]
        center_line_x = frame_width // 2
        self.active_people, self.lost_people = self._track_people(self.active_people, self.lost_people, detections, frame_id, center_line_x)
        if not isinstance(self.active_people, list):
            print(f"track_people returned : {type(self.active_people)}")

        with MappedArray(request, 'main') as m:
            # 起動時の画像を一度だけ保存
            if not self.image_saved:
                modules.save_image_at_startup(m.array, center_line_x, self.directoryInfo.date_dir, self.directoryInfo.output_prefix)
                self.image_saved = True

            # 中央ラインを描画
            cv2.line(m.array, (center_line_x, 0), (center_line_x, frame_height), 
                    (255, 255, 0), 2)

            # CENTER_LINE_MARGINを描画
            center_line_margin_px = self.parameters.center_line_margin_px
            cv2.line(m.array, (center_line_x - center_line_margin_px, 0), (center_line_x - center_line_margin_px, frame_height), 
                    (0, 128, 255), 2)
            cv2.line(m.array, (center_line_x + center_line_margin_px, 0), (center_line_x + center_line_margin_px, frame_height), 
                    (0, 128, 255), 2)

            # 人物の検出ボックスと軌跡を描画
            for person in self.active_people:
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
                cv2.putText(m.array, f"ID: {person.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # ボックスの高さ表示
                cv2.putText(m.array, f"H: {int(h)}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # 軌跡を描画
                if len(person.trajectory) > 1:
                    for i in range(1, len(person.trajectory)):
                        cv2.line(m.array, person.trajectory[i-1], person.trajectory[i], color, 2)
            
            # カウント情報を表示
            total_counts = self.counter.get_total_counts()
            cv2.putText(m.array, f"right_to_left: {total_counts['right_to_left']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(m.array, f"left_to_right: {total_counts['left_to_right']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 時刻とフレームIDを表示
            text_str = f"FrameID: {frame_id} / {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(m.array, text_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            frame = m.array.copy()  # デバッグ画像保存用
            # ========== RTSP非同期配信 ==========
            if self.rtsp.rtsp_server_ip != 'None' and self.rtsp.ffmpeg_proc and self.rtsp.ffmpeg_proc.stdin:
                frame_for_rtsp = m.array
                # BGRA→BGR変換
                if frame_for_rtsp.shape[2] == 4:
                    frame_for_rtsp = cv2.cvtColor(frame_for_rtsp, cv2.COLOR_BGRA2BGR)
                self.rtsp.send_frame_for_rtsp(frame_for_rtsp)
            # ===================================

        # ラインを横切った人をカウント
        for person in self.active_people:
            # 少なくとも2フレーム以上の軌跡がある人物が対象
            if len(person.trajectory) >= 2:
                direction = self._check_line_crossing(person, center_line_x, frame)
                if direction:
                    self.counter.update(direction)

        # --- アクティブ人物リスト整理 ---
        # 古いトラッキング対象を削除 (last_seen が TRACKING_TIMEOUT を超えたもの)
        current_time = time.time()
        self.active_people = [p for p in self.active_people if current_time - p.last_seen < self.parameters.tracking_timeout]

        # 定期的なログ出力
        if current_time - self.last_log_time >= self.parameters.log_interval:
            self.last_log_time = current_time
            total_counts = self.counter.get_total_counts()
            elapsed = int(current_time - self.counter.last_save_time)
            remaining = max(0, int(self.parameters.counting_interval - elapsed)) # 負にならないように
            print(f"--- Status Update ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            print(f"Active tracking: {len(self.active_people)} people")
            print(f"Counts - Period (R->L: {self.counter.right_to_left}, L->R: {self.counter.left_to_right})")
            print(f"Counts - Total (R->L: {total_counts['right_to_left']}, L->R: {total_counts['left_to_right']})")
            print(f"Next save in: {remaining} seconds")
            print(f"--------------------------------------------------")

        # 指定間隔ごとにJSONファイルに保存
        if current_time - self.counter.last_save_time >= self.parameters.counting_interval:
            self.counter.save_to_json()

# ======= メイン処理 =======
if __name__ == "__main__":
    # コマンドライン引数のパーサを作成
    parser = argparse.ArgumentParser(description="IMX500 AIカメラモジュール制御")
    parser.add_argument('--preview', action='store_true', help='プレビュー画面を表示する')
    args = parser.parse_args()
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
    rtsp = RTSP(parameters.rtsp_server_ip, parameters.rtsp_server_port, intrinsics)
    directoryInfo = DirectoryInfo(parameters.output_dir, parameters.camera_name, parameters.debug_images_subdir_name)
    directoryInfo.makedir(parameters.debug_mode)
    counter = PeopleCounter(directoryInfo)

    # フレーム毎に呼ばれるコールバックをPeopleFlowManagerで設定
    manager = PeopleFlowManager(config, rtsp, counter, directoryInfo, intrinsics, camera, parameters)
    picam2.pre_callback = manager.process_frame

    print(f"人流カウント開始 - {parameters.counting_interval}秒ごとにデータを保存します")
    print(f"ログは{parameters.log_interval}秒ごとに出力されます")
    print("Ctrl+Cで終了します")
    
    try:
        # メインループ - コールバックにて処理されるためループ内は待機のみ
        while True:
            time.sleep(0.01)  # CPU使用率抑制

    except KeyboardInterrupt:
        print("終了中...")
        # 最後のデータを保存
        counter.save_to_json()

    finally:
        # カメラ停止とリソース解放
        try:
            if 'picam2' in locals() and picam2: # picam2が初期化されているか確認
                picam2.stop()
                picam2.close()
                print("カメラを停止しました")
            print("プログラムを終了します")
        except Exception as e:
            print(f"終了処理エラー: {e}")