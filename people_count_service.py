import json
import os
import time
from datetime import datetime
import numpy as np
import asyncio
from websockets.asyncio.server import serve
import websockets
from scipy.optimize import linear_sum_assignment
import traceback
import numpy as np

def print_x_axis_line(center_line_x, people_centers_x, width, scale=4):
    """
    x軸上のマーキングを圧縮表示します。

    Args:
        center_line_x (int): 中心線のx座標（元のスケール, 例：ピクセル単位）
        people_centers_x (list of int): 各人物の中心x座標リスト（元のスケール, ピクセル）
        width (int): 描画範囲の幅（元のスケール, ピクセル）
        scale (int): 圧縮度合い（例：4なら1/4スケールで描画）

    出力例:
        width=80, scale=4なら横幅は21文字('-'ベース)で
        ('-', '+', '0', '1', '2'...)等が圧縮された位置に表示される。
    """
    draw_width = width // scale + 1  # 圧縮して表示する幅
    line = ['-' for _ in range(draw_width)]

    center_line_pos = int(center_line_x // scale)
    if 0 <= center_line_pos < draw_width:
        line[center_line_pos] = '+'

    for idx, person_x in enumerate(people_centers_x):
        pos = int(person_x // scale)
        if 0 <= pos < draw_width:
            # 1桁分だけ番号を入れる（10人以上になるとずれるので要注意！）
            line[pos] = str(idx % 10)

    print(''.join(line))

class Person:
    next_id = 0

    def __init__(self, box):
        self.id = Person.next_id
        Person.next_id += 1
        self.box = box  # [x, y, w, h] 形式
        self.trajectory = [self.get_center()]
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.crossed_direction = None

    def get_center(self):
        """バウンディングボックスの中心座標を取得"""
        if len(self.box) != 4:
            print(f"[ERROR] get_center: boxの要素数が4ではありません: {self.box}")
            raise ValueError("Box の要素数が4つでありません")
        x, y, w, h = self.box
        return [x + w / 2, y + h / 2]

    def update(self, box):
        """新しい検出結果で人物の情報を更新"""
        self.box = box  # [x, y, w, h] 形式
        self.trajectory.append(self.get_center())
        if len(self.trajectory) > 30:  # 軌跡は最大30ポイントまで保持
            self.trajectory.pop(0)
        self.last_seen = time.time()


class PeopleCounter:
    def __init__(self, start_time, output_dir, output_prefix, counting_interval=60):
        self.right_to_left = 0  # 右から左へ移動（期間カウント）
        self.left_to_right = 0  # 左から右へ移動（期間カウント）
        self.total_right_to_left = 0  # 累積カウント
        self.total_left_to_right = 0  # 累積カウント
        self.start_time = start_time
        self.last_save_time = start_time
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.counting_interval = counting_interval

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
        if current_time - self.last_save_time >= self.counting_interval:
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
                return False    # 保存失敗

        return False


class PeopleTracker:
    def __init__(self, config_path="config.json", camera_name_path="camera_name.json"):
        # 設定ファイルの読み込み
        self.config = self._load_config(config_path)
        self.camera_name = self._load_config(camera_name_path)
        
        # 設定パラメータ
        self.max_tracking_distance = self.config.get('MAX_TRACKING_DISTANCE', 60)
        # ----------------------------------------------
        # 追跡対象と新しい検出結果の「中心点間距離」の最大許容値（ピクセル単位）。
        # この値以下ならマッチング候補と見なす。
        # - 値を大きくすると、急な移動や検出ずれにも追従しやすくなるが、近くの他人を誤ってマッチさせやすくなる。
        # - 値を小さくすると誤追跡は減るが、カメラ揺れ・一時ロスト・急な移動で追跡が切れやすくなる。
        # - 映像解像度、人物サイズ、フレームレート、移動速度に応じて現場で要チューニング。
        #   目安: 検出ボックスの幅の半分～1倍程度や、1フレームで起きうる最大移動距離
        # ----------------------------------------------

        self.iou_threshold = self.config.get('IOU_THRESHOLD', 0.3)
        # ----------------------------------------------
        # マッチング時、追跡対象と検出結果の「バウンディングボックスの重なり（IoU）」の下限値。
        # この値より大きい場合のみ同一人物候補とする。
        # - 値を大きくすると、ほぼ完全な重なりでのみマッチし、誤追跡減だが途切れやすい。
        # - 値を小さくすると、多少のズレやサイズ変動も許容し、追跡の継続性は増すものの、近距離他人の誤マッチリスク増。
        # - 通常0.2～0.5あたりで調整（高フレームレート＆精度が良いカメラなら大きくできる）。
        #   人物サイズ/動きの激しさ/カメラの安定度で最適値が変わる。
        # ----------------------------------------------

        self.tracking_timeout = self.config.get('TRACKING_TIMEOUT', 5.0)
        # ---------------------------
        # 人物を追跡し続ける最大時間（秒）
        # ---------------------------

        self.counting_interval = self.config.get('COUNTING_INTERVAL', 60)
        # ---------------------------
        # カウントデータを保存する間隔（秒）
        # ---------------------------

        self.output_dir = self.config.get('OUTPUT_DIR', 'people_count_data')
        # ---------------------------
        # データ保存ディレクトリ
        # ---------------------------

        self.output_prefix = self.camera_name.get('CAMERA_NAME', 'cameraA')
        # -----------------------------------------------------------
        # 出力ファイル名のプレフィックス(カメラ名はcamera_name.jsonから取得)
        # -----------------------------------------------------------

        self.debug_mode = str(self.camera_name.get('DEBUG_MODE', 'False')).lower() == 'true'
        # -----------------------------------------------------
        # デバッグモードのオン/オフ - アクティブな人物を標準出力で描画
        # -----------------------------------------------------

        self.log_interval = 5  # ログ出力間隔（秒）
        
        # 状態の初期化
        self.active_people = []
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(os.path.join(self.output_dir, timestamp), exist_ok=True)

        # カウンターの初期化
        self.counter = PeopleCounter(
            self.start_time, 
            self.output_dir, 
            self.output_prefix,
            self.counting_interval,
            self.debug_mode
        )
        
        # キューの初期化
        self.tensor_queue = asyncio.Queue()

    def _load_config(self, path):
        """設定ファイルを読み込む"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config from {path}: {e}")
            return {}

    def calculate_iou(self, box1, box2):
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

    def track_people(self, detections, active_people):
        """
        物体検出で得られた人物候補（detections）と、既存の追跡対象（active_people）を
        効率的かつ精度良くマッチングし、追跡リストを更新します。
        ※ IOUと距離、ハンガリアンアルゴリズムを使用
        """
        # ここで「detectionがdictなら必ず['box']を取り出す」ことで以降処理を一本化
        clean_detections = []
        for d in detections:
            if isinstance(d, dict):
                clean_detections.append(d['box'])
            else:
                clean_detections.append(d)

        detections = clean_detections
        num_people = len(active_people)
        num_detections = len(detections)

        # 検出結果も追跡対象もいない場合はそのまま返す
        if num_detections == 0 and num_people == 0:
            return []

        # 新しい検出結果がない場合、既存の追跡対象は維持
        if num_detections == 0:
            return active_people

        # 追跡対象がいない場合、全ての検出を新しい人物とする
        if num_people == 0:
            return [Person(det) for det in detections]

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
                detection_box = detection
                detection_center = (detection_box[0] + detection_box[2]//2, 
                                   detection_box[1] + detection_box[3]//2)

                # 距離とIOUを計算
                distance = np.sqrt((person_center[0] - detection_center[0])**2 + (person_center[1] - detection_center[1])**2)
                iou = self.calculate_iou(person_box, detection_box)

                # マッチングの条件: 距離が閾値以内 かつ IOUが閾値以上
                if distance < self.max_tracking_distance and iou > self.iou_threshold:
                    # コストの定義 (距離が近いほど、IOUが大きいほど良いマッチング -> コスト小)
                    cost_matrix[i, j] = (1.0 - iou) + (distance / self.max_tracking_distance) * 0.1

        # コスト行列の全要素がinf or どの行or列も全てinfならreturn
        if (
            np.all(np.isinf(cost_matrix)) or 
            np.any(np.all(np.isinf(cost_matrix), axis=0)) or 
            np.any(np.all(np.isinf(cost_matrix), axis=1))
        ):
            # print("Assignment infeasible: some row or column is all inf.")
            return active_people

        else:
            # ハンガリアンアルゴリズムを実行し、最適なマッチングを見つける
            # matched_person_indices: active_peopleのインデックスの配列
            # matched_detection_indices: detectionsのインデックスの配列
            # print("Will run linear_sum_assignment")
            matched_person_indices, matched_detection_indices = linear_sum_assignment(cost_matrix)

            # マッチング結果を処理
            new_people = []
            # マッチした検出結果のインデックスを記録
            used_detections = set(matched_detection_indices)

            # マッチした人物を更新して新しいリストに追加
            for i, j in zip(matched_person_indices, matched_detection_indices):
                if cost_matrix[i, j] == np.inf:
                    continue
                person = active_people[i]
                detection = detections[j]
                
                # detection_boxの取得 (構造に応じて分岐)
                if hasattr(detection, 'box'):
                    detection_box = detection.box
                else:
                    detection_box = detection
                
                person.update(detection_box)
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
                    # detection_boxの取得 (構造に応じて分岐)
                    if hasattr(detection, 'box'):
                        detection_box = detection.box
                    else:
                        detection_box = detection
                        
                    new_people.append(Person(detection_box))

            return new_people

    def check_line_crossing(self, person, center_line_x):
        """中央ラインを横切ったかチェック"""
        if len(person.trajectory) < 2:
            return None

        prev_x = person.trajectory[-2][0]
        curr_x = person.trajectory[-1][0]

        # 左→右: 中央線を未満→以上で通過
        if prev_x < center_line_x and curr_x >= center_line_x:
            person.crossed_direction = "left_to_right"
            return "left_to_right"

        # 右→左: 中央線を以上→未満で通過
        elif prev_x >= center_line_x and curr_x < center_line_x:
            person.crossed_direction = "right_to_left"
            return "right_to_left"

        return None

    def print_status(self):
        """現在の状態を出力"""
        current_time = time.time()
        # 定期的なログ出力
        if current_time - self.last_log_time >= self.log_interval:
            total_counts = self.counter.get_total_counts()
            elapsed = int(current_time - self.counter.last_save_time)
            remaining = max(0, int(self.counting_interval - elapsed))
            
            print(f"--- Status Update ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            print(f"Active tracking: {len(self.active_people)} people")
            print(f"Counts - Period (R->L: {self.counter.right_to_left}, L->R: {self.counter.left_to_right})")
            print(f"Counts - Total (R->L: {total_counts['right_to_left']}, L->R: {total_counts['left_to_right']})")
            print(f"Next save in: {remaining} seconds")
            print(f"--------------------------------------------------")
            
            self.last_log_time = current_time

    async def detect_server(self, websocket):
        """WebSocketサーバー: クライアントからのメッセージを受信してキューに追加"""
        client_id = id(websocket)
        # print(f"[WS] 新しい接続が確立されました。クライアントID: {client_id}")
        try:
            async for message in websocket:
                # print(f"[WS] クライアント{client_id}からメッセージを受信。サイズ: {len(message)}バイト")
                # print(f"[WS] メッセージプレビュー: {message[:100]}..." if len(message) > 100 else f"[WS] メッセージ: {message}")
                await self.tensor_queue.put(message)
        except websockets.ConnectionClosed as e:
            print(f"[WS] クライアント{client_id}との接続が閉じられました。コード: {e.code}, 理由: {e.reason}")
        except Exception as e:
            print(f"[WS] クライアント{client_id}との通信中にエラーが発生: {e}")
            print(f"[WS] エラー詳細: {traceback.format_exc()}")
        finally:
            print(f"[WS] クライアント{client_id}との接続が終了しました")

    async def tensor_worker(self):
        """キューからメッセージを取り出して処理"""
        print("[Worker] テンソルワーカーが開始されました")
        while True:
            try:
                # キューからメッセージを取得
                # print(f"[Worker] メッセージ待機中。キューサイズ: {self.tensor_queue.qsize()}")
                msg = await self.tensor_queue.get()
                # print(f"[Worker] キューからメッセージを取得。サイズ: {len(msg)}バイト")
                
                # JSONパース
                try:
                    start_time = time.time()
                    packet = json.loads(msg)
                    parse_time = time.time() - start_time
                    # print(f"[Worker] JSONの解析に成功しました（{parse_time:.4f}秒）。キー: {list(packet.keys())}")
                except json.JSONDecodeError as e:
                    print(f"[Worker] JSONの解析に失敗: {e}")
                    print(f"[Worker] メッセージプレビュー: {msg[:100]}...")
                    continue
                
                # データ抽出
                center_line_x = packet.get("center_line_x")
                detections = packet.get("detections", [])
                
                # print(f"[Worker] 抽出データ - center_line_x: {center_line_x}, 検出数: {len(detections)}")
                if detections:
                    # print(f"[Worker] 最初の検出サンプル: {detections[0]}")
                    pass
                
                if center_line_x is None:
                    print("[Worker] 警告: パケット内にcenter_line_xが見つかりません")
                    continue
                
                # 人物追跡を更新
                previous_count = len(self.active_people)
                start_time = time.time()
                self.active_people = self.track_people(detections, self.active_people)
                track_time = time.time() - start_time
                
                # フレーム内すべての人物のx座標を集める
                people_centers_x = []
                for person in self.active_people:
                    center_x, center_y = person.trajectory[-1]
                    people_centers_x.append(center_x)  # デバッグ用 x座標だけ集める

                    # print(f"[DEBUG] 人物ID {person.id}: box={person.box}, trajectory(last2)={person.trajectory[-2:]}")
                    # 少なくとも2フレーム以上の軌跡がある人物が対象
                    if len(person.trajectory) >= 2:
                        direction = self.check_line_crossing(person, center_line_x)
                        # print(f"[Worker] 人物ID {person.id} のライン判定")
                        # print(f"[Worker] 軌跡: {person.trajectory[-2:]} (最後の2点を表示)")
                        # print(f"[DEBUG] 直近2点のcenter_line_xまでの距離: {[abs(xy[0] - center_line_x) for xy in person.trajectory[-2:]]}")
                        if direction:
                            self.counter.update(direction)
                            print(f"Person ID {person.id} crossed line: {direction}")
                        else:
                            # print(f"[DEBUG] {person.id} はまだ横断していません")
                            pass

                # 古いトラッキング対象を削除
                current_time = time.time()
                before_cleanup = len(self.active_people)
                self.active_people = [p for p in self.active_people 
                                    if current_time - p.last_seen < self.tracking_timeout]
                
                if before_cleanup != len(self.active_people):
                    print(f"[Worker] 追跡オブジェクトをクリーンアップしました。削除数: {before_cleanup - len(self.active_people)}")
                    print(f"[Worker] 残りのID: {[p.id for p in self.active_people]}")
                
                # ステータス表示
                self.print_status()

                if self.debug_mode:
                    # 全人物分の位置とラインをまとめて横棒で可視化
                    print_x_axis_line(center_line_x, people_centers_x, 640, 4)

                # データ保存
                save_start = time.time()
                self.counter.save_to_json()
                save_time = time.time() - save_start
                # print(f"[Worker] データをJSONに保存しました（{save_time:.4f}秒）")
                
                # 処理完了
                # print("-" * 50)
                
            except Exception as e:
                print(f"[Worker] tensor_workerでエラーが発生: {e}")
                print(f"[Worker] エラー詳細: {traceback.format_exc()}")

    async def run(self):
        """サーバーを起動してワーカーを実行"""
        print(f"[Server] WebSocketサーバーを0.0.0.0:8765で起動します")
        async with serve(self.detect_server, "0.0.0.0", 8765):
            print(f"[Server] WebSocketサーバーが初期化されました")
            worker_task = asyncio.create_task(self.tensor_worker())
            try:
                await asyncio.Future()  # ここで永遠に止まる（Ctrl+Cまで動き続ける）
            except asyncio.CancelledError:
                print("[Server] サーバーの無限待機がキャンセルされました。終了します。")

def main():
    """メイン関数"""
    print("[Main] PeopleTrackerを初期化します")
    tracker = PeopleTracker()
    
    try:
        print("[Main] asyncioイベントループを開始します")
        asyncio.run(tracker.run())
    except KeyboardInterrupt:
        print("[Main] ユーザーによってサーバーが停止されました")
    except Exception as e:
        print(f"[Main] エラー: {e}")
        print(f"[Main] エラー詳細: {traceback.format_exc()}")
    finally:
        print("[Main] サーバーのシャットダウンが完了しました")


if __name__ == "__main__":
    main()