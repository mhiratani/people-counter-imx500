**process_frame_callback全体のサンプル実装**（RTSP配信部を「最新フレーム優先・古いフレームは廃棄」化済み）

---

## process_frame_callbackのテンプレート実装例

必要なもの（例：ffmpeg_proc, counter, active_people, TRACKING_TIMEOUTなど）を外で初期化

```python
import time
import threading
import queue
import cv2
import numpy as np

# ============ RTSPスレッドセットアップ ============

# ffmpeg_proc: 先に (subprocessで) 起動しておくこと！
frame_queue = queue.Queue(maxsize=3)

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

# ============ コールバック本体 ============

def process_frame_callback(request):
    global active_people, counter, last_log_time

    # よく使うグローバルの初期化（初回のみ、状態保持用）
    if not hasattr(process_frame_callback, 'image_saved'):
        process_frame_callback.image_saved = False

    # FPS表示用カウンタ
    now = time.time()
    if not hasattr(process_frame_callback, 'frame_count'):
        process_frame_callback.frame_count = 0
        process_frame_callback.last_fps_time = now
    process_frame_callback.frame_count += 1
    if now - process_frame_callback.last_fps_time > 2.0:
        print(f"[処理側] 上流での実FPS: {process_frame_callback.frame_count / (now - process_frame_callback.last_fps_time):.2f}")
        process_frame_callback.frame_count = 0
        process_frame_callback.last_fps_time = now

    try:
        # --- 検出・追跡 ---
        metadata = request.get_metadata()
        if metadata is None:
            detections = []
        else:
            detections = parse_detections(metadata)

        active_people = track_people(detections, active_people)
        if not isinstance(active_people, list):
            print(f"track_people returned : {type(active_people)}")

        # --- フレーム取得・描画 ---
        with MappedArray(request, 'main') as m:
            frame_height, frame_width = m.array.shape[:2]
            center_line_x = frame_width // 2

            # 起動時一度だけ参考画像保存
            if not process_frame_callback.image_saved:
                modules.save_image_at_startup(m.array, center_line_x, counter.date_dir, counter.output_prefix)
                process_frame_callback.image_saved = True

            cv2.line(m.array, (center_line_x, 0), (center_line_x, frame_height), (255,255,0), 2)
            for person in active_people:
                x, y, w, h = person.box
                if person.crossed_direction == "left_to_right":
                    color = (0,255,0)
                elif person.crossed_direction == "right_to_left":
                    color = (0,0,255)
                else:
                    color = (255,255,255)
                cv2.rectangle(m.array, (x, y), (x + w, y + h), color, 2)
                cv2.putText(m.array, f"ID: {person.id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if len(person.trajectory) > 1:
                    for i in range(1, len(person.trajectory)):
                        cv2.line(m.array, person.trajectory[i-1], person.trajectory[i], color, 2)

            # カウント情報や残り時間
            total_counts = counter.get_total_counts()
            cv2.putText(m.array, f"right_to_left: {total_counts['right_to_left']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(m.array, f"left_to_right: {total_counts['left_to_right']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            elapsed = int(time.time() - counter.last_save_time)
            remaining = COUNTING_INTERVAL - elapsed
            cv2.putText(m.array, f"Remaining time: {remaining}sec", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            # ========== RTSP非同期配信 ==========
            if RTSP_SERVER_IP != 'None' and ffmpeg_proc and ffmpeg_proc.stdin:
                frame_for_rtsp = m.array
                # BGRA→BGR変換
                if frame_for_rtsp.shape[2] == 4:
                    frame_for_rtsp = cv2.cvtColor(frame_for_rtsp, cv2.COLOR_BGRA2BGR)
                send_frame_for_rtsp(frame_for_rtsp)
            # ===================================

        # --- ライン跨ぎ判定 ---
        for person in active_people:
            if len(person.trajectory) >= 2:
                direction = check_line_crossing(person, center_line_x, m.array)
                if direction:
                    counter.update(direction)

        # --- アクティブ人物リスト整理 ---
        current_time = time.time()
        active_people = [p for p in active_people if current_time - p.last_seen < TRACKING_TIMEOUT]
        last_log_time = current_time
        counter.save_to_json()

    except Exception as e:
        print(f"コールバックエラー: {e}")
        import traceback
        traceback.print_exc()

```

---

## 使い方

1. `ffmpeg_proc` を（subprocess等で）**別途起動**しておいてください
2. サンプルの
   ```python
   rtsp_thread = start_rtsp_thread(ffmpeg_proc)
   ```
   を初期化時に**一度だけ**実行（main内等で）

3. 検出・追跡・カウント・描画部は従来通りでOK、  
   RTSP配信部（`send_frame_for_rtsp()` 呼び出し）は**コールバック内**で固定です

---

## 必要に応じて差し替えてください

- `MappedArray`, `parse_detections`, `track_people`, `check_line_crossing`, `modules.save_image_at_startup` などは**既存の実装を各自補完**してください。

---