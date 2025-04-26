import os
import cv2
from datetime import datetime

def save_debug_image(frame, person, center_line_x, direction, debug_images_dir, output_prefix):
    """デバッグ用に画像を保存する関数"""
    try:
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
        filename = os.path.join(debug_images_dir, f"{output_prefix}_{timestamp}_crossing_{person.id}_{direction}.jpg")
        cv2.imwrite(filename, debug_frame)
        print(f"デバッグ画像を保存しました: {filename}")
    except Exception as e:
        print(f"デバッグ画像保存エラー: {e}")


# --- save_image_at_startup  ---
def save_image_at_startup(frame, center_line_x, output_dir):
    """起動時に画像を保存する関数"""
    try:
        # 画像にラインを描画
        debug_frame = frame.copy()

        # 中央ラインを描画
        cv2.line(debug_frame, (center_line_x, 0), (center_line_x, debug_frame.shape[0]), (0, 255, 0), 2)

        # 情報テキストを追加
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        text = f"Start Up Time: {timestamp}, Counting Line X: {center_line_x}"
        cv2.putText(debug_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # タイムスタンプ付きのファイル名で保存
        filename = os.path.join(output_dir, f"{output_dir}_{timestamp}_startupimage.jpg")
        cv2.imwrite(filename, debug_frame)
        print(f"起動時に画像を保存しました: {filename}")
    except Exception as e:
        print(f"起動時に画像を保存する関数の実行エラー: {e}")