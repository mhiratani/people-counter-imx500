from picamera2 import Picamera2
from picamera2.devices import IMX500
import time

try:
    # IMX500の初期化を試みる
    print("IMX500を初期化中...")
    imx500 = IMX500("/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    print("IMX500初期化成功")
    
    # カメラの初期化
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    
    print("カメラを起動中...")
    picam2.start()
    print("カメラ起動成功")
    
    # 数秒間待機
    time.sleep(3)
    
    # カメラを停止
    picam2.stop()
    print("テスト完了：IMX500とpicamera2が正常に動作しています")
    
except Exception as e:
    print(f"エラーが発生しました: {e}")
    import traceback
    traceback.print_exc()