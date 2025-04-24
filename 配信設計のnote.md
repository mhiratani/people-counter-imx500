明日の“自分”への道しるべとして、**設計・実装がサクッと着手できる最小限かつ重要なポイント**をまとめます。

---

# 🎬 目的

- ラズパイでCV2(OpenCV)による処理済み映像をWebRTCストリーム化。
- 別サーバーのStreamlitアプリでリアルタイム受信・表示。

---

# 🏗️ 構成図

```
╔═════════╗     WebRTC+Signaling     ╔════════════════════╗
║ ラズパイ ║ <-------------------→  ║  Dockerサーバー      ║
║ aiortc  ║ (カスタムSignalingなど)  ║  ├─ Nginx（入口）    ║
║ cv2描画 ║                         ║  ├─ signaling(WS等) ║
╚═════════╝                         ║  ├─ streamlit      ║
                                     ╚════════════════════╝
```

---

# 📝 設計・実装ステップ

## 1. 映像ストリームの送信（ラズパイ側）

- `cv2`で描画したフレーム(numpy配列)を`aiortc.MediaStreamTrack`に流す。
- 必要に応じて「WebSocketベース」のSignalingサーバー経由で外部とPeer接続。

### 雛形
```python
import av
from aiortc import MediaStreamTrack

class ProcessedStreamTrack(MediaStreamTrack):
    kind = "video"
    async def recv(self):
        # 描画済みcv2フレーム（numpy配列, RGB）
        frame = get_processed_cv2_frame()
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        return av_frame
```
※「get_processed_cv2_frame」部分は現状のcv2処理を組み込めばOK

---

## 2. Signalingサーバー（streamlitとaiortc間の仲介）

- docker-composeに「signaling」サービスとして追加。
- Python/Node.js等でWebSocketやHTTPベースで実装。  
  `aiortc/examples/server`付属サーバーもそのまま使いやすい。

---

## 3. 受信＆可視化（docker上Streamlit/streamlit-webrtc）

- `docker-compose.yml`にstreamlitサービスを追加
- `streamlit-webrtc`で受信。
- Signalingサーバー接続先（nginx経由`/ws/`等）を指定。

### 雛形
```python
from streamlit_webrtc import webrtc_streamer, WebRtcMode

def video_frame_callback(frame):
    img = frame.to_ndarray(format="rgb24")  # 送信形式に合わせる
    return frame  # またはcv2で更に上書きして可

webrtc_streamer(
    key="recv",
    mode=WebRtcMode.RECVONLY,
    signaling=WebSocketSignaling("wss://yourdomain/ws/"),
)
```

---

## 4. Nginxの役割

- `/` → streamlit
- `/ws/` → signalingサーバー(WSS)

---

# ✅ 明日やるべきToDo

1. **ラズパイ側**
    - [ ] aiortc/av/cv2 で描画済みフレーム配信のクラス化
    - [ ] Signalingサーバーへの接続動作確認
2. **signaling**
    - [ ] サンプルコード動作 or 既存流用
    - [ ] WebSocket経由で双方の「Offer/Answer」仲介テスト
3. **streamlit**
    - [ ] streamlit-webrtcサンプル→signaling先書換え
    - [ ] 受信したフレームがそのまま反映されるか動作確認
4. **docker-compose/nginx**
    - [ ] 各サービス（nginx、signaling、streamlit）記述
    - [ ] Nginxのプロキシ設定（wss含む）
    - [ ] 外部からhttps/443でアクセス＆WebRTC受信動作確認

---

# 💡 KEYポイント速習

- **cv2フレーム→VideoFrame→aiortcで送信**（ラズパイ）
- **Signalingサーバー（WS/HTTP）でピア接続調整**
- **streamlit-webrtcで受信→video_frame_callbackにOpenCV配列で渡る**
- **docker-composeで全サービス一体管理、Nginxリバプロ入口に集約**
- **色形式(format)に注意＆STUNでNAT越え設定**

---

# 🗝️ 明日一歩
「まずラズパイ側の `cv2配信クラス` と、docker-compose(nginx + signaling + streamlit)の最小サンプルから組み立てるべし！」
