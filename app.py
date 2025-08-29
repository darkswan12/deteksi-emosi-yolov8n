import streamlit as st
from ultralytics import YOLO
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# === Load Model YOLOv8 Classification ===
@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

model = load_model()

# Mapping class → label Indo
classes = {
    "anger": "Marah",
    "contempt": "Menghina",
    "disgust": "Jijik",
    "fear": "Takut",
    "happiness": "Bahagia",
    "neutrality": "Netral",
    "sadness": "Sedih",
    "surprise": "Terkejut"
}

st.title("🎭 Deteksi Emosi Wajah Realtime")
st.write("Menggunakan **YOLOv8 Classification** + **Streamlit WebRTC**")

st.subheader("✨ Emosi yang dapat dikenali:")
cols = st.columns(4)
for i, (eng, indo) in enumerate(classes.items()):
    with cols[i % 4]:
        st.markdown(f"- **{indo}** ({eng})")

st.markdown("---")

# Konfigurasi WebRTC (supaya bisa jalan di cloud juga)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# === Video Processor ===
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Prediksi emosi
        results = model.predict(img, imgsz=224, verbose=False)
        probs = results[0].probs
        cls_id = int(probs.top1)
        conf = float(probs.top1conf)
        pred_class = list(classes.keys())[cls_id]
        label = f"{classes[pred_class]} ({conf:.2f})"

        # Tampilkan label di frame
        cv2.putText(img, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Jalankan kamera dengan WebRTC
webrtc_streamer(
    key="emotion-detect",
    mode="recvonly",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
