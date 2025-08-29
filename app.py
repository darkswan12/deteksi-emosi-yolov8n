import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# === Load Model ===
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

# === Fungsi untuk ambil gambar ikon ===
def get_class_image(class_name):
    static_dir = "static/images"
    exts = ["jpg", "jpeg", "png"]
    for ext in exts:
        path = os.path.join(static_dir, f"{class_name}.{ext}")
        if os.path.exists(path):
            return path
    return os.path.join(static_dir, "default.jpg")  # fallback

# === Sidebar Navigasi ===
page = st.sidebar.radio("📌 Navigasi", ["Beranda", "Deteksi Foto", "Deteksi Realtime"])

# ===========================
# HALAMAN 1 - BERANDA
# ===========================
if page == "Beranda":
    st.title("🎭 Deteksi Emosi Wajah")
    st.write("Aplikasi ini dapat mengenali ekspresi wajah menggunakan **YOLOv8 Classification**.")
    
    st.subheader("✨ Emosi yang dapat dikenali:")
    cols = st.columns(4)
    for i, (eng, indo) in enumerate(classes.items()):
        with cols[i % 4]:
            img_path = get_class_image(eng)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((200, 200))
                st.image(img, caption=f"{indo} ({eng})", use_container_width=True)
            except:
                st.write(f"{indo} ({eng})")

# ===========================
# HALAMAN 2 - DETEKSI FOTO
# ===========================
elif page == "Deteksi Foto":
    st.title("📷 Deteksi Emosi dari Foto")
    uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diupload", use_container_width=True)

        # Prediksi
        results = model.predict(img, imgsz=224, conf=0.25, verbose=False)
        probs = results[0].probs
        cls_id = int(probs.top1)
        conf = float(probs.top1conf)
        pred_class = list(classes.keys())[cls_id]
        st.success(f"Prediksi: **{classes[pred_class]}** ({conf:.2f})")

# ===========================
# HALAMAN 3 - DETEKSI REALTIME
# ===========================
elif page == "Deteksi Realtime":
    st.title("📹 Deteksi Emosi Realtime dengan Kamera")

    # Slider threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.5, 0.05)

    # Video Processor
    class EmotionProcessor(VideoProcessorBase):
        def __init__(self, model, conf, iou):
            self.model = model
            self.conf = conf
            self.iou = iou

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(img, imgsz=224, conf=self.conf, iou=self.iou, verbose=False)
            probs = results[0].probs
            cls_id = int(probs.top1)
            conf = float(probs.top1conf)
            pred_class = list(classes.keys())[cls_id]
            label = f"{classes[pred_class]} ({conf:.2f})"
            cv2.putText(img, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Jalankan kamera dengan WebRTC
    webrtc_streamer(
        key="emotion-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: EmotionProcessor(model, confidence_threshold, iou_threshold),
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
                {"urls": ["stun:stun.services.mozilla.com"]},
                {"urls": ["stun:stun.nextcloud.com:3478"]},
                {"urls": ["stun:stun.stunprotocol.org:3478"]},
                {"urls": ["stun:stun.voipbuster.com:3478"]},
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.info("Izinkan akses kamera di browser Anda. "
            "Jika tidak muncul gambar, pastikan menggunakan HTTPS atau localhost, "
            "dan cek permission kamera di browser.")
