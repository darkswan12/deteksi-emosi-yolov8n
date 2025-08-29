import streamlit as st
import cv2
from ultralytics import YOLO
import os
from PIL import Image

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

# === Fungsi untuk mendeteksi kamera ===
def list_cameras(max_test=5):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available.append(i)
            cap.release()
    return available if available else [0]

# === Tampilan Streamlit ===
st.title("🎭 Deteksi Emosi Wajah Realtime")
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

st.markdown("---")

# Deteksi kamera yang tersedia
available_cams = list_cameras()
device_id = st.selectbox("Pilih Kamera:", available_cams, index=0)

# Checkbox untuk nyalakan kamera
run = st.checkbox("▶️ Nyalakan Kamera")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(device_id)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Tidak bisa membuka kamera")
            break

        # Prediksi emosi
        results = model.predict(frame, imgsz=224, verbose=False)
        probs = results[0].probs
        cls_id = int(probs.top1)
        conf = float(probs.top1conf)
        pred_class = list(classes.keys())[cls_id]

        # Tambahkan teks ke frame
        cv2.putText(frame, f"{classes[pred_class]} {conf:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
