from flask import Flask, render_template, Response, request, url_for
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 Classification Model
model = YOLO("model/best.pt")

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

camera = None
device_id = 0  # default kamera 0


@app.route('/')
def index():
    return render_template('index.html', classes=classes)


@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/set_camera', methods=['POST'])
def set_camera():
    global device_id, camera
    new_id = int(request.form.get("device_id", 0))

    if camera is not None:
        camera.release()
        camera = None

    device_id = new_id
    return "Camera set to " + str(device_id)


def gen_frames():
    global camera, device_id
    camera = cv2.VideoCapture(device_id)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model.predict(frame, verbose=False)

            if results and len(results[0].probs) > 0:
                cls_id = int(results[0].probs.top1)
                conf = float(results[0].probs.top1conf)
                class_name = list(classes.keys())[cls_id]
                label_indo = classes[class_name]

                cv2.putText(frame, f"{label_indo} {conf:.2f}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "Camera stopped"


if __name__ == "__main__":
    app.run(debug=True)
