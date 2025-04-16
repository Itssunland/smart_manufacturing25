import cv2
from ultralytics import YOLO
import logging
import time
from paho.mqtt import client as mqtt_client

#TODO: Glasses detection unstable.

# ==== MQTT SETUP ====
broker = 'broker.emqx.io'
port = 1883
topic = "camera/glasses"
client_id = f'publisher-glasses'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("✅ Connected to MQTT Broker!")
        else:
            print(f"❌ Failed to connect, return code {rc}")

    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

client = connect_mqtt()
client.loop_start()

# ==== DETEKSJON SETUP ====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = YOLO("yolov8n.pt")
model.verbose = False
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Bruk mobilkamera (via Camo eller IP-kamera) – endre om nødvendig
cap = cv2.VideoCapture(0)  # Bruk 0 for innebygd, 1 for ekstern, eller IP-stream

frame_counter = 0
glasses_counter = 0
decision_made = False
final_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = model(frame)
    detections = results[0].boxes
    glasses_detected = False

    for box in detections:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        if label in ["glasses", "sunglasses"]:
            glasses_detected = True
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not decision_made and len(faces) > 0:
        frame_counter += 1
        if glasses_detected:
            glasses_counter += 1

        label = f"Analyzing... ({frame_counter}/30)"
        color = (0, 255, 255)

        if frame_counter >= 30:
            ratio = glasses_counter / frame_counter
            final_label = "Glasses 👓" if ratio > 0.3 else "No Glasses 🚫"
            print(f"Finished: {glasses_counter} / {frame_counter} = {ratio}")
            print("Final label:", final_label)

            # ✅ Send til MQTT
            client.publish(topic, final_label)
            decision_made = True

    elif decision_made:
        label = final_label
        color = (0, 255, 0)
    else:
        label = "No face detected"
        color = (0, 0, 255)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("YOLOv8 Glasses Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        frame_counter = 0
        glasses_counter = 0
        decision_made = False
        final_label = ""

cap.release()
cv2.destroyAllWindows()
client.loop_stop()
