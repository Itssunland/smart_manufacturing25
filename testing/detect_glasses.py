import cv2
import time
import logging
from transformers import pipeline
from paho.mqtt import client as mqtt_client

# ==== MQTT CONFIGURATION ====
broker = 'broker.emqx.io'
port = 1883
topic = "camera/glasses"
client_id = 'publisher-glasses'

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

# ==== MAIN PROGRAM ====
def main():
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # === HuggingFace eyeglasses detection model ===
    pipe = pipeline("image-classification", model="youngp5/eyeglasses_detection")

    # === Optional: load YOLO for later use ===
    # from ultralytics import YOLO
    # yolo_model = YOLO("glasses-detection-model.pt") #NOT A REAL MODEL.
    # yolo_model.verbose = False

    # === Optional: Haar cascade for face detection (still used for annotation)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    client = connect_mqtt()
    client.loop_start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access camera.")
        return

    label = "Analyzing..."
    score = 0.0
    last_check = 0
    interval = 1  # seconds between classifications

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received.")
            break

        current_time = time.time()
        if current_time - last_check > interval:
            cv2.imwrite("temp.jpg", frame)
            result = pipe("temp.jpg")[0]
            raw_label = result['label']
            score = result['score']
            print(f"📷 Detected: {raw_label} ({score:.2f})")

            label = "Glasses 👓" if raw_label == "eyeglasses" and score > 0.5 else "No Glasses 🚫"
            client.publish(topic, label)
            last_check = current_time

        # Optional face box (cosmetic only)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw result
        cv2.putText(frame, f"{label} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Eyeglasses Detection (HuggingFace)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            label = "Analyzing..."
            score = 0.0
            last_check = 0

    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()
    print("🛑 Program terminated.")

if __name__ == "__main__":
    main()
