#!/usr/bin/env python3
import random
import json
import time
import pickle
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO
from collections import deque
import numpy as np

# ——— Configuration ———
BROKER      = "broker.emqx.io"
PORT        = 1883
TOPIC       = "drill/data"
CLIENT_ID   = f"pi-subscriber-{random.randint(0,999)}"
MODEL_FILE  = "/home/pi/models/rf_material_clf.pkl"
BUZZER_PIN  = 27
WINDOW_SIZE = 256

# ——— Load trained RF model ———
with open(MODEL_FILE, 'rb') as f:
    clf = pickle.load(f)

# ——— GPIO setup ———
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)

# buffer for last N RMS values
rms_buf = deque(maxlen=WINDOW_SIZE)
last_label = None

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker:", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    global last_label
    try:
        data = json.loads(msg.payload.decode())
        rms = data["rms"]
    except (ValueError, KeyError):
        return

    rms_buf.append(rms)
    # once we have a full window we could compute entropy+centroid,
    # but here we just demo with RMS alone:
    if len(rms_buf) == WINDOW_SIZE:
        features = np.array([rms, 0.0, 0.0]).reshape(1, -1)
        mat = clf.predict(features)[0]
        print(f"Predicted material: {mat}")

        if last_label and mat != last_label:
            # beep on change
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(BUZZER_PIN, GPIO.LOW)

        last_label = mat

client = mqtt.Client(CLIENT_ID)
# If you need authentication on EMQX, uncomment:
# client.username_pw_set("emqx", "public")

client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("Shutting down")
finally:
    GPIO.cleanup()
