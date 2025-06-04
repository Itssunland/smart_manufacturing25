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

# Bytt ut buzzeren med en LED på BCM 17
LED_PIN     = 17        # GPIO17 (pinne 11 på header)
WINDOW_SIZE = 256

# ——— Last inn trenet RF-modell ———
with open(MODEL_FILE, 'rb') as f:
    clf = pickle.load(f)

# ——— GPIO-setup ———
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

# buffer for siste N RMS-verdier
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
    # når vi har WINDOW_SIZE elementer, kjør klassifisering
    if len(rms_buf) == WINDOW_SIZE:
        # Her demonstrerer vi med bare RMS som eneste feature
        features = np.array([rms, 0.0, 0.0]).reshape(1, -1)
        mat = clf.predict(features)[0]
        print(f"Predicted material: {mat}")

        if last_label and mat != last_label:
            # blink LED ved endring
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(LED_PIN, GPIO.LOW)

        last_label = mat

client = mqtt.Client(CLIENT_ID)
# Hvis EMQX-broker krever brukernavn/passord, sett det her:
# client.username_pw_set("brukernavn", "passord")

client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("Shutting down")
finally:
    GPIO.cleanup()
