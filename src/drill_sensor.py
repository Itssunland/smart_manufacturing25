#!/usr/bin/env python3
import os
import json
import math
import time
import sqlite3
import pickle
import logging

from collections import deque
from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt
from smbus2 import SMBus

# ——— KONFIGURASJON ———
MQTT_BROKER    = "broker.emqx.io"
DATA_TOPIC     = "drill/data"
CTRL_TOPIC     = "drill/control"
mqtt_client = mqtt.Client(f"pi-publisher-{np.random.randint(0,1000)}")
mqtt_client.connect(MQTT_BROKER, 1883, 60)

BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR       = os.path.join(BASE_DIR, 'data')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
LOG_DIR        = os.path.join(BASE_DIR, 'logs')
DB_FILE        = os.path.join(DATA_DIR, 'demodata.db')
MODEL_FILE     = os.path.join(MODELS_DIR, 'rf_material_clf.pkl')

WINDOW_SIZE    = 256
SAMPLE_RATE    = 1000            # Hz
SENSOR_INTERVAL= 1.0 / SAMPLE_RATE
IDLE_THRESHOLD = 0.2             # m/s²
IDLE_WINDOW_SECS = 1.0           # sekunder

# Lag nødvendige mapper
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ——— LAST MODELL ———
with open(MODEL_FILE, 'rb') as f:
    clf = pickle.load(f)

# ——— DATABASEFUNKSJON ———
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
          session_id INTEGER PRIMARY KEY,
          start_ts   TEXT NOT NULL,
          end_ts     TEXT,
          snapshot   TEXT,
          snap_ts    TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS features (
          id         INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id INTEGER NOT NULL,
          window_idx INTEGER NOT NULL,
          rms        REAL NOT NULL,
          entropy    REAL NOT NULL,
          centroid   REAL NOT NULL,
          label      TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn

def extract_features(window):
    arr      = np.array(window) - np.mean(window)
    rms      = math.sqrt(np.mean(arr**2))
    yf       = np.abs(np.fft.rfft(arr))
    xf       = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps       = yf / np.sum(yf)
    entropy  = -np.sum(ps * np.log(ps + 1e-12)) / math.log(len(ps))
    centroid = np.sum(xf * ps)
    return rms, entropy, centroid

# ——— SENSORFUNKSJON ———
def init_sensor():
    bus = SMBus(1)
    bus.write_byte_data(0x68, 0x6B, 0)  # wake up MPU-6050
    time.sleep(0.1)
    logging.info("MPU-6050 initialized")
    return bus

def read_vibration(bus):
    data = bus.read_i2c_block_data(0x68, 0x3B, 6)
    def conv(h, l):
        v = (h << 8) | l
        return v - 65536 if (v & 0x8000) else v
    ax = conv(data[0], data[1]) / 16384.0 * 9.81
    ay = conv(data[2], data[3]) / 16384.0 * 9.81
    az = conv(data[4], data[5]) / 16384.0 * 9.81
    return math.sqrt(ax*ax + ay*ay + az*az)

# ——— GLOBAL STATE ———
session_conn = None
session_id   = None
in_session   = False
idle_start   = None
vib_buf      = deque(maxlen=WINDOW_SIZE)
last_label   = None
window_idx   = 0

# ——— MQTT CALLBACKS ———
def on_connect(client, userdata, flags, rc):
    logging.info("Connected to MQTT, subscribing to %s", DATA_TOPIC)
    client.subscribe(DATA_TOPIC)

def on_message(client, userdata, msg):
    global in_session, idle_start, vib_buf, last_label, window_idx, session_conn, session_id

    if msg.topic != DATA_TOPIC:
        return

    try:
        window = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        logging.error("Invalid JSON data window")
        return

    if len(window) != WINDOW_SIZE:
        logging.warning("Expected window %d, got %d", WINDOW_SIZE, len(window))
        return

    rms, ent, cen = extract_features(window)

    if not in_session and rms >= IDLE_THRESHOLD:
        session_conn = init_db()
        cur = session_conn.cursor()
        cur.execute("INSERT INTO sessions(start_ts) VALUES(datetime('now'))")
        session_conn.commit()
        session_id = cur.lastrowid
        logging.info("Session %d started", session_id)
        in_session = True
        idle_start = None
        vib_buf.clear()
        last_label = None
        window_idx = 0


    if in_session:
        vib_buf.append(rms)  

        label = clf.predict([[rms, ent, cen]])[0]
        if last_label and label != last_label:
            client.publish(CTRL_TOPIC, "CHANGE")
            logging.info("Published CHANGE for %s→%s @ window %d", last_label, label, window_idx)
        last_label = label

        session_conn.execute(
            "INSERT INTO features(session_id,window_idx,rms,entropy,centroid,label) VALUES(?,?,?,?,?,?)",
            (session_id, window_idx, rms, ent, cen, label)
        )
        session_conn.commit()
        window_idx += 1


        if rms < IDLE_THRESHOLD:
            if idle_start is None:
                idle_start = time.time()
            elif time.time() - idle_start >= IDLE_WINDOW_SECS:
                session_conn.execute("UPDATE sessions SET end_ts=datetime('now') WHERE session_id=?", (session_id,))
                session_conn.commit()
                session_conn.close()
                logging.info("Session %d ended", session_id)
                in_session = False
        else:
            idle_start = None

def main():
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    bus = init_sensor()
    mqtt_client.loop_start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        mqtt_client.loop_stop()
        if session_conn:
            session_conn.close()


if __name__ == '__main__':
    main()
