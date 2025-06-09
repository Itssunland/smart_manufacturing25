#!/usr/bin/env python3
import os
import math
import time
import sqlite3
import pickle
import logging

from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt

# ——— MQTT SETUP ———
MQTT_BROKER = "broker.emqx.io"
MQTT_TOPIC  = "drill/data"
mqtt_client = mqtt.Client(f"pi-publisher-{np.random.randint(0,1000)}")
mqtt_client.connect(MQTT_BROKER, 1883, 60)

# ——— PATH & LOGGING SETUP ———
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
SNAPSHOT_DIR = os.path.join(BASE_DIR, 'snapshots')
LOG_DIR      = os.path.join(BASE_DIR, 'logs')

os.makedirs(DATA_DIR,     exist_ok=True)
os.makedirs(MODELS_DIR,   exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,      exist_ok=True)

DB_FILE    = os.path.join(DATA_DIR, 'drill_sessions.db')
MODEL_FILE = os.path.join(MODELS_DIR, 'rf_material_clf.pkl')

# ——— USER PARAMETERS ———
WINDOW_SIZE       = 256      # samples per analysis window
SAMPLE_RATE       = 1000     # Hz
SENSOR_INTERVAL   = 1.0 / SAMPLE_RATE
IDLE_THRESHOLD    = 0.2      # m/s² below which we consider "not drilling"
IDLE_WINDOW_SECS  = 1.0      # seconds of idle before ending session
LED_PIN           = 0        # BCM pin for LED

# ——— LOAD RF MODEL ———
with open(MODEL_FILE, 'rb') as mf:
    clf = pickle.load(mf)

# ——— REAL SENSOR FUNCTIONS ———
from smbus2 import SMBus
I2C_ADDR = 0x68

def init_sensor():
    bus = SMBus(1)
    bus.write_byte_data(I2C_ADDR, 0x6B, 0)  # wake up MPU-6050
    time.sleep(0.1)
    logging.info("MPU-6050 initialized on I2C bus 1")
    return bus


def read_vibration(bus):
    data = bus.read_i2c_block_data(I2C_ADDR, 0x3B, 6)
    def conv(h, l):
        v = (h << 8) | l
        return v - 65536 if (v & 0x8000) else v
    ax = conv(data[0],data[1]) / 16384.0 * 9.81
    ay = conv(data[2],data[3]) / 16384.0 * 9.81
    az = conv(data[4],data[5]) / 16384.0 * 9.81
    return math.sqrt(ax*ax + ay*ay + az*az)

# ——— FEATURE EXTRACTION ———
def extract_features(buffer):
    arr      = np.array(buffer) - np.mean(buffer)
    rms      = math.sqrt(np.mean(arr**2))
    yf       = np.abs(np.fft.rfft(arr))
    xf       = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps       = yf / np.sum(yf)
    entropy  = -np.sum(ps * np.log(ps + 1e-12)) / math.log(len(ps))
    centroid = np.sum(xf * ps)
    return rms, entropy, centroid

# ——— DATABASE INIT ———
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute('''
      CREATE TABLE IF NOT EXISTS sessions (
        session_id    INTEGER PRIMARY KEY,
        start_ts      TEXT    NOT NULL,
        end_ts        TEXT,
        snapshot_path TEXT,
        snapshot_ts   TEXT
      )''')
    c.execute('''
      CREATE TABLE IF NOT EXISTS features (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   INTEGER NOT NULL,
        window_idx   INTEGER NOT NULL,
        rms          REAL    NOT NULL,
        entropy      REAL    NOT NULL,
        centroid     REAL    NOT NULL,
        label        TEXT    NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
      )''')
    conn.commit()
    logging.info(f"Database ready at {DB_FILE}")
    return conn

# ——— MAIN LOOP ———
def main():
    start_time = time.time()

    bus = init_sensor()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    logging.info(f"LED initialized on BCM pin {LED_PIN}")

    in_session    = False
    idle_start    = None
    vib_buf       = deque(maxlen=WINDOW_SIZE)
    last_label    = None
    change_points = []
    idx           = 0

    plt.ion()
    fig, (ax_r, ax_e, ax_c) = plt.subplots(3,1,sharex=True,figsize=(6,9))
    ax_r.set_ylabel('RMS Amplitude')
    ax_e.set_ylabel('Spectral Entropy')
    ax_c.set_ylabel('Spectral Centroid (Hz)')
    ax_c.set_xlabel('Window Index')

    try:
        while True:
            vib = read_vibration(bus)

            if not in_session and vib > IDLE_THRESHOLD:
                in_session = True
                idle_start = None
                vib_buf.clear()
                change_points.clear()
                last_label = None
                idx = 0
                conn = init_db()
                cur  = conn.cursor()
                cur.execute("INSERT INTO sessions(start_ts) VALUES(datetime('now'))")
                conn.commit()
                session_id = cur.lastrowid
                logging.info(f"Session {session_id} started")

            if in_session:
                vib_buf.append(vib)
                if len(vib_buf) == WINDOW_SIZE:
                    rms, ent, cen = extract_features(vib_buf)
                    label = clf.predict([[rms, ent, cen]])[0]

                    if last_label and label != last_label:
                        change_points.append((idx, f"{last_label}→{label}"))
                        GPIO.output(LED_PIN, GPIO.HIGH)
                        time.sleep(0.2)
                        GPIO.output(LED_PIN, GPIO.LOW)
                        logging.info(f"Material change: {last_label}→{label} @ window {idx}")
                    last_label = label

                    conn.execute(
                        "INSERT INTO features(session_id,window_idx,rms,entropy,centroid,label) VALUES(?,?,?,?,?,?)",
                        (session_id, idx, rms, ent, cen, label)
                    )
                    conn.commit()

                    # update live plot omitted for brevity
                    idx += 1

                if vib < IDLE_THRESHOLD:
                    if idle_start is None:
                        idle_start = time.time()
                    elif time.time() - idle_start >= IDLE_WINDOW_SECS:
                        conn.execute("UPDATE sessions SET end_ts=datetime('now') WHERE session_id=?", (session_id,))
                        conn.commit()
                        snapshot = os.path.join(SNAPSHOT_DIR, f"session_{session_id}_live.png")
                        fig.savefig(snapshot)
                        logging.info(f"Session {session_id} ended; snapshot → {snapshot}")
                        conn.execute(
                          "UPDATE sessions SET snapshot_path=?,snapshot_ts=datetime('now') WHERE session_id= ?",
                          (snapshot, session_id)
                        )
                        conn.commit()
                        conn.close()
                        in_session = False
                        GPIO.cleanup()
                        plt.pause(2)
                        fig.clear()
                        logging.info("Waiting for next drilling session…")
                else:
                    idle_start = None

            time.sleep(SENSOR_INTERVAL)

    except KeyboardInterrupt:
        logging.info("Interrupted by user; exiting.")
    finally:
        GPIO.cleanup()
        plt.close('all')
        if in_session:
            conn.close()

if __name__ == '__main__':
    main()
