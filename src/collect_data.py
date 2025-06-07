#!/usr/bin/env python3
import math
import time
import sqlite3
from collections import deque
from datetime import datetime

import numpy as np
from smbus2 import SMBus

# ——— PARAMETERS ———
WINDOW_SIZE     = 256
SAMPLE_RATE     = 1000               # Hz
SENSOR_INTERVAL = 1.0 / SAMPLE_RATE
DB_FILE         = "data/training_data.db"

# Materials you can choose from:
LABELS = ['Plywood', 'Marble', 'Gypsum']
# ↑ If you want to rename or add labels, do it here.

# ——— I²C / MPU-6050 setup (same as in drill_sim.py) ———
I2C_ADDR = 0x68

def init_sensor():
    bus = SMBus(1)
    # Wake up MPU-6050 by writing 0 to register 0x6B
    bus.write_byte_data(I2C_ADDR, 0x6B, 0)
    time.sleep(0.1)
    return bus

def read_vibration(bus):
    data = bus.read_i2c_block_data(I2C_ADDR, 0x3B, 6)
    def conv(h, l):
        v = (h << 8) | l
        return v - 65536 if (v & 0x8000) else v

    ax = conv(data[0], data[1]) / 16384.0 * 9.81
    ay = conv(data[2], data[3]) / 16384.0 * 9.81
    az = conv(data[4], data[5]) / 16384.0 * 9.81
    return math.sqrt(ax * ax + ay * ay + az * az)

def extract_features(buf):
    arr = np.array(buf) - np.mean(buf)
    rms = math.sqrt(np.mean(arr**2))
    yf = np.abs(np.fft.rfft(arr))
    xf = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps = yf / np.sum(yf)
    entropy = -np.sum(ps * np.log(ps + 1e-12)) / math.log(len(ps))
    centroid = np.sum(xf * ps)
    return rms, entropy, centroid

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
      CREATE TABLE IF NOT EXISTS features (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT,
        rms         REAL,
        entropy     REAL,
        centroid    REAL,
        label       TEXT
      )
    ''')
    conn.commit()
    return conn

def main():
    bus = init_sensor()
    conn = init_db()
    buf = deque(maxlen=WINDOW_SIZE)

    print("Prepare to drill in one material. Choose a label from:", LABELS)
    label = None
    while label not in LABELS:
        label = input("Label for this session: ").strip()

    print(f"Starting recording for {label}. Press Ctrl+C to stop…")
    try:
        while True:
            vib = read_vibration(bus)
            buf.append(vib)
            if len(buf) == WINDOW_SIZE:
                rms, ent, cen = extract_features(buf)
                ts = datetime.now().isoformat()
                conn.execute(
                    "INSERT INTO features (timestamp, rms, entropy, centroid, label) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ts, rms, ent, cen, label)
                )
                conn.commit()
            time.sleep(SENSOR_INTERVAL)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        conn.close()

if __name__ == '__main__':
    main()
