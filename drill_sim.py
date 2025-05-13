#!/usr/bin/env python3
import math
import time
import sqlite3
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# ——— USER CONFIGURATION ———
SIMULATE = True                # Use simulated data or real sensor
SESSION_DURATION = 5           # seconds to run the session
WINDOW_SIZE = 256              # samples per analysis window
SAMPLE_RATE = 1000             # samples per second
SENSOR_INTERVAL = 1.0 / SAMPLE_RATE
DB_FILE = "drill_sessions.db"
SNAPSHOT_FOLDER = "snapshots"
MODEL_FILE = 'rf_material_clf.pkl'  # path to trained RF model

# Load trained Random Forest model
with open(MODEL_FILE, 'rb') as mf:
    clf = pickle.load(mf)

# Sensor or simulation
if not SIMULATE:
    from smbus2 import SMBus
    I2C_ADDR = 0x68
    def init_sensor():
        bus = SMBus(1)
        bus.write_byte_data(I2C_ADDR, 0x6B, 0)
        time.sleep(0.1)
        return bus
    def read_vibration(bus):
        data = bus.read_i2c_block_data(I2C_ADDR, 0x3B, 6)
        def conv(h, l): v = (h << 8) | l; return v - 65536 if v & 0x8000 else v
        ax = conv(data[0], data[1]) / 16384.0 * 9.81
        ay = conv(data[2], data[3]) / 16384.0 * 9.81
        az = conv(data[4], data[5]) / 16384.0 * 9.81
        return math.sqrt(ax*ax + ay*ay + az*az)
else:
    # same simulate_vib as before
    SIM_SEQUENCE = [
        ('Wood',    0.4, 200, 2.0),
        ('Plastic', 0.8, 400, 2.0),
        ('Brick',   1.2, 800, 2.0),
    ]
    CUM_DUR = np.cumsum([d for *_, d in SIM_SEQUENCE])
    TOTAL_TIME = CUM_DUR[-1]
    def simulate_vib(elapsed):
        t = min(elapsed, TOTAL_TIME)
        idx = int(np.searchsorted(CUM_DUR, t))
        _, amp, freq, dur = SIM_SEQUENCE[idx]
        t_in = t - (CUM_DUR[idx] - dur)
        return amp * math.sin(2 * math.pi * freq * t_in)

# Feature extraction
def extract_features(buf):
    arr = np.array(buf) - np.mean(buf)
    rms = math.sqrt(np.mean(arr**2))
    yf = np.abs(np.fft.rfft(arr))
    xf = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps = yf / np.sum(yf)
    entropy = -np.sum(ps * np.log(ps + 1e-12)) / math.log(len(ps))
    centroid = np.sum(xf * ps)
    return rms, entropy, centroid

# Database setup
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Recreate features table to include label column (session_id, window_idx, rms, entropy, centroid, label)
    c.execute('DROP TABLE IF EXISTS features')
    c.execute('''
        CREATE TABLE features (
            session_id INTEGER,
            window_idx INTEGER,
            rms REAL,
            entropy REAL,
            centroid REAL,
            label TEXT,
            PRIMARY KEY(session_id, window_idx)
        )
    ''')
    # Sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY,
            start_ts TEXT,
            end_ts TEXT
        )
    ''')
    conn.commit()
    return conn

# Main loop
def main():
    conn = init_db()
    sid = conn.cursor().execute("INSERT INTO sessions(start_ts) VALUES(datetime('now'))").lastrowid
    conn.commit()
    print(f"Session {sid} started at {datetime.now()}")

    vib_buf = deque(maxlen=WINDOW_SIZE)
    if not SIMULATE:
        bus = init_sensor()

    plt.ion()
    fig_live, ax_live = plt.subplots(figsize=(6,4))
    times, rms_hist = [], []

    start = time.time()
    idx = 0
    while time.time() - start < SESSION_DURATION:
        vib = simulate_vib(time.time()-start) if SIMULATE else read_vibration(bus)
        vib_buf.append(vib)
        if len(vib_buf) == WINDOW_SIZE:
            rms, ent, cen = extract_features(vib_buf)
            # predict via RF model
            mat_pred = clf.predict([[rms, ent, cen]])[0]
            print(f"Window {idx}: Predicted material = {mat_pred}")
            # store features and prediction
            conn.execute('INSERT OR REPLACE INTO features VALUES(?,?,?,?,?,?)',
                         (sid, idx, rms, ent, cen, mat_pred))
            conn.commit()
            # update live RMS plot
            times.append(idx)
            rms_hist.append(rms)
            ax_live.clear()
            ax_live.plot(times, rms_hist, '-o')
            ax_live.set_ylabel('RMS Amplitude')
            ax_live.set_xlabel('Window Index')
            fig_live.canvas.draw(); fig_live.canvas.flush_events()
            idx += 1
        time.sleep(SENSOR_INTERVAL)

    # finalize session
    conn.execute("UPDATE sessions SET end_ts=datetime('now') WHERE session_id=?", (sid,))
    conn.commit()
    conn.close()
    print(f"Session {sid} ended at {datetime.now()}")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
