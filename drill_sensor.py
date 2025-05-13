#!/usr/bin/env python3
import math
import time
import sqlite3
import os
import pickle
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ——— USER CONFIGURATION ———
SIMULATE = False               # Use real sensor
SESSION_DURATION = 5           # seconds to run the session
WINDOW_SIZE = 256              # samples per analysis window
SAMPLE_RATE = 1000             # samples per second
SENSOR_INTERVAL = 1.0 / SAMPLE_RATE
DB_FILE = "drill_sessions.db"
MODEL_FILE = "rf_material_clf.pkl"
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# ——— Load trained Random Forest model ———
with open(MODEL_FILE, "rb") as mf:
    clf = pickle.load(mf)

# ——— Sensor setup for MPU-6050 over I²C ———
if not SIMULATE:
    from smbus2 import SMBus
    I2C_ADDR = 0x68

    def init_sensor():
        bus = SMBus(1)
        # Wake up MPU-6050
        bus.write_byte_data(I2C_ADDR, 0x6B, 0)
        time.sleep(0.1)
        return bus

    def read_vibration(bus):
        data = bus.read_i2c_block_data(I2C_ADDR, 0x3B, 6)
        def conv(h, l): 
            v = (h << 8) | l
            return v - 65536 if v & 0x8000 else v
        ax = conv(data[0], data[1]) / 16384.0 * 9.81
        ay = conv(data[2], data[3]) / 16384.0 * 9.81
        az = conv(data[4], data[5]) / 16384.0 * 9.81
        return math.sqrt(ax*ax + ay*ay + az*az)
else:
    # Simulation stub (not used in sensor mode)
    def simulate_vib(elapsed):
        return 0.0

# ——— Feature extraction ———
def extract_features(buf):
    arr = np.array(buf) - np.mean(buf)
    rms = math.sqrt(np.mean(arr**2))
    yf = np.abs(np.fft.rfft(arr))
    xf = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps = yf / np.sum(yf)
    entropy = -np.sum(ps * np.log(ps + 1e-12)) / math.log(len(ps))
    centroid = np.sum(xf * ps)
    return rms, entropy, centroid

# ——— Database setup ———
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
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
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id      INTEGER PRIMARY KEY,
            start_ts        TEXT,
            end_ts          TEXT,
            snapshot_path   TEXT,
            snapshot_ts     TEXT
        )
    ''')
    conn.commit()
    return conn

# ——— Main loop ———
def main():
    # Initialize sensor bus
    bus = init_sensor() if not SIMULATE else None

    # Buffers & state
    vib_buf = deque(maxlen=WINDOW_SIZE)
    last_mat = None
    change_points = []
    times, rms_hist, ent_hist, cen_hist = [], [], [], []
    idx = 0
    start = time.time()
    saw_change = False
    sid = None

    # Background colors per material
    seg_colors = {
        'Wood': 'burlywood',
        'Plastic': 'lightgreen',
        'Rubber': 'lightcoral',
        'Brick': 'lightgray',
    }

    # Live plot setup
    plt.ion()
    fig_live, (ax_rms, ax_ent, ax_cen) = plt.subplots(
        3, 1, sharex=True, figsize=(6, 9), constrained_layout=True
    )
    ax_rms.set_ylabel('RMS Amplitude')
    ax_ent.set_ylabel('Spectral Entropy')
    ax_cen.set_ylabel('Spectral Centroid (Hz)')
    ax_cen.set_xlabel('Window Index')

    # Main acquisition loop
    while time.time() - start < SESSION_DURATION:
        vib = read_vibration(bus)
        vib_buf.append(vib)

        if len(vib_buf) == WINDOW_SIZE:
            rms, ent, cen = extract_features(vib_buf)
            mat_pred = clf.predict([[rms, ent, cen]])[0]

            # Lazy-init DB session
            if sid is None:
                conn = init_db()
                sid = conn.cursor().execute(
                    "INSERT INTO sessions(start_ts) VALUES(datetime('now'))"
                ).lastrowid
                conn.commit()

            # Detect material change
            if last_mat is not None and mat_pred != last_mat:
                saw_change = True
                change_points.append((idx, f"{last_mat}→{mat_pred}"))
            last_mat = mat_pred

            # Store features only after first change
            if saw_change:
                conn.execute(
                    'INSERT OR REPLACE INTO features '
                    '(session_id, window_idx, rms, entropy, centroid, label) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (sid, idx, rms, ent, cen, mat_pred)
                )
                conn.commit()

            # Update history
            times.append(idx)
            rms_hist.append(rms)
            ent_hist.append(ent)
            cen_hist.append(cen)

            # Build shading segments
            segments = []
            prev = 0
            prev_label = None
            for cp_idx, cp_label in change_points:
                frm, to = cp_label.split('→')
                segments.append((prev, cp_idx, frm))
                prev = cp_idx
                prev_label = to
            if prev_label:
                segments.append((prev, idx+1, prev_label))

            # Redraw plots
            for ax, data in zip((ax_rms, ax_ent, ax_cen),
                                (rms_hist, ent_hist, cen_hist)):
                ax.clear()
                ax.plot(times, data, '-o')
                # shade
                for s, e, mat in segments:
                    ax.axvspan(s, e, color=seg_colors.get(mat, 'white'), alpha=0.2)
                # change lines and labels
                for cp_idx, cp_label in change_points:
                    ax.axvline(cp_idx, color='red', linestyle='--', linewidth=2)
                    y_m = max(data) * 1.02
                    ax.text(cp_idx, y_m, cp_label,
                            ha='center', va='bottom',
                            bbox=dict(facecolor='white', alpha=0.7))
            ax_rms.set_ylabel('RMS Amplitude')
            ax_ent.set_ylabel('Spectral Entropy')
            ax_cen.set_ylabel('Spectral Centroid (Hz)')
            ax_cen.set_xlabel('Window Index')

            fig_live.canvas.draw()
            fig_live.canvas.flush_events()
            idx += 1

        time.sleep(SENSOR_INTERVAL)

    # Tear-down
    if sid is None or not saw_change:
        print("No material change detected; nothing saved.")
        if sid is not None:
            conn.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
            conn.commit()
        conn.close()
        plt.close('all')
        return

    # Finalize session
    conn.execute(
        "UPDATE sessions SET end_ts=datetime('now') WHERE session_id=?", (sid,)
    )
    conn.commit()

    # Save snapshot
    snapshot_path = os.path.join(SNAPSHOT_FOLDER, f"session_{sid}_live.png")
    fig_live.savefig(snapshot_path)
    print(f"Saved visualization to {snapshot_path}")

    conn.execute(
        "UPDATE sessions SET snapshot_path=?, snapshot_ts=datetime('now') "
        "WHERE session_id=?", (snapshot_path, sid)
    )
    conn.commit()
    conn.close()

    # Display briefly and exit
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    print(f"Session {sid} completed with changes and resources saved.")

if __name__ == '__main__':
    main()
