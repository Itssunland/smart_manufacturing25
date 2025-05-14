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
import logging, os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR  = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'drill_sim.log')

logging.basicConfig(
    level=logging.INFO, #cana be DEBUG or INFO
    format='%(asctime)s %(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()   # still prints to console
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING) #Silence matplotlib font-finder chatter


# ——— PATH SETUP ———
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR       = os.path.join(BASE_DIR, 'data')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
SNAPSHOT_DIR   = os.path.join(BASE_DIR, 'snapshots')

DB_FILE        = os.path.join(DATA_DIR,    'drill_sessions.db')
MODEL_FILE     = os.path.join(MODELS_DIR,  'rf_material_clf.pkl')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ——— USER CONFIGURATION ———
SIMULATE        = True   # set False when running on real sensor
SESSION_DURATION =   5   # seconds
WINDOW_SIZE      = 256   # samples per analysis window
SAMPLE_RATE      =1000   # Hz
SENSOR_INTERVAL = 1.0/SAMPLE_RATE


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
            session_id      INTEGER PRIMARY KEY,
            start_ts        TEXT,
            end_ts          TEXT,
            snapshot_path   TEXT,
            snapshot_ts     TEXT
        )
    ''')
    conn.commit()
    return conn

# Main loop
def main():
    import pickle, sys

    # ——— Load RF model ———
    with open('../models/rf_material_clf.pkl', 'rb') as mf:
        clf = pickle.load(mf)

    # ——— State & Buffers ———
    vib_buf = deque(maxlen=WINDOW_SIZE)
    last_mat = None
    change_points = []      # list of (idx, "from→to")
    times, rms_hist, ent_hist, cen_hist = [], [], [], []
    idx = 0
    start = time.time()
    saw_change = False
    sid = None

    # segment background colors
    seg_colors = {
        'Wood':    'burlywood',
        'Plastic': 'lightgreen',
        'Rubber':  'lightcoral',
        'Brick':   'lightgray',
    }

    # ——— Live plot setup ———
    plt.ion()
    fig_live, (ax_rms, ax_ent, ax_cen) = plt.subplots(
        3, 1, sharex=True, figsize=(6, 9),
        constrained_layout=True
    )
    ax_rms.set_ylabel('RMS Amplitude')
    ax_ent.set_ylabel('Spectral Entropy')
    ax_cen.set_ylabel('Spectral Centroid (Hz)')
    ax_cen.set_xlabel('Window Index')

    # ——— Main loop ———
    while time.time() - start < SESSION_DURATION:
        vib = simulate_vib(time.time() - start) if SIMULATE else read_vibration(bus)
        vib_buf.append(vib)

        if len(vib_buf) == WINDOW_SIZE:
            rms, ent, cen = extract_features(vib_buf)
            mat_pred = clf.predict([[rms, ent, cen]])[0]

            # Lazy‐init session on first window
            if sid is None:
                conn = init_db()
                sid = conn.cursor().execute(
                    "INSERT INTO sessions(start_ts) VALUES(datetime('now'))"
                ).lastrowid
                conn.commit()

            # Detect change
            if last_mat is not None and mat_pred != last_mat:
                saw_change = True
                change_points.append((idx, f"{last_mat}→{mat_pred}"))
            last_mat = mat_pred

            # Only store once we’ve seen at least one change
            if saw_change:
                conn.execute(
                    'INSERT OR REPLACE INTO features '
                    '(session_id, window_idx, rms, entropy, centroid, label) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (sid, idx, rms, ent, cen, mat_pred)
                )
                conn.commit()

            # Update histories
            times.append(idx)
            rms_hist.append(rms)
            ent_hist.append(ent)
            cen_hist.append(cen)

            # Build segments for shading
            segments = []
            prev_start = 0
            prev_mat = None
            for cp_idx, cp_label in change_points:
                frm, to = cp_label.split('→')
                segments.append((prev_start, cp_idx, frm))
                prev_start = cp_idx
                prev_mat = to
            if prev_mat is not None:
                segments.append((prev_start, idx+1, prev_mat))

            # Redraw panels
            for ax, data in zip((ax_rms, ax_ent, ax_cen),
                                (rms_hist, ent_hist, cen_hist)):
                ax.clear()
                ax.plot(times, data, '-o')
                # shade segments
                for s, e, m in segments:
                    ax.axvspan(s, e, color=seg_colors[m], alpha=0.2)
                # draw change lines & labels
                for cp_idx, cp_label in change_points:
                    ax.axvline(cp_idx, color='red', linestyle='--', linewidth=2)
                    y_max = max(data) * 1.02
                    ax.text(cp_idx, y_max, cp_label,
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

    # ——— Tear‐down ———
    if sid is None or not saw_change:
        # No valid session → clean up and exit
        logger.info("No material change detected; nothing saved.")
        if sid is not None:
            # remove the session row we created
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            conn.commit()
        conn.close()
        plt.close('all')
        return

    # We did see a change → finalize
    conn.execute(
        "UPDATE sessions SET end_ts = datetime('now') WHERE session_id = ?",
        (sid,)
    )
    conn.commit()

    # Save and record the final plot
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"session_{sid}_live.png")
    fig_live.savefig(snapshot_path)
    logger.info(f"Saved visualization to {snapshot_path}")

    conn.execute(
        "UPDATE sessions SET snapshot_path = ?, snapshot_ts = datetime('now') "
        "WHERE session_id = ?",
        (snapshot_path, sid)
    )
    conn.commit()
    conn.close()
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')
    logger.info(f"Session {sid} completed with changes and resources saved.")


if __name__ == '__main__':
    main()
