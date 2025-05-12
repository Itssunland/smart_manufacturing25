#!/usr/bin/env python3
import math
import time
import sqlite3
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

# ——— USER CONFIGURATION ———
SIMULATE = True                # Use simulated data or real sensor
SESSION_DURATION = 5           # seconds to run the session
WINDOW_SIZE = 256              # samples per analysis window
SAMPLE_RATE = 1000             # samples per second
SENSOR_INTERVAL = 1.0 / SAMPLE_RATE

# Thresholds: (name, rms_th, entropy_th, centroid_th)
MATERIAL_THRESHOLDS = [
    ('Wood',    0.5,  0.8, 200),
    ('Plastic', 1.0,  0.9, 400),
    ('Brick',   1.5,  0.7, 800),
]
DB_FILE = "drill_sessions.db"
SNAPSHOT_FOLDER = "snapshots"

# Simulation sequence
SIM_SEQUENCE = [
    ('Wood',    0.4, 200, 2.0),
    ('Plastic', 0.8, 400, 2.0),
    ('Brick',   1.2, 800, 2.0),
]
CUM_DUR = np.cumsum([d for _,_,_,d in SIM_SEQUENCE])
TOTAL_TIME = CUM_DUR[-1]

os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

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
        def conv(h,l): v=(h<<8)|l; return v-65536 if v&0x8000 else v
        ax = conv(data[0],data[1])/16384.0*9.81
        ay = conv(data[2],data[3])/16384.0*9.81
        az = conv(data[4],data[5])/16384.0*9.81
        return math.sqrt(ax*ax+ay*ay+az*az)
else:
    def simulate_vib(elapsed):
        t = min(elapsed, TOTAL_TIME)
        idx = int(np.searchsorted(CUM_DUR, t))
        mat, amp, freq, dur = SIM_SEQUENCE[idx]
        t_in = t - (CUM_DUR[idx]-dur)
        return amp*math.sin(2*math.pi*freq*t_in), mat

# Feature extraction & classification
def extract_features(buf):
    arr = np.array(buf)-np.mean(buf)
    rms = math.sqrt(np.mean(arr**2))
    yf = np.abs(np.fft.rfft(arr))
    xf = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps = yf/np.sum(yf)
    entropy = -np.sum(ps*np.log(ps+1e-12))/math.log(len(ps))
    centroid = np.sum(xf*ps)
    return rms, entropy, centroid

def classify(rms, entropy, centroid):
    for name, r_th, e_th, c_th in MATERIAL_THRESHOLDS:
        if rms<r_th and entropy<e_th and centroid<c_th:
            return name
    return MATERIAL_THRESHOLDS[-1][0]

# Database setup
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS sessions(session_id INTEGER PRIMARY KEY, start_ts TEXT, end_ts TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS features(session_id INTEGER, window_idx INTEGER, rms REAL, entropy REAL, centroid REAL, PRIMARY KEY(session_id, window_idx))')
    conn.commit()
    return conn

# Main loop with final snapshot
def main():
    conn = init_db()
    sid = conn.cursor().execute("INSERT INTO sessions(start_ts) VALUES(datetime('now'))").lastrowid
    conn.commit()
    print(f"Session {sid} started at {datetime.now()}")

    vib_buf = deque(maxlen=WINDOW_SIZE)
    if not SIMULATE:
        bus = init_sensor()

    hist_r, hist_e, hist_c = [], [], []
    times = []
    transition_indices = []
    transition_labels = []
    last_mat = None
    start_time = time.time()

    plt.ion()
    fig_live, ax_live = plt.subplots(figsize=(6, 4))

    while time.time() - start_time < SESSION_DURATION:
        vib, mat = (simulate_vib(time.time()-start_time) if SIMULATE else (read_vibration(bus), None))
        vib_buf.append(vib)
        if len(vib_buf) == WINDOW_SIZE:
            rms, ent, cen = extract_features(vib_buf)
            conn.execute('INSERT OR REPLACE INTO features VALUES(?,?,?,?,?)', (sid, len(hist_r), rms, ent, cen))
            conn.commit()
            # detect transitions
            if last_mat is not None and mat != last_mat:
                idx = len(hist_r)
                transition_indices.append(idx)
                transition_labels.append(f"{last_mat}→{mat}")
            last_mat = mat
            hist_r.append(rms)
            hist_e.append(ent)
            hist_c.append(cen)
            times.append(len(hist_r)-1)
            # live plot
            ax_live.clear()
            ax_live.plot(times, hist_r, '-o')
            for idx in transition_indices:
                ax_live.axvline(idx, color='red', linestyle='--', linewidth=2)
            ax_live.set_ylabel('RMS Amplitude')
            ax_live.set_xlabel('Window Index')
            fig_live.canvas.draw()
            fig_live.canvas.flush_events()
        time.sleep(SENSOR_INTERVAL)

    # end session
    conn.cursor().execute("UPDATE sessions SET end_ts=datetime('now') WHERE session_id=?", (sid,))
    conn.commit()
    conn.close()
    print(f"Session {sid} ended at {datetime.now()} - saving snapshot...")

    # final 3-panel snapshot with multiple red lines and labels
    fig2, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    datasets = [(hist_r, 'RMS Amplitude'), (hist_e, 'Spectral Entropy'), (hist_c, 'Spectral Centroid (Hz)')]
    for ax, (data, title) in zip(axes, datasets):
        ax.plot(times, data, '-o')
        for idx, label in zip(transition_indices, transition_labels):
            ax.axvline(idx, color='red', linestyle='--', linewidth=2)
            ax.text(idx, max(data)*1.02, label, ha='center', va='bottom', backgroundcolor='white')
        ax.set_ylabel(title)
    axes[-1].set_xlabel('Window Index')
    fig2.tight_layout()
    fname = f"{SNAPSHOT_FOLDER}/session_{sid}_final_3panel.png"
    fig2.savefig(fname)
    plt.show()

if __name__ == '__main__':
    main()
