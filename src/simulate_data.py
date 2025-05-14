#!/usr/bin/env python3
"""
simulate_data.py

Simulates feature data for each material and inserts into
`data/drill_sim_sessions.db`. Generates non-overlapping windows
for Wood, Plastic, Rubber and Brick, and retains all rows.
"""

import math
import os
import sqlite3
from collections import deque
from datetime import datetime

import numpy as np

# ——— CONFIGURATION ———
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
DB_FILE      = os.path.join(DATA_DIR, 'drill_sim_sessions.db')

WINDOW_SIZE           = 256     # samples per analysis window
SAMPLE_RATE           = 1000    # samples per second
SENSOR_INTERVAL       = 1.0 / SAMPLE_RATE
DURATION_PER_MATERIAL = 5.0     # seconds per material segment

MATERIALS = [
    ('Wood',    0.4, 200),
    ('Plastic', 0.8, 400),
    ('Rubber',  0.3, 150),
    ('Brick',   1.2, 800),
]

# ——— DATABASE INITIALIZATION ———
def init_db():
    # 1) Ensure data folder exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2) Connect / create the database file
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    # Use an autoincrement id as primary key, so we never overwrite
    c.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER    NOT NULL,
            window_idx  INTEGER    NOT NULL,
            rms         REAL       NOT NULL,
            entropy     REAL       NOT NULL,
            centroid    REAL       NOT NULL,
            label       TEXT       NOT NULL
        )
    ''')
    conn.commit()
    return conn

# ——— FEATURE EXTRACTION ———
def extract_features(buf):
    arr      = np.array(buf) - np.mean(buf)
    rms      = math.sqrt(np.mean(arr**2))
    yf       = np.abs(np.fft.rfft(arr))
    xf       = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps       = yf / np.sum(yf)
    entropy  = -np.sum(ps * np.log(ps + 1e-12)) / math.log(len(ps))
    centroid = np.sum(xf * ps)
    return rms, entropy, centroid

# ——— SIMULATE A SINGLE MATERIAL SEGMENT ———
def simulate_material(conn, session_id, name, amplitude, freq, duration, start_idx):
    total_samples = int(duration * SAMPLE_RATE)
    buf           = deque(maxlen=WINDOW_SIZE)
    idx           = start_idx

    for i in range(total_samples):
        t   = i / SAMPLE_RATE
        vib = amplitude * math.sin(2 * math.pi * freq * t)
        buf.append(vib)
        if len(buf) == WINDOW_SIZE:
            rms, entropy, centroid = extract_features(buf)
            conn.execute(
                '''
                INSERT INTO features
                  (session_id, window_idx, rms, entropy, centroid, label)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (session_id, idx, rms, entropy, centroid, name)
            )
            idx += 1

    conn.commit()
    return idx

# ——— MAIN SCRIPT ———
def main():
    conn       = init_db()
    session_id = int(datetime.now().timestamp())
    print(f"Simulating session {session_id} into {DB_FILE}")

    next_idx = 0
    for name, amp, freq in MATERIALS:
        print(f"  → Generating segment for {name}")
        next_idx = simulate_material(
            conn, session_id, name, amp, freq,
            DURATION_PER_MATERIAL, start_idx=next_idx
        )

    conn.close()
    print("Simulation complete. All materials inserted.")

if __name__ == '__main__':
    main()
