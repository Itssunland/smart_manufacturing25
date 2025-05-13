#!/usr/bin/env python3
"""
simulate_data.py

A simple script to simulate feature data for each material and insert into the
`features` table of `drill_sessions.db`. Generates non-overlapping windows
for Wood, Plastic, and Brick with known amplitudes and frequencies.

Usage:
  python simulate_data.py

This will:
 1. Create the `features` table if not exists (with label column).
 2. Simulate SESSION_DURATION per material segment (default 5s each).
 3. Compute RMS, spectral entropy, spectral centroid for each window.
 4. Insert all generated feature rows with correct `label` into the database.
"""
import math
import sqlite3
from collections import deque
import numpy as np
from datetime import datetime

# Configuration
db_file = 'drill_sessions.db'
WINDOW_SIZE = 256
SAMPLE_RATE = 1000  # Hz
SENSOR_INTERVAL = 1.0 / SAMPLE_RATE
DURATION_PER_MATERIAL = 5.0  # seconds

# Material definitions: (name, amplitude, frequency Hz)
MATERIALS = [
    ('Wood',    0.4, 200),
    ('Plastic', 0.8, 400),
    ('Rubber',  0.3, 150),
    ('Brick',   1.2, 800)
]

# Create or open database and ensure features table exists
def init_db():
    conn = sqlite3.connect(db_file)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS features (
            session_id INTEGER,
            window_idx INTEGER,
            rms REAL,
            entropy REAL,
            centroid REAL,
            label TEXT,
            PRIMARY KEY(session_id, window_idx, label)
        )
    ''')
    conn.commit()
    return conn

# Feature extraction
def extract_features(buffer):
    arr = np.array(buffer) - np.mean(buffer)
    rms = math.sqrt(np.mean(arr**2))
    yf = np.abs(np.fft.rfft(arr))
    xf = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps = yf / np.sum(yf)
    entropy = -np.sum(ps * np.log(ps + 1e-12)) / math.log(len(ps))
    centroid = np.sum(xf * ps)
    return rms, entropy, centroid

# Simulate one material segment and insert into DB
def simulate_material(conn, sid, material_name, amplitude, freq, duration):
    total_samples = int(duration * SAMPLE_RATE)
    buffer = deque(maxlen=WINDOW_SIZE)
    window_idx = 0
    for i in range(total_samples):
        t = i / SAMPLE_RATE
        vib = amplitude * math.sin(2 * math.pi * freq * t)
        buffer.append(vib)
        if len(buffer) == WINDOW_SIZE:
            rms, entropy, centroid = extract_features(buffer)
            conn.execute(
                'INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?, ?, ?)',
                (sid, window_idx, rms, entropy, centroid, material_name)
            )
            window_idx += 1
    conn.commit()

# Main simulation
if __name__ == '__main__':
    conn = init_db()
    # create a new session_id as timestamp
    sid = int(datetime.now().timestamp())
    print(f"Simulating session {sid} at {datetime.now()}")
    for name, amp, freq in MATERIALS:
        print(f"  Generating {name} segment... ")
        simulate_material(conn, sid, name, amp, freq, DURATION_PER_MATERIAL)
    conn.close()
    print("Simulation complete. Features table populated.")
