import subprocess
import json
import time
import math
import sqlite3
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import os

# RUN in terminal: sqlite3 anomalies.db
# SELECT * FROM anomalies;

# Configuration
URL = "http://192.168.11.120:8080/get?"
WHAT_TO_GET = ['accX', 'accY', 'accZ']
THRESHOLD = 12.0  # Vibration threshold
WAIT_AFTER_ANOMALY = 50  # Number of points to wait before saving a snapshot
WINDOW_SIZE = 100  # For plotting

DB_FILE = "anomalies.db"
SNAPSHOT_FOLDER = "snapshots"

# Store vibration values for plotting
vibration_history = deque(maxlen=WINDOW_SIZE)

# Helper variables
anomaly_active = False
anomaly_counter = 0
last_detected_vibration = 0.0

# Create snapshot folder if it doesn't exist
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Connect to the database and create table if needed
def init_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vibration REAL,
            snapshot_file TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Save anomaly event to database
def save_to_database(timestamp, vibration, filename):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO anomalies (timestamp, vibration, snapshot_file)
        VALUES (?, ?, ?)
    ''', (timestamp, vibration, filename))
    conn.commit()
    conn.close()
    print(f"🗂️ Saved to database: {filename}")

# Fetch data from Phypox
def phypox_data():
    try:
        output = subprocess.check_output(["curl", "-s", URL + '&'.join(WHAT_TO_GET)])
        data = json.loads(output)
        readings = {}
        for item in WHAT_TO_GET:
            readings[item] = data['buffer'][item]['buffer'][0]
        return readings
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None

# Analyze total vibration and detect anomaly
def analyze(readings):
    accX = readings['accX']
    accY = readings['accY']
    accZ = readings['accZ']

    total_vibration = math.sqrt(accX**2 + accY**2 + accZ**2)

    print(f"Vibration: {total_vibration:.2f} m/s²", end='\t')

    is_anomaly = total_vibration > THRESHOLD

    if is_anomaly:
        print("🚨 ANOMALY DETECTED!")
    else:
        print("✅ Normal vibration.")

    return total_vibration, is_anomaly

# Save plot image and record event
def save_plot(fig, timestamp, vibration):
    filename = f"{SNAPSHOT_FOLDER}/anomaly_{timestamp}.png"
    fig.savefig(filename)
    print(f"💾 Saved plot: {filename}")
    save_to_database(timestamp, vibration, filename)

# Main execution loop
def main_loop():
    global anomaly_active, anomaly_counter, last_detected_vibration

    init_database()

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(0, 20)
    ax.set_xlim(0, WINDOW_SIZE)
    ax.set_ylabel('Vibration (m/s²)')
    ax.set_xlabel('Time (points)')
    ax.set_title('Live Vibration Monitoring')

    while True:
        readings = phypox_data()
        if readings:
            total_vibration, is_anomaly = analyze(readings)
            vibration_history.append(total_vibration)

            # Update plot
            line.set_ydata(vibration_history)
            line.set_xdata(range(len(vibration_history)))
            ax.relim()
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            fig.canvas.flush_events()

            if is_anomaly:
                if not anomaly_active:
                    anomaly_active = True
                    anomaly_counter = 0
                    last_detected_vibration = total_vibration
                    print("📍 Snapshot countdown started.")
                elif total_vibration > last_detected_vibration:
                    anomaly_counter = 0
                    last_detected_vibration = total_vibration
                    print("🔄 Stronger anomaly detected, resetting countdown.")

            if anomaly_active:
                anomaly_counter += 1
                if anomaly_counter >= WAIT_AFTER_ANOMALY:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_plot(fig, timestamp, last_detected_vibration)
                    anomaly_active = False
                    anomaly_counter = 0
                    last_detected_vibration = 0.0
                    print("✅ Snapshot saved and database updated.")

        time.sleep(0.5)

if __name__ == "__main__":
    main_loop()
