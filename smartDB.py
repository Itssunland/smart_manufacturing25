import subprocess
import json
import time
import math
import sqlite3
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import os

#RUN sqlite3 anomalies.db
#SELECT * FROM anomalies;


# Konfigurasjon
URL = "http://192.168.11.120:8080/get?"
WHAT_TO_GET = ['accX', 'accY', 'accZ']
THRESHOLD = 12.0  # Vibrasjonsterskel
WAIT_AFTER_ANOMALY = 50  # Antall punkter å vente før snapshot
WINDOW_SIZE = 100  # For plott

DB_FILE = "anomalies.db"
SNAPSHOT_FOLDER = "snapshots"

# Lagre data for plotting
vibration_history = deque(maxlen=WINDOW_SIZE)

# Hjelpevariabler
anomaly_active = False
anomaly_counter = 0
last_detected_vibration = 0.0

# Opprett snapshot-mappe hvis den ikke finnes
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Koble til database og lag tabell hvis nødvendig
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

# Lagre hendelse i database
def save_to_database(timestamp, vibration, filename):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO anomalies (timestamp, vibration, snapshot_file)
        VALUES (?, ?, ?)
    ''', (timestamp, vibration, filename))
    conn.commit()
    conn.close()
    print(f"🗂️ Lagret i database: {filename}")

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

def analyze(readings):
    accX = readings['accX']
    accY = readings['accY']
    accZ = readings['accZ']

    total_vibration = math.sqrt(accX**2 + accY**2 + accZ**2)

    print(f"Vibrasjon: {total_vibration:.2f} m/s²", end='\t')

    is_anomaly = total_vibration > THRESHOLD

    if is_anomaly:
        print("🚨 ANOMALI DETEKTERT!")
    else:
        print("✅ Normal vibrasjon.")

    return total_vibration, is_anomaly

def save_plot(fig, timestamp, vibration):
    filename = f"{SNAPSHOT_FOLDER}/anomaly_{timestamp}.png"
    fig.savefig(filename)
    print(f"💾 Saved plot: {filename}")
    save_to_database(timestamp, vibration, filename)

def main_loop():
    global anomaly_active, anomaly_counter, last_detected_vibration

    init_database()

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(0, 20)
    ax.set_xlim(0, WINDOW_SIZE)
    ax.set_ylabel('Vibrasjon (m/s²)')
    ax.set_xlabel('Tid (punkter)')
    ax.set_title('Live Vibrasjonsmåling')

    while True:
        readings = phypox_data()
        if readings:
            total_vibration, is_anomaly = analyze(readings)
            vibration_history.append(total_vibration)

            # Oppdater graf
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
                    print("📍 Starter snapshot-teller.")
                elif total_vibration > last_detected_vibration:
                    anomaly_counter = 0
                    last_detected_vibration = total_vibration
                    print("🔄 Kraftigere anomali oppdaget, restart teller.")

            if anomaly_active:
                anomaly_counter += 1
                if anomaly_counter >= WAIT_AFTER_ANOMALY:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_plot(fig, timestamp, last_detected_vibration)
                    anomaly_active = False
                    anomaly_counter = 0
                    last_detected_vibration = 0.0
                    print("✅ Snapshot lagret og database oppdatert.")

        time.sleep(0.5)

if __name__ == "__main__":
    main_loop()
