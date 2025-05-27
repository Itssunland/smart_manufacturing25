#!/usr/bin/env python3
import math, time, sqlite3
from collections import deque
from datetime import datetime
import numpy as np
from smbus2 import SMBus

WINDOW_SIZE = 256
SAMPLE_RATE = 1000
SENSOR_INTERVAL = 1.0 / SAMPLE_RATE
DB_FILE = "data/training_data.db"

LABELS = ['Wood', 'Plastic', 'Rubber', 'Brick']

# … I2C‐init som i drill_sim.py …
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

def extract_features(buf):
    arr = np.array(buf)-np.mean(buf)
    rms = math.sqrt(np.mean(arr**2))
    yf = np.abs(np.fft.rfft(arr))
    xf = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps = yf/np.sum(yf)
    ent = -np.sum(ps*np.log(ps+1e-12))/math.log(len(ps))
    cen = np.sum(xf*ps)
    return rms, ent, cen

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

    print("Klargjør boring i ett materiale. Velg label fra:", LABELS)
    label = None
    while label not in LABELS:
        label = input("Label for denne økten: ").strip()

    print(f"Starter opptak for {label}. Trykk Ctrl+C for å avslutte…")
    try:
        while True:
            vib = read_vibration(bus)
            buf.append(vib)
            if len(buf)==WINDOW_SIZE:
                rms, ent, cen = extract_features(buf)
                ts = datetime.now().isoformat()
                conn.execute(
                    "INSERT INTO features (timestamp,rms,entropy,centroid,label) "
                    "VALUES (?,?,?,?,?)",
                    (ts, rms, ent, cen, label)
                )
                conn.commit()
            time.sleep(SENSOR_INTERVAL)
    except KeyboardInterrupt:
        print("\nOpptak avsluttet.")
    finally:
        conn.close()

if __name__=='__main__':
    main()
