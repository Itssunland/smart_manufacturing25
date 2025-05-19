#!/usr/bin/env python3
"""
drill_sensor_preemptive.py

Like drill_sensor.py but gives a preemptive buzzer warning
when we're estimated to be ~T_PREWARN seconds from penetrating
into the next material layer.
"""
import os
import math
import time
import sqlite3
import pickle
import logging

from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ——— GPIO SETUP (fallback on macOS) ———
try:
    import RPi.GPIO as GPIO
except ModuleNotFoundError:
    class _FakeGPIO:
        BCM = OUT = LOW = HIGH = None
        def setmode(self,*a,**k): pass
        def setup(self,*a,**k): pass
        def output(self,*a,**k): pass
        def cleanup(self,*a,**k): pass
    GPIO = _FakeGPIO()
    print("⚠️  RPi.GPIO not found—running without buzzer support")

# ——— PATH & LOGGING SETUP ———
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
SNAPSHOT_DIR = os.path.join(BASE_DIR, 'snapshots')
LOG_DIR      = os.path.join(BASE_DIR, 'logs')
for d in (DATA_DIR, MODELS_DIR, SNAPSHOT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

LOG_FILE   = os.path.join(LOG_DIR, 'drill_preemptive.log')
DB_FILE    = os.path.join(DATA_DIR, 'drill_sessions.db')
MODEL_FILE = os.path.join(MODELS_DIR, 'rf_material_clf.pkl')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ——— USER PARAMETERS ———
SIMULATE         = True     # True→use synthetic vib, False→real MPU-6050
WINDOW_SIZE      = 256      # samples per analysis window
SAMPLE_RATE      = 1000     # Hz
SENSOR_INTERVAL  = 1.0/SAMPLE_RATE
IDLE_THRESHOLD   = 0.2      # below this → consider “not drilling”
IDLE_WINDOW_SECS = 1.0      # secs of idle → end session
BUZZER_PIN       = 27       # BCM pin

# preemptive parameters
T_PREWARN   = 0.2           # seconds before actual threshold to pre-warn
SLOPE_TH    = 0.0           # if slope <=0, skip
PREWARNED   = False

# your material thresholds (must match classifier)
MATERIAL_THRESHOLDS = {
    'Wood':    (0.5,  0.8, 200),
    'Plastic': (1.0,  0.9, 400),
    'Rubber':  (0.3, 0.95, 150),
    'Brick':   (math.inf, math.inf, math.inf)  # fallback
}

# ——— LOAD RF MODEL ———
with open(MODEL_FILE,'rb') as f:
    clf = pickle.load(f)
logger.info(f"Loaded RF model from {MODEL_FILE}")

# ——— SENSOR or SIMULATION ———
if not SIMULATE:
    from smbus2 import SMBus
    I2C_ADDR = 0x68
    def init_sensor():
        bus = SMBus(1)
        bus.write_byte_data(I2C_ADDR,0x6B,0)  # wake MPU
        time.sleep(0.1)
        return bus
    def read_vibration(bus):
        data = bus.read_i2c_block_data(I2C_ADDR,0x3B,6)
        def conv(h,l):
            v=(h<<8)|l
            return v-65536 if v&0x8000 else v
        ax = conv(data[0],data[1])/16384*9.81
        ay = conv(data[2],data[3])/16384*9.81
        az = conv(data[4],data[5])/16384*9.81
        return math.sqrt(ax*ax+ay*ay+az*az)
else:
    # synthetic sequence
    SIM_SEQUENCE = [('Wood',0.4,200,2.0),('Plastic',0.8,400,2.0),
                    ('Rubber',0.3,150,2.0),('Brick',1.2,800,2.0)]
    CUM_DUR = np.cumsum([d for *_,d in SIM_SEQUENCE])
    TOTAL   = float(CUM_DUR[-1])
    def init_sensor(): return None
    def read_vibration(_bus=None):
        t = min(time.time()-start_time, TOTAL)
        idx = int(np.searchsorted(CUM_DUR, t))
        _, amp, freq, dur = SIM_SEQUENCE[idx]
        t_in = t-(CUM_DUR[idx]-dur)
        return amp*math.sin(2*math.pi*freq*t_in)

# ——— FEATURE EXTRACTION ———
def extract_features(buf):
    arr     = np.array(buf)-np.mean(buf)
    rms     = math.sqrt(np.mean(arr**2))
    yf      = np.abs(np.fft.rfft(arr))
    xf      = np.fft.rfftfreq(len(arr), d=SENSOR_INTERVAL)
    ps      = yf/np.sum(yf)
    ent     = -np.sum(ps*np.log(ps+1e-12))/math.log(len(ps))
    cen     = np.sum(xf*ps)
    return rms, ent, cen

# ——— DB INIT ———
def init_db():
    conn=sqlite3.connect(DB_FILE)
    c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions(
                   id INTEGER PRIMARY KEY, start_ts TEXT, end_ts TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS features(
                   id INTEGER PRIMARY KEY, sess INTEGER,
                   win INTEGER, rms REAL, ent REAL, cen REAL, lbl TEXT,
                   FOREIGN KEY(sess) REFERENCES sessions(id))''')
    conn.commit(); return conn

# ——— MAIN ———
def main():
    global start_time
    start_time = time.time()
    bus = init_sensor()

    # buzzer setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
    logger.info(f"Buzzer on pin {BUZZER_PIN}")

    # session state
    in_sess      = False
    idle_start   = None
    buf          = deque(maxlen=WINDOW_SIZE)
    prev_rms     = 0.0
    prewarned    = False
    final_warned = False
    idx          = 0

    # live plot
    plt.ion()
    fig, (ax_r,ax_e,ax_c) = plt.subplots(3,1,sharex=True,figsize=(6,9))
    ax_r.set_ylabel('RMS'); ax_e.set_ylabel('Entropy'); ax_c.set_ylabel('Centroid (Hz)')
    ax_c.set_xlabel('Window')

    try:
        while True:
            vib = read_vibration(bus)
            # detect session start
            if not in_sess:
                if vib> IDLE_THRESHOLD:
                    in_sess = True
                    buf.clear(); idx=0; prewarned=final_warned=False
                    conn=init_db(); cur=conn.cursor()
                    cur.execute("INSERT INTO sessions(start_ts) VALUES(datetime('now'))")
                    conn.commit(); sess=cur.lastrowid
                    logger.info(f"Session {sess} start")
                else:
                    time.sleep(SENSOR_INTERVAL); continue

            # inside session
            buf.append(vib)
            if len(buf)==WINDOW_SIZE:
                rms,ent,cen = extract_features(buf)
                lbl = clf.predict([[rms,ent,cen]])[0]

                # --- PREEMPTIVE LOGIC ---
                # slope
                slope = (rms-prev_rms)/SENSOR_INTERVAL if prev_rms>0 else 0
                # distance to threshold
                r_th = MATERIAL_THRESHOLDS[lbl][0]
                if slope>0:
                    t_to = (r_th - rms)/slope
                else:
                    t_to = math.inf
                # prewarn
                if not prewarned and t_to< T_PREWARN:
                    GPIO.output(BUZZER_PIN,GPIO.HIGH); time.sleep(0.1)
                    GPIO.output(BUZZER_PIN,GPIO.LOW)
                    prewarned = True
                    logger.info(f"🔔 Pre-warn at window {idx}, ~{t_to:.2f}s to {lbl}")

                # final warn when actual cross
                if not final_warned and rms>=r_th:
                    GPIO.output(BUZZER_PIN,GPIO.HIGH); time.sleep(0.3)
                    GPIO.output(BUZZER_PIN,GPIO.LOW)
                    final_warned=True
                    logger.info(f"🚨 Final warn at window {idx}, crossed {lbl}")

                prev_rms = rms

                # store only after first prewarn
                if prewarned:
                    conn.execute("INSERT INTO features(sess,win,rms,ent,cen,lbl) VALUES(?,?,?,?,?,?)",
                                 (sess,idx,rms,ent,cen,lbl))
                    conn.commit()

                # update live plot
                for ax, series in ((ax_r,'r'),(ax_e,'e'),(ax_c,'c')):
                    pass  # you can copy your shading+plot logic here

                idx+=1

            # detect session end
            if vib< IDLE_THRESHOLD:
                if idle_start is None: idle_start=time.time()
                elif time.time()-idle_start> IDLE_WINDOW_SECS:
                    conn.execute("UPDATE sessions SET end_ts=datetime('now') WHERE id=?", (sess,))
                    conn.commit(); conn.close()
                    logger.info(f"Session {sess} end")
                    break
            else:
                idle_start=None

            time.sleep(SENSOR_INTERVAL)

    except KeyboardInterrupt:
        logger.info("User stopped.")
    finally:
        GPIO.cleanup()
        plt.close('all')

if __name__=='__main__':
    main()
