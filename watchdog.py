#!/usr/bin/env python3
import time, subprocess
from sensor_module import extract_vibration  # en liten wrapper som gir deg én RMS-måling

THRESHOLD = 0.2
HYSTERESIS_TIME = 3.0   # sekunder uten vibrasjon før vi stopper

def main():
    active = False
    last_high = 0.0
    proc = None

    while True:
        rms = extract_vibration()
        now = time.time()

        if not active and rms > THRESHOLD:
            # START økt
            proc = subprocess.Popen(['./drill_sim.py'], env={'VIRTUAL_ENV':'/path/to/.venv', 'PATH':'/path/to/.venv/bin:'+os.environ['PATH']})
            active = True
            print("→ Økt startet")

        if active and rms < THRESHOLD:
            # hvis stabilt lav RMS
            if now - last_high > HYSTERESIS_TIME:
                # STOPP økt
                proc.send_signal(signal.SIGINT)
                proc.wait()
                active = False
                print("← Økt avsluttet")
        else:
            last_high = now

        time.sleep(0.1)

if __name__=='__main__':
    main()
