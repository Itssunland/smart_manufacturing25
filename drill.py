#!/usr/bin/env python3
import numpy as np
import math
import time
from collections import deque
from smbus2 import SMBus

# ——— USER CONFIGURATION ———
I2C_ADDR = 0x68                 # I2C address for MPU6050
WINDOW_SIZE = 256               # samples per analysis window
SAMPLE_RATE = 1000              # samples per second
# Calibration thresholds: (name, rms_th, entropy_th, centroid_th)
MATERIAL_THRESHOLDS = [
    ('Wood',    0.5,  0.8, 300),
    ('Plastic', 1.0,  0.9, 500),
    ('Brick',   1.5,  0.7, 700),
]
# ————————————————————————

# Initialize I2C bus and sensor
def init_sensor():
    bus = SMBus(1)
    # Wake up MPU6050
    bus.write_byte_data(I2C_ADDR, 0x6B, 0)
    time.sleep(0.1)
    return bus

# Read raw accelerometer data and return total vibration (m/s^2)
def read_vibration(bus):
    data = bus.read_i2c_block_data(I2C_ADDR, 0x3B, 6)
    def conv(high, low):
        v = (high << 8) | low
        return v - 65536 if v & 0x8000 else v
    ax = conv(data[0], data[1]) / 16384.0 * 9.81
    ay = conv(data[2], data[3]) / 16384.0 * 9.81
    az = conv(data[4], data[5]) / 16384.0 * 9.81
    return math.sqrt(ax*ax + ay*ay + az*az)

# Feature extraction
SENSOR_INTERVAL = 1.0 / SAMPLE_RATE
def extract_features(buffer):
    vals = np.array(buffer) - np.mean(buffer)
    rms = math.sqrt(np.mean(vals**2))
    yf = np.abs(np.fft.rfft(vals))
    xf = np.fft.rfftfreq(len(vals), d=SENSOR_INTERVAL)
    ps = yf / np.sum(yf)
    entropy = -np.sum(ps * np.log(ps + 1e-12)) / math.log(len(ps))
    centroid = np.sum(xf * ps)
    return rms, entropy, centroid

# Classify material based on calibrated thresholds
def classify_material(rms, entropy, centroid):
    for name, rms_th, ent_th, cent_th in MATERIAL_THRESHOLDS:
        if rms < rms_th and entropy < ent_th and centroid < cent_th:
            return name
    return MATERIAL_THRESHOLDS[-1][0]

if __name__ == '__main__':
    bus = init_sensor()
    vib_buffer = deque(maxlen=WINDOW_SIZE)
    last_material = None

    print("Starting real-time material detection...")
    while True:
        vib = read_vibration(bus)
        vib_buffer.append(vib)
        if len(vib_buffer) == WINDOW_SIZE:
            rms, ent, cen = extract_features(vib_buffer)
            material = classify_material(rms, ent, cen)
            if last_material is None:
                print(f"Detected material: {material}")
            elif material != last_material:
                print(f"MATERIAL CHANGE: {last_material} -> {material}")
            last_material = material
        time.sleep(SENSOR_INTERVAL)
