#!/usr/bin/env python3
import numpy as np
import math
import time
from collections import deque
import matplotlib.pyplot as plt

# ——— USER CONFIGURATION ———
MATERIALS = [
    ('Wood',    1.0, 200),  # 200 Hz
    ('Plastic', 0.6, 400),  # 400 Hz
    ('Brick',   1.8, 800),  # 800 Hz
]
# Define test sequence for simulation
TEST_SEQUENCE = MATERIALS + MATERIALS[::-1] + MATERIALS
WINDOW_SIZE      = 256      # samples per analysis window
SAMPLE_RATE      = 1000     # samples per second
SEGMENT_DURATION = 0.5      # seconds each material lasts
PAUSE_BETWEEN    = 0.0      # no pause between segments
# ————————————————————————


def generate_stream(seq, sample_rate, duration, window_size):
    total_samples = int(sample_rate * duration)
    buffer = deque(maxlen=window_size)
    for name, amp, freq in seq:
        t = np.arange(total_samples) / sample_rate
        segment = amp * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(total_samples)
        for s in segment:
            buffer.append(s)
            if len(buffer) == window_size:
                yield list(buffer), name
        time.sleep(PAUSE_BETWEEN)

def generate_real_stream(sample_rate, window_size):
    buffer = deque(maxlen=window_size)
    
# Compute features: RMS, spectral entropy, spectral centroid

def extract_features(buf):
    vals = np.array(buf) - np.mean(buf)
    rms = math.sqrt(np.mean(vals**2))
    yf = np.abs(np.fft.rfft(vals))
    xf = np.fft.rfftfreq(len(vals), d=1/SAMPLE_RATE)
    ps = yf / np.sum(yf)
    entropy = -np.sum(ps * np.log(ps + 1e-12)) / np.log(len(ps))
    centroid = np.sum(xf * ps)
    return rms, entropy, centroid

if __name__ == '__main__':
    # Collect features
    rms_vals, ent_vals, cen_vals, times, mats = [], [], [], [], []
    for idx, (buf, mat) in enumerate(generate_stream(TEST_SEQUENCE, SAMPLE_RATE, SEGMENT_DURATION, WINDOW_SIZE)):
        rms, ent, cen = extract_features(buf)
        t = idx * SEGMENT_DURATION + SEGMENT_DURATION/2
        rms_vals.append(rms)
        ent_vals.append(ent)
        cen_vals.append(cen)
        times.append(t)
        mats.append(mat)
        print(f"Segment {idx+1} ({mat}): RMS={rms:.2f}, Ent={ent:.3f}, Centroid={cen:.1f} Hz")

    total_duration = len(mats) * SEGMENT_DURATION
    # Detect change boundaries and labels
    boundaries, labels = [], []
    prev_mat = mats[0]
    labels.append(prev_mat)
    for i in range(1, len(mats)):
        if mats[i] != prev_mat:
            boundary = i * SEGMENT_DURATION
            boundaries.append(boundary)
            prev_mat = mats[i]
            labels.append(prev_mat)

    # Prepare color mapping per material
    cmap = plt.get_cmap('tab10')
    unique = [m[0] for m in MATERIALS]
    color_map = {name: cmap(i) for i, name in enumerate(unique)}

    # Plot: three panels
    fig, axes = plt.subplots(3,1, figsize=(10,8), sharex=True)
    # Panel 1: RMS
    axes[0].plot(times, rms_vals, 'o-')
    axes[0].set_ylabel('RMS')
    axes[0].set_title('RMS Amplitude Over Time')
    # Panel 2: Entropy
    axes[1].plot(times, ent_vals, 'o-', color='C1')
    axes[1].set_ylabel('Spectral Entropy')
    axes[1].set_title('Spectral Entropy Over Time')
    # Panel 3: Spectral Centroid
    axes[2].plot(times, cen_vals, 'o-', color='C2')
    axes[2].set_ylabel('Spectral Centroid (Hz)')
    axes[2].set_title('Spectral Centroid Over Time')
    axes[2].set_xlabel('Time (s)')

    # Shade segments: same material = same color
    for ax in axes:
        start = 0
        for boundary, label in zip(boundaries + [total_duration], labels):
            ax.axvspan(start, boundary, alpha=0.3, color=color_map[label])
            ax.axvline(boundary, color='k', linestyle='--')
            ax.text((start + boundary)/2, ax.get_ylim()[1]*0.9, label,
                    ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6))
            start = boundary

    plt.tight_layout()
    plt.show()