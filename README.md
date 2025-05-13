# Drill Material Detection

This project measures drilling vibration data and classifies the material being drilled based on three key signal features:

## Measured Features

1. **RMS Amplitude**

   * Definition: Root-mean-square of the vibration signal in each analysis window.
   * What it captures: Overall vibration energy. Higher mass-density materials generate larger vibration amplitudes.

2. **Spectral Entropy**

   * Definition: Normalized Shannon entropy of the power spectral density.
   * What it captures: Signal complexity. More heterogeneous materials produce broader spectra and higher entropy.

3. **Spectral Centroid**

   * Definition: Frequency-domain center of mass of the power spectrum (in Hz).
   * What it captures: Dominant vibration frequency. Harder materials shift spectral energy toward higher frequencies.

## Material Classification

Classification thresholds are calibrated per material:

| Material | RMS Threshold | Entropy Threshold | Centroid Threshold (Hz) |
| -------- | ------------- | ----------------- | ----------------------- |
| Wood     | 0.5           | 0.8               | 200                     |
| Plastic  | 1.0           | 0.9               | 400                     |
| Brick    | 1.5           | 0.7               | 800                     |
| Rubber   | 1.5           | 0.7               | 800                     |


1. Compute the three features for each window of sensor data.
2. Compare to thresholds in order: Wood → Plastic → Brick.
3. Assign the first material whose all thresholds (`rms <`, `entropy <`, `centroid <`) are met.

## Usage

1. Install dependencies:

   ```bash
   pip install smbus2 numpy matplotlib
   ```
2. Run simulation or real sensor mode:

   ```bash
   python drill_sim.py
   ```
3. Review live plot, transitions snapshots, and final visualization.

## Output

* **SQLite Database (`drill_sessions.db`)**:

  * `sessions`: start/end timestamps
  * `features`: per-window RMS, entropy, centroid
  * `transitions`: detected material changes and snapshot image paths
* **Snapshots Folder**: PNG images showing the exact transition point in RMS curve.

## File explination
rf_syntetic_data/simulate_data.py -> simulate trainingdata
rf_syntetic_data/train_RF -> training random forest model
--> rf_material_cld.pkl -> trained model
drill_sim.py -> simulates test and uses trained model

---

*Erlend Dragland, 2025*
