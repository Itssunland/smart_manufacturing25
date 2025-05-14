# Drill Material Detection

This project captures drilling vibration data with a Raspberry Pi + MPU-6050 accelerometer, extracts three signal features, and classifies the drilled material in real time using a pretrained Random Forest model.

---

## Features

1. **RMS Amplitude**  
   *Definition:* Root-mean-square of the vibration signal in each analysis window.  
   *Captures:* Overall vibration energy—denser or harder materials yield higher RMS.

2. **Spectral Entropy**  
   *Definition:* Shannon entropy of the normalized power spectral density.  
   *Captures:* Signal complexity—more heterogeneous materials produce broader spectra.

3. **Spectral Centroid**  
   *Definition:* “Center of mass” of the power spectrum (Hz).  
   *Captures:* Dominant vibration frequency—harder materials shift energy upward.

---

## Repo Structure

```text
smart_manufacturing25/
├── data/  
│   └── drill_sim_sessions.db       # SQLite DB for live sessions
├── models/  
│   └── rf_material_clf.pkl         # Pretrained RandomForest model
├── snapshots/                      # Auto-saved PNGs per session
├── src/  
│   ├── drill_sim.py                # Live sensor acquisition & plotting
│   ├── simulate_data.py            # Simulate & populate database
│   ├── collect_data.py             # (Optional) manual data collection
│   └── train_rf.py                 # Train RF model on `data/training_data.db`
├── requirements.txt                # Python dependencies
├── .gitignore                      
└── README.md

## Material Classification

Classification thresholds are calibrated per material:

| Material | RMS Threshold | Entropy Threshold | Centroid Threshold (Hz) |
| -------- | ------------- | ----------------- | ----------------------- |
| Wood     | 0.5           | 0.8               | 200                     |
| Plastic  | 1.0           | 0.9               | 400                     |
| Rubber   | 0.3           | 0.95              | 150                     |
| Brick    | 1.5           | 0.7               | 800                     |


1. Compute the three features for each window of sensor data.
2. Compare to thresholds in order: Wood → Plastic → Rubber → Brick.
3. Assign the first material whose all thresholds (`rms <`, `entropy <`, `centroid <`) are met.

## Usage

1. Install dependencies:

   ```bash
   pip install smbus2 numpy matplotlib scikit-learn paho-mqtt
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
