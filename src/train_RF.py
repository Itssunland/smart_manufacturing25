#!/usr/bin/env python3
import os
import sqlite3
import pickle

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ——— PATHS ———
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
DB_FILE    = os.path.join(DATA_DIR, 'drill_sim_sessions.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILE = os.path.join(MODELS_DIR, 'rf_material_clf.pkl')

# ——— LOAD DATA ———
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute("""
    SELECT rms, entropy, centroid, label
      FROM features
     WHERE label IS NOT NULL
""")
rows = cur.fetchall()
conn.close()

if not rows:
    raise RuntimeError(
        "No labeled data found in features table — run collect_data.py first."
    )

X = np.array([r[:3] for r in rows])
y = np.array([r[3]   for r in rows])

# ——— TRAIN/TEST SPLIT ———
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# ——— PIPELINE & GRID SEARCH ———
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    RandomForestClassifier(random_state=42))
])

param_grid = {
    'clf__n_estimators':    [50, 100, 200],
    'clf__max_depth':       [None, 10, 20],
    'clf__min_samples_leaf':[1, 3, 5],
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='f1_macro',
    verbose=2,
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print(f"Train  F1-macro: {grid.best_score_:.3f}")

# ——— EVALUATE ON TEST SET ———
y_pred = grid.predict(X_test)
print("\nTest Classification Report:")
print(classification_report(y_test, y_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ——— SAVE MODEL ———
os.makedirs(MODELS_DIR, exist_ok=True)
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
print(f"Model saved to {MODEL_FILE}")
