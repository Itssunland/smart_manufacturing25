#!/usr/bin/env python3
"""
train_RF.py

Train a RandomForestClassifier on your collected drill-session data.

Usage:
  cd <project_root>
  source venv/bin/activate
  python src/train_RF.py
"""

import os
import sqlite3
import pickle

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Paths (relativt til prosjektets rotmappe)
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DB_FILE     = os.path.join(DATA_DIR, 'training_data.db')
OUT_MODEL   = os.path.join(MODELS_DIR, 'rf_material_clf.pkl')

#Output directory
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(db_path):
    """
    Fetch feature-matrix X and label-vector y from SQLite.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT rms, entropy, centroid, label
          FROM features
         WHERE label IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(f"No data found in {db_path}. Run collect_data.py first")

    X = np.array([row[:3] for row in rows], dtype=float)
    y = np.array([row[3]    for row in rows], dtype=str)
    return X, y

def main():
    print(f"Loading data from {DB_FILE} …")
    X, y = load_data(DB_FILE)
    print(f"Total samples: {len(y)}; classes: {np.unique(y)}")

    # Split 70% train, 30% test, stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

    # Pipeline: scaling + RF
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(random_state=42))
    ])

    # GridSearch hyperparametre
    param_grid = {
        'clf__n_estimators':    [50, 100, 200],
        'clf__max_depth':       [None, 10, 20],
        'clf__min_samples_leaf':[1, 3, 5],
    }

    grid = GridSearchCV(
        pipe, param_grid,
        cv=5, scoring='f1_macro',
        verbose=2, n_jobs=-1
    )

    print("Starting GridSearchCV …")
    grid.fit(X_train, y_train)

    print("\n=== Best parameters ===")
    print(grid.best_params_)
    print(f"Train F1-macro: {grid.best_score_:.4f}")

    # Evaluer på test-settet
    print("\n=== Test evaluation ===")
    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Lagre beste modell
    best_model = grid.best_estimator_
    with open(OUT_MODEL, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nSaved trained model to {OUT_MODEL}")

if __name__ == '__main__':
    main()
