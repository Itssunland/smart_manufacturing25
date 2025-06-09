#!/usr/bin/env python3
"""
train_MLP.py

Train a small Multilayer Perceptron på drill‐session data,
med LabelEncoder, økt max_iter, early_stopping og utvidet grid av skjulte lag.

Usage:
  cd <project_root>
  source venv/bin/activate
  python src/train_MLP.py
"""

import os
import sqlite3
import pickle

import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DB_FILE     = os.path.join(DATA_DIR, 'drilldata.db')
OUT_MODEL   = os.path.join(MODELS_DIR, 'mlp_material_clf.pkl')

os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(db_path):
    """
    Hent X og y (rms, entropy, centroid, label) fra SQLite.
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
        raise RuntimeError(f"Ingen data funnet i {db_path}. Kjør data-generering først!")

    X = np.array([row[:3] for row in rows], dtype=float)
    y = np.array([row[3]    for row in rows], dtype=str)
    return X, y

def main():
    print(f"Loading data from {DB_FILE} …")
    X, y = load_data(DB_FILE)
    print(f"Total samples: {len(y)}; classes: {np.unique(y)}")


    le = LabelEncoder()
    y_enc = le.fit_transform(y)  
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_enc, test_size=0.3, stratify=y_enc, random_state=42
    )
    print(f"Train size: {len(y_train_enc)}, Test size: {len(y_test_enc)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scalers = {
        'standard': StandardScaler(),
        'robust':   RobustScaler()
    }

    best_overall = {
        'score': 0,
        'params': None,
        'scaler': None,
        'model': None
    }

    for scaler_name, scaler in scalers.items():
        print(f"\n=== Tester MLP med {scaler_name} scaler ===")

        pipe = Pipeline([
            ('scaler', scaler),
            ('mlp',    MLPClassifier(
                            hidden_layer_sizes=(100, 50, 25),
                            activation='relu',
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            max_iter=1500,
                            early_stopping=True,
                            tol=1e-6,
                            random_state=42
             ))
        ])


        param_grid = {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 50, 25)],
            'mlp__activation':         ['relu', 'tanh'],
            'mlp__alpha':              [1e-4, 1e-3, 1e-2],
            'mlp__learning_rate_init': [1e-3, 5e-3, 1e-2],
        }

        grid = GridSearchCV(
            pipe, param_grid,
            cv=skf, scoring='f1_macro',
            verbose=2, n_jobs=-1
        )

        print("Starting GridSearchCV for MLP …")
        grid.fit(X_train, y_train_enc)

        print("\n--- Resultater for", scaler_name, "scaler ---")
        print("Best params:", grid.best_params_)
        print(f"Train F1-macro: {grid.best_score_:.4f}")

        y_pred_enc = grid.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        y_true = le.inverse_transform(y_test_enc)

        print("\n=== Test evaluation ===")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

        if grid.best_score_ > best_overall['score']:
            best_overall.update({
                'score': grid.best_score_,
                'params': grid.best_params_,
                'scaler': scaler_name,
                'model': grid.best_estimator_
            })

    print("\n=== Beste overall ===")
    print("Scaler:", best_overall['scaler'])
    print("Params:", best_overall['params'])
    print(f"Validation F1-macro: {best_overall['score']:.4f}")

    with open(OUT_MODEL, 'wb') as f:
        pickle.dump(best_overall['model'], f)
    print(f"\nSaved best MLP model to {OUT_MODEL}")

if __name__ == '__main__':
    main()
