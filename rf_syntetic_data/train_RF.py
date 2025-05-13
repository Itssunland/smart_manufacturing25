import sqlite3
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1) Hent data fra SQLite
DB_FILE = 'drill_sessions.db'
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
    raise RuntimeError("Ingen merkede data funnet. Kjør features_pipeline.py og label først.")

# Del ut X og y
X = np.array([row[:3] for row in rows])
y = np.array([row[3]   for row in rows])

# 2) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 3) Pipeline + GridSearch
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
    pipe, param_grid,
    cv=5, scoring='f1_macro', verbose=2, n_jobs=-1
)
grid.fit(X_train, y_train)

print("Beste parametre:", grid.best_params_)
print("Train F1-macro:", grid.best_score_)

# 4) Evaluer på testsett
y_pred = grid.predict(X_test)
print("Test Classification Report:")
print(classification_report(y_test, y_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5) Lagring av modellen
with open('rf_material_clf.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
print("Modell lagret til rf_material_clf.pkl")
