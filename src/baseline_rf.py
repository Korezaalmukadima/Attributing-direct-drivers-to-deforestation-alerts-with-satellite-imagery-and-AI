"""Benchmark metode konvensional (Random Forest) - saran Dr. Slachter: mulai dari baseline."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def features(X):
    f = [X.mean((2, 3)), X.std((2, 3)), X.min((2, 3)), X.max((2, 3))]
    return np.concatenate(f, axis=1)


def fit_rf(Xtr, ytr, seed=0):
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed)
    return rf.fit(features(Xtr), ytr)


def predict_rf(rf, X):
    return rf.predict(features(X))
