"""
indoor_localization.py — Multi-floor KNN WiFi localization.
Supports: ground, floor1, floor2, floor3, floor4
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'static', 'data')

FLOOR_DATASETS = {
    'ground': 'wifi_dataset_ground.csv',
    'floor1': 'wifi_dataset_floor1.csv',
    'floor2': 'wifi_dataset_floor2.csv',
    'floor3': 'wifi_dataset_floor3.csv',
    'floor4': 'wifi_dataset_floor4.csv',
}

_models         = {}
_label_encoders = {}
_feature_cols   = {}


def train_model(floor: str):
    if floor not in FLOOR_DATASETS:
        raise ValueError(f'Unknown floor: {floor}')
    csv_path = os.path.join(DATA_DIR, FLOOR_DATASETS[floor])
    df = pd.read_csv(csv_path).dropna()
    feat_cols = [c for c in df.columns if c != 'location']
    X = df[feat_cols].values
    y = df['location'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    mdl = KNeighborsClassifier(n_neighbors=3)
    mdl.fit(X, y_enc)
    _models[floor]         = mdl
    _label_encoders[floor] = le
    _feature_cols[floor]   = feat_cols
    print(f"[IndoorLocalization] {floor}: {len(df)} samples, {len(le.classes_)} locations")


def _ensure(floor: str):
    if floor not in _models:
        train_model(floor)


def predict_location(rssi_values: list, floor: str = 'ground') -> dict:
    _ensure(floor)
    X = np.array(rssi_values).reshape(1, -1)
    pred_enc   = _models[floor].predict(X)[0]
    location   = _label_encoders[floor].inverse_transform([pred_enc])[0]
    proba      = _models[floor].predict_proba(X)[0]
    confidence = round(float(proba.max()), 2)
    return {'location': location, 'confidence': confidence}


def get_all_locations(floor: str = 'ground') -> list:
    _ensure(floor)
    return list(_label_encoders[floor].classes_)


def simulate_rssi(location_name: str, floor: str = 'ground') -> list:
    _ensure(floor)
    csv_path = os.path.join(DATA_DIR, FLOOR_DATASETS[floor])
    df = pd.read_csv(csv_path).dropna()
    subset = df[df['location'] == location_name]
    if subset.empty:
        return [-70, -70, -70]
    feat_cols = [c for c in df.columns if c != 'location']
    return subset[feat_cols].mean().round(0).astype(int).tolist()


# Pre-train all floors on import
for _fl in FLOOR_DATASETS:
    train_model(_fl)
