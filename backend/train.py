"""
train.py — XGBoost Model Training Script
==========================================
Run this script to train the risk model on your labeled dataset.
Saves weights/xgb_risk_model.json and weights/scaler.pkl

Usage:
  python train.py --data data/labeled_degradation.csv

CSV columns required:
  lat, lon, date, ndvi, ndsi, si1, si2, si_swir,
  ndvi_anomaly, ndvi_trend_30d, ndvi_trend_90d,
  wind_speed, wind_direction, soil_moisture,
  temperature, precipitation, sand_fraction,
  clay_fraction, silt_fraction, soil_org_carbon,
  dist_aralkum_km, dust_freq, land_cover,
  degradation_risk  ← TARGET (0.0–1.0)
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

from model import FEATURE_COLUMNS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Circular encoding of wind direction (removes 0/360 discontinuity)
    df["wind_direction_sin"] = np.sin(np.radians(df["wind_direction"]))
    df["wind_direction_cos"] = np.cos(np.radians(df["wind_direction"]))

    X = df[FEATURE_COLUMNS].fillna(0).values
    y = df["degradation_risk"].clip(0, 1).values
    return X, y, df


def train(csv_path: str):
    X, y, df = load_data(csv_path)

    # ── Normalize ──────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Split ──────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ── Train XGBoost ──────────────────────────────────────────
    # Hyperparameters tuned via 5-fold CV on Aral Sea degradation dataset
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:logistic",  # bounds output to [0,1]
        eval_metric="rmse",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # ── Evaluate ───────────────────────────────────────────────
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    r2   = r2_score(y_val, preds)
    print(f"\nValidation RMSE: {rmse:.4f}")
    print(f"Validation R²:   {r2:.4f}")

    # Feature importance
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 features:")
    for name, imp in top5:
        print(f"  {name:<30} {imp:.4f}")

    # ── Save weights ───────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path  = os.path.join(OUTPUT_DIR, "xgb_risk_model.json")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")

    model.save_model(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nModel saved  → {model_path}")
    print(f"Scaler saved → {scaler_path}")
    print("\nRestart api.py to load the new weights.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    args = parser.parse_args()
    train(args.data)
