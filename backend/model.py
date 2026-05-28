"""
model.py — ML Risk Prediction Model
=====================================
XGBoost-based degradation risk predictor.
Outputs continuous probability 0.00–1.00.

Training: labeled historical degradation events (Indoitu et al. dataset +
          remote sensing time series 1990–2023).
Features: 20+ normalized indices from Sentinel-2, ERA5, SoilGrids.
"""

import numpy as np
import pickle
import os
from typing import Dict, Tuple, List

# Optional: sklearn for preprocessing pipeline
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ── FEATURE ORDER ─────────────────────────────────────────────────────────────
# Must match training data column order exactly.
FEATURE_COLUMNS = [
    "ndvi",
    "ndvi_anomaly",
    "ndvi_trend_30d",
    "ndvi_trend_90d",
    "ndsi",
    "si1",
    "si2",
    "si_swir",
    "wind_speed",
    "wind_direction_sin",   # sin(dir) — circular encoding
    "wind_direction_cos",   # cos(dir) — circular encoding
    "soil_moisture",
    "temperature",
    "precipitation",
    "sand_fraction",
    "clay_fraction",
    "silt_fraction",
    "soil_org_carbon",
    "dist_aralkum_km",
    "dust_freq",
    "land_cover",
]

# ── MODEL PATHS ───────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "weights", "xgb_risk_model.json")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "weights", "scaler.pkl")


class RiskModel:
    """
    XGBoost degradation risk model.

    In production: model is loaded from pre-trained weights.
    In demo mode: uses an analytical approximation derived from
                  feature importances in the literature.
    """

    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.model = None
        self.scaler = None

        if not demo_mode:
            self._load_model()

    def _load_model(self):
        """Load trained XGBoost model and scaler from disk."""
        if not HAS_XGB:
            raise ImportError("xgboost is required. Install: pip install xgboost")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights not found at {MODEL_PATH}. "
                "Run train.py first or set demo_mode=True."
            )
        self.model = xgb.XGBRegressor()
        self.model.load_model(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)

    def _encode_wind_direction(self, features_raw: Dict) -> Dict:
        """
        Circular encoding of wind direction to avoid 0/360° discontinuity.
        wind_sin = sin(direction_radians)
        wind_cos = cos(direction_radians)
        This preserves angular relationships in the feature space.
        """
        direction = features_raw.get("wind_direction", 0.0)
        rad = np.radians(direction)
        return {
            **features_raw,
            "wind_direction_sin": float(np.sin(rad)),
            "wind_direction_cos": float(np.cos(rad)),
        }

    def _build_input_array(self, features_raw: Dict) -> np.ndarray:
        """Convert feature dict to ordered numpy array for model input."""
        features = self._encode_wind_direction(features_raw)
        return np.array([features.get(col, 0.0) for col in FEATURE_COLUMNS]).reshape(1, -1)

    def predict(self, features_raw: Dict) -> Dict:
        """
        Predict land degradation risk probability.

        Args:
            features_raw : dict from FeatureEngine.build_feature_vector()['raw']

        Returns:
            {
              "degradation_risk_probability": float 0–1,
              "salinity_risk":               float 0–1,
              "vegetation_loss_risk":         float 0–1,
              "salt_dust_exposure_risk":      float 0–1,
              "confidence":                  float 0–1,
            }
        """
        if self.demo_mode:
            return self._demo_predict(features_raw)

        X = self._build_input_array(features_raw)
        if self.scaler:
            X = self.scaler.transform(X)

        risk = float(self.model.predict(X)[0])
        risk = max(0.0, min(1.0, risk))

        return {
            "degradation_risk_probability": round(risk, 4),
            "salinity_risk":               round(self._salinity_risk(features_raw), 4),
            "vegetation_loss_risk":         round(self._veg_risk(features_raw), 4),
            "salt_dust_exposure_risk":      round(self._dust_risk(features_raw), 4),
            "confidence":                  round(self._confidence(features_raw), 4),
        }

    # ── DEMO PREDICTOR ────────────────────────────────────────────────────────
    def _demo_predict(self, f: Dict) -> Dict:
        """
        Analytical approximation used when trained model is not available.
        Weights derived from Indoitu et al. 2012 feature importances.

        DO NOT USE IN PRODUCTION — replace with trained XGBoost.
        """
        from feature_engineering import REGIONAL_STATS, FeatureEngine

        def z(key):
            return FeatureEngine.zscore(f.get(key, REGIONAL_STATS.get(key, {}).get("mean", 0)), key)

        # Salt source contribution (NDSI + SI_SWIR)
        salt = 0.28 * max(0, z("ndsi")) + 0.08 * max(0, z("si_swir"))
        # Wind erosion × dryness interaction
        wind_drought = 0.22 * max(0, z("wind_speed")) * max(0.1, 1 - f.get("soil_moisture", 0.1) / 0.2)
        # Vegetation degradation (current + trend)
        veg  = 0.18 * max(0, -z("ndvi")) + 0.05 * max(0, -(f.get("ndvi_trend_30d", 0) / 0.03))
        # Proximity to Aralkum source
        prox = 0.14 * max(0, -z("dist_aralkum_km"))
        # Historical dust frequency
        dust = 0.09 * f.get("dust_freq", 0.3)
        # Sandy soil amplifier (erodibility)
        soil = 0.04 * max(0, (f.get("sand_fraction", 0.5) - 0.5))

        raw = salt + wind_drought + veg + prox + dust + soil
        # Sigmoid normalization
        prob = 1 / (1 + np.exp(-2.8 * (raw - 0.55)))

        return {
            "degradation_risk_probability": round(max(0, min(1, float(prob))), 4),
            "salinity_risk":               round(self._salinity_risk(f), 4),
            "vegetation_loss_risk":         round(self._veg_risk(f), 4),
            "salt_dust_exposure_risk":      round(self._dust_risk(f), 4),
            "confidence":                  round(self._confidence(f), 4),
            "demo_mode":                   True,
        }

    # ── SUB-RISK COMPONENTS ───────────────────────────────────────────────────

    def _salinity_risk(self, f: Dict) -> float:
        ndsi = f.get("ndsi", 0)
        si_swir = f.get("si_swir", 1)
        # Threshold: NDSI > 0.25 = high risk; SI_SWIR > 2.0 = high risk
        s = (ndsi / 0.45) * 0.6 + (max(0, si_swir - 1.0) / 2.0) * 0.4
        return max(0.0, min(1.0, s))

    def _veg_risk(self, f: Dict) -> float:
        ndvi = f.get("ndvi", 0.2)
        trend = f.get("ndvi_trend_30d", 0)
        # Risk increases as NDVI falls below regional mean (0.18)
        base = max(0, (0.35 - ndvi) / 0.35)
        trend_factor = max(0, -trend / 0.02)
        return max(0.0, min(1.0, base * 0.7 + trend_factor * 0.3))

    def _dust_risk(self, f: Dict) -> float:
        wind = f.get("wind_speed", 8)
        moisture = f.get("soil_moisture", 0.12)
        dist_km = f.get("dist_aralkum_km", 200)
        dust_freq = f.get("dust_freq", 0.3)
        # High wind + dry soil + near source = high dust exposure
        wind_factor  = max(0, (wind - 7.5) / 12)
        dry_factor   = max(0, 1 - moisture / 0.15)
        prox_factor  = max(0, 1 - dist_km / 400)
        return max(0.0, min(1.0,
            wind_factor * 0.35 + dry_factor * 0.25 + prox_factor * 0.25 + dust_freq * 0.15
        ))

    def _confidence(self, f: Dict) -> float:
        """
        Confidence score based on data completeness and recency.
        Lower confidence if bands are zero (cloud-masked or missing).
        """
        non_zero = sum(1 for k, v in f.items() if isinstance(v, (int, float)) and v != 0)
        completeness = non_zero / max(len(FEATURE_COLUMNS), 1)
        # Additional penalty for cloud-masked bands (B8 = 0 → no NIR)
        nir_valid = 1.0 if f.get("ndvi", 0) != 0 else 0.5
        return round(completeness * nir_valid, 3)


    # ── TRAINING ──────────────────────────────────────────────────────────────

    @staticmethod
    def train(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> "RiskModel":
        """
        Train XGBoost regressor on labeled degradation dataset.

        Args:
            X_train : (n_samples, n_features) training features
            y_train : (n_samples,) labels — continuous risk 0–1
            X_val   : validation features
            y_val   : validation labels

        Returns:
            Fitted RiskModel instance

        XGBoost hyperparameters tuned via 5-fold cross-validation on
        historical Aral Sea degradation dataset.
        """
        if not HAS_XGB:
            raise ImportError("Install xgboost: pip install xgboost")

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:logistic",  # bounds output to 0–1
            eval_metric="rmse",
            early_stopping_rounds=30,
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save_model(MODEL_PATH)

        rm = RiskModel(demo_mode=False)
        rm.model = model
        return rm
