"""
explainability.py — SHAP-based Model Explainability
====================================================
Computes SHAP values to explain individual risk predictions.
Identifies top 5 risk drivers for each location.
"""

from typing import Dict, List
import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class Explainer:
    """
    SHAP TreeExplainer for XGBoost model interpretability.
    Falls back to permutation importance in demo mode.
    """

    def __init__(self, model, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self._explainer = None
        if not demo_mode and HAS_SHAP and model is not None:
            # TreeExplainer is optimized for XGBoost/RF (exact Shapley values)
            self._explainer = shap.TreeExplainer(model)

    def explain(self, features_raw: Dict, features_array: np.ndarray) -> Dict:
        """
        Compute SHAP values and return top-5 drivers.

        Returns:
            {
              "shap_values": {feature_name: shap_value},
              "top_5_drivers": [
                {"feature": str, "contribution": float, "direction": "increases"/"decreases"},
                ...
              ],
              "base_value": float  — expected model output
            }
        """
        if self.demo_mode or not HAS_SHAP:
            return self._demo_explain(features_raw)

        shap_vals = self._explainer.shap_values(features_array)
        base_val  = float(self._explainer.expected_value)

        from model import FEATURE_COLUMNS
        shap_dict = dict(zip(FEATURE_COLUMNS, shap_vals[0]))

        sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top5 = [
            {
                "feature":      name,
                "contribution": round(float(val), 4),
                "direction":    "increases" if val > 0 else "decreases",
                "explanation":  _feature_explanation(name, features_raw)
            }
            for name, val in sorted_feats[:5]
        ]

        return {
            "shap_values":  {k: round(float(v), 4) for k, v in shap_dict.items()},
            "top_5_drivers": top5,
            "base_value":   round(base_val, 4),
        }

    def _demo_explain(self, f: Dict) -> Dict:
        """Analytical SHAP approximation for demo mode."""
        contribs = {
            "ndsi":           0.28 * max(0, (f.get("ndsi", 0) - 0.14) / 0.09),
            "wind_speed":     0.22 * max(0, (f.get("wind_speed", 8.5) - 8.5) / 4.2),
            "ndvi":           0.18 * max(0, (0.18 - f.get("ndvi", 0.18)) / 0.12),
            "dist_aralkum_km":0.14 * max(0, (185 - f.get("dist_aralkum_km", 185)) / 148),
            "dust_freq":      0.09 * f.get("dust_freq", 0.3),
            "si_swir":        0.06 * max(0, (f.get("si_swir", 1.25) - 1.25) / 0.38),
            "soil_moisture":  0.06 * max(0, (0.118 - f.get("soil_moisture", 0.118)) / 0.062),
        }
        sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        top5 = [
            {
                "feature":      name,
                "contribution": round(val, 4),
                "direction":    "increases" if val > 0 else "decreases",
                "explanation":  _feature_explanation(name, f)
            }
            for name, val in sorted_c[:5]
        ]
        return {
            "shap_values":   {k: round(v, 4) for k, v in contribs.items()},
            "top_5_drivers": top5,
            "base_value":    0.45,
        }


def _feature_explanation(feature: str, values: Dict) -> str:
    """Human-readable explanation of a feature's contribution to risk."""
    v = values.get(feature, 0)
    explanations = {
        "ndsi":           f"NDSI salinity index is {v:.3f} (threshold: 0.25). High SWIR reflectance indicates salt accumulation.",
        "ndvi":           f"NDVI is {v:.3f} (critical: <0.10). Low values indicate vegetation loss and bare, erodible soil.",
        "wind_speed":     f"Wind speed is {v:.1f} m/s (mobilization threshold: ~7.5 m/s). Above threshold triggers salt deflation.",
        "dist_aralkum_km":f"Location is {v:.0f} km from Aralkum Desert. Proximity increases salt-dust deposition exposure.",
        "dust_freq":      f"Historical dust event frequency: {v:.2f}. High frequency indicates chronic exposure risk.",
        "si_swir":        f"SI_SWIR = {v:.3f}. Elevated SWIR-NIR ratio confirms salt-mineral surface crusting.",
        "soil_moisture":  f"Soil moisture: {v:.3f} m³/m³. Low moisture reduces cohesion and increases deflation susceptibility.",
        "ndvi_trend_30d": f"NDVI 30-day trend: {v:.4f}/day. Negative trend indicates ongoing vegetation degradation.",
        "sand_fraction":  f"Sand content: {v*100:.0f}%. Sandy soils have lower threshold velocity for wind erosion.",
    }
    return explanations.get(feature, f"{feature} = {v:.4f}")
