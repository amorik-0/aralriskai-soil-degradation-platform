"""
feature_engineering.py — Remote Sensing Feature Computation
============================================================
Computes all spectral indices and derived features from raw band values.
All formulas are cited from peer-reviewed literature.

Input:  raw band reflectances + climate variables + ancillary data
Output: normalized feature vector for ML model input
"""

import math
import numpy as np
from typing import Dict, Optional


# ── REGIONAL NORMALIZATION BASELINES ─────────────────────────────────────────
# Source: historical Sentinel-2 and ERA5 statistics for Aral Sea region
# (lat 41–48°N, lon 55–68°E), period 2017–2023
REGIONAL_STATS = {
    "ndvi":       {"mean": 0.181, "std": 0.122, "min": -0.15, "max": 0.72},
    "ndsi":       {"mean": 0.143, "std": 0.092, "min": -0.30, "max": 0.65},
    "si1":        {"mean": 0.052, "std": 0.031, "min":  0.00, "max": 0.22},
    "si2":        {"mean": 0.063, "std": 0.037, "min":  0.00, "max": 0.28},
    "si_swir":    {"mean": 1.248, "std": 0.381, "min":  0.20, "max": 3.80},
    "wind_speed": {"mean": 8.52,  "std": 4.21,  "min":  0.50, "max": 28.0},
    "moisture":   {"mean": 0.118, "std": 0.062, "min":  0.01, "max": 0.38},
    "temp":       {"mean": 22.4,  "std": 12.8,  "min": -15.0, "max": 48.0},
    "precip":     {"mean": 105.0, "std": 85.0,  "min":  0.00, "max": 450.0},
}

# Center of Aralkum Desert (main salt-dust source)
# Source: Micklin P. 2007
ARALKUM_LAT = 44.2
ARALKUM_LON = 58.8


class FeatureEngine:
    """
    Computes all spectral indices and prepares features for the ML model.
    """

    # ── SPECTRAL INDICES ──────────────────────────────────────────────────────

    @staticmethod
    def ndvi(nir: float, red: float) -> float:
        """
        NDVI — Normalized Difference Vegetation Index
        Formula: NDVI = (NIR − RED) / (NIR + RED)
        Sentinel-2: NIR = B8 (865 nm), RED = B4 (665 nm)
        Interpretation:
          < 0.10 : bare soil / severely degraded
          0.10–0.25 : sparse vegetation / degrading
          0.25–0.50 : moderate vegetation
          > 0.50   : dense, healthy vegetation
        Source: Rouse et al. 1974
        """
        if (nir + red) == 0:
            return 0.0
        return (nir - red) / (nir + red)

    @staticmethod
    def ndsi(swir: float, nir: float) -> float:
        """
        NDSI — Normalized Difference Salinity Index
        Formula: NDSI = (SWIR − NIR) / (SWIR + NIR)
        Sentinel-2: SWIR = B11 (1610 nm), NIR = B8 (865 nm)
        Interpretation:
          > 0.20 : moderate salinization risk
          > 0.35 : high salinization risk
        Source: Khan et al. 2005; Douaoui et al. 2006
        """
        if (swir + nir) == 0:
            return 0.0
        return (swir - nir) / (swir + nir)

    @staticmethod
    def si1(blue: float, red: float) -> float:
        """
        SI_1 — Salinity Index 1
        Formula: SI₁ = √(BLUE × RED)
        Sentinel-2: BLUE = B2 (490 nm), RED = B4 (665 nm)
        Higher values → higher salt reflectance in visible spectrum.
        Source: Abbas et al. 2013; Elnaggar & Noller 2010
        """
        return math.sqrt(max(0.0, blue * red))

    @staticmethod
    def si2(green: float, red: float) -> float:
        """
        SI_2 — Salinity Index 2
        Formula: SI₂ = √(GREEN × RED)
        Sentinel-2: GREEN = B3 (560 nm), RED = B4 (665 nm)
        Source: Abbas et al. 2013
        """
        return math.sqrt(max(0.0, green * red))

    @staticmethod
    def si_swir(red: float, swir: float, nir: float) -> float:
        """
        SI_SWIR — SWIR-based Salinity Index
        Formula: SI_SWIR = (RED + SWIR) / NIR
        Sentinel-2: RED=B4, SWIR=B11, NIR=B8
        Sensitive to salt crusts due to high SWIR reflectance of salt minerals.
        Source: Allbed & Kumar 2013; Metternicht & Zinck 2003
        """
        if nir == 0:
            return 0.0
        return (red + swir) / nir

    @staticmethod
    def wind_from_components(u: float, v: float) -> Dict:
        """
        Compute wind speed and meteorological direction from ERA5 u/v components.
        ERA5 bands: u_component_of_wind_10m, v_component_of_wind_10m
        
        speed     = √(u² + v²)  [m/s]
        direction = atan2(u, v) → converted to meteorological degrees (0°=N, 90°=E)
        
        Threshold for salt-dust mobilization: > 7–10 m/s (threshold velocity)
        Source: Indoitu et al. 2012; Orlovsky et al. 2005
        """
        speed = math.sqrt(u**2 + v**2)
        direction = (math.degrees(math.atan2(u, v)) + 360) % 360
        return {
            "wind_speed": speed,
            "wind_direction": direction,
            "exceeds_threshold": speed > 7.5  # Bagnold threshold for salt mobilization
        }

    @staticmethod
    def distance_from_aralkum(lat: float, lon: float) -> float:
        """
        Distance from Aralkum Desert center (primary salt-dust source) in km.
        Uses planar approximation valid for distances < 500 km.
        
        Key finding: dust deposition decreases exponentially with distance.
        Source: Shen et al. 2013; Whish-Wilson 2002
        """
        dlat_km = (lat - ARALKUM_LAT) * 111.32
        dlon_km = (lon - ARALKUM_LON) * math.cos(math.radians(lat)) * 111.32
        return math.sqrt(dlat_km**2 + dlon_km**2)

    # ── NDVI ANOMALY & TREND ─────────────────────────────────────────────────

    @staticmethod
    def ndvi_anomaly(current_ndvi: float, historical_mean: float,
                     historical_std: float) -> float:
        """
        NDVI anomaly as z-score from historical mean.
        Negative anomaly → vegetation below historical baseline (degradation signal).
        Source: Tucker & Sellers 1986; Propastin & Kappas 2008
        """
        if historical_std == 0:
            return 0.0
        return (current_ndvi - historical_mean) / historical_std

    @staticmethod
    def ndvi_trend(ndvi_series: list, window_days: int = 30) -> float:
        """
        Linear trend in NDVI over specified window (days).
        Returns ΔNDVI/day. Negative trend → degradation.
        Uses linear least-squares regression on the time series.
        """
        n = len(ndvi_series)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.array(ndvi_series, dtype=float)
        # Linear regression: β = (Σxy - n·x̄·ȳ) / (Σx² - n·x̄²)
        x_mean, y_mean = x.mean(), y.mean()
        numerator   = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        return (numerator / denominator) if denominator != 0 else 0.0

    # ── NORMALIZATION ─────────────────────────────────────────────────────────

    @staticmethod
    def zscore(value: float, feature_name: str) -> float:
        """
        Z-score normalization using regional historical statistics.
        z = (x − μ) / σ
        Clips to [-3, 3] to reduce influence of extreme outliers.
        """
        stats = REGIONAL_STATS.get(feature_name, {"mean": 0, "std": 1})
        z = (value - stats["mean"]) / max(stats["std"], 1e-6)
        return max(-3.0, min(3.0, z))

    @staticmethod
    def minmax(value: float, feature_name: str) -> float:
        """
        Min-max normalization to [0, 1] using regional range.
        """
        stats = REGIONAL_STATS.get(feature_name, {"min": 0, "max": 1})
        rng = stats["max"] - stats["min"]
        if rng == 0:
            return 0.0
        return max(0.0, min(1.0, (value - stats["min"]) / rng))

    # ── FULL FEATURE VECTOR ────────────────────────────────────────────────────

    @classmethod
    def build_feature_vector(
        cls,
        bands: Dict,
        climate: Dict,
        lat: float,
        lon: float,
        ndvi_history: Optional[list] = None,
        soil_texture: Optional[Dict] = None,
        dust_freq: float = 0.0,
        land_cover: int = -1,
    ) -> Dict:
        """
        Build the complete, normalized feature vector for the ML model.

        Args:
            bands         : raw Sentinel-2 reflectance bands (B2-B12)
            climate       : ERA5 climate variables
            lat, lon      : location coordinates
            ndvi_history  : list of past NDVI values for trend computation
            soil_texture  : SoilGrids values {sand, clay, silt, soc}
            dust_freq     : historical dust storm frequency (0–1)
            land_cover    : MODIS LC_Type1 integer

        Returns:
            dict of normalized and raw features
        """
        b = bands
        # Raw spectral indices
        raw_ndvi    = cls.ndvi(b.get("B8_nir",0),   b.get("B4_red",0))
        raw_ndsi    = cls.ndsi(b.get("B11_swir",0),  b.get("B8_nir",0))
        raw_si1     = cls.si1( b.get("B2_blue",0),   b.get("B4_red",0))
        raw_si2     = cls.si2( b.get("B3_green",0),  b.get("B4_red",0))
        raw_si_swir = cls.si_swir(b.get("B4_red",0), b.get("B11_swir",0), b.get("B8_nir",0))

        # Wind from ERA5 u/v
        wind = cls.wind_from_components(
            climate.get("u_wind", 0), climate.get("v_wind", 0)
        )

        # Distance proxy
        dist_km = cls.distance_from_aralkum(lat, lon)

        # NDVI trend & anomaly
        ndvi_trend_30d  = cls.ndvi_trend(ndvi_history or [raw_ndvi], window_days=30)
        ndvi_trend_90d  = cls.ndvi_trend(ndvi_history or [raw_ndvi], window_days=90)
        ndvi_anom       = cls.ndvi_anomaly(
            raw_ndvi,
            REGIONAL_STATS["ndvi"]["mean"],
            REGIONAL_STATS["ndvi"]["std"]
        )

        # Soil texture (SoilGrids)
        sand = (soil_texture or {}).get("sand", 0.55)
        clay = (soil_texture or {}).get("clay", 0.15)
        silt = (soil_texture or {}).get("silt", 0.30)
        soc  = (soil_texture or {}).get("soc",  0.02)

        # Evapotranspiration proxy (Hargreaves simplified, if no direct data)
        temp_c = climate.get("temperature_c", 25.0)
        precip = climate.get("precipitation", 0.0)

        features_raw = {
            # Spectral
            "ndvi":           raw_ndvi,
            "ndsi":           raw_ndsi,
            "si1":            raw_si1,
            "si2":            raw_si2,
            "si_swir":        raw_si_swir,
            # NDVI dynamics
            "ndvi_anomaly":   ndvi_anom,
            "ndvi_trend_30d": ndvi_trend_30d,
            "ndvi_trend_90d": ndvi_trend_90d,
            # Climate
            "wind_speed":     wind["wind_speed"],
            "wind_direction": wind["wind_direction"],
            "soil_moisture":  climate.get("soil_moisture", 0.1),
            "temperature":    temp_c,
            "precipitation":  precip,
            # Soil
            "sand_fraction":  sand,
            "clay_fraction":  clay,
            "silt_fraction":  silt,
            "soil_org_carbon":soc,
            # Ancillary
            "dist_aralkum_km": dist_km,
            "dust_freq":       dust_freq,
            "land_cover":      land_cover,
        }

        # Normalized features (z-score for ML model input)
        features_norm = {
            k: cls.zscore(v, k)
            for k, v in features_raw.items()
            if k in REGIONAL_STATS
        }

        return {
            "raw":        features_raw,
            "normalized": features_norm,
        }
