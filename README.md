# AralRiskAI
### Scientific Land Degradation Risk Platform — Aral Sea / Aralkum Desert Region

> Predicts micro-regional land degradation risk caused by salt-dust storms,
> salinization, drought and vegetation loss using Sentinel-2, ERA5 and XGBoost.

---

## Quick Start

### 1. Open the dashboard (no backend needed)
```bash
open frontend/aralriskai.html
```
The dashboard runs fully in the browser using demo data with real scientific formulas.

### 2. Run the backend API
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

### 3. Configure environment
```bash
cp .env.example .env
# Add your GEE credentials and Anthropic API key
```

---

## Project Structure

```
AralRiskAI/
├── frontend/
│   ├── aralriskai.html          # Full dashboard (map + charts + AI chat)
│   ├── css/styles.css           # Dark scientific theme
│   └── js/
│       ├── map.js               # Leaflet + gradient risk overlay
│       ├── charts.js            # Chart.js: NDVI, forecast, salinity, SHAP
│       └── ai.js                # Claude API scientific assistant
│
├── backend/
│   ├── api.py                   # FastAPI: /risk /forecast /explain /recommendations
│   ├── gee_service.py           # Google Earth Engine data extraction
│   ├── feature_engineering.py  # NDVI, NDSI, SI_SWIR, wind, soil features
│   ├── model.py                 # XGBoost training + prediction
│   ├── explainability.py        # SHAP top-5 drivers
│   ├── database.py              # SQLite predictions history
│   ├── train.py                 # Model training script
│   └── requirements.txt
│
├── weights/
│   ├── xgb_risk_model.json      # Trained XGBoost model
│   └── scaler.pkl               # Feature normalizer
│
├── .env                         # API keys (not committed)
├── .gitignore
├── docker-compose.yml
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/risk?lat=&lon=&date=` | Risk score for a point |
| GET | `/risk/region?bbox=&date=` | Risk grid for a region |
| GET | `/forecast?lat=&lon=&days=7` | 30-day risk forecast |
| GET | `/explain?lat=&lon=&date=` | SHAP top-5 drivers |
| GET | `/recommendations?lat=&lon=` | Agronomic recommendations |

### Example
```bash
curl "http://localhost:8000/risk?lat=44.5&lon=59.2&date=2024-06-01"
```
```json
{
  "degradation_risk_probability": 0.7821,
  "salinity_risk": 0.6943,
  "vegetation_loss_risk": 0.7102,
  "salt_dust_exposure_risk": 0.8214,
  "risk_label": "HIGH",
  "confidence": 0.87
}
```

---

## Scientific Formulas

All indices are computed from Sentinel-2 Surface Reflectance bands:

| Index | Formula | Source |
|-------|---------|--------|
| NDVI | `(B8 − B4) / (B8 + B4)` | Rouse et al. 1974 |
| NDSI | `(B11 − B8) / (B11 + B8)` | Khan et al. 2005 |
| SI₁ | `√(B2 × B4)` | Abbas et al. 2013 |
| SI₂ | `√(B3 × B4)` | Abbas et al. 2013 |
| SI_SWIR | `(B4 + B11) / B8` | Allbed & Kumar 2013 |
| Wind speed | `√(u² + v²)` | ERA5 u/v components |

Risk is predicted by XGBoost trained on labeled degradation events.
Features are normalized using regional historical baselines (1991–2023).

---

## Data Sources

| Data | Dataset | Provider |
|------|---------|----------|
| Satellite imagery | `COPERNICUS/S2_SR_HARMONIZED` | ESA / GEE |
| Climate (wind, moisture, temp) | `ECMWF/ERA5_LAND/HOURLY` | ECMWF / GEE |
| Wind components | `ECMWF/ERA5/HOURLY` | ECMWF / GEE |
| Soil texture | SoilGrids v2.0 | ISRIC |
| Land cover | `MODIS/006/MCD12Q1` | NASA / GEE |

---

## Training Your Own Model

```bash
# Prepare labeled CSV (see train.py for required columns)
python backend/train.py --data data/labeled_degradation.csv
# → saves weights/xgb_risk_model.json and weights/scaler.pkl
```

---

## References

- Micklin P. (2007). The Aral Sea Disaster. *Annual Review of Earth and Planetary Sciences*
- Indoitu R. et al. (2012). Dust storms in Central Asia. *Journal of Arid Environments*
- Khan N.M. et al. (2005). Assessment of soil salinity using remote sensing. *Pakistan Journal of Botany*
- Abbas A. et al. (2013). Characterizing soil salinity in irrigated agriculture. *Pedosphere*
- Propastin P. (2012). Spatial non-stationarity and scale dependency. *Remote Sensing of Environment*
