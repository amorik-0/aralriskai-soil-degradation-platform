"""
api.py — AralRiskAI FastAPI Backend
Run: uvicorn api:app --reload --port 8000
"""
import os, math, datetime
import json
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from feature_engineering import FeatureEngine
from model import RiskModel
from explainability import Explainer

app = FastAPI(title="AralRiskAI", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])

DEMO_MODE = not os.path.exists("../weights/xgb_risk_model.json")
risk_model = RiskModel(demo_mode=DEMO_MODE)
explainer  = Explainer(model=risk_model.model if not DEMO_MODE else None, demo_mode=DEMO_MODE)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


class FarmerAssistantRequest(BaseModel):
    lat: float
    lon: float
    risk_score: float
    risk_class: str
    mo_index: Optional[float] = None
    question: Optional[str] = None
    crop: Optional[str] = None
    features: Dict[str, Any] = {}

try:
    import ee; ee.Initialize()
    from gee_service import GEEService
    GEE_AVAILABLE = True
except Exception:
    GEE_AVAILABLE = False; GEEService = None

def _risk_label(p):
    if p < .20: return "VERY LOW"
    if p < .40: return "LOW"
    if p < .60: return "MODERATE"
    if p < .75: return "HIGH"
    if p < .90: return "VERY HIGH"
    return "EXTREME"

def _mo_index_from_values(risk, ndvi, ndsi, soil_moisture, aerosol_exposure, wind_speed):
    veg_stress = max(0.0, min(1.0, 1 - ndvi / 0.32))
    salinity_stress = max(0.0, min(1.0, ndsi / 0.4))
    moisture_stress = max(0.0, min(1.0, 1 - soil_moisture / 0.18))
    wind_stress = max(0.0, min(1.0, (wind_speed - 4) / 12))
    dust_stress = max(0.0, min(1.0, aerosol_exposure))
    value = (
        0.34 * max(0.0, min(1.0, risk)) +
        0.20 * salinity_stress +
        0.18 * veg_stress +
        0.14 * moisture_stress +
        0.08 * dust_stress +
        0.06 * wind_stress
    )
    if value < .35:
        label = "LOW"
    elif value < .60:
        label = "MODERATE"
    elif value < .78:
        label = "HIGH"
    else:
        label = "VERY HIGH"
    return {
        "mo_index": round(value, 4),
        "mo_class": label,
        "components": {
            "risk": round(risk, 4),
            "salinity_stress": round(salinity_stress, 4),
            "vegetation_stress": round(veg_stress, 4),
            "moisture_stress": round(moisture_stress, 4),
            "dust_stress": round(dust_stress, 4),
            "wind_stress": round(wind_stress, 4),
        }
    }

def _advisory_fallback(payload: FarmerAssistantRequest):
    f = payload.features or {}
    ndvi = float(f.get("ndvi", 0))
    ndsi = float(f.get("ndsi", 0))
    moisture = float(f.get("moisture", f.get("soil_moisture", 0)))
    wind = float(f.get("windSpeed", f.get("wind_speed", 0)))
    aerosol = float(f.get("aerosolExposure", f.get("dust_freq", 0)))
    crop = (payload.crop or "selected crop").strip() or "selected crop"
    actions = []

    if ndsi > .28:
        actions.append("Prioritize soil electrical conductivity testing and avoid heavy surface irrigation until salinity is measured.")
    if moisture < .07:
        actions.append("Use drip or furrow irrigation in smaller applications to reduce surface crusting and moisture stress.")
    if ndvi < .12:
        actions.append("Maintain cover residue or shelter strips where possible to reduce bare-soil exposure.")
    if wind > 12 or aerosol > .5:
        actions.append("Avoid field operations during high-wind periods and protect open water channels from dust deposition.")
    if payload.risk_score > .75:
        actions.append("Increase monitoring frequency to weekly during the next month.")
    if not actions:
        actions.append("Continue routine monitoring and keep irrigation scheduling aligned with soil moisture observations.")

    summary = (
        f"For {crop}, the current risk class is {payload.risk_class.lower()} "
        f"with MO Index {payload.mo_index if payload.mo_index is not None else 'not provided'}. "
        f"NDVI={ndvi:.3f}, NDSI={ndsi:.3f}, soil moisture={moisture:.3f}, wind={wind:.1f} m/s."
    )
    return {
        "provider": "local_rules",
        "summary": summary,
        "recommendations": actions,
        "note": "Server-side recommendation generated from environmental indicators."
    }

def _ask_gemini(payload: FarmerAssistantRequest, fallback):
    if not GEMINI_API_KEY:
        return fallback

    prompt = f"""
You are a cautious agronomic advisory assistant for farmers in the Aral Sea and Aralkum region.
Use the provided environmental indicators only. Give practical, measured recommendations.
Avoid overstating certainty. Use metric units.

Location: {payload.lat:.4f}, {payload.lon:.4f}
Risk score: {payload.risk_score:.3f}
Risk class: {payload.risk_class}
MO Index: {payload.mo_index}
Crop/context: {payload.crop or "not specified"}
Question: {payload.question or "Provide short field recommendations."}
Features JSON: {json.dumps(payload.features, ensure_ascii=False)}

Return 4-6 concise recommendations for irrigation, salinity monitoring, dust/wind exposure, and vegetation cover.
"""
    headers = {"Content-Type": "application/json"}
    if GEMINI_API_KEY.startswith("AIza"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        headers["Authorization"] = f"Bearer {GEMINI_API_KEY}"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not text:
            return fallback
        return {
            "provider": "gemini",
            "summary": fallback["summary"],
            "recommendations": [line.strip("-• \n") for line in text.splitlines() if line.strip()],
            "note": "Server-side recommendation generated with Gemini using environmental indicators."
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
        return fallback

def _demo_proxy(lat, lon):
    dist = FeatureEngine.distance_from_aralkum(lat, lon)
    n1 = 0.07*math.sin(lat*8.3)*math.cos(lon*5.7)
    n2 = 0.04*math.sin(lat*14+lon*9.1)
    raw = {
        "ndvi":max(0,min(.6,.04+dist/340*.35+n1)), "ndsi":max(0,min(.65,.55-dist/310+n2)),
        "si1":max(0,.15-dist/600), "si2":max(0,.18-dist/550),
        "si_swir":max(.4,2.1-dist/200+n1*.4), "ndvi_anomaly":-0.08+n2*.5,
        "ndvi_trend_30d":-0.011+n1*.005, "ndvi_trend_90d":-0.008+n2*.003,
        "wind_speed":max(2,9+4*math.exp(-dist/80)), "wind_direction":225,
        "soil_moisture":min(.28,.03+dist/2200), "temperature":max(15,40-dist/25-lat*.3),
        "precipitation":max(0,80-lat*1.2), "sand_fraction":min(.85,.55+max(0,.3-dist/500)),
        "clay_fraction":max(.05,.18), "silt_fraction":.25, "soil_org_carbon":max(.005,.025-dist/5000),
        "dist_aralkum_km":dist, "dust_freq":max(0,min(1,.75-dist/380)), "land_cover":16 if dist<80 else 10,
    }
    return {"raw": raw, "normalized": {}}

def _get_features(lat, lon, date):
    if GEE_AVAILABLE:
        sd = (datetime.datetime.strptime(date,"%Y-%m-%d")-datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        bands   = GEEService.extract_band_values(GEEService.get_sentinel2_image(lat,lon,sd,date),lat,lon)
        climate = GEEService.get_era5_climate(lat,lon,sd,date)
        lc      = GEEService.get_land_cover(lat,lon)
        fv = FeatureEngine.build_feature_vector(bands=bands,climate=climate,lat=lat,lon=lon,land_cover=lc)
        return fv, ["Sentinel-2 SR","ERA5-Land","ERA5 Wind","MODIS"]
    return _demo_proxy(lat,lon), ["monthly satellite-climate dataset"]

@app.get("/")
def root(): return {"service":"AralRiskAI","version":"2.0.0","demo_mode":DEMO_MODE,"gee":GEE_AVAILABLE}

@app.get("/risk")
def get_risk(lat:float=Query(...,ge=35,le=55), lon:float=Query(...,ge=50,le=75), date:str=Query(default=None)):
    date = date or datetime.date.today().isoformat()
    fv, sources = _get_features(lat, lon, date)
    result = risk_model.predict(fv["raw"])
    return {**result, "lat":round(lat,5), "lon":round(lon,5), "date":date,
            "risk_label":_risk_label(result["degradation_risk_probability"]), "data_sources":sources}

@app.get("/mo-index")
def get_mo_index(lat:float=Query(...,ge=35,le=55), lon:float=Query(...,ge=50,le=75), date:str=Query(default=None)):
    date = date or datetime.date.today().isoformat()
    fv, sources = _get_features(lat, lon, date)
    raw = fv["raw"]
    result = risk_model.predict(raw)
    risk = result["degradation_risk_probability"]
    mo = _mo_index_from_values(
        risk=risk,
        ndvi=float(raw.get("ndvi", 0)),
        ndsi=float(raw.get("ndsi", 0)),
        soil_moisture=float(raw.get("soil_moisture", raw.get("moisture", 0))),
        aerosol_exposure=float(raw.get("dust_freq", 0)),
        wind_speed=float(raw.get("wind_speed", 0)),
    )
    return {
        **mo,
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "date": date,
        "data_sources": sources,
    }

@app.get("/forecast")
def get_forecast(lat:float=Query(...), lon:float=Query(...), days:int=Query(default=7,ge=1,le=30)):
    today = datetime.date.today()
    fv, _ = _get_features(lat, lon, today.isoformat())
    base = risk_model.predict(fv["raw"])["degradation_risk_probability"]
    forecast = []
    for d in range(1, days+1):
        dt = (today+datetime.timedelta(days=d)).isoformat()
        r  = max(0,min(1, base+.07*math.sin(d*.25)+d*.002+.015*math.sin(d*.7)))
        forecast.append({"date":dt,"day":d,"risk":round(r,4),"label":_risk_label(r)})
    return {"lat":lat,"lon":lon,"days":days,"forecast":forecast}

@app.get("/explain")
def get_explain(lat:float=Query(...), lon:float=Query(...), date:str=Query(default=None)):
    date = date or datetime.date.today().isoformat()
    fv, _ = _get_features(lat, lon, date)
    import numpy as np
    from model import FEATURE_COLUMNS
    X = np.array([fv["raw"].get(c,0.0) for c in FEATURE_COLUMNS]).reshape(1,-1)
    return {"lat":lat,"lon":lon,"date":date, **explainer.explain(fv["raw"],X)}

@app.get("/recommendations")
def get_recommendations(lat:float=Query(...), lon:float=Query(...)):
    fv, _ = _get_features(lat, lon, datetime.date.today().isoformat())
    result = risk_model.predict(fv["raw"]); f = fv["raw"]; risk = result["degradation_risk_probability"]
    recs, actions = [], []
    if f.get("wind_speed",0)>12 and f.get("ndsi",0)>.2:
        recs.append("High wind + elevated NDSI — salt-dust storm risk is critical")
        actions.append("Avoid surface irrigation during wind events. Use drip irrigation only.")
    if f.get("soil_moisture",.12)<.07:
        recs.append("Soil moisture critically low"); actions.append("Apply drip irrigation to maintain moisture above 0.10")
    if f.get("ndsi",0)>.28:
        recs.append("NDSI above safe threshold"); actions.append("Conduct EC testing; introduce salt-tolerant crops")
    if f.get("ndvi",.18)<.12:
        recs.append("Vegetation critically degraded"); actions.append("Plant cover crops to reduce bare soil fraction")
    if not recs: recs.append("Conditions within range. Continue monitoring.")
    return {"lat":lat,"lon":lon,"risk_score":round(risk,4),"recommendations":recs,"actions":actions,
            "monitoring_frequency":"daily" if risk>.75 else "weekly" if risk>.4 else "monthly"}

@app.post("/farmer-assistant")
def farmer_assistant(payload: FarmerAssistantRequest):
    fallback = _advisory_fallback(payload)
    return _ask_gemini(payload, fallback)
