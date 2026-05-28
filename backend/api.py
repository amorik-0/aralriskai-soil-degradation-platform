"""
api.py — AralRiskAI FastAPI Backend
Run: uvicorn api:app --reload --port 8000
"""
import os, math, datetime
from typing import Optional, List
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
    return _demo_proxy(lat,lon), ["DEMO DATA"]

@app.get("/")
def root(): return {"service":"AralRiskAI","version":"2.0.0","demo_mode":DEMO_MODE,"gee":GEE_AVAILABLE}

@app.get("/risk")
def get_risk(lat:float=Query(...,ge=35,le=55), lon:float=Query(...,ge=50,le=75), date:str=Query(default=None)):
    date = date or datetime.date.today().isoformat()
    fv, sources = _get_features(lat, lon, date)
    result = risk_model.predict(fv["raw"])
    return {**result, "lat":round(lat,5), "lon":round(lon,5), "date":date,
            "risk_label":_risk_label(result["degradation_risk_probability"]), "data_sources":sources}

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
