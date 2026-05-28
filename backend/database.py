"""
database.py — Historical Predictions Storage
=============================================
SQLite database for storing risk predictions, enabling:
- Historical trend analysis
- Model performance tracking
- Audit trail for field decisions
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "aralriskai.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables on first run."""
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS predictions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT    NOT NULL,
        lat         REAL    NOT NULL,
        lon         REAL    NOT NULL,
        date        TEXT    NOT NULL,
        risk        REAL    NOT NULL,
        salinity_risk   REAL,
        veg_risk        REAL,
        dust_risk       REAL,
        confidence      REAL,
        features_json   TEXT,
        shap_json       TEXT,
        demo_mode       INTEGER DEFAULT 1
    );

    CREATE INDEX IF NOT EXISTS idx_latlon ON predictions(lat, lon);
    CREATE INDEX IF NOT EXISTS idx_date   ON predictions(date);

    CREATE TABLE IF NOT EXISTS alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL,
        lat         REAL NOT NULL,
        lon         REAL NOT NULL,
        risk        REAL NOT NULL,
        message     TEXT NOT NULL,
        sent        INTEGER DEFAULT 0
    );
    """)
    conn.commit()
    conn.close()


def save_prediction(lat: float, lon: float, date: str,
                    result: Dict, features: Dict, shap: Dict) -> int:
    """Persist a risk prediction. Returns inserted row id."""
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO predictions
            (created_at, lat, lon, date, risk, salinity_risk, veg_risk,
             dust_risk, confidence, features_json, shap_json, demo_mode)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
        round(lat, 5), round(lon, 5), date,
        result.get("degradation_risk_probability"),
        result.get("salinity_risk"),
        result.get("vegetation_loss_risk"),
        result.get("salt_dust_exposure_risk"),
        result.get("confidence"),
        json.dumps(features),
        json.dumps(shap),
        int(result.get("demo_mode", True)),
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()

    # Auto-create alert if risk is critical
    if result.get("degradation_risk_probability", 0) > 0.75:
        save_alert(lat, lon, result["degradation_risk_probability"],
                   f"CRITICAL risk {result['degradation_risk_probability']:.2f} at {lat:.3f}N {lon:.3f}E")
    return row_id


def get_history(lat: float, lon: float,
                radius_deg: float = 0.05, limit: int = 30) -> List[Dict]:
    """
    Fetch prediction history for a location (within radius_deg degrees).
    Used for NDVI trend and risk trend charts.
    """
    conn = get_conn()
    rows = conn.execute("""
        SELECT date, risk, salinity_risk, veg_risk, dust_risk, features_json
        FROM predictions
        WHERE ABS(lat - ?) < ? AND ABS(lon - ?) < ?
        ORDER BY date DESC
        LIMIT ?
    """, (lat, radius_deg, lon, radius_deg, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_alert(lat: float, lon: float, risk: float, message: str):
    conn = get_conn()
    conn.execute("""
        INSERT INTO alerts (created_at, lat, lon, risk, message)
        VALUES (?,?,?,?,?)
    """, (datetime.utcnow().isoformat(), lat, lon, risk, message))
    conn.commit()
    conn.close()


def get_pending_alerts() -> List[Dict]:
    """Fetch unsent alerts (for SMS/email notification pipeline)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM alerts WHERE sent = 0 ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_alert_sent(alert_id: int):
    conn = get_conn()
    conn.execute("UPDATE alerts SET sent = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


# Initialize on import
init_db()
