# weights/

This folder stores trained ML model files.

## Files

| File | Description | Size (approx) |
|------|-------------|---------------|
| `xgb_risk_model.json` | Trained XGBoost model (500 trees) | ~2–5 MB |
| `scaler.pkl` | StandardScaler fitted on training data | ~1 KB |

## How to generate

```bash
# 1. Prepare your labeled dataset (CSV format — see train.py)
# 2. Run training:
python backend/train.py --data data/labeled_degradation.csv

# Files will appear here automatically.
```

## Note

These files are excluded from git (see .gitignore).
Store them in cloud storage (S3, GCS) and download before running the API.

While weights are missing, the API runs in **demo_mode=True**
which uses an analytical formula instead of the trained model.
