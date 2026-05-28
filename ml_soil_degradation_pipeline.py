"""
ML pipeline for next-month soil degradation risk in the Aral Sea region.

The script:
1. loads the prepared CSV dataset,
2. prints and saves basic EDA,
3. trains several regression models with a time-based split,
4. evaluates the models,
5. saves predictions, feature importance, plots, and the best model.

Run from the project root:
    python ml_soil_degradation_pipeline.py

Optional:
    python ml_soil_degradation_pipeline.py --data /path/to/file.csv --output-dir ml_outputs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Keep matplotlib cache inside the project so the script works cleanly in
# restricted environments where the user home cache is not writable.
_MPL_CACHE_DIR = Path(__file__).resolve().parent / "ml_outputs" / "matplotlib_cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_MPL_CACHE_DIR.parent / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR.parent / "xdg_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR.parent / "aral_ml_ready_next_month_risk_2019_2024.csv"

TARGET = "target_risk_next_month"
CLASS_TARGET = "target_class_next_month"

# These are the only columns used as model inputs. Target columns, class labels,
# coordinates, dates, and service/text columns are intentionally excluded.
FEATURE_COLUMNS = [
    "aerosol_index",
    "aod",
    "ndvi",
    "bsi",
    "salinity_index",
    "ndsi",
    "temperature_c",
    "precipitation_mm",
    "soil_moisture",
    "wind_speed",
    "year",
    "month",
    "risk_score",
]


def print_and_save(lines: List[str], output_path: Path) -> None:
    """Print EDA text to console and save the same text into a file."""
    text = "\n".join(lines)
    print(text)
    output_path.write_text(text, encoding="utf-8")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load CSV and normalize key columns to numeric types."""
    df = pd.read_csv(csv_path)

    numeric_columns = FEATURE_COLUMNS + [TARGET, "lat", "lon"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Show basic EDA required for the research project."""
    lines: List[str] = []
    lines.append("BASIC EDA")
    lines.append("=" * 80)
    lines.append(f"Dataset shape: {df.shape}")
    lines.append("\nColumns:")
    lines.extend([f"- {col}" for col in df.columns])

    lines.append("\nMissing values:")
    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("missing_count", ascending=False)
    lines.append(missing.to_string())

    lines.append("\nDescribe numeric columns:")
    lines.append(df.describe().T.to_string())

    if "risk_score" in df.columns:
        lines.append("\nrisk_score distribution:")
        lines.append(df["risk_score"].describe().to_string())

    if TARGET in df.columns:
        lines.append(f"\n{TARGET} distribution:")
        lines.append(df[TARGET].describe().to_string())

    if "risk_class" in df.columns:
        lines.append("\nrisk_class distribution:")
        lines.append(df["risk_class"].value_counts(dropna=False).to_string())

    if CLASS_TARGET in df.columns:
        lines.append(f"\n{CLASS_TARGET} distribution:")
        lines.append(df[CLASS_TARGET].value_counts(dropna=False).to_string())

    print_and_save(lines, output_dir / "eda_summary.txt")


def validate_columns(df: pd.DataFrame) -> None:
    """Fail early if required columns are missing."""
    required = FEATURE_COLUMNS + [TARGET, "year", "month"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use 2019-2023 for training and 2024 for final testing."""
    train_df = df[(df["year"] >= 2019) & (df["year"] <= 2023)].copy()
    test_df = df[df["year"] == 2024].copy()

    # Rows without target cannot be used for supervised training/evaluation.
    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.dropna(subset=[TARGET])

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train or test split is empty. Expected train years 2019-2023 and test year 2024."
        )

    return train_df, test_df


def build_models() -> Dict[str, Pipeline]:
    """Create baseline and tree-based regression models."""
    models: Dict[str, Pipeline] = {
        "LinearRegression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=160,
                        max_depth=18,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBRegressor"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=450,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        print("xgboost is installed: using XGBRegressor.")
    except ImportError:
        models["GradientBoostingRegressor"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=260,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        )
        print("xgboost is not installed: using GradientBoostingRegressor.")

    return models


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(
    models: Dict[str, Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], str, np.ndarray]:
    """Train all models and choose the best one by RMSE, then R2."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET]

    fitted_models: Dict[str, Pipeline] = {}
    rows: List[Dict[str, float | str]] = []
    predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test.to_numpy(), y_pred)
        rows.append({"model": name, **metrics})
        fitted_models[name] = model
        predictions[name] = y_pred
        print(
            f"{name}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}"
        )

    metrics_df = pd.DataFrame(rows).sort_values(["RMSE", "R2"], ascending=[True, False])
    best_name = str(metrics_df.iloc[0]["model"])
    best_pred = predictions[best_name]

    return metrics_df, fitted_models, best_name, best_pred


def get_feature_importance(model: Pipeline, model_name: str) -> pd.DataFrame:
    """Extract feature importance from a tree model or coefficients from linear baseline."""
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        importance_type = "tree_feature_importance"
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_)
        importance_type = "absolute_linear_coefficient"
    else:
        importance = np.zeros(len(FEATURE_COLUMNS))
        importance_type = "not_available"

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": importance,
            "model": model_name,
            "importance_type": importance_type,
        }
    ).sort_values("importance", ascending=False)

    return importance_df


def choose_tree_model_for_importance(
    fitted_models: Dict[str, Pipeline], best_name: str
) -> Tuple[str, Pipeline]:
    """Prefer the best model if it has feature_importances_, otherwise use another tree model."""
    best_estimator = fitted_models[best_name].named_steps["model"]
    if hasattr(best_estimator, "feature_importances_"):
        return best_name, fitted_models[best_name]

    for name, model in fitted_models.items():
        if hasattr(model.named_steps["model"], "feature_importances_"):
            return name, model

    return best_name, fitted_models[best_name]


def save_predictions(test_df: pd.DataFrame, y_pred: np.ndarray, output_dir: Path) -> pd.DataFrame:
    """Save 2024 predictions with coordinates and actual values for analysis."""
    keep_columns = [
        col
        for col in [
            "lat",
            "lon",
            "year",
            "month",
            "date",
            "point_id",
            "risk_score",
            "risk_class",
            TARGET,
            CLASS_TARGET,
        ]
        if col in test_df.columns
    ]

    pred_df = test_df[keep_columns].copy()
    pred_df["predicted_risk_next_month"] = y_pred
    pred_df["residual"] = pred_df[TARGET] - pred_df["predicted_risk_next_month"]
    pred_df.to_csv(output_dir / "aral_predictions_2024.csv", index=False)
    return pred_df


def save_plots(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Create and save all required PNG plots."""
    plt.style.use("default")

    # Distribution of current and next-month risk scores.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["risk_score"].dropna(), bins=35, alpha=0.65, label="risk_score")
    ax.hist(df[TARGET].dropna(), bins=35, alpha=0.65, label=TARGET)
    ax.set_title("Distribution of risk_score and target_risk_next_month")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "risk_score_distribution.png", dpi=160)
    plt.close(fig)

    # Distribution of class labels.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column in zip(axes, ["risk_class", CLASS_TARGET]):
        if column in df.columns:
            counts = df[column].value_counts(dropna=False)
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_title(f"{column} distribution")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=25)
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    # Actual vs predicted.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        pred_df[TARGET],
        pred_df["predicted_risk_next_month"],
        s=10,
        alpha=0.45,
        edgecolors="none",
    )
    min_value = float(min(pred_df[TARGET].min(), pred_df["predicted_risk_next_month"].min()))
    max_value = float(max(pred_df[TARGET].max(), pred_df["predicted_risk_next_month"].max()))
    ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)
    ax.set_title("Actual vs predicted risk, 2024")
    ax.set_xlabel("Actual target risk next month")
    ax.set_ylabel("Predicted risk next month")
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Residuals plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        pred_df["predicted_risk_next_month"],
        pred_df["residual"],
        s=10,
        alpha=0.45,
        edgecolors="none",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Residuals vs predicted risk, 2024")
    ax.set_xlabel("Predicted risk next month")
    ax.set_ylabel("Actual - predicted")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_plot.png", dpi=160)
    plt.close(fig)

    # Feature importance for a tree-based model.
    top_importance = feature_importance_df.head(13).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top_importance["feature"], top_importance["importance"])
    ax.set_title("Feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    # Mean predicted risk by month.
    monthly = (
        pred_df.groupby("month", as_index=False)["predicted_risk_next_month"]
        .mean()
        .sort_values("month")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(monthly["month"], monthly["predicted_risk_next_month"], marker="o")
    ax.set_xticks(monthly["month"])
    ax.set_title("Mean predicted risk by month, 2024")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean predicted risk")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "mean_predicted_risk_by_month.png", dpi=160)
    plt.close(fig)

    # Map-like scatter plot with coordinates. Coordinates are not model features.
    if {"lon", "lat"}.issubset(pred_df.columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            pred_df["lon"],
            pred_df["lat"],
            c=pred_df["predicted_risk_next_month"],
            s=12,
            cmap="YlOrRd",
            alpha=0.75,
            edgecolors="none",
        )
        ax.set_title("Predicted next-month risk map, 2024")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Predicted risk")
        fig.tight_layout()
        fig.savefig(plots_dir / "predicted_risk_map_2024.png", dpi=160)
        plt.close(fig)


def build_summary(
    metrics_df: pd.DataFrame,
    best_name: str,
    feature_importance_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> Dict[str, object]:
    """Create a compact final summary for console and JSON."""
    best_metrics = metrics_df.loc[metrics_df["model"] == best_name].iloc[0].to_dict()
    top_features = feature_importance_df.head(5)[["feature", "importance"]].to_dict("records")

    high_risk_months = (
        pred_df.groupby("month", as_index=False)["predicted_risk_next_month"]
        .mean()
        .sort_values("predicted_risk_next_month", ascending=False)
        .head(3)
        .to_dict("records")
    )

    zone_columns = [col for col in ["lat", "lon", "month", "point_id", "predicted_risk_next_month"] if col in pred_df.columns]
    high_risk_zones = (
        pred_df.sort_values("predicted_risk_next_month", ascending=False)
        .head(8)[zone_columns]
        .to_dict("records")
    )

    return {
        "best_model": best_name,
        "best_metrics": {
            "MAE": round(float(best_metrics["MAE"]), 4),
            "RMSE": round(float(best_metrics["RMSE"]), 4),
            "R2": round(float(best_metrics["R2"]), 4),
        },
        "top_features": top_features,
        "high_risk_months": high_risk_months,
        "high_risk_zones": high_risk_zones,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aral Sea soil degradation risk models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to input CSV.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "ml_outputs", help="Directory for outputs.")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data}")
    df = load_dataset(args.data)
    validate_columns(df)

    run_eda(df, output_dir)
    train_df, test_df = make_time_split(df)
    print(f"\nTrain split: {train_df.shape} (years 2019-2023)")
    print(f"Test split:  {test_df.shape} (year 2024)")
    print(f"Features used: {FEATURE_COLUMNS}")

    models = build_models()
    metrics_df, fitted_models, best_name, best_pred = train_and_evaluate(models, train_df, test_df)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

    best_model = fitted_models[best_name]
    joblib.dump(best_model, output_dir / "best_aral_soil_degradation_model.pkl")

    pred_df = save_predictions(test_df, best_pred, output_dir)

    importance_model_name, importance_model = choose_tree_model_for_importance(fitted_models, best_name)
    feature_importance_df = get_feature_importance(importance_model, importance_model_name)
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    save_plots(df, test_df, pred_df, feature_importance_df, plots_dir)

    summary = build_summary(metrics_df, best_name, feature_importance_df, pred_df)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL COMPARISON")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    print("\nFINAL SUMMARY")
    print("=" * 80)
    print(f"Best model: {summary['best_model']}")
    print(f"Metrics: {summary['best_metrics']}")
    print("Most important features:")
    for item in summary["top_features"]:
        print(f"- {item['feature']}: {item['importance']:.4f}")
    print("Highest-risk months by mean predicted risk:")
    for item in summary["high_risk_months"]:
        print(f"- month {int(item['month'])}: {item['predicted_risk_next_month']:.2f}")
    print("Highest-risk zones in 2024:")
    for item in summary["high_risk_zones"][:5]:
        lat = item.get("lat", "NA")
        lon = item.get("lon", "NA")
        month = item.get("month", "NA")
        risk = item.get("predicted_risk_next_month", "NA")
        print(f"- lat={lat}, lon={lon}, month={month}, predicted risk={risk:.2f}")

    print(f"\nSaved outputs to: {output_dir}")
    print(f"Saved plots to:   {plots_dir}")


if __name__ == "__main__":
    main()
