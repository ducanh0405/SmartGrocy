#!/usr/bin/env python3
"""
Simple Chart Generator
======================
Generate report charts using matplotlib (no heavy dependencies).

Author: SmartGrocy Team
Date: 2025-11-18
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("WARNING: matplotlib not available")
    MATPLOTLIB_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def generate_feature_importance():
    """Generate feature importance chart."""
    logger.info("Generating feature importance chart...")

    # Sample SHAP data
    features = {
        "rolling_mean_24_lag_1": 0.384,
        "sales_quantity_lag_1": 0.176,
        "dow_sin": 0.103,
        "rolling_mean_168_lag_1": 0.093,
        "sales_quantity_lag_24": 0.020,
        "rolling_std_168_lag_1": 0.019,
        "dow_cos": 0.016,
        "rolling_std_24_lag_1": 0.016,
        "sales_quantity_lag_48": 0.009,
    }

    # Sort and format
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [f.replace("_", " ").title() for f, _ in sorted_features]
    values = [v for _, v in sorted_features]

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#2196F3", edgecolor="#1976D2")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title("Top 10 Features by Importance")
    plt.tight_layout()

    output_dir = Path("reports/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved: {output_dir / 'feature_importance.png'}")


def generate_model_performance():
    """Generate model performance chart."""
    logger.info("Generating model performance chart...")

    # Sample metrics
    metrics = {
        "Q05": {"mae": 0.750, "rmse": 1.196},
        "Q25": {"mae": 0.462, "rmse": 0.771},
        "Q50": {"mae": 0.384, "rmse": 0.653},
        "Q75": {"mae": 0.438, "rmse": 0.716},
        "Q95": {"mae": 0.761, "rmse": 1.111},
    }

    quantiles = list(metrics.keys())
    mae_values = [metrics[q]["mae"] for q in quantiles]
    rmse_values = [metrics[q]["rmse"] for q in quantiles]

    # Create chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MAE
    ax1.bar(quantiles, mae_values, color="#2196F3", edgecolor="#1976D2")
    ax1.set_ylabel("MAE")
    ax1.set_title("Mean Absolute Error by Quantile")
    ax1.grid(axis="y", alpha=0.3)

    # RMSE
    ax2.bar(quantiles, rmse_values, color="#4CAF50", edgecolor="#388E3C")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Root Mean Square Error by Quantile")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_dir = Path("reports/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "model_performance.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved: {output_dir / 'model_performance.png'}")


def generate_predictions_distribution():
    """Generate predictions distribution chart."""
    logger.info("Generating predictions distribution...")

    # Sample prediction data
    np.random.seed(42)
    forecasts = np.random.gamma(2, 50, 1000)  # Realistic distribution

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(forecasts, bins=50, color="#2196F3", edgecolor="#1976D2", alpha=0.7)
    ax.set_xlabel("Forecast Value (units)")
    ax.set_ylabel("Frequency")
    ax.set_title("Demand Forecast Distribution")
    ax.axvline(
        np.mean(forecasts), color="red", linestyle="--", label=f"Mean: {np.mean(forecasts):.1f}"
    )
    ax.axvline(
        np.median(forecasts),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(forecasts):.1f}",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_dir = Path("reports/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "predictions_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved: {output_dir / 'predictions_distribution.png'}")


def main():
    """Generate all charts."""

    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available. Install: pip install matplotlib")
        return

    logger.info("\n" + "=" * 70)
    logger.info("GENERATING CHARTS")
    logger.info("=" * 70 + "\n")

    try:
        generate_feature_importance()
        generate_model_performance()
        generate_predictions_distribution()

        logger.info("\n" + "=" * 70)
        logger.info("✅ All charts generated successfully")
        logger.info("📁 Location: reports/charts/")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error generating charts: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
