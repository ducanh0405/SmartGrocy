import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_pinball_loss
import warnings
import time
import sys
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, List

# === DEFINE PROJECT ROOT ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# ===============================

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# -----------------------------------------------------------------
CONFIG = {
    "data_file": PROJECT_ROOT / 'data' / '3_processed' / 'master_feature_table.parquet',
    "model_output_path": PROJECT_ROOT / 'models' / 'final_forecaster.joblib',
    "features_output_path": PROJECT_ROOT / 'models' / 'model_features.json',
    "metrics_output_path": PROJECT_ROOT / 'reports' / 'metrics' / 'final_model_metrics.json',
    "tuning_iterations": 20,
    "cv_folds": 3
}


# -----------------------------------------------------------------
# 2. FUNCTIONAL DEFINITIONS (All print/logging in English)
# -----------------------------------------------------------------

def load_data(filepath):
    """Loads the clean feature table from the processing pipeline."""
    print(f"[load_data] Loading data from: {filepath}...")  # SỬA LỖI TV
    start_time = time.time()
    try:
        if str(filepath).endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif str(filepath).endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        print(f"OK. Load complete. Shape: {df.shape}. (Took {time.time() - start_time:.2f}s)")  # SỬA LỖI EMOJI
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found {filepath}.")  # SỬA LỖI EMOJI
        print("Please run the data processing pipeline (_02_feature_enrichment.py) first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading file: {e}")  # SỬA LỖI EMOJI
        sys.exit(1)


def prepare_data(df):
    """
    Filters, creates target variable (sales), selects features
    from all integrated Workstreams, and splits data BY TIME (leak-safe).
    
    CRITICAL CHANGE: Uses time-based split instead of random shuffle.
    Cutoff: 80th percentile of WEEK_NO (Option A from requirements).
    """
    print("[prepare_data] Preparing data for modeling...")  # SỬA LỖI TV

    # === DEFINE TARGET VARIABLE ===
    target_col = 'SALES_VALUE'  # Assuming Dunnhumby

    if target_col not in df.columns:
        if 'sales' in df.columns:  # Fallback for M5
            target_col = 'sales'
        else:
            print(f"ERROR: Target column '{target_col}' or 'sales' not found.")  # SỬA LỖI TV
            sys.exit(1)

    print(f"Target variable (Y) set to: {target_col}")  # SỬA LỖI TV

    df_model = df.dropna(subset=[target_col]).copy()

    if df_model.empty:
        print("ERROR: No data left to train after dropping NaN target values.")  # SỬA LỖI TV
        sys.exit(1)

    # === DEFINE FEATURES (ALL 4 WORKSTREAMS) ===
    numeric_features = [
        # --- WS0 (Aggregated) ---
        'QUANTITY',
        
        # --- WS2 (Time-Series) Features ---
        'sales_value_lag_1', 'sales_value_lag_4', 'sales_value_lag_8', 'sales_value_lag_12',
        'rolling_mean_4_lag_1', 'rolling_std_4_lag_1',
        'rolling_mean_8_lag_1', 'rolling_std_8_lag_1',
        'rolling_mean_12_lag_1', 'rolling_std_12_lag_1',
        'week_of_year', 'month_proxy', 'quarter', 'week_sin', 'week_cos',

        # --- WS4 (Price/Promo) Features ---
        'base_price', 'total_discount', 'discount_pct',
    ]

    categorical_features = [
        # --- WS1 (Relational) Features ---
        'DEPARTMENT', 'COMMODITY_DESC',
        
        # --- WS4 (Price/Promo) Features ---
        'is_on_display', 'is_on_mailer', 'is_on_retail_promo', 'is_on_coupon_promo',
    ]
    # === END OF FEATURE LIST ===

    all_features = [col for col in (numeric_features + categorical_features) if col in df.columns]
    categorical_features = [col for col in categorical_features if col in all_features]

    missing_features = set(numeric_features + categorical_features) - set(df.columns)
    if missing_features:
        print(f"Warning: Missing expected features (WS may be toggled off): {missing_features}")  # SỬA LỖI TV

    if not all_features:
        print("ERROR: No valid features found in the input file.")  # SỬA LỖI TV
        sys.exit(1)

    print(f"Found {len(all_features)} valid features for training.")  # SỬA LỖI TV

    X = df_model[all_features]
    y = df_model[target_col]
    
    # Add WEEK_NO for time-based split
    if 'WEEK_NO' not in df_model.columns:
        print("ERROR: WEEK_NO column required for time-based split!")
        sys.exit(1)
    
    week_no = df_model['WEEK_NO']

    print(f"Converting {len(categorical_features)} columns to 'category' dtype...")  # SỬA LỖI TV
    for col in categorical_features:
        X[col] = X[col].astype('category')

    # === TIME-BASED SPLIT (NO RANDOM SHUFFLE!) ===
    print("\n" + "=" * 70)
    print("PERFORMING TIME-BASED SPLIT (Leak-Safe)")
    print("=" * 70)
    
    # Calculate 80th percentile cutoff
    cutoff_week = week_no.quantile(0.8)
    print(f"Time cutoff: WEEK_NO < {cutoff_week:.0f} = TRAIN, >= {cutoff_week:.0f} = TEST")
    
    # Split by time
    train_mask = week_no < cutoff_week
    test_mask = week_no >= cutoff_week
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Train set: {len(X_train):,} samples (weeks {week_no[train_mask].min():.0f}-{week_no[train_mask].max():.0f})")
    print(f"Test set:  {len(X_test):,} samples (weeks {week_no[test_mask].min():.0f}-{week_no[test_mask].max():.0f})")
    print(f"Split ratio: {len(X_train)/len(X)*100:.1f}% train / {len(X_test)/len(X)*100:.1f}% test")
    print("=" * 70)
    print("OK. Data preparation complete (TIME-BASED, NO LEAKAGE).")  # SỬA LỖI TV

    return X_train, X_test, y_train, y_test, all_features, categorical_features


def train_quantile_models(X_train, y_train, categorical_features, quantiles=[0.05, 0.50, 0.95]):
    """
    Trains separate LightGBM models for each quantile (probabilistic forecasting).
    
    CRITICAL CHANGE: Uses objective='quantile' instead of 'regression_l1'.
    This enables prediction intervals for inventory optimization.
    
    Args:
        X_train: Training features
        y_train: Training target
        categorical_features: List of categorical column names
        quantiles: List of quantile levels to train (default: [0.05, 0.50, 0.95])
    
    Returns:
        Dict mapping quantile -> trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING QUANTILE MODELS (Probabilistic Forecasting)")
    print("=" * 70)
    print(f"Training {len(quantiles)} separate models for quantiles: {quantiles}")
    start_train = time.time()
    
    models = {}
    
    # Base hyperparameters (can be tuned further)
    base_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for alpha in quantiles:
        print(f"\n--- Training Q{int(alpha*100):02d} model (alpha={alpha}) ---")
        
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            **base_params
        )
        
        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_features
        )
        
        models[alpha] = model
        print(f"  [OK] Q{int(alpha*100):02d} model trained successfully")
    
    print(f"\n[OK] All quantile models trained (Took {time.time() - start_train:.2f}s)")
    print("=" * 70)
    
    return models


def evaluate_quantile_models(models: Dict[float, lgb.LGBMRegressor], X_test, y_test):
    """
    Evaluates quantile models using pinball loss (the correct metric for quantile regression).
    
    CRITICAL CHANGE: Uses mean_pinball_loss instead of RMSE.
    
    Args:
        models: Dict mapping quantile -> model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dict of evaluation metrics
    """
    print("\n" + "=" * 70)
    print("EVALUATING QUANTILE MODELS (Pinball Loss)")
    print("=" * 70)
    
    metrics = {}
    predictions = {}
    
    for alpha, model in models.items():
        print(f"\n--- Evaluating Q{int(alpha*100):02d} (alpha={alpha}) ---")
        
        y_pred = model.predict(X_test)
        y_pred[y_pred < 0] = 0  # Clip negative predictions
        
        predictions[alpha] = y_pred
        
        # Pinball loss (primary metric for quantile regression)
        pinball = mean_pinball_loss(y_test, y_pred, alpha=alpha)
        
        # Also calculate RMSE for reference
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics[f'q{int(alpha*100):02d}_pinball_loss'] = pinball
        metrics[f'q{int(alpha*100):02d}_rmse'] = rmse
        
        print(f"  Pinball Loss: {pinball:.4f}")
        print(f"  RMSE (reference): {rmse:.4f}")
    
    # Calculate prediction interval coverage (if we have q05 and q95)
    if 0.05 in predictions and 0.95 in predictions:
        lower = predictions[0.05]
        upper = predictions[0.95]
        coverage = ((y_test >= lower) & (y_test <= upper)).mean()
        metrics['prediction_interval_coverage'] = coverage
        print(f"\nPrediction Interval Coverage (90%): {coverage*100:.2f}%")
        print(f"  (Target: ~90%, Actual: {coverage*100:.1f}%)")
    
    print("=" * 70)
    
    return metrics


def save_artifacts(models: Dict[float, lgb.LGBMRegressor], features_config: Dict, metrics: Dict):
    """
    Saves quantile models, features, and metrics to disk.
    
    CRITICAL CHANGE: Saves 3 separate model files (q05, q50, q95) instead of single model.
    
    Args:
        models: Dict mapping quantile -> model
        features_config: Feature configuration dict
        metrics: Evaluation metrics dict
    """
    print("\n[save_artifacts] Saving model artifacts...")

    (PROJECT_ROOT / 'models').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'reports' / 'metrics').mkdir(parents=True, exist_ok=True)

    # Save each quantile model separately
    for alpha, model in models.items():
        model_path = PROJECT_ROOT / 'models' / f'q{int(alpha*100):02d}_forecaster.joblib'
        try:
            joblib.dump(model, model_path)
            print(f"  [OK] Q{int(alpha*100):02d} model saved to: {model_path}")
        except Exception as e:
            print(f"  ERROR saving Q{int(alpha*100):02d} model: {e}")

    # Save feature config
    try:
        features_path = PROJECT_ROOT / 'models' / 'model_features.json'
        with open(features_path, 'w') as f:
            json.dump(features_config, f, indent=4)
        print(f"  [OK] Feature config saved to: {features_path}")
    except Exception as e:
        print(f"  ERROR saving feature config: {e}")

    # Save metrics
    try:
        metrics_path = PROJECT_ROOT / 'reports' / 'metrics' / 'quantile_model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"  [OK] Metrics saved to: {metrics_path}")
    except Exception as e:
        print(f"  ERROR saving metrics: {e}")


# -----------------------------------------------------------------
# 3. MAIN ORCHESTRATOR (All English)
# -----------------------------------------------------------------

def main():
    """
    Orchestrates the entire training pipeline.
    
    MAJOR CHANGES:
    1. Time-based split (no random shuffle)
    2. Quantile regression (3 models: q05, q50, q95)
    3. Pinball loss evaluation
    """
    print("========== STARTING MODEL TRAINING PIPELINE (QUANTILE REGRESSION) ==========")
    total_start_time = time.time()

    print("\n--- STEP 1: LOAD DATA ---")
    df = load_data(CONFIG['data_file'])

    print("\n--- STEP 2: PREPARE DATA & TIME-BASED SPLIT ---")
    X_train, X_test, y_train, y_test, features, cat_features = prepare_data(df)

    print("\n--- STEP 3: TRAIN QUANTILE MODELS (Q05, Q50, Q95) ---")
    quantile_models = train_quantile_models(X_train, y_train, cat_features)

    print("\n--- STEP 4: EVALUATE QUANTILE MODELS (PINBALL LOSS) ---")
    metrics = evaluate_quantile_models(quantile_models, X_test, y_test)

    print("\n--- STEP 5: SAVE ARTIFACTS (MODELS, FEATURES, METRICS) ---")
    features_config = {
        "all_features": features,
        "categorical_features": cat_features,
        "quantiles": [0.05, 0.50, 0.95],
        "model_type": "LightGBM_Quantile_Regression"
    }
    save_artifacts(quantile_models, features_config, metrics)

    print("\n========================================================")
    print(f"COMPLETE! Total runtime: {time.time() - total_start_time:.2f} seconds.")
    print(f"Artifacts saved to: {PROJECT_ROOT / 'models'} and {PROJECT_ROOT / 'reports' / 'metrics'}")
    print("========================================================")


if __name__ == "__main__":
    main()