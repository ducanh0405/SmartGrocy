"""
WS3: ADVANCED Model Training with Hyperparameter Tuning
========================================================
IMPROVEMENTS:
1. Optuna-based hyperparameter search
2. Time-series cross-validation
3. Separate tuning per quantile
4. Better calibration for sparse data

EXPECTED:
- Pinball loss: 0.000116 -> <0.00008
- Coverage: 99.98% -> 88-92%
"""
import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Tuple, Any
import joblib
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import Optuna (optional but recommended)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Install with: pip install optuna")


def prepare_data(
    feature_table_path: str, 
    time_cutoff_percentile: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    Time-based train/test split (leak-safe).
    """
    logging.info(f"Loading feature table from: {feature_table_path}")
    df = pd.read_parquet(feature_table_path)
    
    logging.info(f"Data shape: {df.shape}")
    
    # Time-based split
    cutoff_week = df['WEEK_NO'].quantile(time_cutoff_percentile)
    logging.info(f"Time cutoff (p={time_cutoff_percentile}): week {cutoff_week}")
    
    train = df[df['WEEK_NO'] < cutoff_week].copy()
    test = df[df['WEEK_NO'] >= cutoff_week].copy()
    
    logging.info(f"Train: {train.shape} | Test: {test.shape}")
    logging.info(f"Train weeks: {train['WEEK_NO'].min()}-{train['WEEK_NO'].max()}")
    logging.info(f"Test weeks: {test['WEEK_NO'].min()}-{test['WEEK_NO'].max()}")
    
    # Define features
    exclude_cols = ['SALES_VALUE', 'QUANTITY', 'PRODUCT_ID', 'STORE_ID', 'WEEK_NO']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    categorical_features = ['month_proxy', 'quarter']
    
    logging.info(f"Features: {len(feature_cols)} total")
    logging.info(f"Categorical: {categorical_features}")
    
    return train, test, feature_cols, categorical_features


def create_time_series_cv_splits(
    train_df: pd.DataFrame,
    n_splits: int = 3
) -> list:
    """
    Time-series cross-validation splits (expanding window).
    
    Example with n_splits=3:
    - Fold 1: weeks 1-54 train, 55-68 val
    - Fold 2: weeks 1-68 train, 69-81 val
    - Fold 3: weeks 1-75 train, 76-81 val
    """
    # Ensure index is reset to positional
    train_df = train_df.reset_index(drop=True)
    
    weeks = sorted(train_df['WEEK_NO'].unique())
    n_weeks = len(weeks)
    
    if n_splits < 2:
        logging.warning("n_splits < 2, using single split at 80%")
        n_splits = 2
    
    splits = []
    
    for i in range(1, n_splits + 1):
        # Expanding window
        val_end_idx = int(n_weeks * (0.6 + 0.2 * i / n_splits))
        val_start_idx = max(0, val_end_idx - int(n_weeks * 0.15))
        
        val_end_week = weeks[min(val_end_idx, n_weeks - 1)]
        val_start_week = weeks[val_start_idx]
        
        train_mask = train_df['WEEK_NO'] < val_start_week
        val_mask = (train_df['WEEK_NO'] >= val_start_week) & (train_df['WEEK_NO'] < val_end_week)
        
        # Use numpy where to get positional indices
        train_idx = np.where(train_mask)[0].tolist()
        val_idx = np.where(val_mask)[0].tolist()
        
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
            logging.info(f"  Fold {i}: Train weeks <{val_start_week}, Val weeks {val_start_week}-{val_end_week}")
    
    return splits


def tune_quantile_hyperparameters(
    train_df: pd.DataFrame,
    feature_cols: list,
    categorical_features: list,
    alpha: float,
    n_trials: int = 50,
    n_cv_splits: int = 3
) -> Dict[str, Any]:
    """
    Optuna-based hyperparameter tuning for one quantile.
    """
    if not OPTUNA_AVAILABLE:
        logging.warning(f"Optuna not available, using default params for alpha={alpha}")
        return {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    
    logging.info(f"[TUNING] Starting Optuna for alpha={alpha}, {n_trials} trials, {n_cv_splits}-fold CV")
    
    # Reset index to ensure positional indexing works
    train_df = train_df.reset_index(drop=True)
    
    cv_splits = create_time_series_cv_splits(train_df, n_splits=n_cv_splits)
    
    def objective(trial):
        params = {
            'objective': 'quantile',
            'alpha': alpha,
            'metric': 'quantile',
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        
        cv_scores = []
        
        for train_idx, val_idx in cv_splits:
            # Use iloc for positional indexing
            X_tr = train_df.iloc[train_idx][feature_cols]
            y_tr = train_df.iloc[train_idx]['SALES_VALUE']
            X_val = train_df.iloc[val_idx][feature_cols]
            y_val = train_df.iloc[val_idx]['SALES_VALUE']
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                categorical_feature=categorical_features,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            preds = model.predict(X_val)
            score = pinball_loss(y_val.values, preds, alpha)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize', study_name=f'quantile_{alpha}')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logging.info(f"[TUNING] Best score: {study.best_value:.6f}")
    logging.info(f"[TUNING] Best params: {study.best_params}")
    
    best_params = study.best_params
    best_params['objective'] = 'quantile'
    best_params['alpha'] = alpha
    best_params['metric'] = 'quantile'
    best_params['verbosity'] = -1
    
    return best_params


def train_quantile_models_tuned(
    train_df: pd.DataFrame,
    feature_cols: list,
    categorical_features: list,
    alphas: list = [0.05, 0.50, 0.95],
    models_dir: str = 'models',
    tune: bool = True,
    n_trials: int = 50
) -> Dict[str, Any]:
    """
    Train quantile models with optional hyperparameter tuning.
    """
    logging.info("=" * 70)
    logging.info(f"[TRAINING] QUANTILE MODELS (Tuning: {tune})")
    logging.info("=" * 70)
    
    models = {}
    all_best_params = {}
    
    X_train = train_df[feature_cols]
    y_train = train_df['SALES_VALUE']
    
    for alpha in alphas:
        logging.info(f"\n[MODEL] Quantile {alpha:.2f}")
        
        if tune:
            best_params = tune_quantile_hyperparameters(
                train_df,
                feature_cols,
                categorical_features,
                alpha,
                n_trials=n_trials,
                n_cv_splits=3
            )
        else:
            # Default params
            best_params = {
                'objective': 'quantile',
                'alpha': alpha,
                'metric': 'quantile',
                'n_estimators': 300,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'verbosity': -1
            }
        
        all_best_params[f'q{int(alpha*100):02d}'] = best_params
        
        # Train final model on full train set
        logging.info(f"Training final model with best params...")
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, categorical_feature=categorical_features)
        
        models[f'q{int(alpha*100):02d}'] = model
        
        # Save model
        model_name = f'q{int(alpha*100):02d}_forecaster_tuned.joblib'
        model_path = Path(models_dir) / model_name
        joblib.dump(model, model_path)
        
        file_size = model_path.stat().st_size / 1024 / 1024
        logging.info(f"  Saved: {model_path} ({file_size:.2f} MB)")
    
    # Save hyperparameters
    params_path = Path(models_dir) / 'best_hyperparameters.json'
    with open(params_path, 'w') as f:
        json.dump(all_best_params, f, indent=2)
    logging.info(f"\nSaved hyperparameters: {params_path}")
    
    return models


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball loss for quantile regression."""
    residual = y_true - y_pred
    return np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual))


def evaluate_quantile_models(
    models: Dict[str, Any],
    test_df: pd.DataFrame,
    feature_cols: list,
    alphas: list = [0.05, 0.50, 0.95]
) -> Dict[str, float]:
    """
    Evaluate quantile models with pinball loss and coverage.
    """
    logging.info("\n" + "=" * 70)
    logging.info("[EVALUATION] Quantile Models on Test Set")
    logging.info("=" * 70)
    
    X_test = test_df[feature_cols]
    y_test = test_df['SALES_VALUE'].values
    
    results = {}
    predictions = {}
    
    # Get predictions from all models
    for alpha in alphas:
        model_key = f'q{int(alpha*100):02d}'
        model = models[model_key]
        
        preds = model.predict(X_test)
        predictions[model_key] = preds
        
        loss = pinball_loss(y_test, preds, alpha)
        results[f'{model_key}_pinball_loss'] = loss
        
        logging.info(f"{model_key.upper()} | Pinball Loss: {loss:.6f}")
    
    # Coverage analysis
    if 'q05' in predictions and 'q95' in predictions:
        lower = predictions['q05']
        upper = predictions['q95']
        
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        results['coverage_90pct'] = coverage
        
        status = "[OK]" if 0.88 <= coverage <= 0.92 else "[WARN]"
        logging.info(f"\nCoverage (90% interval): {coverage:.2%} {status}")
        
        if coverage > 0.95:
            logging.warning("  Too conservative! Models may need recalibration.")
        elif coverage < 0.85:
            logging.warning("  Too narrow! Models may be overconfident.")
    
    # Median performance
    if 'q50' in predictions:
        mae = np.mean(np.abs(y_test - predictions['q50']))
        rmse = np.sqrt(np.mean((y_test - predictions['q50']) ** 2))
        
        results['mae'] = mae
        results['rmse'] = rmse
        
        logging.info(f"\nQ50 (Median) Metrics:")
        logging.info(f"  MAE:  {mae:.6f}")
        logging.info(f"  RMSE: {rmse:.6f}")
    
    logging.info("=" * 70)
    
    return results


def run_training_pipeline(
    feature_table_path: str = 'data/3_processed/master_feature_table.parquet',
    models_dir: str = 'models',
    time_cutoff: float = 0.80,
    tune_hyperparameters: bool = True,
    n_trials: int = 50
):
    """
    Complete training pipeline with optional tuning.
    
    Args:
        tune_hyperparameters: If True, run Optuna tuning (slow but better results)
        n_trials: Number of Optuna trials per quantile
    """
    # Create directories
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and split data
    train_df, test_df, feature_cols, categorical_features = prepare_data(
        feature_table_path,
        time_cutoff_percentile=time_cutoff
    )
    
    # Train models
    models = train_quantile_models_tuned(
        train_df,
        feature_cols,
        categorical_features,
        alphas=[0.05, 0.50, 0.95],
        models_dir=models_dir,
        tune=tune_hyperparameters,
        n_trials=n_trials
    )
    
    # Evaluate
    results = evaluate_quantile_models(
        models,
        test_df,
        feature_cols,
        alphas=[0.05, 0.50, 0.95]
    )
    
    # Save results
    results_path = Path(models_dir) / 'tuned_model_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\n[OK] Saved metrics: {results_path}")
    
    # Save feature config
    feature_config = {
        'feature_columns': feature_cols,
        'categorical_features': categorical_features,
        'n_features': len(feature_cols)
    }
    
    config_path = Path(models_dir) / 'tuned_feature_config.json'
    with open(config_path, 'w') as f:
        json.dump(feature_config, f, indent=2)
    
    logging.info(f"[OK] Saved config: {config_path}")
    logging.info("\n[COMPLETE] Training pipeline finished!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train quantile models with optional tuning')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--quick', action='store_true', help='Quick mode (no tuning, 10 trials)')
    
    args = parser.parse_args()
    
    if args.quick:
        tune = False
        n_trials = 10
    else:
        tune = args.tune
        n_trials = args.trials
    
    run_training_pipeline(
        tune_hyperparameters=tune,
        n_trials=n_trials
    )
