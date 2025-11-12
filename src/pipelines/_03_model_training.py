"""
Unified Model Training Pipeline (Config-Driven)
=================================================
Trains quantile models based on the active dataset config.
"""
import sys
import warnings
import logging
from pathlib import Path

# Setup project path FIRST before any other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import config
try:
    from src.config import (
        setup_project_path, setup_logging, ensure_directories,
        OUTPUT_FILES, TRAINING_CONFIG, ALL_FEATURES_CONFIG,
        get_model_config, get_dataset_config, MODEL_CONFIGS
    )
    setup_project_path()
    setup_logging()
    ensure_directories()
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"Error: Cannot import config. Please ensure src/config.py exists.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Import error: {e}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

# Import other dependencies after config is loaded
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
import time
import joblib
import json
from typing import Dict, Tuple, List, Any, Optional
import argparse

# Import optional models
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    # CatBoost installation on Windows may require Rust compiler
    # Users can install via: pip install catboost --no-build-isolation
    # Or use conda: conda install -c conda-forge catboost
    logger.warning("CatBoost not available. CatBoost will be skipped. "
                   "To install on Windows, try: pip install catboost --no-build-isolation "
                   "or use conda: conda install -c conda-forge catboost")

warnings.filterwarnings('ignore')

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the clean feature table."""
    logger.info(f"Loading data from: {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        logger.info(f"Load complete. Shape: {df.shape}.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}. Run _02_feature_enrichment.py first.")
        sys.exit(1)

def get_features_from_config(config: Dict) -> Tuple[List[str], List[str]]:
    """
    Builds the feature lists dynamically based on dataset config toggles.
    """
    logger.info("Building feature list from config toggles...")
    all_features = []
    
    # Base timeseries features (luôn có)
    all_features.extend(ALL_FEATURES_CONFIG['timeseries_base'])
    
    # Các feature tùy chọn
    if config['has_relational']:
        all_features.extend(ALL_FEATURES_CONFIG['relational'])
    if config['has_intraday_patterns']:
        all_features.extend(ALL_FEATURES_CONFIG['intraday_patterns'])
    if config['has_behavior']:
        all_features.extend(ALL_FEATURES_CONFIG['behavior'])
    if config['has_price_promo']:
        all_features.extend(ALL_FEATURES_CONFIG['price_promo'])
    if config['has_stockout']:
        all_features.extend(ALL_FEATURES_CONFIG['stockout'])
    if config['has_weather']:
        all_features.extend(ALL_FEATURES_CONFIG['weather'])

    # Lấy danh sách categorical features từ config
    all_categorical = []
    for ws_features in ALL_FEATURES_CONFIG.values():
        all_categorical.extend(ws_features) # Tạm lấy tất cả
        
    # Lọc features thực sự tồn tại trong DataFrame
    logger.info("Finding categorical features...")
    
    # Tạo một DataFrame mẫu để kiểm tra kiểu dữ liệu (để tìm categorical)
    # Đây là một cách đơn giản, chúng ta sẽ cải thiện nó trong prepare_data
    temp_numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Giả định: Bất kỳ feature nào KHÔNG phải là số đều là categorical
    # Chúng ta sẽ làm sạch danh sách này trong prepare_data
    
    return all_features, all_categorical


def prepare_data(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Prepares data for modeling: selects features and performs time-based split.
    """
    logger.info("Preparing data for modeling (config-driven)...")

    # 1. Get Target and Time columns from config
    target_col = config['target_column']
    time_col_name = config['time_column']
    
    logger.info(f"Target (Y): {target_col}")
    logger.info(f"Time Column: {time_col_name}")

    if target_col not in df.columns or time_col_name not in df.columns:
        logger.error(f"FATAL: Target '{target_col}' or Time '{time_col_name}' not in DataFrame.")
        sys.exit(1)

    df_model = df.dropna(subset=[target_col]).copy()

    # 2. Build Feature List
    all_features_config, all_categorical_config = get_features_from_config(config)
    
    # Lọc: chỉ giữ lại các features có trong df
    all_features = [col for col in all_features_config if col in df.columns]
    
    # Xác định chính xác categorical features
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    categorical_features = []
    for col in all_features:
        if df_model[col].dtype.name not in numeric_dtypes:
             categorical_features.append(col)
             
    # Thêm các cột số nhưng thực ra là categorical (ví dụ: hour_of_day)
    manual_cats = ['hour_of_day', 'day_of_week', 'is_morning_peak', 'is_evening_peak', 
                   'is_weekend', 'temp_category', 'is_rainy', 'rain_intensity', 
                   'is_extreme_heat', 'is_extreme_cold', 'is_high_humidity',
                   'is_on_display', 'is_on_mailer', 'is_on_retail_promo', 'is_on_coupon_promo']
                   
    for col in manual_cats:
        if col in all_features and col not in categorical_features:
            categorical_features.append(col)

    logger.info(f"Found {len(all_features)} total features.")
    logger.info(f"Found {len(categorical_features)} categorical features: {categorical_features}")

    X = df_model[all_features]
    y = df_model[target_col]
    
    # CHỐT CHẶN CUỐI: Fill tất cả NaN còn sót lại
    # --------------------------------------------------
    numeric_features = X.select_dtypes(include=[np.number]).columns
    X[numeric_features] = X[numeric_features].fillna(0)
    
    logger.info(f"Final safeguard: Filled NaNs in {len(numeric_features)} numeric columns with 0.")
    # --------------------------------------------------
    
    # Chuyển đổi categorical features sang 'category' dtype
    for col in categorical_features:
        X[col] = X[col].astype('category')
        # Fill 'Unknown' cho categorical
        if X[col].isnull().any():
            X[col] = X[col].cat.add_categories(['Unknown']).fillna('Unknown')
    
    logger.info("Final safeguard: Filled NaNs in categorical columns with 'Unknown'.")

    # 3. Time-based split (NO SHUFFLE!)
    logger.info("=" * 70)
    logger.info("PERFORMING TIME-BASED SPLIT (Leak-Safe)")
    
    time_col_data = pd.to_datetime(df_model[time_col_name])
    cutoff_percentile = TRAINING_CONFIG['train_test_split']['cutoff_percentile']
    cutoff_time = time_col_data.quantile(cutoff_percentile)
    
    logger.info(f"Time cutoff: < {cutoff_time} = TRAIN, >= {cutoff_time} = TEST")
    
    train_mask = time_col_data < cutoff_time
    test_mask = time_col_data >= cutoff_time
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    logger.info(f"Train set: {len(X_train):,} samples (up to {time_col_data[train_mask].max()})")
    logger.info(f"Test set:  {len(X_test):,} samples (from {time_col_data[test_mask].min()})")
    logger.info(f"Split ratio: {len(X_train)/len(X)*100:.1f}% train / {len(X_test)/len(X)*100:.1f}% test")
    logger.info("=" * 70)

    return X_train, X_test, y_train, y_test, all_features, categorical_features

def create_model(model_type: str, quantile: float, categorical_features: List[str]) -> Any:
    """
    Tạo model instance dựa trên model type và quantile.
    
    Args:
        model_type: Model type (lightgbm, catboost, random_forest)
        quantile: Quantile level
        categorical_features: List categorical features
        
    Returns:
        Model instance
    """
    model_config = MODEL_CONFIGS[model_type]
    params = get_model_config(quantile, model_type=model_type)
    
    if model_type == 'lightgbm':
        model = lgb.LGBMRegressor(**params)
    elif model_type == 'catboost':
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available. Install with: pip install catboost")
        # CatBoost cũng cần wrapper
        model = cb.CatBoostRegressor(**{k: v for k, v in params.items() if k != 'quantile'})
    elif model_type == 'random_forest':
        # Random Forest cần wrapper
        model = RandomForestRegressor(**{k: v for k, v in params.items() if k != 'quantile'})
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, model_config['quantile_support']


def train_quantile_model(
    model_type: str,
    quantile: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: List[str]
) -> Any:
    """
    Train một quantile model.
    
    Args:
        model_type: Model type
        quantile: Quantile level
        X_train: Training features
        y_train: Training target
        categorical_features: List categorical features
        
    Returns:
        Trained model
    """
    model, quantile_support = create_model(model_type, quantile, categorical_features)
    
    # Encode categorical features cho các model không hỗ trợ categorical trực tiếp
    def encode_categorical(X: pd.DataFrame, cat_features: List[str]) -> pd.DataFrame:
        """Encode categorical features thành numeric codes."""
        X_encoded = X.copy()
        for col in cat_features:
            if col in X_encoded.columns:
                if X_encoded[col].dtype.name == 'category':
                    X_encoded[col] = X_encoded[col].cat.codes
                elif X_encoded[col].dtype.name == 'object':
                    X_encoded[col] = pd.Categorical(X_encoded[col]).codes
        return X_encoded
    
    # Train model dựa trên model type
    if model_type == 'lightgbm' and quantile_support:
        # LightGBM hỗ trợ quantile regression trực tiếp
        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_features
        )
    elif model_type == 'catboost':
        # CatBoost hỗ trợ categorical features trực tiếp
        cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
        model.fit(
            X_train, y_train,
            cat_features=cat_indices,
            verbose=False
        )
    else:
        # Random Forest và các model khác
        X_train_encoded = encode_categorical(X_train, categorical_features)
        model.fit(X_train_encoded, y_train)
    
    return model


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: List[str],
    model_types: Optional[List[str]] = None
) -> Dict[str, Dict[float, Any]]:
    """
    Trains multiple models for each quantile.
    
    Args:
        X_train: Training features
        y_train: Training target
        categorical_features: List categorical features
        model_types: List model types to train. If None, use TRAINING_CONFIG['model_types']
        
    Returns:
        Dict of {model_type: {quantile: model}}
    """
    quantiles = TRAINING_CONFIG['quantiles']
    model_types = model_types or TRAINING_CONFIG.get('model_types', ['lightgbm'])
    
    # Validate: CHỈ TRAIN 5 QUANTILES (Q05, Q25, Q50, Q75, Q95)
    if len(quantiles) != 5:
        logger.warning(f"Expected 5 quantiles, but found {len(quantiles)}: {quantiles}")
    expected_quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    if set(quantiles) != set(expected_quantiles):
        logger.warning(f"Quantiles do not match expected values. Expected: {expected_quantiles}, Got: {quantiles}")
    
    logger.info("=" * 70)
    logger.info(f"TRAINING MODELS - 5 QUANTILES ONLY")
    logger.info(f"Model types: {model_types}")
    logger.info(f"Quantiles: {quantiles} (Total: {len(quantiles)} quantiles)")
    logger.info("=" * 70)
    
    # Filter available models
    available_models = []
    for model_type in model_types:
        if model_type == 'catboost' and not CATBOOST_AVAILABLE:
            logger.warning(f"Skipping {model_type}: not available")
            continue
        if model_type not in MODEL_CONFIGS:
            logger.warning(f"Skipping {model_type}: not in MODEL_CONFIGS")
            continue
        available_models.append(model_type)
    
    if not available_models:
        raise ValueError("No available models to train")
    
    logger.info(f"Training {len(available_models)} model types: {available_models}")
    
    all_models = {}
    total_start_time = time.time()
    
    for model_type in available_models:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training {model_type.upper()} models")
        logger.info(f"{'='*70}")
        all_models[model_type] = {}
        model_start_time = time.time()
        
        for quantile in quantiles:
            logger.info(f"Training {model_type} Q{int(quantile*100):02d} model (alpha={quantile})...")
            
            try:
                model = train_quantile_model(
                    model_type, quantile, X_train, y_train, categorical_features
                )
                all_models[model_type][quantile] = model
                logger.info(f"  ✓ {model_type} Q{int(quantile*100):02d} trained")
            except Exception as e:
                logger.error(f"  ✗ Error training {model_type} Q{int(quantile*100):02d}: {e}")
                raise
        
        logger.info(f"{model_type.upper()} models trained in {time.time() - model_start_time:.2f}s")
    
    total_time = time.time() - total_start_time
    total_models = sum(len(models) for models in all_models.values())
    logger.info(f"\n{'='*70}")
    logger.info(f"All {total_models} models trained in {total_time:.2f}s")
    logger.info(f"{'='*70}")
    
    return all_models

def encode_categorical_for_prediction(X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
    """Encode categorical features thành numeric codes cho prediction."""
    X_encoded = X.copy()
    for col in categorical_features:
        if col in X_encoded.columns:
            if X_encoded[col].dtype.name == 'category':
                X_encoded[col] = X_encoded[col].cat.codes
            elif X_encoded[col].dtype.name == 'object':
                X_encoded[col] = pd.Categorical(X_encoded[col]).codes
    return X_encoded


def predict_with_model(model: Any, model_type: str, X: pd.DataFrame, 
                      categorical_features: List[str]) -> np.ndarray:
    """
    Predict với model, xử lý categorical features cho từng model type.
    
    Args:
        model: Trained model
        model_type: Model type
        X: Features
        categorical_features: List categorical features
        
    Returns:
        Predictions
    """
    if model_type == 'lightgbm':
        return model.predict(X)
    elif model_type == 'catboost':
        return model.predict(X)
    else:
        # Random Forest và các model khác cần encode categorical
        X_encoded = encode_categorical_for_prediction(X, categorical_features)
        return model.predict(X_encoded)


def evaluate_quantile_models(
    all_models: Dict[str, Dict[float, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_features: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates all models using pinball loss.
    
    Args:
        all_models: Dict of {model_type: {quantile: model}}
        X_test: Test features
        y_test: Test target
        categorical_features: List categorical features
        
    Returns:
        Dict of {model_type: {metric: value}}
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    logger.info("=" * 70)
    logger.info("EVALUATING QUANTILE MODELS")
    logger.info("=" * 70)
    
    quantiles = TRAINING_CONFIG['quantiles']
    all_metrics = {}
    
    for model_type, models in all_models.items():
        logger.info(f"\nEvaluating {model_type.upper()} models...")
        metrics = {}
        predictions = {}
        
        for quantile in quantiles:
            if quantile not in models:
                continue
            
            model = models[quantile]
            y_pred = predict_with_model(model, model_type, X_test, categorical_features)
            y_pred = np.maximum(y_pred, 0)  # Clip negative predictions
            predictions[quantile] = y_pred
            
            # Pinball loss
            pinball = mean_pinball_loss(y_test, y_pred, alpha=quantile)
            metrics[f'q{int(quantile*100):02d}_pinball_loss'] = pinball
            
            # MAE
            mae = mean_absolute_error(y_test, y_pred)
            metrics[f'q{int(quantile*100):02d}_mae'] = mae
            
            # RMSE
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics[f'q{int(quantile*100):02d}_rmse'] = rmse
            
            logger.info(f"  Q{int(quantile*100):02d} - Pinball: {pinball:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Coverage
        if len(predictions) >= 2:
            lower_q = min(quantiles)
            upper_q = max(quantiles)
            if lower_q in predictions and upper_q in predictions:
                lower_pred = predictions[lower_q]
                upper_pred = predictions[upper_q]
                coverage = ((y_test >= lower_pred) & (y_test <= upper_pred)).mean()
                metrics[f'coverage_{(upper_q-lower_q)*100:.0f}%'] = coverage
                logger.info(f"  Coverage ({ (upper_q-lower_q)*100:.0f}%): {coverage*100:.2f}%")
        
        # R2 score (median quantile)
        median_q = 0.50
        if median_q in predictions:
            r2 = r2_score(y_test, predictions[median_q])
            metrics['r2_score'] = r2
            logger.info(f"  R2 Score: {r2:.4f}")
        
        all_metrics[model_type] = metrics
    
    return all_metrics

def save_artifacts(
    all_models: Dict[str, Dict[float, Any]],
    features_config: Dict,
    all_metrics: Dict[str, Dict[str, float]]
) -> None:
    """Saves models, features, and metrics."""
    logger.info("Saving model artifacts...")

    # Save models
    for model_type, models in all_models.items():
        for quantile, model in models.items():
            model_filename = f"{model_type}_q{int(quantile*100):02d}_forecaster.joblib"
            model_path = OUTPUT_FILES['models_dir'] / model_filename
            joblib.dump(model, model_path)
            logger.info(f"  {model_filename} saved to: {model_path}")

    # Save feature config
    features_path = OUTPUT_FILES['model_features']
    with open(features_path, 'w') as f:
        json.dump(features_config, f, indent=4)
    logger.info(f"  Feature config saved to: {features_path}")

    # Save metrics
    metrics_path = OUTPUT_FILES['model_metrics']
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"  Metrics saved to: {metrics_path}")

def main(args) -> None:
    """Orchestrates the entire training pipeline."""
    logger.info("=" * 70)
    logger.info("STARTING MODEL TRAINING PIPELINE (Config-Driven)")
    logger.info("=" * 70)
    total_start_time = time.time()
    
    # 1. Lấy config
    config = get_dataset_config()
    logger.info(f"Active Dataset: {config['name']}")

    # 2. Load Data
    logger.info("STEP 1: LOAD DATA")
    df = load_data(OUTPUT_FILES['master_feature_table'])

    # 3. Prepare Data
    logger.info("STEP 2: PREPARE DATA & TIME-BASED SPLIT")
    X_train, X_test, y_train, y_test, features, cat_features = prepare_data(df, config)

    # 4. Train Models
    logger.info("STEP 3: TRAIN QUANTILE MODELS")
    model_types = args.model_types if args.model_types else TRAINING_CONFIG.get('model_types', ['lightgbm'])
    
    if args.tune:
        logger.warning("Hyperparameter tuning not implemented. Using standard params.")
    
    all_models = train_quantile_models(X_train, y_train, cat_features, model_types=model_types)

    # 5. Evaluate
    logger.info("STEP 4: EVALUATE QUANTILE MODELS")
    all_metrics = evaluate_quantile_models(all_models, X_test, y_test, cat_features)

    # 6. Save
    logger.info("STEP 5: SAVE ARTIFACTS")
    final_features_config = {
        "all_features": features,
        "categorical_features": cat_features,
        "quantiles": TRAINING_CONFIG['quantiles'],
        "model_types": list(all_models.keys()),
        "dataset_trained_on": config['name']
    }
    save_artifacts(all_models, final_features_config, all_metrics)

    logger.info("=" * 70)
    logger.info(f"COMPLETE! Total runtime: {time.time() - total_start_time:.2f} seconds.")
    logger.info(f"Artifacts saved to: {OUTPUT_FILES['models_dir']}")
    logger.info("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train quantile regression models')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning (nếu được implement)')
    parser.add_argument('--model-types', nargs='+', type=str, default=None,
                       help='Model types to train (lightgbm, catboost, random_forest)')
    args = parser.parse_args()
    
    main(args)