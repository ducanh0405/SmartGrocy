"""
WS2: Leak-Safe Time-Series Features
====================================
Creates lag and rolling features WITHOUT data leakage.
All features are calculated on LAGGED data only (never including current row).

CRITICAL RULES:
1. All lags start from t-1 (never t-0)
2. Rolling windows are calculated on LAGGED series
3. Data must be sorted by [PRODUCT_ID, STORE_ID, WEEK_NO] before calling these functions
"""
import pandas as pd
import numpy as np
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_lag_features(
    df: pd.DataFrame, 
    target_col: str = 'SALES_VALUE',
    lags: List[int] = [1, 4, 8, 12]
) -> pd.DataFrame:
    """
    Creates lag features for the target column.
    
    LEAK-SAFE: All lags are >= 1 (never use current value).
    
    Args:
        df: DataFrame sorted by [PRODUCT_ID, STORE_ID, WEEK_NO]
        target_col: Column to lag (default: SALES_VALUE)
        lags: List of lag periods (default: [1, 4, 8, 12] weeks)
    
    Returns:
        DataFrame with new lag columns
    """
    logging.info(f"[WS2] Creating lag features for '{target_col}'...")
    
    if target_col not in df.columns:
        logging.warning(f"SKIPPING: Column '{target_col}' not found")
        return df
    
    df_out = df.copy()
    
    for lag in lags:
        col_name = f'{target_col.lower()}_lag_{lag}'
        df_out[col_name] = df_out.groupby(['PRODUCT_ID', 'STORE_ID'])[target_col].shift(lag)
        logging.info(f"  Created: {col_name}")
    
    return df_out


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'SALES_VALUE',
    base_lag: int = 1,
    windows: List[int] = [4, 8, 12]
) -> pd.DataFrame:
    """
    Creates rolling statistics on LAGGED data (leak-safe).
    
    CRITICAL: Rolling window is calculated on lag_{base_lag}, NOT on current value.
    Example: rolling_mean_4_lag_1 = mean of [t-1, t-2, t-3, t-4]
    
    Args:
        df: DataFrame sorted by [PRODUCT_ID, STORE_ID, WEEK_NO]
        target_col: Column to calculate rolling stats on
        base_lag: Base lag to apply before rolling (default: 1)
        windows: List of window sizes (default: [4, 8, 12] weeks)
    
    Returns:
        DataFrame with new rolling features
    """
    logging.info(f"[WS2] Creating rolling features on lag_{base_lag} of '{target_col}'...")
    
    if target_col not in df.columns:
        logging.warning(f"SKIPPING: Column '{target_col}' not found")
        return df
    
    df_out = df.copy()
    
    # First create base lag if not exists
    lag_col = f'{target_col.lower()}_lag_{base_lag}'
    if lag_col not in df_out.columns:
        df_out[lag_col] = df_out.groupby(['PRODUCT_ID', 'STORE_ID'])[target_col].shift(base_lag)
    
    # Calculate rolling stats on the LAGGED column
    grouped = df_out.groupby(['PRODUCT_ID', 'STORE_ID'])[lag_col]
    
    for window in windows:
        # Rolling mean
        col_mean = f'rolling_mean_{window}_lag_{base_lag}'
        df_out[col_mean] = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
        logging.info(f"  Created: {col_mean}")
        
        # Rolling std
        col_std = f'rolling_std_{window}_lag_{base_lag}'
        df_out[col_std] = grouped.transform(lambda x: x.rolling(window, min_periods=1).std())
        logging.info(f"  Created: {col_std}")
        
        # Rolling max
        col_max = f'rolling_max_{window}_lag_{base_lag}'
        df_out[col_max] = grouped.transform(lambda x: x.rolling(window, min_periods=1).max())
        logging.info(f"  Created: {col_max}")
        
        # Rolling min
        col_min = f'rolling_min_{window}_lag_{base_lag}'
        df_out[col_min] = grouped.transform(lambda x: x.rolling(window, min_periods=1).min())
        logging.info(f"  Created: {col_min}")
    
    return df_out


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates calendar-based features from WEEK_NO.
    
    Args:
        df: DataFrame with WEEK_NO column
    
    Returns:
        DataFrame with calendar features
    """
    logging.info("[WS2] Creating calendar features from WEEK_NO...")
    
    if 'WEEK_NO' not in df.columns:
        logging.warning("SKIPPING: Column 'WEEK_NO' not found")
        return df
    
    df_out = df.copy()
    
    # Week of year (cyclical: 1-52)
    df_out['week_of_year'] = ((df_out['WEEK_NO'] - 1) % 52) + 1
    
    # Month proxy (assuming ~4.33 weeks per month)
    df_out['month_proxy'] = ((df_out['WEEK_NO'] - 1) // 4) % 12 + 1
    
    # Quarter
    df_out['quarter'] = ((df_out['month_proxy'] - 1) // 3) + 1
    
    # Cyclical encoding for week_of_year (sin/cos for capturing cyclical patterns)
    df_out['week_sin'] = np.sin(2 * np.pi * df_out['week_of_year'] / 52)
    df_out['week_cos'] = np.cos(2 * np.pi * df_out['week_of_year'] / 52)
    
    logging.info("  Created: week_of_year, month_proxy, quarter, week_sin, week_cos")
    
    return df_out


def add_lag_rolling_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function for WS2: Adds all time-series features.
    
    This is the function called by _02_feature_enrichment.py.
    
    REQUIREMENTS:
    - master_df MUST be sorted by [PRODUCT_ID, STORE_ID, WEEK_NO]
    - master_df MUST have been processed by WS0 (aggregation & grid)
    
    Args:
        master_df: Master DataFrame from WS0/WS1
    
    Returns:
        DataFrame with time-series features added
    """
    logging.info("=" * 70)
    logging.info("[WS2] STARTING: Leak-Safe Time-Series Feature Engineering")
    logging.info("=" * 70)
    
    # Verify required columns
    required_cols = ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO', 'SALES_VALUE']
    missing = [col for col in required_cols if col not in master_df.columns]
    if missing:
        logging.error(f"SKIPPING WS2: Missing required columns: {missing}")
        return master_df
    
    # Verify sorting (CRITICAL for leak-safe features)
    is_sorted = master_df.groupby(['PRODUCT_ID', 'STORE_ID'])['WEEK_NO'].apply(
        lambda x: (x.diff().dropna() >= 0).all()
    ).all()
    
    if not is_sorted:
        logging.warning("WARNING: Data not sorted properly! Sorting now...")
        master_df = master_df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)
    
    # Step 1: Create lag features for SALES_VALUE
    master_df = create_lag_features(
        master_df, 
        target_col='SALES_VALUE',
        lags=[1, 4, 8, 12]  # 1 week, 1 month, 2 months, 3 months
    )
    
    # Step 2: Create lag features for QUANTITY (if exists)
    if 'QUANTITY' in master_df.columns:
        master_df = create_lag_features(
            master_df,
            target_col='QUANTITY',
            lags=[1, 4]
        )
    
    # Step 3: Create rolling features on lagged SALES_VALUE
    master_df = create_rolling_features(
        master_df,
        target_col='SALES_VALUE',
        base_lag=1,  # Calculate on lag_1 to avoid leakage
        windows=[4, 8, 12]
    )
    
    # Step 4: Create calendar features
    master_df = create_calendar_features(master_df)
    
    logging.info("=" * 70)
    logging.info(f"[WS2] COMPLETE: Added time-series features. Shape: {master_df.shape}")
    logging.info("=" * 70)
    
    return master_df


# Backward compatibility alias
add_timeseries_features = add_lag_rolling_features
