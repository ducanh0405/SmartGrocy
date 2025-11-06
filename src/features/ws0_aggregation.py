"""
WS0: Data Aggregation & Master Grid Creation
============================================
This module handles the critical pre-processing step before feature engineering:
1. Aggregates raw transactions to target granularity [PRODUCT_ID, STORE_ID, WEEK_NO]
2. Creates complete grid of all combinations (zero-fill missing periods)
3. Ensures strict time ordering for leak-safe feature creation

This prevents time leakage and ensures consistent forecasting granularity.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction-level data to weekly granularity.
    
    Target granularity: [PRODUCT_ID, STORE_ID, WEEK_NO]
    
    Args:
        df: Raw transaction data with columns:
            - PRODUCT_ID, STORE_ID, WEEK_NO (grouping keys)
            - SALES_VALUE, QUANTITY, RETAIL_DISC, COUPON_DISC (values to aggregate)
    
    Returns:
        Aggregated DataFrame at weekly level
    """
    logging.info("[WS0] Aggregating transactions to WEEKLY granularity...")
    
    # Define grouping keys
    groupby_keys = ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']
    
    # Verify required columns exist
    required_cols = groupby_keys + ['SALES_VALUE', 'QUANTITY']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Define aggregation rules
    agg_rules = {
        'SALES_VALUE': 'sum',
        'QUANTITY': 'sum',
    }
    
    # Add optional discount columns if they exist
    if 'RETAIL_DISC' in df.columns:
        agg_rules['RETAIL_DISC'] = 'sum'
    if 'COUPON_DISC' in df.columns:
        agg_rules['COUPON_DISC'] = 'sum'
    if 'COUPON_MATCH_DISC' in df.columns:
        agg_rules['COUPON_MATCH_DISC'] = 'sum'
    
    # Perform aggregation
    df_agg = df.groupby(groupby_keys, as_index=False).agg(agg_rules)
    
    logging.info(f"  Aggregation complete: {len(df):,} transactions -> {len(df_agg):,} weekly records")
    
    return df_agg


def create_master_grid(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a complete grid of all [PRODUCT_ID, STORE_ID, WEEK_NO] combinations.
    Zero-fills missing periods to ensure continuity for time-series features.
    
    This is CRITICAL for:
    1. Preventing data leakage (missing weeks shouldn't be skipped in lag calculations)
    2. Enabling proper rolling window calculations
    3. Consistent forecasting across all products/stores
    
    Args:
        df_agg: Aggregated weekly data
    
    Returns:
        Complete grid with zero-filled missing periods
    """
    logging.info("[WS0] Creating complete master grid (all PRODUCT × STORE × WEEK combinations)...")
    
    original_rows = len(df_agg)
    
    # Get all unique values for each dimension
    all_products = df_agg['PRODUCT_ID'].unique()
    all_stores = df_agg['STORE_ID'].unique()
    all_weeks = df_agg['WEEK_NO'].unique()
    
    # Sort weeks to ensure proper time ordering
    all_weeks = np.sort(all_weeks)
    
    logging.info(f"  Grid dimensions: {len(all_products):,} products × {len(all_stores):,} stores × {len(all_weeks):,} weeks")
    
    # Create complete grid using MultiIndex
    from itertools import product
    
    grid_index = pd.MultiIndex.from_product(
        [all_products, all_stores, all_weeks],
        names=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']
    )
    
    master_grid = pd.DataFrame(index=grid_index).reset_index()
    
    logging.info(f"  Complete grid size: {len(master_grid):,} rows")
    
    # Left join aggregated data onto the grid
    master_df = pd.merge(
        master_grid,
        df_agg,
        on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
        how='left'
    )
    
    # Zero-fill missing values (weeks where product wasn't sold in that store)
    fill_cols = ['SALES_VALUE', 'QUANTITY']
    if 'RETAIL_DISC' in master_df.columns:
        fill_cols.append('RETAIL_DISC')
    if 'COUPON_DISC' in master_df.columns:
        fill_cols.append('COUPON_DISC')
    if 'COUPON_MATCH_DISC' in master_df.columns:
        fill_cols.append('COUPON_MATCH_DISC')
    
    for col in fill_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna(0)
    
    # Sort by time dimension (CRITICAL for lag/rolling features)
    master_df = master_df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)
    
    filled_rows = len(master_df) - original_rows
    logging.info(f"  Zero-filled {filled_rows:,} missing period records ({filled_rows/len(master_df)*100:.1f}% of grid)")
    logging.info(f"  Final master_df shape: {master_df.shape}")
    
    return master_df


def prepare_master_dataframe(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Main orchestrator for WS0 (Data Aggregation & Grid Creation).
    
    Pipeline:
    1. Aggregate transactions to weekly level
    2. Create complete grid with zero-filling
    3. Sort by time for downstream feature engineering
    
    Args:
        raw_transactions: Raw transaction data from transaction_data.csv
    
    Returns:
        Master DataFrame ready for feature enrichment (WS1-4)
    """
    logging.info("=" * 70)
    logging.info("[WS0] STARTING: Data Aggregation & Master Grid Creation")
    logging.info("=" * 70)
    
    # Step 1: Aggregate to weekly granularity
    df_weekly = aggregate_to_weekly(raw_transactions)
    
    # Step 2: Create complete grid with zero-filling
    master_df = create_master_grid(df_weekly)
    
    # Step 3: Verify time ordering
    is_sorted = master_df.groupby(['PRODUCT_ID', 'STORE_ID'])['WEEK_NO'].apply(
        lambda x: (x.diff().dropna() >= 0).all()
    ).all()
    
    if not is_sorted:
        logging.warning("WARNING: master_df is not properly sorted by WEEK_NO within groups!")
    else:
        logging.info("  ✓ Verified: master_df is properly sorted for time-series features")
    
    logging.info("=" * 70)
    logging.info("[WS0] COMPLETE: Master DataFrame ready for WS1-4 enrichment")
    logging.info("=" * 70)
    
    return master_df


# Backward compatibility: export with alternative name
enrich_aggregation_features = prepare_master_dataframe
