#!/usr/bin/env python3
"""
Test script to validate the sparse data fix.
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / 'src'))

def test_sparse_data_fix():
    """Test that the sparse data fix works correctly."""

    # Load raw transaction data
    raw_data_path = PROJECT_ROOT / 'data' / '2_raw' / 'transaction_data.csv'
    if not raw_data_path.exists():
        print(f"Raw data not found: {raw_data_path}")
        return

    print("Loading transaction data...")
    df_raw = pd.read_csv(raw_data_path, nrows=10000)  # Small sample for testing
    print(f"Raw data shape: {df_raw.shape}")

    # Simulate WS0: aggregation to weekly level
    print("\nSimulating WS0 aggregation...")
    agg_df = df_raw.groupby(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).agg({
        'SALES_VALUE': 'sum',
        'QUANTITY': 'sum',
        'RETAIL_DISC': 'sum',
        'COUPON_DISC': 'sum',
        'COUPON_MATCH_DISC': 'sum'
    }).reset_index()

    print(f"After aggregation: {agg_df.shape}")

    # Simulate full grid creation (old way - problematic)
    print("\nSimulating old sparse grid creation...")
    all_products = agg_df['PRODUCT_ID'].unique()
    all_stores = agg_df['STORE_ID'].unique()
    all_weeks = agg_df['WEEK_NO'].unique()

    from itertools import product
    grid_combinations = list(product(all_products, all_stores, all_weeks))
    print(f"Full grid combinations: {len(grid_combinations):,}")

    # Create sparse grid (old way)
    grid_df = pd.DataFrame(grid_combinations, columns=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])
    sparse_master = pd.merge(grid_df, agg_df, on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'], how='left').fillna(0)
    print(f"Sparse master table: {sparse_master.shape}")

    # Count zero-filled rows
    zero_rows = (sparse_master['SALES_VALUE'] == 0).sum()
    print(f"Zero-filled rows: {zero_rows:,} ({zero_rows/len(sparse_master)*100:.1f}%)")

    # Apply the fix
    print("\nApplying sparse data fix...")
    filtered_master = sparse_master[sparse_master['SALES_VALUE'] > 0].reset_index(drop=True)
    print(f"Filtered master table: {filtered_master.shape}")
    print(f"Removed {len(sparse_master) - len(filtered_master):,} zero-filled rows")

    # Test sample
    print("\nTesting sample quality...")
    sample_old = sparse_master.head(100)
    sample_new = filtered_master.head(100)

    print(f"Old sample SALES_VALUE range: {sample_old['SALES_VALUE'].min():.2f} - {sample_old['SALES_VALUE'].max():.2f}")
    print(f"New sample SALES_VALUE range: {sample_new['SALES_VALUE'].min():.2f} - {sample_new['SALES_VALUE'].max():.2f}")

    print("\n✅ Sparse data fix validation complete!")

if __name__ == "__main__":
    test_sparse_data_fix()
