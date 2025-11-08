#!/usr/bin/env python3
"""
Demo script showing the sparse data fix logic.
"""
import pandas as pd

def demo_sparse_data_fix():
    """Demonstrate the sparse data fix logic."""

    print("🔍 DEMO: Sparse Data Fix Logic")
    print("=" * 50)

    # Create sample transaction data
    print("\n📊 Step 1: Sample Transaction Data")
    transaction_data = pd.DataFrame({
        'PRODUCT_ID': [1, 1, 1, 2, 2],
        'STORE_ID': [100, 100, 100, 100, 100],
        'WEEK_NO': [1, 2, 3, 1, 2],
        'SALES_VALUE': [10.5, 15.2, 8.7, 12.1, 9.8],
        'QUANTITY': [2, 3, 1, 2, 1]
    })
    print(transaction_data)

    # Simulate aggregation (WS0)
    print("\n🔄 Step 2: After Aggregation (WS0)")
    agg_data = transaction_data.groupby(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).agg({
        'SALES_VALUE': 'sum',
        'QUANTITY': 'sum'
    }).reset_index()
    print(agg_data)

    # OLD WAY: Create full grid (problematic)
    print("\n❌ Step 3: OLD WAY - Full Grid Creation")
    all_products = agg_data['PRODUCT_ID'].unique()
    all_stores = agg_data['STORE_ID'].unique()
    all_weeks = [1, 2, 3, 4]  # Assume 4 weeks total

    from itertools import product
    grid_combinations = list(product(all_products, all_stores, all_weeks))
    print(f"Grid combinations: {len(grid_combinations)}")

    grid_df = pd.DataFrame(grid_combinations, columns=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])
    sparse_master = pd.merge(grid_df, agg_data, on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'], how='left').fillna(0)
    print(f"Sparse master table shape: {sparse_master.shape}")
    print(sparse_master)

    zero_rows = (sparse_master['SALES_VALUE'] == 0).sum()
    print(f"Zero-filled rows: {zero_rows} ({zero_rows/len(sparse_master)*100:.0f}%)")

    # NEW WAY: Filter zero sales
    print("\n✅ Step 4: NEW WAY - Filter Zero Sales")
    filtered_master = sparse_master[sparse_master['SALES_VALUE'] > 0].reset_index(drop=True)
    print(f"Filtered master table shape: {filtered_master.shape}")
    print(filtered_master)

    print("
📋 Comparison:"    print(f"Old approach: {len(sparse_master)} rows, {zero_rows} zeros ({zero_rows/len(sparse_master)*100:.0f}%)")
    print(f"New approach: {len(filtered_master)} rows, 0 zeros")
    print(f"Data retention: {len(filtered_master)/len(agg_data)*100:.0f}% of aggregated data")

    return True

def demo_feature_config():
    """Show the updated feature config."""

    print("\n🔧 UPDATED FEATURE CONFIG")
    print("=" * 50)

    # Import the updated config
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PROJECT_ROOT / 'src'))

    try:
        from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

        print(f"✅ Numeric features: {len(NUMERIC_FEATURES)}")
        print("   Key additions:"
        print("   - Rolling max/min for all windows (4,8,12)")
        print("   - Quantity lags (1,4 weeks)")
        print("   - Trend features (wow_change, momentum, volatility)")

        print(f"\n✅ Categorical features: {len(CATEGORICAL_FEATURES)}")
        print("   - DEPARTMENT, COMMODITY_DESC")
        print("   - Promotion flags (is_on_display, is_on_mailer, etc.)")

        # Show sample features
        print("
📋 Sample Numeric Features:"        sample_features = [
            'sales_value_lag_1', 'quantity_lag_4',
            'rolling_mean_4_lag_1', 'rolling_max_4_lag_1', 'rolling_min_4_lag_1',
            'wow_change', 'momentum', 'base_price', 'discount_pct'
        ]
        for feat in sample_features:
            status = "✅" if feat in NUMERIC_FEATURES else "❌"
            print(f"   {status} {feat}")

    except ImportError:
        print("❌ Cannot import config - run from project root")

if __name__ == "__main__":
    demo_sparse_data_fix()
    demo_feature_config()

    print("\n🎉 DEMO COMPLETE!")
    print("\n📝 Summary of Changes:")
    print("1. ✅ Added sparse data filter: SALES_VALUE > 0")
    print("2. ✅ Updated config with all rolling stats, quantity lags, trend features")
    print("3. ✅ Created validation script")
    print("\n🚀 Ready for pipeline testing!")
