#!/usr/bin/env python3
"""
Simple test to check validation logic.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / 'src'))

def test_config_import():
    """Test if config can be imported."""
    print("Testing config import...")
    try:
        from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        print(f"✅ Config imported: {len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical")

        # Check expected features
        expected_rolling = [
            'rolling_mean_4_lag_1', 'rolling_std_4_lag_1', 'rolling_max_4_lag_1', 'rolling_min_4_lag_1',
            'rolling_mean_8_lag_1', 'rolling_std_8_lag_1', 'rolling_max_8_lag_1', 'rolling_min_8_lag_1',
            'rolling_mean_12_lag_1', 'rolling_std_12_lag_1', 'rolling_max_12_lag_1', 'rolling_min_12_lag_1'
        ]

        expected_trend = ['wow_change', 'wow_pct_change', 'momentum', 'volatility']
        expected_quantity = ['quantity_lag_1', 'quantity_lag_4']

        print(f"Checking {len(expected_rolling)} rolling features...")
        rolling_ok = all(feat in NUMERIC_FEATURES for feat in expected_rolling)
        print(f"Rolling features: {'✅' if rolling_ok else '❌'}")

        print(f"Checking {len(expected_trend)} trend features...")
        trend_ok = all(feat in NUMERIC_FEATURES for feat in expected_trend)
        print(f"Trend features: {'✅' if trend_ok else '❌'}")

        print(f"Checking {len(expected_quantity)} quantity features...")
        quantity_ok = all(feat in NUMERIC_FEATURES for feat in expected_quantity)
        print(f"Quantity features: {'✅' if quantity_ok else '❌'}")

        if not quantity_ok:
            print("Missing quantity features:")
            for feat in expected_quantity:
                if feat not in NUMERIC_FEATURES:
                    print(f"  - {feat}")

        return rolling_ok and trend_ok and quantity_ok

    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_data_files():
    """Test if data files exist."""
    print("\nTesting data files...")

    files_to_check = [
        'data/2_raw/transaction_data.csv',
        'data/3_processed/master_feature_table.parquet',
        'data/3_processed/master_feature_table_improved_sample.csv'
    ]

    all_exist = True
    for file_path in files_to_check:
        full_path = PROJECT_ROOT / file_path
        exists = full_path.exists()
        print(f"{'✅' if exists else '❌'} {file_path}: {'exists' if exists else 'missing'}")
        if not exists:
            all_exist = False

    return all_exist

def test_sample_quality():
    """Test the quality of the improved sample."""
    print("\nTesting sample quality...")

    sample_path = PROJECT_ROOT / 'data' / '3_processed' / 'master_feature_table_improved_sample.csv'
    if not sample_path.exists():
        print("❌ Sample file not found")
        return False

    try:
        import pandas as pd
        df = pd.read_csv(sample_path)

        # Check basic stats
        sales_values = df['SALES_VALUE']
        print(f"Sample size: {len(df)} rows")
        print(f"SALES_VALUE range: {sales_values.min():.2f} - {sales_values.max():.2f}")
        print(f"SALES_VALUE mean: {sales_values.mean():.2f}")

        # Check for zero sales
        zero_sales = (sales_values == 0).sum()
        print(f"Zero sales rows: {zero_sales} ({zero_sales/len(df)*100:.1f}%)")

        # Check feature availability
        lag_features = [col for col in df.columns if 'lag' in col]
        rolling_features = [col for col in df.columns if 'rolling' in col]

        print(f"Lag features: {len(lag_features)}")
        print(f"Rolling features: {len(rolling_features)}")

        # Sample some rows to check data quality
        sample_rows = df.head(3)
        print("\nSample rows:")
        for i, row in sample_rows.iterrows():
            print(f"  Row {i+1}: SALES_VALUE={row['SALES_VALUE']:.2f}, lag_1={row.get('sales_value_lag_1', 'N/A')}")

        return zero_sales == 0 and len(lag_features) > 0 and len(rolling_features) > 0

    except Exception as e:
        print(f"❌ Error reading sample: {e}")
        return False

if __name__ == "__main__":
    print("🧪 VALIDATION TESTS")
    print("=" * 50)

    results = []

    results.append(("Config Import", test_config_import()))
    results.append(("Data Files", test_data_files()))
    results.append(("Sample Quality", test_sample_quality()))

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY:")
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 50)
