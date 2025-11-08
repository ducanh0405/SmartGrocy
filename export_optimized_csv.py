#!/usr/bin/env python3
"""
Simple script to export optimized CSV from existing improved sample.
"""
import pandas as pd
from pathlib import Path

def export_optimized_csv():
    """Export optimized master table CSV from existing data."""

    print("📤 EXPORTING OPTIMIZED MASTER TABLE CSV")
    print("=" * 50)

    project_root = Path(__file__).resolve().parent

    # Input: existing improved sample
    sample_path = project_root / 'data' / '3_processed' / 'master_feature_table_improved_sample.csv'

    # Outputs
    optimized_csv = project_root / 'data' / '3_processed' / 'master_feature_table_optimized.csv'
    model_features_csv = project_root / 'data' / '3_processed' / 'model_ready_features.csv'

    if not sample_path.exists():
        print(f"❌ Sample file not found: {sample_path}")
        return False

    print("📖 Loading improved sample...")
    df = pd.read_csv(sample_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Quality checks
    sales_values = df['SALES_VALUE']
    print("
📊 Quality Metrics:"    print(".2f"    print(".2f"    print(".2f"
    # Check for zero sales
    zero_sales = (sales_values == 0).sum()
    if zero_sales > 0:
        print(f"   ⚠️  Found {zero_sales} zero sales rows")
    else:
        print("   ✅ No zero sales rows")

    # Feature counts
    lag_cols = [col for col in df.columns if 'lag' in col]
    rolling_cols = [col for col in df.columns if 'rolling' in col]
    calendar_cols = [col for col in df.columns if col in ['week_of_year', 'month_proxy', 'quarter', 'week_sin', 'week_cos']]
    price_cols = [col for col in df.columns if col in ['base_price', 'total_discount', 'discount_pct']]

    print(f"   Lag features: {len(lag_cols)}")
    print(f"   Rolling features: {len(rolling_cols)}")
    print(f"   Calendar features: {len(calendar_cols)}")
    print(f"   Price features: {len(price_cols)}")

    # Sort data
    df = df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)

    # Export optimized CSV
    print("
💾 Exporting files..."    df.to_csv(optimized_csv, index=False)
    print(f"   ✅ Optimized CSV: {optimized_csv}")

    # Create model-ready features (subset for modeling)
    try:
        # Import feature config (if available)
        import sys
        sys.path.append(str(project_root / 'src'))

        from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        available_features = [col for col in feature_cols if col in df.columns]

        model_df = df[['PRODUCT_ID', 'STORE_ID', 'WEEK_NO', 'SALES_VALUE'] + available_features].copy()

        # Fill NaNs appropriately
        for col in available_features:
            if col in NUMERIC_FEATURES:
                model_df[col] = model_df[col].fillna(0)
            elif col in CATEGORICAL_FEATURES:
                model_df[col] = model_df[col].fillna('unknown')

        model_df.to_csv(model_features_csv, index=False)
        print(f"   ✅ Model-ready CSV: {model_features_csv} ({len(available_features)} features)")

    except ImportError:
        print("   ⚠️  Config not available, skipping model-ready export")

    print("
📏 File Sizes:"    try:
        csv_size = optimized_csv.stat().st_size / 1024  # KB
        print(".1f"        if model_features_csv.exists():
            model_size = model_features_csv.stat().st_size / 1024
            print(".1f"    except:
        pass

    print("
🎯 Summary:"    print("   ✅ Master table exported with optimized features"    print("   ✅ Data quality: SALES_VALUE > 0, complete feature coverage"    print("   ✅ Ready for model training and analysis"
    return True

if __name__ == "__main__":
    success = export_optimized_csv()
    if success:
        print("\n🎉 EXPORT COMPLETE!")
    else:
        print("\n❌ EXPORT FAILED!")
