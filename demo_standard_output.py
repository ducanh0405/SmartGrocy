#!/usr/bin/env python3
"""
Demo script to show standardized master table output.
"""
from pathlib import Path
import pandas as pd

def demo_standard_output():
    """Demonstrate the standardized master table output."""

    print("📋 DEMO: Standardized Master Table Output")
    print("=" * 50)

    project_root = Path(__file__).resolve().parent
    processed_dir = project_root / 'data' / '3_processed'

    # Expected output files
    parquet_file = processed_dir / 'master_feature_table.parquet'
    csv_file = processed_dir / 'master_feature_table.csv'

    print("🎯 Standardized Output Files:")
    print(f"   📁 {parquet_file}")
    print(f"   📄 {csv_file}")
    print()

    # Check current files
    if parquet_file.exists():
        parquet_size = parquet_file.stat().st_size / (1024**2)  # MB
        print(".1f"
        # Load and show info
        try:
            df = pd.read_parquet(parquet_file)
            print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(".2f"            print(".2f"
            # Check data quality
            zero_sales = (df['SALES_VALUE'] == 0).sum()
            print(f"   ✅ Zero sales rows: {zero_sales}")

            # Feature counts
            lag_features = [col for col in df.columns if 'lag' in col]
            rolling_features = [col for col in df.columns if 'rolling' in col]
            print(f"   ✅ Lag features: {len(lag_features)}")
            print(f"   ✅ Rolling features: {len(rolling_features)}")

        except Exception as e:
            print(f"   ❌ Error reading parquet: {e}")
    else:
        print("   ❌ Parquet file not found")

    if csv_file.exists():
        csv_size = csv_file.stat().st_size / (1024**2)  # MB
        print(".1f"    else:
        print("   ❌ CSV file not found")

    # Show cleanup of old files
    print("
🧹 Auto-Cleanup:"    old_files = list(processed_dir.glob("master_feature_table_*"))
    old_files = [f for f in old_files if not (f.name.endswith('.parquet') or f.name.endswith('.csv'))]

    if old_files:
        print(f"   🗑️  Old sample files found: {len(old_files)}")
        for old_file in old_files:
            print(f"      - {old_file.name}")
        print("   📝 These will be cleaned up on next pipeline run")
    else:
        print("   ✅ No old sample files to cleanup")

    print("
🚀 How to Run:"    print("   python scripts/run_optimized_pipeline.py")
    print("   # This will generate both .parquet and .csv files")

    print("
📋 Usage:"    print("   # Load parquet (fast, compressed)")
    print("   df = pd.read_parquet('data/3_processed/master_feature_table.parquet')")
    print("   ")
    print("   # Load CSV (universal, readable)")
    print("   df = pd.read_csv('data/3_processed/master_feature_table.csv')")

if __name__ == "__main__":
    demo_standard_output()
