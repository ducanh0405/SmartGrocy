#!/usr/bin/env python3
"""
Test script to verify pipeline output standardization.
"""
import subprocess
import sys
from pathlib import Path

def test_pipeline_output():
    """Test that pipeline generates standardized output files."""

    print("🧪 TESTING PIPELINE OUTPUT STANDARDIZATION")
    print("=" * 50)

    project_root = Path(__file__).resolve().parent

    # Check if we can run pipeline (Python available)
    try:
        result = subprocess.run([sys.executable, "--version"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            python_available = True
            print("✅ Python available for testing")
        else:
            python_available = False
            print("❌ Python not available")
    except:
        python_available = False
        print("❌ Python execution failed")

    if not python_available:
        print("\n⚠️  Cannot run full pipeline test (Python not available)")
        print("Manual verification needed:")
        print("   python scripts/run_optimized_pipeline.py")
        return

    # Check current state
    processed_dir = project_root / 'data' / '3_processed'
    parquet_file = processed_dir / 'master_feature_table.parquet'
    csv_file = processed_dir / 'master_feature_table.csv'

    print("
📁 Current Files:"    if parquet_file.exists():
        size_mb = parquet_file.stat().st_size / (1024**2)
        print(".1f"    else:
        print("   ❌ master_feature_table.parquet: missing")

    if csv_file.exists():
        size_mb = csv_file.stat().st_size / (1024**2)
        print(".1f"    else:
        print("   ❌ master_feature_table.csv: missing")

    # Check for old files that should be cleaned up
    old_files = list(processed_dir.glob("master_feature_table_*"))
    old_files = [f for f in old_files if not (f.name.endswith('.parquet') or f.name.endswith('.csv'))]

    print("
🧹 Cleanup Status:"    if old_files:
        print(f"   📋 Old sample files: {len(old_files)}")
        for old_file in old_files:
            print(f"      - {old_file.name}")
    else:
        print("   ✅ No old sample files")

    # Simulate pipeline run (just check code structure)
    pipeline_file = project_root / 'src' / 'pipelines' / '_02_feature_enrichment.py'

    print("
🔧 Pipeline Code Check:"    if pipeline_file.exists():
        with open(pipeline_file, 'r') as f:
            content = f.read()

        # Check for key features
        checks = [
            ('CSV export', 'master_df.to_csv(output_csv' in content),
            ('Cleanup code', 'glob.glob' in content and 'unlink' in content),
            ('Quality logging', 'SALES_VALUE range' in content),
            ('Zero sales check', 'zero_sales' in content),
        ]

        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"   {status} {check_name}")

    print("
🎯 Expected Output After Pipeline Run:"    print("   ✅ data/3_processed/master_feature_table.parquet")
    print("   ✅ data/3_processed/master_feature_table.csv")
    print("   ✅ Old sample files cleaned up")
    print("   ✅ Quality metrics logged")

    print("
🚀 Run Pipeline:"    print("   python scripts/run_optimized_pipeline.py")

    print("
📋 Verification Commands:"    print("   # Check files")
    print("   dir data\\3_processed\\")
    print("   ")
    print("   # Verify CSV content")
    print("   head -5 data/3_processed/master_feature_table.csv")

if __name__ == "__main__":
    test_pipeline_output()
