#!/usr/bin/env python3
"""
Optimize Predictions File
=========================
Convert existing predictions CSV to Parquet format for better storage efficiency.

Usage:
    python scripts/optimize_predictions_file.py
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_FILES

def main():
    predictions_csv = OUTPUT_FILES['predictions_test']
    predictions_parquet = predictions_csv.with_suffix('.parquet')
    
    if not predictions_csv.exists():
        print(f"❌ Predictions CSV not found: {predictions_csv}")
        return
    
    print(f"📊 Loading predictions from: {predictions_csv}")
    print(f"   File size: {predictions_csv.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load CSV
    try:
        df = pd.read_csv(predictions_csv)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return
    
    # Save as Parquet
    print(f"\n💾 Saving to Parquet: {predictions_parquet}")
    try:
        df.to_parquet(predictions_parquet, index=False, compression='snappy')
        parquet_size = predictions_parquet.stat().st_size / 1024 / 1024
        csv_size = predictions_csv.stat().st_size / 1024 / 1024
        
        print(f"✓ Saved successfully!")
        print(f"   Parquet size: {parquet_size:.2f} MB")
        print(f"   CSV size: {csv_size:.2f} MB")
        print(f"   Compression ratio: {csv_size / parquet_size:.2f}x")
        print(f"\n💡 Tip: Use Parquet format for better performance and smaller file size")
        
    except Exception as e:
        print(f"❌ Failed to save Parquet: {e}")
        return
    
    # Optionally compress CSV
    print(f"\n🗜️ Compressing CSV...")
    try:
        compressed_csv = str(predictions_csv) + '.gz'
        df.to_csv(compressed_csv, index=False, compression='gzip')
        compressed_size = Path(compressed_csv).stat().st_size / 1024 / 1024
        print(f"✓ Compressed CSV saved: {compressed_csv}")
        print(f"   Compressed size: {compressed_size:.2f} MB")
        print(f"   Original CSV can be deleted if not needed")
    except Exception as e:
        print(f"⚠️ Failed to compress CSV: {e}")

if __name__ == "__main__":
    main()

