import duckdb
import os
from pathlib import Path

# Constants
SAMPLE_SIZE = 200000
DATA_DIR = Path("data/raw/Dunnhumby")
CSV_FILES = [
    "campaign_desc.csv",
    "campaign_table.csv",
    "causal_data.csv",
    "coupon.csv",
    "coupon_redempt.csv",
    "hh_demographic.csv",
    "product.csv",
    "transaction_data.csv"
]

def peek_csv(file_path: Path):
    """Preview a CSV file using DuckDB."""
    print(f"\n{'='*80}\n{file_path.name}\n{'='*80}")
    
    # Get file size in MB
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB\n")
    
    # Create connection and read file
    con = duckdb.connect(":memory:")
    
    # Get schema
    print("SCHEMA:")
    schema = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{file_path}', SAMPLE_SIZE={SAMPLE_SIZE})").fetchall()
    for col in schema:
        print(f"- {col[0]:<20} {col[1]}")
    
    # Get sample rows
    print("\nSAMPLE DATA (first 5 rows):")
    sample = con.execute(f"""
        SELECT *
        FROM read_csv_auto('{file_path}', SAMPLE_SIZE={SAMPLE_SIZE})
        LIMIT 5
    """).fetchall()
    if sample:
        # Get column names
        cols = [col[0] for col in schema]
        print("| " + " | ".join(f"{col:<15}" for col in cols) + " |")
        print("|" + "|".join("-" * 17 for _ in cols) + "|")
        # Print rows
        for row in sample:
            print("| " + " | ".join(f"{str(val)[:15]:<15}" for val in row) + " |")
    
    con.close()

def main():
    for csv_file in CSV_FILES:
        file_path = DATA_DIR / csv_file
        if file_path.exists():
            peek_csv(file_path)
        else:
            print(f"\nWARNING: {csv_file} not found!")

if __name__ == "__main__":
    main()