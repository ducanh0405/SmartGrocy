"""
Modular Data Preview Tool

This script demonstrates how to use the new dataset interface to preview data.
It can work with any dataset that implements the BaseDatasetInterface.
"""

import duckdb
import os
from pathlib import Path
import sys

# Add the src directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from dataset_dunnhumby import create_dataset

# Constants for preview
SAMPLE_SIZE = 200000


def peek_csv_with_dataset(dataset, table_name: str):
    """Preview a specific table using dataset interface."""
    print(f"\n{'='*80}\n{table_name.upper()}\n{'='*80}")
    
    # Get file mapping from dataset
    table_mapping = dataset.get_table_mapping()
    
    if table_name not in table_mapping:
        print(f"Table '{table_name}' not found in dataset mapping!")
        print(f"Available tables: {list(table_mapping.keys())}")
        return
    
    file_name = table_mapping[table_name]
    file_path = dataset.raw_dir / file_name
    
    if not file_path.exists():
        print(f"WARNING: {file_name} not found at {file_path}!")
        return
    
    # Get file size in MB
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File: {file_name}")
    print(f"Path: {file_path}")
    print(f"Size: {size_mb:.2f} MB\n")
    
    # Create connection and read file
    con = duckdb.connect(":memory:")
    
    try:
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
        
        # Get row count
        row_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{file_path}', SAMPLE_SIZE={SAMPLE_SIZE})").fetchall()[0][0]
        print(f"\nTotal rows: {row_count:,}")
        
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
    finally:
        con.close()


def peek_all_tables(dataset):
    """Preview all available tables in the dataset."""
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset.__class__.__name__}")
    print(f"Raw directory: {dataset.raw_dir}")
    print(f"{'='*80}")
    
    table_mapping = dataset.get_table_mapping()
    print(f"\nAvailable tables: {len(table_mapping)}")
    
    for table_name in table_mapping.keys():
        peek_csv_with_dataset(dataset, table_name)


def main():
    """Main function to demonstrate modular data preview."""
    print("Modular Data Preview Tool")
    print("=========================")
    
    # Create dataset instance (can be easily changed to other datasets)
    dataset_config = "configs/dataset_dunnhumby_paths.yaml"
    dataset = create_dataset(dataset_config)
    
    print(f"\nUsing dataset: {dataset.__class__.__name__}")
    print(f"Configuration: {dataset_config}")
    
    # Validate data first
    if dataset.validate_data():
        print("✓ Data validation passed")
    else:
        print("✗ Data validation failed - some files may be missing")
        print("Proceeding with available files...")
    
    # Preview all tables
    peek_all_tables(dataset)
    
    print(f"\n{'='*80}")
    print("Preview completed!")
    print("To switch datasets:")
    print("1. Create a new dataset class implementing BaseDatasetInterface")
    print("2. Change the create_dataset() call in main()")
    print("3. Update the configuration file path")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
