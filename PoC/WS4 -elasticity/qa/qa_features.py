import os
import logging
import duckdb
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parents[3]
FEATURES_PATH = PROJECT_ROOT / "data/processed/features_weekly_store_sku.parquet"
DB_PATH = PROJECT_ROOT / "data/processed/dunnhumby.duckdb"
QA_REPORT_PATH = PROJECT_ROOT / "reports/qa/features_weekly_store_sku.md"

def connect_db():
    """Connect to DuckDB and create features schema"""
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("CREATE SCHEMA IF NOT EXISTS features;")
    return conn

def load_features_to_db(conn):
    """Load parquet file into DuckDB table"""
    query = f"""
    CREATE OR REPLACE TABLE features.weekly_store_sku AS 
    SELECT * FROM read_parquet('{str(FEATURES_PATH)}');
    """
    conn.execute(query)
    logger.info(f"Loaded features into table: features.weekly_store_sku")

def run_qa_checks(conn):
    """Run all QA checks and return results as string"""
    qa_results = []
    
    # 1. Row counts and distinct values
    counts = conn.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT store_id) as distinct_stores,
            COUNT(DISTINCT product_id) as distinct_products,
            COUNT(DISTINCT week_no) as distinct_weeks
        FROM features.weekly_store_sku;
    """).fetchone()
    
    qa_results.append("## 1. Basic Counts")
    qa_results.append(f"- Total rows: {counts[0]:,}")
    qa_results.append(f"- Distinct stores: {counts[1]:,}")
    qa_results.append(f"- Distinct products: {counts[2]:,}")
    qa_results.append(f"- Distinct weeks: {counts[3]:,}")
    
    # 2. Check for duplicate keys
    duplicates = conn.execute("""
        SELECT COUNT(*) as duplicate_count
        FROM (
            SELECT store_id, product_id, week_no, COUNT(*) as cnt
            FROM features.weekly_store_sku
            GROUP BY store_id, product_id, week_no
            HAVING COUNT(*) > 1
        );
    """).fetchone()[0]
    
    qa_results.append("\n## 2. Duplicate Key Check")
    qa_results.append(f"- Rows with duplicate (store_id, product_id, week_no): {duplicates:,}")
    
    # 3. Value range checks
    value_checks = conn.execute("""
        SELECT 
            SUM(CASE WHEN units < 0 THEN 1 ELSE 0 END) as negative_units,
            SUM(CASE WHEN baskets < 0 THEN 1 ELSE 0 END) as negative_baskets,
            SUM(CASE WHEN avg_net_price <= 0 THEN 1 ELSE 0 END) as non_positive_net_price,
            SUM(CASE WHEN avg_gross_price <= 0 THEN 1 ELSE 0 END) as non_positive_gross_price
        FROM features.weekly_store_sku;
    """).fetchone()
    
    qa_results.append("\n## 3. Value Range Checks")
    qa_results.append(f"- Rows with negative units: {value_checks[0]:,}")
    qa_results.append(f"- Rows with negative baskets: {value_checks[1]:,}")
    qa_results.append(f"- Rows with non-positive avg_net_price: {value_checks[2]:,}")
    qa_results.append(f"- Rows with non-positive avg_gross_price: {value_checks[3]:,}")
    
    # 4. Discount rate check
    discount_check = conn.execute("""
        SELECT 
            COUNT(*) as invalid_discount_rate
        FROM features.weekly_store_sku
        WHERE units > 0 
        AND avg_gross_price > 0
        AND (avg_discount_rate < 0 OR avg_discount_rate > 1);
    """).fetchone()[0]
    
    qa_results.append("\n## 4. Discount Rate Check")
    qa_results.append("For rows with units > 0 and avg_gross_price > 0:")
    qa_results.append(f"- Rows with avg_discount_rate outside [0,1]: {discount_check:,}")
    
    # 5. Promo distribution
    promo_stats = conn.execute("""
        SELECT 
            AVG(CAST(promo_display AS FLOAT))*100 as display_rate,
            AVG(CAST(promo_mailer AS FLOAT))*100 as mailer_rate
        FROM features.weekly_store_sku;
    """).fetchone()
    
    qa_results.append("\n## 5. Promotion Distribution")
    qa_results.append(f"- Rows with promo_display=1: {promo_stats[0]:.2f}%")
    qa_results.append(f"- Rows with promo_mailer=1: {promo_stats[1]:.2f}%")
    
    # 6. Null check for categorical columns
    null_checks = conn.execute("""
        SELECT 
            (1 - COUNT(CASE WHEN department IS NULL THEN 1 END)::FLOAT/COUNT(*))*100 as dept_rate,
            (1 - COUNT(CASE WHEN brand IS NULL THEN 1 END)::FLOAT/COUNT(*))*100 as brand_rate,
            (1 - COUNT(CASE WHEN commodity IS NULL THEN 1 END)::FLOAT/COUNT(*))*100 as commodity_rate,
            (1 - COUNT(CASE WHEN sub_commodity IS NULL THEN 1 END)::FLOAT/COUNT(*))*100 as sub_commodity_rate
        FROM features.weekly_store_sku;
    """).fetchone()
    
    qa_results.append("\n## 6. Non-null Rates for Categorical Columns")
    qa_results.append(f"- department: {null_checks[0]:.2f}%")
    qa_results.append(f"- brand: {null_checks[1]:.2f}%")
    qa_results.append(f"- commodity: {null_checks[2]:.2f}%")
    qa_results.append(f"- sub_commodity: {null_checks[3]:.2f}%")
    
    return "\n".join(qa_results)

def main():
    # Connect to DB and load features
    conn = connect_db()
    load_features_to_db(conn)
    
    # Run QA checks
    qa_results = run_qa_checks(conn)
    
    # Write QA report
    with open(QA_REPORT_PATH, 'w') as f:
        f.write("# QA Report: Weekly Store-SKU Features\n\n")
        f.write(qa_results)
    
    logger.info(f"\nFeatures table location: {DB_PATH} (table: features.weekly_store_sku)")
    logger.info(f"QA report saved to: {QA_REPORT_PATH}")

if __name__ == "__main__":
    main()