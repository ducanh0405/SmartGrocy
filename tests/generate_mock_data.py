"""Generate Mock Data for CI Testing
=====================================
Creates minimal fake CSV data for CI pipeline testing without requiring real data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_freshretail_data(output_dir: Path, n_rows: int = 1000):
    """Create mock FreshRetail-format data for testing."""
    logger.info(f"Generating {n_rows} rows of mock FreshRetail data...")
    
    np.random.seed(42)
    
    # Generate realistic timestamp range (1 month of hourly data)
    start_date = pd.Timestamp('2024-01-01')
    hours = pd.date_range(start_date, periods=720, freq='h')  # 30 days
    
    # Generate data
    data = {
        'product_id': np.random.randint(1000, 1050, n_rows),  # 50 products
        'store_id': np.random.randint(1, 11, n_rows),  # 10 stores
        'city_id': np.random.randint(1, 4, n_rows),  # 3 cities
        'dt': np.random.choice(hours, n_rows),
        'sale_amount': np.random.uniform(0, 100, n_rows).round(2),
    }
    
    df = pd.DataFrame(data)
    
    # Add derived columns to match expected format
    df['hour_timestamp'] = pd.to_datetime(df['dt'])
    df['sales_quantity'] = df['sale_amount']
    
    # Sort by time
    df = df.sort_values(['product_id', 'store_id', 'hour_timestamp']).reset_index(drop=True)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'freshretail_train.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved mock data to {output_path} ({len(df)} rows)")
    
    return df

def create_mock_dunnhumby_data(output_dir: Path, n_rows: int = 1000):
    """Create mock Dunnhumby-format data for testing."""
    logger.info(f"Generating {n_rows} rows of mock Dunnhumby data...")
    
    np.random.seed(42)
    
    # Transaction data
    transaction_data = {
        'PRODUCT_ID': np.random.choice([f'P{i}' for i in range(1, 21)], n_rows),  # 20 products
        'STORE_ID': np.random.choice([f'S{i}' for i in range(1, 6)], n_rows),  # 5 stores
        'WEEK_NO': np.random.randint(1, 53, n_rows),
        'SALES_VALUE': np.random.uniform(10, 500, n_rows).round(2),
        'QUANTITY': np.random.randint(1, 20, n_rows),
        'RETAIL_DISC': np.random.uniform(-10, 0, n_rows).round(2),
        'COUPON_DISC': np.random.uniform(-5, 0, n_rows).round(2),
    }
    
    df = pd.DataFrame(transaction_data)
    df = df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'transaction_data.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved transaction_data to {output_path} ({len(df)} rows)")
    
    # Product data
    products = pd.DataFrame({
        'PRODUCT_ID': [f'P{i}' for i in range(1, 21)],
        'DEPARTMENT': np.random.choice(['D1', 'D2', 'D3'], 20),
        'COMMODITY_DESC': np.random.choice(['C1', 'C2', 'C3', 'C4'], 20),
    })
    products.to_csv(output_dir / 'product.csv', index=False)
    logger.info(f"✓ Saved product.csv ({len(products)} rows)")
    
    return df

def create_mock_poc_data(output_dir: Path, n_rows: int = 500):
    """Create minimal POC/integration test data."""
    logger.info(f"Generating {n_rows} rows of mock POC data...")
    
    np.random.seed(42)
    
    data = {
        'product_id': np.random.randint(1000, 1020, n_rows),
        'store_id': np.random.randint(1, 5, n_rows),
        'hour_timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='h'),
        'sales_quantity': np.random.uniform(0, 50, n_rows).round(2),
    }
    
    df = pd.DataFrame(data)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'transaction_data.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved POC data to {output_path} ({len(df)} rows)")
    
    return df

def main():
    """Generate all mock data files for CI testing."""
    project_root = Path(__file__).resolve().parent.parent
    
    logger.info("="*70)
    logger.info("MOCK DATA GENERATOR FOR CI TESTING")
    logger.info("="*70)
    
    # Create mock data directories
    raw_data_dir = project_root / 'data' / '2_raw'
    poc_data_dir = project_root / 'data' / 'poc_data'
    processed_dir = project_root / 'data' / '3_processed'
    models_dir = project_root / 'models'
    reports_dir = project_root / 'reports' / 'metrics'
    
    # Create all required directories
    for directory in [raw_data_dir, poc_data_dir, processed_dir, models_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Ensured directory exists: {directory}")
    
    # Generate mock data for both datasets
    create_mock_freshretail_data(raw_data_dir, n_rows=1000)
    create_mock_dunnhumby_data(raw_data_dir, n_rows=1000)
    create_mock_poc_data(poc_data_dir, n_rows=500)
    
    logger.info("="*70)
    logger.info("✅ MOCK DATA GENERATION COMPLETE")
    logger.info("="*70)
    logger.info("\nGenerated files:")
    logger.info(f"  - {raw_data_dir}/freshretail_train.csv")
    logger.info(f"  - {raw_data_dir}/transaction_data.csv")
    logger.info(f"  - {raw_data_dir}/product.csv")
    logger.info(f"  - {poc_data_dir}/transaction_data.csv")
    logger.info("\nCI tests can now run without requiring real data!")

if __name__ == '__main__':
    main()
