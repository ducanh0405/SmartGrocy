"""Pytest Configuration and Shared Fixtures
==========================================
Provides mock data fixtures that work without requiring real data files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def ensure_directories(project_root):
    """Ensure all required directories exist before tests run."""
    required_dirs = [
        project_root / 'data' / '2_raw',
        project_root / 'data' / '3_processed',
        project_root / 'data' / 'poc_data',
        project_root / 'models',
        project_root / 'reports' / 'metrics',
    ]
    
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    return required_dirs


@pytest.fixture
def mock_freshretail_data():
    """Generate mock FreshRetail data for testing."""
    np.random.seed(42)
    n_rows = 200
    
    hours = pd.date_range('2024-01-01', periods=168, freq='h')  # 1 week
    
    data = {
        'product_id': np.random.randint(1000, 1010, n_rows),
        'store_id': np.random.randint(1, 4, n_rows),
        'city_id': np.random.randint(1, 3, n_rows),
        'dt': np.random.choice(hours, n_rows),
        'sale_amount': np.random.uniform(0, 100, n_rows).round(2),
    }
    
    df = pd.DataFrame(data)
    df['hour_timestamp'] = pd.to_datetime(df['dt'])
    df['sales_quantity'] = df['sale_amount']
    
    return df.sort_values(['product_id', 'store_id', 'hour_timestamp']).reset_index(drop=True)


@pytest.fixture
def mock_transaction_data():
    """Generate mock Dunnhumby transaction data."""
    np.random.seed(42)
    n_rows = 200
    
    data = {
        'PRODUCT_ID': np.random.choice(['P1', 'P2', 'P3', 'P4', 'P5'], n_rows),
        'STORE_ID': np.random.choice(['S1', 'S2', 'S3'], n_rows),
        'WEEK_NO': np.tile(range(1, 21), 10)[:n_rows],
        'SALES_VALUE': np.random.uniform(10, 200, n_rows).round(2),
        'QUANTITY': np.random.randint(1, 15, n_rows),
        'RETAIL_DISC': np.random.uniform(-10, 0, n_rows).round(2),
        'COUPON_DISC': np.random.uniform(-5, 0, n_rows).round(2),
    }
    
    return pd.DataFrame(data).sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)


@pytest.fixture
def mock_product_data():
    """Generate mock product data."""
    return pd.DataFrame({
        'PRODUCT_ID': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'DEPARTMENT': ['D1', 'D2', 'D1', 'D3', 'D2'],
        'COMMODITY_DESC': ['C1', 'C2', 'C1', 'C3', 'C2'],
    })


@pytest.fixture
def sample_data_dir(project_root, ensure_directories):
    """Sample data directory - returns existing directory or raw data dir."""
    poc_data_dir = project_root / 'data' / 'poc_data'
    raw_data_dir = project_root / 'data' / '2_raw'
    
    # If POC data has CSV files, use it; otherwise use raw
    if poc_data_dir.exists() and list(poc_data_dir.glob('*.csv')):
        return poc_data_dir
    return raw_data_dir


@pytest.fixture
def sample_freshretail_data(sample_data_dir, mock_freshretail_data):
    """Load sample FreshRetail data or return mock data."""
    possible_files = [
        'freshretail_train_sample.csv',
        'freshretail_train.csv',
        'freshretail_train.parquet'
    ]
    
    for filename in possible_files:
        data_path = sample_data_dir / filename
        if data_path.exists():
            if filename.endswith('.parquet'):
                return pd.read_parquet(data_path)
            else:
                return pd.read_csv(data_path)
    
    # Return mock data if no real data found
    return mock_freshretail_data


@pytest.fixture
def sample_transaction_data(sample_data_dir, mock_transaction_data):
    """Load sample transaction data or return mock data."""
    data_path = sample_data_dir / 'transaction_data.csv'
    
    if data_path.exists():
        return pd.read_csv(data_path)
    
    # Return mock data if no real data found
    return mock_transaction_data


@pytest.fixture
def sample_product_data(sample_data_dir, mock_product_data):
    """Load sample product data or return mock data."""
    data_path = sample_data_dir / 'product.csv'
    
    if data_path.exists():
        return pd.read_csv(data_path)
    
    # Return mock data if no real data found
    return mock_product_data


@pytest.fixture
def sample_master_data(sample_transaction_data):
    """Create master data from transaction data (after WS0 aggregation)."""
    df = sample_transaction_data
    
    # Check if already in master format or needs aggregation
    if 'PRODUCT_ID' in df.columns and 'WEEK_NO' in df.columns:
        master = df.groupby(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).agg({
            'SALES_VALUE': 'sum',
            'QUANTITY': 'sum',
            'RETAIL_DISC': 'sum',
            'COUPON_DISC': 'sum',
        }).reset_index()
    else:
        # Already in master format or different format
        master = df.copy()
    
    return master.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)


# Configure pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests (quick validation)"
    )
    config.addinivalue_line(
        "markers", "feature: marks tests as feature engineering tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requires data)"
    )


# Auto-use fixtures for all tests
@pytest.fixture(autouse=True)
def setup_test_environment(ensure_directories):
    """Automatically setup test environment for all tests."""
    pass
