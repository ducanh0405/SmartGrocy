"""
Dunnhumby Data Cleaning Pipeline

This script handles the cleaning and transformation of Dunnhumby data:
1. Load raw CSVs
2. Clean & validate
3. Save as optimized Parquet files

Key cleaning tasks:
- Handle missing values
- Fix data types
- Remove duplicates
- Validate referential integrity
- Create derived features
"""

import duckdb
import pandas as pd
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DunnhumbyDataCleaner:
    def __init__(self, config_path: str = "configs/paths.yaml"):
        """Initialize data cleaner with paths configuration."""
        self.config = self._load_config(config_path)
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.interim_dir = Path(self.config['data']['interim_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        
        # Initialize DuckDB connection
        self.con = duckdb.connect(":memory:")
        
        # Register tables we'll clean
        self.tables = {
            # Core transaction data
            'transactions': 'transaction_data.csv',
            'products': 'product.csv',
            # Customer data
            'demographics': 'hh_demographic.csv',
            # Promotional data
            'causal': 'causal_data.csv',
            # Coupon data
            'coupons': 'coupon.csv',
            'coupon_redemptions': 'coupon_redempt.csv',
            'campaigns': 'campaign_table.csv',
            'campaign_desc': 'campaign_desc.csv'
        }

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def clean_transactions(self) -> pd.DataFrame:
        """Clean transaction_data.csv."""
        logger.info("Cleaning transactions data...")
        
        # Load with DuckDB for efficient processing
        df = self.con.execute(f"""
            SELECT *
            FROM read_csv_auto('{self.raw_dir}/transaction_data.csv')
        """).df()
        
        # Data cleaning steps
        df = df.assign(
            # Convert transaction time to proper datetime
            TRANS_DATETIME = lambda x: pd.to_datetime(
                x['DAY'].astype(str) + ' ' + x['TRANS_TIME'].str.zfill(4),
                format='%j %H%M'
            ),
            # Extract hour from transaction time
            TRANS_HOUR = lambda x: x['TRANS_TIME'].str[:2].astype(int),
            # Calculate total discount
            TOTAL_DISCOUNT = lambda x: x['RETAIL_DISC'] + x['COUPON_DISC'] + x['COUPON_MATCH_DISC'],
            # Calculate final price
            FINAL_PRICE = lambda x: x['SALES_VALUE'] + x['TOTAL_DISCOUNT']
        )
        
        # Validate values
        assert (df['QUANTITY'] > 0).all(), "Found negative quantities"
        assert (df['FINAL_PRICE'] >= 0).all(), "Found negative final prices"
        
        return df

    def clean_products(self) -> pd.DataFrame:
        """Clean product.csv."""
        logger.info("Cleaning products data...")
        
        df = self.con.execute(f"""
            SELECT *
            FROM read_csv_auto('{self.raw_dir}/product.csv')
        """).df()
        
        # Clean size information
        df['UNIT'] = df['CURR_SIZE_OF_PRODUCT'].str.extract(r'([A-Za-z]+)$')
        df['SIZE_VALUE'] = pd.to_numeric(
            df['CURR_SIZE_OF_PRODUCT'].str.extract(r'([\d.]+)')[0],
            errors='coerce'
        )
        
        # Fill missing values
        df['MANUFACTURER'] = df['MANUFACTURER'].fillna(-1)
        df['DEPARTMENT'] = df['DEPARTMENT'].fillna('Unknown')
        df['BRAND'] = df['BRAND'].fillna('Unknown')
        
        return df

    def clean_demographics(self) -> pd.DataFrame:
        """Clean hh_demographic.csv."""
        logger.info("Cleaning demographics data...")
        
        df = self.con.execute(f"""
            SELECT *
            FROM read_csv_auto('{self.raw_dir}/hh_demographic.csv')
        """).df()
        
        # Convert categorical columns
        cat_cols = ['AGE_DESC', 'MARITAL_STATUS_CODE', 'INCOME_DESC', 
                   'HOMEOWNER_DESC', 'HH_COMP_DESC', 'HOUSEHOLD_SIZE_DESC',
                   'KID_CATEGORY_DESC']
        
        df[cat_cols] = df[cat_cols].fillna('Unknown')
        
        # Extract numeric values where possible
        df['HOUSEHOLD_SIZE'] = df['HOUSEHOLD_SIZE_DESC'].str.extract(r'(\d+)').astype(float)
        df['INCOME_MIN'] = df['INCOME_DESC'].str.extract(r'(\d+)').astype(float)
        
        return df

    def clean_causal(self) -> pd.DataFrame:
        """Clean causal_data.csv."""
        logger.info("Cleaning causal (promotional) data...")
        
        df = self.con.execute(f"""
            SELECT *
            FROM read_csv_auto('{self.raw_dir}/causal_data.csv')
        """).df()
        
        # Convert display to boolean
        df['display'] = df['display'].astype(int).astype(bool)
        
        # Validate week numbers
        assert df['WEEK_NO'].between(1, 53).all(), "Invalid week numbers found"
        
        return df

    def clean_coupons(self) -> Dict[str, pd.DataFrame]:
        """Clean coupon-related tables."""
        logger.info("Cleaning coupon data...")
        
        # Load all coupon-related tables
        coupons = self.con.execute(f"""
            SELECT *
            FROM read_csv_auto('{self.raw_dir}/coupon.csv')
        """).df()
        
        redemptions = self.con.execute(f"""
            SELECT *
            FROM read_csv_auto('{self.raw_dir}/coupon_redempt.csv')
        """).df()
        
        campaigns = self.con.execute(f"""
            SELECT *
            FROM read_csv_auto('{self.raw_dir}/campaign_table.csv')
        """).df()
        
        campaign_desc = self.con.execute(f"""
            SELECT *
            FROM read_csv_auto('{self.raw_dir}/campaign_desc.csv')
        """).df()
        
        # Validate referential integrity
        assert set(redemptions['COUPON_UPC']).issubset(set(coupons['COUPON_UPC'])), \
            "Found redemptions for non-existent coupons"
        
        assert set(redemptions['CAMPAIGN']).issubset(set(campaign_desc['CAMPAIGN'])), \
            "Found redemptions for non-existent campaigns"
        
        return {
            'coupons': coupons,
            'redemptions': redemptions,
            'campaigns': campaigns,
            'campaign_desc': campaign_desc
        }

    def save_parquet(self, df: pd.DataFrame, name: str):
        """Save DataFrame as Parquet with optimal settings."""
        output_path = self.processed_dir / f"{name}.parquet"
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logger.info(f"Saved {name}.parquet ({df.shape[0]:,} rows)")

    def process_all(self):
        """Run full cleaning pipeline."""
        # Clean core transaction data
        transactions = self.clean_transactions()
        products = self.clean_products()
        
        # Clean customer data
        demographics = self.clean_demographics()
        
        # Clean promotional data
        causal = self.clean_causal()
        
        # Clean coupon data
        coupon_data = self.clean_coupons()
        
        # Save all cleaned datasets
        self.save_parquet(transactions, 'transactions')
        self.save_parquet(products, 'products')
        self.save_parquet(demographics, 'demographics')
        self.save_parquet(causal, 'causal')
        
        for name, df in coupon_data.items():
            self.save_parquet(df, name)
        
        logger.info("Data cleaning pipeline completed!")

if __name__ == "__main__":
    cleaner = DunnhumbyDataCleaner()
    cleaner.process_all()