"""
Build Feature Pipeline for Workstream 4
- Create weekly store-SKU level features
- Export to parquet for analysis
"""

import logging
from pathlib import Path
import duckdb
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DunnhumbyW4Features:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[3]
        self.config = self._load_config()
        self.processed_dir = self.base_dir / self.config['data']['processed_dir']
        
        # Connect to DuckDB
        self.db_path = self.processed_dir / "dunnhumby.duckdb"
        self.con = duckdb.connect(str(self.db_path))
        
        # SQL file for feature creation
        self.sql_path = Path(__file__).parent.parent / "cleaning" / "sql" / "weekly_features.sql"

    def _load_config(self) -> dict:
        """Load paths configuration."""
        config_path = self.base_dir / "configs" / "paths.yaml"
        if not config_path.exists():
            return {
                'data': {
                    'processed_dir': 'data/processed'
                }
            }
        with open(config_path) as f:
            return yaml.safe_load(f)

    def build_features(self):
        """Create weekly store-SKU feature table."""
        logger.info("Building weekly store-SKU features")
        
        # Execute feature SQL
        sql = self.sql_path.read_text()
        self.con.execute(sql)
        
        # Export to parquet
        output_path = self.processed_dir / "features_weekly_store_sku.parquet"
        self.con.execute(f"""
            COPY features.weekly_store_sku 
            TO '{output_path}' 
            (FORMAT PARQUET)
        """)
        
        # Print summary
        row_count = self.con.execute(
            "SELECT COUNT(*) FROM features.weekly_store_sku"
        ).fetchone()[0]
        
        sample = self.con.execute("""
            SELECT * 
            FROM features.weekly_store_sku 
            LIMIT 5
        """).df()
        
        logger.info(f"Created features table with {row_count:,} rows")
        logger.info("\nSample rows:")
        print(sample)
        
        logger.info(f"\nFeature file saved to: {output_path}")

if __name__ == "__main__":
    features = DunnhumbyW4Features()
    features.build_features()