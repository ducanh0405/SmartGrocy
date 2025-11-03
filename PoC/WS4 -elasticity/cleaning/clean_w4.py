"""
Dunnhumby Data Cleaning Pipeline (Workstream 4)
- Clean raw CSVs to core tables
- Validate referential integrity
- Generate QA reports
"""

import logging
from pathlib import Path
import yaml
import duckdb
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DunnhumbyW4Cleaner:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[3]
        self.config = self._load_config()
        self.raw_dir = self.base_dir / self.config['data']['raw_dir']
        self.interim_dir = self.base_dir / self.config['data']['interim_dir']
        self.processed_dir = self.base_dir / self.config['data']['processed_dir']
        self.qa_dir = self.base_dir / "reports" / "qa"
        
        # Ensure directories exist
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.qa_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to DuckDB
        self.db_path = self.processed_dir / "dunnhumby.duckdb"
        self.con = duckdb.connect(str(self.db_path))
        
        # SQL files
        self.sql_dir = Path(__file__).parent / "sql"
        
        # Schema definitions
        self.schemas = ['stg', 'core', 'features']
        
        # Tables to process
        self.tables = [
            'product',
            'transaction',
            'causal',
            'campaign_desc',
            'campaign_table',
            'coupon',
            'coupon_redempt',
            'hh_demo'
        ]

    def _load_config(self) -> dict:
        """Load paths configuration."""
        config_path = self.base_dir / "configs" / "paths.yaml"
        if not config_path.exists():
            return {
                'data': {
                    'raw_dir': 'data/raw/Dunnhumby',
                    'interim_dir': 'data/interim',
                    'processed_dir': 'data/processed'
                }
            }
        with open(config_path) as f:
            return yaml.safe_load(f)

    def ensure_schemas(self):
        """Create schemas if not exist."""
        for schema in self.schemas:
            self.con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    def clean_table(self, table: str):
        """Clean a single table using its SQL definition."""
        logger.info(f"Cleaning {table}")
        
        # Read and parameterize SQL
        sql_path = self.sql_dir / f"{table}.sql"
        sql = sql_path.read_text().replace("${RAW_DIR}", str(self.raw_dir))
        
        # Execute cleaning SQL
        self.con.execute(sql)
        
        # Export staging to parquet
        self.con.execute(f"""
            COPY stg.{table} 
            TO '{self.interim_dir}/{table}.parquet' 
            (FORMAT PARQUET)
        """)
        
        # Generate QA report
        self._generate_qa(table)

    def _generate_qa(self, table: str):
        """Generate QA report for a table."""
        qa_path = self.qa_dir / f"{table}_qa.md"
        
        # Get table info
        row_count = self.con.execute(f"SELECT COUNT(*) FROM core.{table}").fetchone()[0]
        cols = self.con.execute(f"DESCRIBE core.{table}").fetchall()
        sample = self.con.execute(f"SELECT * FROM core.{table} LIMIT 5").fetchdf()
        
        # Calculate NULL percentages
        null_stats = []
        for col in cols:
            col_name = col[0]
            null_count = self.con.execute(f"""
                SELECT COUNT(*) 
                FROM core.{table} 
                WHERE {col_name} IS NULL
            """).fetchone()[0]
            null_pct = (null_count / row_count * 100) if row_count > 0 else 0
            null_stats.append(f"- {col_name}: {null_pct:.2f}%")
        
        # Write report
        with open(qa_path, 'w') as f:
            f.write(f"# QA Report: {table}\n")
            f.write(f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_\n\n")
            
            f.write("## Basic Stats\n")
            f.write(f"- Row count: {row_count:,}\n")
            f.write(f"- Column count: {len(cols)}\n\n")
            
            f.write("## Schema\n")
            for col in cols:
                f.write(f"- {col[0]}: {col[1]}\n")
            f.write("\n")
            
            f.write("## NULL Percentages\n")
            f.write("\n".join(null_stats))
            f.write("\n\n")
            
            f.write("## Sample Rows\n")
            f.write("```\n")
            f.write(sample.to_string())
            f.write("\n```\n")

    def validate_integrity(self):
        """Check referential integrity between tables."""
        logger.info("Validating referential integrity")
        
        # transaction.product_id ⊆ product.product_id
        orphaned = self.con.execute("""
            SELECT COUNT(DISTINCT t.product_id)
            FROM core.transaction t
            LEFT JOIN core.product p ON t.product_id = p.product_id
            WHERE p.product_id IS NULL
        """).fetchone()[0]
        
        if orphaned > 0:
            logger.warning(f"Found {orphaned} transactions with invalid product_id")
        
        # coupon_redempt.coupon_upc ⊆ coupon.coupon_upc
        orphaned = self.con.execute("""
            SELECT COUNT(DISTINCT r.coupon_upc)
            FROM core.coupon_redempt r
            LEFT JOIN core.coupon c ON r.coupon_upc = c.coupon_upc
            WHERE c.coupon_upc IS NULL
        """).fetchone()[0]
        
        if orphaned > 0:
            logger.warning(f"Found {orphaned} redemptions with invalid coupon_upc")
        
        # campaign_hh.campaign ⊆ campaign_desc.campaign
        orphaned = self.con.execute("""
            SELECT COUNT(DISTINCT h.campaign)
            FROM core.campaign_hh h
            LEFT JOIN core.campaign_desc d ON h.campaign = d.campaign
            WHERE d.campaign IS NULL
        """).fetchone()[0]
        
        if orphaned > 0:
            logger.warning(f"Found {orphaned} household campaigns with invalid campaign_id")

    def process_all(self):
        """Run full cleaning pipeline."""
        logger.info("Starting cleaning pipeline")
        
        # Create schemas
        self.ensure_schemas()
        
        # Clean each table
        for table in self.tables:
            self.clean_table(table)
        
        # Validate integrity
        self.validate_integrity()
        
        logger.info("Cleaning pipeline completed!")

if __name__ == "__main__":
    cleaner = DunnhumbyW4Cleaner()
    cleaner.process_all()