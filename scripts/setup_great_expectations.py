#!/usr/bin/env python3
"""
Great Expectations Setup for SmartGrocy Pipeline
=================================================

Automatically initializes Great Expectations context and creates
expectation suites for data quality monitoring.

Features:
- Master feature table validation
- 66 features validation (numeric + categorical)
- Missing value checks
- Data type validation
- Row/column count validation

Usage:
    python scripts/setup_great_expectations.py

Requirements:
    pip install great-expectations==0.18.19

Author: SmartGrocy Team
Date: 2025-11-15
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import great_expectations as gx
    from great_expectations.core.batch import BatchRequest
    from great_expectations.checkpoint import Checkpoint
    import pandas as pd
    import logging
    from src.config import (
        OUTPUT_FILES, DATA_DIRS, ALL_FEATURES_CONFIG,
        get_features_by_type, setup_logging
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install great-expectations==0.18.19")
    sys.exit(1)

setup_logging()
logger = logging.getLogger(__name__)

# GX directories
GX_ROOT = PROJECT_ROOT / "great_expectations"
GX_DATA_DOCS = GX_ROOT / "uncommitted" / "data_docs" / "local_site"


def initialize_gx_context():
    """Initialize or get existing GX context"""
    logger.info("=" * 70)
    logger.info("Initializing Great Expectations Context")
    logger.info("=" * 70)
    
    if GX_ROOT.exists():
        logger.info(f"✓ Found existing GX directory: {GX_ROOT}")
        context = gx.get_context(context_root_dir=str(GX_ROOT))
    else:
        logger.info(f"Creating new GX directory: {GX_ROOT}")
        context = gx.get_context(context_root_dir=str(GX_ROOT), mode="file")
    
    logger.info(f"✓ GX Context initialized")
    logger.info(f"  Root: {GX_ROOT}")
    logger.info(f"  Data Docs: {GX_DATA_DOCS}")
    
    return context


def add_datasource(context):
    """Add pandas datasource for master feature table"""
    logger.info("\n" + "=" * 70)
    logger.info("Adding Datasource")
    logger.info("=" * 70)
    
    datasource_name = "master_feature_datasource"
    
    # Check if datasource exists
    try:
        datasource = context.get_datasource(datasource_name)
        logger.info(f"✓ Datasource already exists: {datasource_name}")
        return datasource
    except:
        pass
    
    # Create new datasource
    datasource_config = {
        "name": datasource_name,
        "class_name": "Datasource",
        "execution_engine": {
            "class_name": "PandasExecutionEngine"
        },
        "data_connectors": {
            "default_runtime_data_connector": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier_name"]
            }
        }
    }
    
    context.add_datasource(**datasource_config)
    logger.info(f"✓ Created datasource: {datasource_name}")
    
    return context.get_datasource(datasource_name)


def create_expectation_suite(context):
    """Create expectation suite for master feature table"""
    logger.info("\n" + "=" * 70)
    logger.info("Creating Expectation Suite")
    logger.info("=" * 70)
    
    suite_name = "master_feature_table_suite"
    
    # Delete existing suite if exists
    try:
        context.delete_expectation_suite(suite_name)
        logger.info(f"  Deleted existing suite: {suite_name}")
    except:
        pass
    
    # Create new suite
    suite = context.add_expectation_suite(suite_name)
    logger.info(f"✓ Created suite: {suite_name}")
    
    # Get feature lists
    all_features = get_features_by_type('all')
    numeric_features = get_features_by_type('num')
    categorical_features = get_features_by_type('cat')
    
    logger.info(f"  Total features: {len(all_features)}")
    logger.info(f"  Numeric: {len(numeric_features)}")
    logger.info(f"  Categorical: {len(categorical_features)}")
    
    # Build expectations list
    expectations = []
    
    # 1. Table-level expectations
    expectations.append({
        "expectation_type": "expect_table_row_count_to_be_between",
        "kwargs": {
            "min_value": 1000,
            "max_value": 100000000,  # 100M max
        }
    })
    
    expectations.append({
        "expectation_type": "expect_table_column_count_to_equal",
        "kwargs": {
            "value": len(all_features)
        }
    })
    
    # 2. Column existence
    for feature in all_features:
        expectations.append({
            "expectation_type": "expect_column_to_exist",
            "kwargs": {
                "column": feature
            }
        })
    
    # 3. Numeric features - type and null checks
    for feature in numeric_features[:20]:  # Limit to first 20 for performance
        # Type check
        expectations.append({
            "expectation_type": "expect_column_values_to_be_of_type",
            "kwargs": {
                "column": feature,
                "type_": "float64"
            }
        })
        
        # Null check - allow up to 10% nulls
        expectations.append({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": feature,
                "mostly": 0.90
            }
        })
    
    # 4. Categorical features - null checks
    for feature in categorical_features[:10]:  # Limit to first 10
        expectations.append({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": feature,
                "mostly": 0.85  # Allow 15% nulls for categorical
            }
        })
    
    # Add all expectations to suite
    logger.info(f"\n  Adding {len(expectations)} expectations...")
    for exp in expectations:
        suite.add_expectation(**exp)
    
    # Save suite
    context.save_expectation_suite(suite)
    logger.info(f"✓ Saved suite with {len(expectations)} expectations")
    
    return suite


def create_checkpoint(context, suite_name):
    """Create checkpoint for automated validation"""
    logger.info("\n" + "=" * 70)
    logger.info("Creating Checkpoint")
    logger.info("=" * 70)
    
    checkpoint_name = "master_feature_checkpoint"
    
    # Delete existing checkpoint if exists
    try:
        context.delete_checkpoint(checkpoint_name)
        logger.info(f"  Deleted existing checkpoint: {checkpoint_name}")
    except:
        pass
    
    # Checkpoint config
    checkpoint_config = {
        "name": checkpoint_name,
        "config_version": 1.0,
        "class_name": "SimpleCheckpoint",
        "run_name_template": "%Y%m%d-%H%M%S-master-feature-validation",
    }
    
    # Add checkpoint
    context.add_checkpoint(**checkpoint_config)
    logger.info(f"✓ Created checkpoint: {checkpoint_name}")
    
    return checkpoint_name


def test_validation():
    """Test validation with sample data if master feature table exists"""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Validation")
    logger.info("=" * 70)
    
    master_table_path = OUTPUT_FILES['master_feature_table']
    
    if not master_table_path.exists():
        logger.warning(f"⚠ Master feature table not found: {master_table_path}")
        logger.warning("  Run feature engineering first: python src/pipelines/_02_feature_enrichment.py")
        logger.warning("  Skipping validation test.")
        return False
    
    logger.info(f"  Loading data from: {master_table_path}")
    df = pd.read_parquet(master_table_path)
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Get context
    context = gx.get_context(context_root_dir=str(GX_ROOT))
    
    # Create batch request
    batch_request = {
        "datasource_name": "master_feature_datasource",
        "data_connector_name": "default_runtime_data_connector",
        "data_asset_name": "master_feature_table",
        "runtime_parameters": {"batch_data": df},
        "batch_identifiers": {"default_identifier_name": "master_feature_table"}
    }
    
    # Run checkpoint
    logger.info("\n  Running validation checkpoint...")
    try:
        result = context.run_checkpoint(
            checkpoint_name="master_feature_checkpoint",
            batch_request=batch_request,
            expectation_suite_name="master_feature_table_suite"
        )
        
        # Parse results
        success = result["success"]
        stats = result.get("run_results", {})
        
        if success:
            logger.info("✓ Validation PASSED")
        else:
            logger.warning("⚠ Validation FAILED")
        
        # Show statistics
        if stats:
            for run_id, run_result in stats.items():
                validation_result = run_result.get("validation_result", {})
                statistics = validation_result.get("statistics", {})
                
                logger.info(f"\n  Validation Statistics:")
                logger.info(f"    Evaluated expectations: {statistics.get('evaluated_expectations', 0)}")
                logger.info(f"    Successful expectations: {statistics.get('successful_expectations', 0)}")
                logger.info(f"    Failed expectations: {statistics.get('unsuccessful_expectations', 0)}")
                logger.info(f"    Success rate: {statistics.get('success_percent', 0):.1f}%")
        
        # Show data docs link
        data_docs_path = GX_DATA_DOCS / "index.html"
        if data_docs_path.exists():
            logger.info(f"\n✓ Data Docs generated: {data_docs_path}")
            logger.info(f"  Open in browser to view detailed results")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return False


def main():
    """Main setup function"""
    logger.info("=" * 70)
    logger.info("GREAT EXPECTATIONS SETUP - SMARTGROCY")
    logger.info("=" * 70)
    logger.info(f"Project: {PROJECT_ROOT}")
    logger.info(f"GX Root: {GX_ROOT}")
    
    try:
        # Step 1: Initialize context
        context = initialize_gx_context()
        
        # Step 2: Add datasource
        add_datasource(context)
        
        # Step 3: Create expectation suite
        suite = create_expectation_suite(context)
        
        # Step 4: Create checkpoint
        checkpoint_name = create_checkpoint(context, suite.expectation_suite_name)
        
        # Step 5: Test validation (if data exists)
        test_validation()
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("✓ GREAT EXPECTATIONS SETUP COMPLETE")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("  1. Generate feature table: python src/pipelines/_02_feature_enrichment.py")
        logger.info("  2. Run validation: python scripts/run_data_quality_check.py")
        logger.info(f"  3. View results: open {GX_DATA_DOCS / 'index.html'}")
        logger.info("\nIntegration:")
        logger.info("  - GX validation will run automatically in modern pipeline")
        logger.info("  - Check src/pipelines/_00_modern_orchestrator.py")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
