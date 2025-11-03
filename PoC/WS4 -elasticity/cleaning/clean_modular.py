"""
Modular Data Cleaning Pipeline

This script demonstrates how to use the new dataset interface for data cleaning.
It can work with any dataset that implements the BaseDatasetInterface.

Usage:
    python clean_modular.py [dataset_config_path]

Example:
    python clean_modular.py configs/dataset_dunnhumby_paths.yaml
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add the src directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataset_dunnhumby import create_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_with_dataset(dataset_config: Optional[str] = None, dataset_type: str = "dunnhumby"):
    """
    Clean data using the specified dataset implementation.
    
    Args:
        dataset_config: Path to dataset configuration file
        dataset_type: Type of dataset to use ('dunnhumby', etc.)
    """
    logger.info(f"Starting modular data cleaning pipeline...")
    logger.info(f"Dataset type: {dataset_type}")
    logger.info(f"Configuration: {dataset_config or 'default'}")
    
    # Create dataset instance based on type
    if dataset_type == "dunnhumby":
        dataset = create_dataset(dataset_config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    logger.info(f"Using dataset class: {dataset.__class__.__name__}")
    logger.info(f"Raw data directory: {dataset.raw_dir}")
    logger.info(f"Processed data directory: {dataset.processed_dir}")
    
    # Validate data before processing
    logger.info("Validating data...")
    if not dataset.validate_data():
        logger.error("Data validation failed!")
        logger.error("Please check that all required files are present and valid.")
        return False
    
    logger.info("✓ Data validation passed")
    
    # Run the cleaning pipeline
    try:
        dataset.process_all()
        logger.info("✓ Data cleaning pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data cleaning pipeline failed: {e}")
        return False


def main():
    """
    Main function to run modular data cleaning.
    """
    print("Modular Data Cleaning Pipeline")
    print("==============================")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/dataset_dunnhumby_paths.yaml"
        print(f"Using default config: {config_path}")
    
    # Determine dataset type from config path
    if "dunnhumby" in config_path.lower():
        dataset_type = "dunnhumby"
    else:
        dataset_type = "dunnhumby"  # default
        print(f"Warning: Could not determine dataset type from config path, using default: {dataset_type}")
    
    # Run cleaning pipeline
    success = clean_with_dataset(config_path, dataset_type)
    
    if success:
        print("\n" + "="*50)
        print("✓ SUCCESS: Data cleaning completed!")
        print("\nTo switch to a different dataset:")
        print("1. Implement BaseDatasetInterface for your dataset")
        print("2. Create a config file for your dataset")
        print("3. Add dataset creation logic to this script")
        print("4. Run: python clean_modular.py your_config.yaml")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("✗ FAILED: Data cleaning encountered errors!")
        print("Check the logs above for details.")
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    main()
