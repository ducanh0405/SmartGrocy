#!/usr/bin/env python3
"""
SmartGrocy End-to-End Pipeline Runner
======================================
Script chạy toàn bộ pipeline từ đầu đến cuối:
1. ML Pipeline (Module 1): Data loading → Feature Engineering → Model Training → Prediction
2. Business Modules: Inventory Optimization → Dynamic Pricing → LLM Insights

Usage:
    # Chạy full pipeline với full data
    python run_end_to_end.py --full-data
    
    # Chạy với 10% sample (nhanh hơn, để test)
    python run_end_to_end.py --full-data --sample 0.1
    
    # Chạy với v2 orchestrator (có GX validation)
    python run_end_to_end.py --full-data --use-v2
    
    # Chỉ chạy ML pipeline (không chạy business modules)
    python run_end_to_end.py --full-data --ml-only
    
    # Chỉ chạy business modules (cần có forecasts từ trước)
    python run_end_to_end.py --business-only
    
    # Không dùng LLM API (rule-based only)
    python run_end_to_end.py --full-data --no-llm
"""

import sys
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import time

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import setup_project_path, setup_logging
    setup_project_path()
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def print_banner(text: str, char: str = "="):
    """Print formatted banner"""
    logger.info("\n" + char * 70)
    logger.info(f"  {text}")
    logger.info(char * 70 + "\n")


def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("🔍 Checking prerequisites...")
    
    all_good = True
    
    # Check data directories
    from src.config import DATA_DIRS, ensure_directories
    ensure_directories()
    logger.info("✓ Data directories: Ready")
    
    # Check for training data
    raw_data_dir = DATA_DIRS['raw_data']
    if not raw_data_dir.exists() or not list(raw_data_dir.glob('*.parquet')):
        logger.warning("⚠️ No training data found in data/2_raw/")
        logger.warning("   Pipeline may fail at data loading stage")
        all_good = False
    else:
        data_files = list(raw_data_dir.glob('*.parquet'))
        logger.info(f"✓ Training data: {len(data_files)} file(s) found")
    
    # Check for required Python packages
    required_packages = ['pandas', 'numpy', 'lightgbm', 'sklearn']
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        logger.warning("   Install with: pip install -r requirements.txt")
        all_good = False
    else:
        logger.info("✓ Required packages: Installed")
    
    return all_good


def run_ml_pipeline(args):
    """Run ML Pipeline (Module 1: Demand Forecasting)"""
    print_banner("🚀 MODULE 1: ML PIPELINE (Demand Forecasting)")
    
    start_time = time.time()
    
    try:
        # Build command
        cmd = [sys.executable, 'run_modern_pipeline_v2.py']
        
        if args.full_data:
            cmd.append('--full-data')
        
        if args.sample < 1.0:
            cmd.extend(['--sample', str(args.sample)])
        
        if args.use_v2:
            cmd.append('--use-v2')
        
        if args.no_cache:
            cmd.append('--no-cache')
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info("=" * 70)
        
        # Run pipeline
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False  # Show output in real-time
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info("\n" + "=" * 70)
            logger.info("✅ ML PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info("=" * 70)
            
            # Check if predictions file exists
            predictions_path = Path('reports/predictions_test_set.csv')
            if predictions_path.exists():
                import pandas as pd
                df = pd.read_csv(predictions_path)
                logger.info(f"✓ Predictions generated: {len(df):,} records")
                logger.info(f"  File: {predictions_path}")
            else:
                logger.warning(f"⚠️ Predictions file not found: {predictions_path}")
            
            return True
        else:
            logger.error("❌ ML Pipeline failed")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ ML Pipeline failed with return code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"❌ Error running ML Pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_business_modules(args):
    """Run Business Modules (Modules 2-4)"""
    print_banner("💼 BUSINESS MODULES (Inventory, Pricing, LLM Insights)")
    
    start_time = time.time()
    
    # Check if forecasts file exists
    forecasts_path = Path(args.forecasts)
    if not forecasts_path.exists():
        logger.error(f"❌ Forecasts file not found: {forecasts_path}")
        logger.info("Please run ML pipeline first or specify correct path with --forecasts")
        return False
    
    try:
        # Build command
        cmd = [sys.executable, 'run_business_modules.py']
        cmd.extend(['--forecasts', str(forecasts_path)])
        
        if args.no_llm:
            cmd.append('--no-llm')
        
        if args.top_n:
            cmd.extend(['--top-n', str(args.top_n)])
        
        if args.output_dir:
            cmd.extend(['--output-dir', args.output_dir])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info("=" * 70)
        
        # Run business modules
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info("\n" + "=" * 70)
            logger.info("✅ BUSINESS MODULES COMPLETED SUCCESSFULLY")
            logger.info(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info("=" * 70)
            
            # Check output files
            output_dir = Path(args.output_dir or 'reports')
            output_files = {
                'Inventory': output_dir / 'inventory_recommendations.csv',
                'Pricing': output_dir / 'pricing_recommendations.csv',
                'LLM Insights': output_dir / 'llm_insights.csv'
            }
            
            logger.info("\n📊 Generated Files:")
            for module_name, file_path in output_files.items():
                if file_path.exists():
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    logger.info(f"  ✓ {module_name}: {len(df):,} records → {file_path}")
                else:
                    logger.warning(f"  ⚠ {module_name}: File not found → {file_path}")
            
            return True
        else:
            logger.error("❌ Business Modules failed")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Business Modules failed with return code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"❌ Error running Business Modules: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_summary_report(args, ml_success: bool, business_success: bool, total_duration: float):
    """Generate summary report"""
    print_banner("📊 END-TO-END PIPELINE SUMMARY", "=")
    
    logger.info(f"Start Time: {args.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    logger.info("")
    
    logger.info("Pipeline Stages:")
    logger.info(f"  {'✅' if ml_success else '❌'} Module 1: ML Pipeline (Demand Forecasting)")
    if not args.ml_only:
        logger.info(f"  {'✅' if business_success else '❌'} Module 2: Inventory Optimization")
        logger.info(f"  {'✅' if business_success else '❌'} Module 3: Dynamic Pricing")
        logger.info(f"  {'✅' if business_success else '❌'} Module 4: LLM Insights")
    
    logger.info("")
    
    # Output files summary
    output_dir = Path(args.output_dir or 'reports')
    logger.info("Output Files:")
    
    # ML Pipeline outputs
    ml_outputs = [
        ('Predictions', 'reports/predictions_test_set.csv'),
        ('Model Metrics', 'reports/metrics/model_metrics.json'),
        ('Feature Importance', 'reports/chart4_feature_importance.png'),
    ]
    
    for name, path in ml_outputs:
        if Path(path).exists():
            logger.info(f"  ✓ {name}: {path}")
    
    # Business module outputs
    if not args.ml_only:
        business_outputs = [
            ('Inventory Recommendations', output_dir / 'inventory_recommendations.csv'),
            ('Pricing Recommendations', output_dir / 'pricing_recommendations.csv'),
            ('LLM Insights', output_dir / 'llm_insights.csv'),
        ]
        
        for name, path in business_outputs:
            if path.exists():
                logger.info(f"  ✓ {name}: {path}")
    
    logger.info("")
    
    # Overall status
    if ml_success and (args.ml_only or business_success):
        logger.info("🎉 END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("")
        logger.info("Next Steps:")
        logger.info("  1. Review predictions: reports/predictions_test_set.csv")
        logger.info("  2. Check inventory recommendations: reports/inventory_recommendations.csv")
        logger.info("  3. Review pricing recommendations: reports/pricing_recommendations.csv")
        logger.info("  4. View LLM insights: reports/llm_insights.csv")
        logger.info("  5. Open dashboard: reports/dashboard/forecast_dashboard.html")
    else:
        logger.warning("⚠️ Pipeline completed with errors. Please check logs above.")
    
    logger.info("=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SmartGrocy End-to-End Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full end-to-end pipeline với full data
  python run_end_to_end.py --full-data
  
  # Quick test với 10% sample
  python run_end_to_end.py --full-data --sample 0.1
  
  # Chạy với v2 orchestrator (có GX validation)
  python run_end_to_end.py --full-data --use-v2
  
  # Chỉ chạy ML pipeline
  python run_end_to_end.py --full-data --ml-only
  
  # Chỉ chạy business modules (cần có forecasts từ trước)
  python run_end_to_end.py --business-only --forecasts reports/predictions_test_set.csv
  
  # Không dùng LLM API
  python run_end_to_end.py --full-data --no-llm
        """
    )
    
    # ML Pipeline options
    parser.add_argument(
        '--full-data', action='store_true',
        help='Use full dataset (default: sample mode)'
    )
    
    parser.add_argument(
        '--sample', type=float, default=1.0,
        help='Sample fraction (0.1 = 10%% of data). Default: 1.0 (100%%)'
    )
    
    parser.add_argument(
        '--use-v2', action='store_true',
        help='Use v2 orchestrator with GX integration (recommended)'
    )
    
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable caching for fresh run'
    )
    
    # Business modules options
    parser.add_argument(
        '--forecasts', type=str,
        default='reports/predictions_test_set.csv',
        help='Path to forecasts file (for business modules)'
    )
    
    parser.add_argument(
        '--no-llm', action='store_true',
        help='Do not use LLM API (rule-based only)'
    )
    
    parser.add_argument(
        '--top-n', type=int, default=10,
        help='Number of products for LLM insights (default: 10)'
    )
    
    parser.add_argument(
        '--output-dir', type=str, default='reports',
        help='Output directory for results (default: reports)'
    )
    
    # Execution mode
    parser.add_argument(
        '--ml-only', action='store_true',
        help='Only run ML pipeline (skip business modules)'
    )
    
    parser.add_argument(
        '--business-only', action='store_true',
        help='Only run business modules (skip ML pipeline)'
    )
    
    args = parser.parse_args()
    
    # Store start time
    args.start_time = datetime.now()
    
    # Print header
    print_banner("🚀 SMARTGROCY END-TO-END PIPELINE", "=")
    logger.info(f"Start Time: {args.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration:")
    logger.info(f"  Full Data: {args.full_data}")
    logger.info(f"  Sample Fraction: {args.sample}")
    logger.info(f"  Use V2 Orchestrator: {args.use_v2}")
    logger.info(f"  ML Only: {args.ml_only}")
    logger.info(f"  Business Only: {args.business_only}")
    logger.info(f"  Use LLM: {not args.no_llm}")
    logger.info("")
    
    # Check prerequisites (unless business-only mode)
    if not args.business_only:
        prereq_ok = check_prerequisites()
        if not prereq_ok:
            logger.warning("\n⚠️ Some prerequisites not met. Pipeline may fail.")
            user_input = input("Continue anyway? (y/N): ")
            if user_input.lower() != 'y':
                logger.info("Pipeline cancelled by user.")
                sys.exit(0)
        logger.info("")
    
    # Track success
    ml_success = False
    business_success = False
    total_start_time = time.time()
    
    # Run ML Pipeline
    if not args.business_only:
        ml_success = run_ml_pipeline(args)
        
        if not ml_success:
            logger.error("\n❌ ML Pipeline failed. Stopping execution.")
            if not args.ml_only:
                logger.info("Business modules require successful ML pipeline.")
            sys.exit(1)
    
    # Run Business Modules
    if not args.ml_only:
        if args.business_only or ml_success:
            business_success = run_business_modules(args)
            
            if not business_success:
                logger.warning("\n⚠️ Business Modules failed or incomplete.")
                logger.warning("ML Pipeline completed successfully, but business modules had issues.")
        else:
            logger.warning("⚠️ Skipping business modules due to ML pipeline failure")
    
    # Generate summary
    total_duration = time.time() - total_start_time
    generate_summary_report(args, ml_success, business_success, total_duration)
    
    # Exit with appropriate code
    if args.ml_only:
        sys.exit(0 if ml_success else 1)
    elif args.business_only:
        sys.exit(0 if business_success else 1)
    else:
        sys.exit(0 if (ml_success and business_success) else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

