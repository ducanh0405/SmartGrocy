# E-Grocery Forecaster - Project Content and Structure Documentation

## Project Overview

This repository contains a comprehensive machine learning solution for e-grocery demand forecasting and business optimization. The system integrates multiple modules to provide end-to-end forecasting, inventory optimization, dynamic pricing, and business intelligence capabilities.

## Directory Structure

### Root Level
- **`main.py`** - Main entry point for the application
- **`run_end_to_end.py`** - Complete pipeline execution script
- **`run_business_modules.py`** - Business logic execution script
- **`run_all_tests.py`** - Test suite runner
- **`run_complete_validation.py`** - Validation and quality checks
- **`regenerate_reports.py`** - Report regeneration utility
- **`pyproject.toml`** - Python project configuration
- **`requirements.txt`** - Core dependencies
- **`requirements-dev.txt`** - Development dependencies
- **`README.md`** - Project documentation (marketing-focused)
- **`TECHNICAL_REPORT.md`** - Technical documentation
- **`COMPREHENSIVE_FINAL_REPORT.md`** - Comprehensive project report
- **`LICENSE`** - MIT License

### Core Directories

#### `src/` - Source Code
Main application codebase organized into logical modules:

- **`core/`** - Core business logic and utilities
  - `forecasting.py` - Forecasting engine
  - `inventory.py` - Inventory optimization logic
  - `pricing.py` - Dynamic pricing algorithms
  - `insights.py` - LLM insights generation

- **`modules/`** - Business modules (5 main modules)
  - `module1_forecasting.py` - Demand forecasting module
  - `module2_inventory.py` - Inventory optimization module
  - `module3_pricing.py` - Dynamic pricing module
  - `module4_insights.py` - LLM insights module
  - `module5_visualization.py` - Visualization and reporting

- **`pipelines/`** - ML Pipeline Components
  - `_01_load_data.py` - Data loading pipeline
  - `_02_feature_enrichment.py` - Feature engineering pipeline
  - `_03_model_training.py` - Model training pipeline
  - `_04_model_evaluation.py` - Model evaluation pipeline
  - `_05_prediction.py` - Prediction generation pipeline
  - Other pipeline utilities

- **`features/`** - Feature Engineering (66 features across 7 workstreams)
  - `ws0_base_features.py` - Base feature extraction
  - `ws1_lag_features.py` - Lag-based features
  - `ws2_rolling_features.py` - Rolling statistics features
  - `ws3_time_features.py` - Time-based features
  - `ws4_categorical_features.py` - Categorical encoding
  - `ws5_interaction_features.py` - Feature interactions
  - `ws6_domain_features.py` - Domain-specific features

- **`preprocessing/`** - Data Preprocessing
  - `data_processor.py` - Main data processing logic

- **`utils/`** - Utility Functions
  - `data_utils.py` - Data manipulation utilities
  - `model_utils.py` - Model-related utilities
  - `validation_utils.py` - Validation and quality checks
  - `logging_utils.py` - Logging configuration
  - `config_utils.py` - Configuration management
  - Other specialized utilities

- **`cli/`** - Command Line Interface
  - CLI utilities for different operations

#### `data/` - Data Management
- **`2_raw/`** - Raw input data
  - `freshretail_train.csv/parquet` - Training dataset
  - `freshretail_eval.csv/parquet` - Evaluation dataset
  - `freshretail_datasets_metadata.json` - Dataset metadata

- **`3_processed/`** - Processed data
  - `master_feature_table.csv/parquet` - Feature-engineered dataset

- **`poc_data/`** - Proof of concept data samples

#### `models/` - Trained Models
- `lightgbm_q*_forecaster.joblib` - Trained LightGBM models for different quantiles (5%, 25%, 50%, 75%, 95%)
- `model_features.json` - Model feature specifications

#### `reports/` - Generated Reports and Outputs
- **`backtesting/`** - Backtesting results
  - `estimated_results.csv` - Backtesting performance
  - `strategy_comparison.csv` - Strategy comparison metrics

- **`dashboard/`** - Dashboard-specific reports
- **`market_analysis/`** - Market analysis reports
- **`metrics/`** - Model performance metrics
- **`report_charts/`** - Generated visualization charts (8 PNG files)
- **`shap_values/`** - SHAP explainability outputs

- Core report files:
  - `predictions_test_set.parquet/csv` - Forecast predictions
  - `business_report_summary.csv` - Business impact summary
  - `inventory_recommendations.csv` - Inventory optimization results
  - `pricing_recommendations.csv` - Dynamic pricing results
  - `llm_insights.csv` - Generated business insights
  - `feature_selection_report.json` - Feature selection results
  - `validation_report.json` - Validation results
  - `summary_statistics.json` - Statistical summaries

#### `scripts/` - Utility Scripts
- **`analysis/`** - Analysis utilities
- `compute_shap_values.py` - SHAP value computation
- `generate_charts_simple.py` - Simple chart generation
- `generate_market_analysis.py` - Market analysis generation
- `generate_report_charts.py` - Main chart generation script
- `generate_summary_statistics.py` - Statistics computation
- `generate_technical_report.py` - Technical report generation
- `run_backtesting_analysis.py` - Backtesting execution
- `run_data_quality_check.py` - Data quality validation
- `run_feature_selection.py` - Feature selection
- `run_full_backtesting.py` - Full backtesting pipeline
- `setup_data_quality.py` - Data quality setup
- `setup_great_expectations.py` - Great Expectations setup
- `validate_report_metrics.py` - Report validation

#### `dashboard/` - Interactive Dashboard
- `streamlit_app.py` - Main Streamlit application
- `templates/` - Dashboard templates

#### `tests/` - Test Suite
- Unit tests for all major components
- Integration tests
- Validation tests
- Configuration tests

#### `docs/` - Documentation
- **`archive/`** - Archived documentation
- **`guides/`** - User guides and tutorials
- **`technical/`** - Technical documentation
- `README.md` - Documentation index
- Various markdown files for different aspects

#### `config/` - Configuration Files
- `pipeline_config.json` - Pipeline configuration

#### `logs/` - Logging and Monitoring
- **`alerts/`** - Alert logs
- `pipeline.log` - Main pipeline logs
- `pipeline_run.log` - Individual run logs

#### `cache/` - Caching
- `cache.db` - SQLite cache database

## Key Components Description

### 1. Forecasting Module (Module 1)
- **Purpose**: Generate demand forecasts using LightGBM quantile regression
- **Models**: 5 quantile models (5%, 25%, 50%, 75%, 95%)
- **Features**: 66 engineered features across 7 workstreams
- **Performance**: 85.68% R² score, 87.03% coverage

### 2. Inventory Optimization (Module 2)
- **Purpose**: Optimize inventory levels and reduce costs
- **Metrics**: 18 optimization metrics
- **Risk Assessment**: 4-level categorization (LOW/MEDIUM/HIGH/CRITICAL)
- **Impact**: 32.5% stockout reduction, 40% spoilage reduction

### 3. Dynamic Pricing (Module 3)
- **Purpose**: Optimize pricing for profit maximization
- **Metrics**: 14 optimization metrics
- **Analysis**: Price elasticity, competitive positioning
- **Impact**: 37.5% profit margin increase

### 4. LLM Insights (Module 4)
- **Purpose**: Generate actionable business intelligence
- **Generation**: Risk-based insights with priority levels
- **Output**: 392 insights with confidence scoring

### 5. Visualization (Module 5)
- **Purpose**: Interactive dashboards and reporting
- **Tools**: Streamlit dashboard, 8 professional charts
- **Features**: Real-time filtering, export capabilities

## Data Flow

```
Raw Data (CSV/Parquet)
    ↓
Data Loading & Preprocessing
    ↓
Feature Engineering (66 features)
    ↓
Model Training (LightGBM Quantiles)
    ↓
Prediction Generation
    ↓
Business Modules (Inventory/Pricing/Insights)
    ↓
Reports & Visualizations
```

## Technology Stack

- **ML Framework**: LightGBM 4.5.0
- **Data Processing**: Pandas 2.3.3, NumPy
- **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
- **Quality Assurance**: Pytest, Black, Pre-commit
- **Infrastructure**: Python 3.10/3.11

## Configuration

Key configuration parameters in `src/config.py`:
- Dataset selection
- Quantile levels
- Service level targets
- Business constraints

## File Organization Principles

1. **Separation of Concerns**: Each module handles specific business logic
2. **Pipeline Architecture**: Sequential processing with clear dependencies
3. **Configuration Management**: Centralized configuration with environment support
4. **Testing Coverage**: Comprehensive test suite for all components
5. **Documentation**: Extensive documentation for maintenance and onboarding

This structure ensures maintainability, scalability, and ease of deployment across different environments.
