#!/usr/bin/env python3
"""
Unit Tests for Module 4: Metrics Validation
==========================================

Tests:
- Forecast metrics validation
- Inventory metrics validation  
- Pricing metrics validation
- SHAP validation
- Integrated insights generation

Author: SmartGrocy Team
Date: 2025-11-18
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.metrics_validator import MetricsValidator, ValidationError, ValidationResult


class TestForecastValidation:
    """Test forecast metrics validation."""
    
    def test_valid_complete_forecast(self):
        """Test with all required fields."""
        data = {'q50': 150, 'q05': 100, 'q95': 200}
        result = MetricsValidator.validate_forecast_metrics(data)
        
        assert result.is_valid
        assert result.validated_data['q50'] == 150
        assert result.validated_data['uncertainty'] == 50
        assert result.validated_data['uncertainty_pct'] > 0
        assert result.validated_data['confidence_level'] in ['HIGH', 'MODERATE', 'LOW']
    
    def test_missing_q50(self):
        """Test with missing required field."""
        data = {'q05': 100, 'q95': 200}
        result = MetricsValidator.validate_forecast_metrics(data)
        
        assert not result.is_valid
        assert any('q50' in err for err in result.errors)
    
    def test_negative_values(self):
        """Test with invalid negative values."""
        data = {'q50': -150, 'q05': 100, 'q95': 200}
        result = MetricsValidator.validate_forecast_metrics(data)
        
        assert not result.is_valid
        assert any('negative' in err.lower() for err in result.errors)
    
    def test_missing_quantiles_estimated(self):
        """Test automatic estimation of missing quantiles."""
        data = {'q50': 150}
        result = MetricsValidator.validate_forecast_metrics(data)
        
        assert result.is_valid
        assert 'q05' in result.validated_data
        assert 'q95' in result.validated_data
        assert len(result.warnings) == 2  # Warnings for both estimates
        assert result.validated_data['q05'] == 150 * 0.7
        assert result.validated_data['q95'] == 150 * 1.3


class TestInventoryValidation:
    """Test inventory metrics validation."""
    
    def test_valid_inventory(self):
        """Test with valid inventory data."""
        data = {
            'current_inventory': 120,
            'safety_stock': 30,
            'reorder_point': 100
        }
        result = MetricsValidator.validate_inventory_metrics(data)
        
        assert result.is_valid
        assert 'should_reorder' in result.validated_data
        assert isinstance(result.validated_data['should_reorder'], bool)
    
    def test_missing_fields_estimated(self):
        """Test handling of missing optional fields."""
        data = {'current_inventory': 120}
        result = MetricsValidator.validate_inventory_metrics(data)
        
        assert result.is_valid
        assert 'safety_stock' in result.validated_data
        assert 'reorder_point' in result.validated_data
        assert len(result.warnings) >= 1
    
    def test_negative_inventory(self):
        """Test invalid negative inventory."""
        data = {'current_inventory': -50, 'safety_stock': 30, 'reorder_point': 100}
        result = MetricsValidator.validate_inventory_metrics(data)
        
        assert not result.is_valid


class TestPricingValidation:
    """Test pricing metrics validation."""
    
    def test_valid_pricing(self):
        """Test with valid pricing data."""
        data = {
            'current_price': 50000,
            'unit_cost': 30000,
            'recommended_price': 45000
        }
        result = MetricsValidator.validate_pricing_metrics(data)
        
        assert result.is_valid
        assert 'current_margin' in result.validated_data
        assert 'new_margin' in result.validated_data
        assert 'price_change_pct' in result.validated_data
    
    def test_margin_calculation(self):
        """Test margin calculations."""
        data = {'current_price': 100, 'unit_cost': 60, 'recommended_price': 90}
        result = MetricsValidator.validate_pricing_metrics(data)
        
        assert result.is_valid
        assert result.validated_data['current_margin'] == 0.4  # 40%
        assert abs(result.validated_data['new_margin'] - 0.333) < 0.01  # ~33%
    
    def test_negative_margin_warning(self):
        """Test warning for negative margins."""
        data = {'current_price': 50, 'unit_cost': 60}  # Selling at loss
        result = MetricsValidator.validate_pricing_metrics(data)
        
        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) > 0
        assert any('loss' in w.lower() for w in result.warnings)


class TestSHAPValidation:
    """Test SHAP values validation."""
    
    def test_valid_shap(self):
        """Test with valid SHAP values."""
        data = {'promo_active': 0.35, 'price': -0.15, 'day_of_week': 0.10}
        result = MetricsValidator.validate_shap_values(data)
        
        assert result.is_valid
        assert len(result.validated_data) == 3
        # Should be sorted by absolute value
        values = list(result.validated_data.values())
        abs_values = [abs(v) for v in values]
        assert abs_values == sorted(abs_values, reverse=True)
    
    def test_empty_shap(self):
        """Test with empty SHAP values."""
        result = MetricsValidator.validate_shap_values({})
        
        assert result.is_valid
        assert len(result.validated_data) == 0
        assert len(result.warnings) > 0


class TestComprehensiveValidation:
    """Test comprehensive validation of all metrics."""
    
    def test_all_valid(self):
        """Test with all valid metrics."""
        forecast = {'q50': 150, 'q05': 100, 'q95': 200}
        inventory = {'current_inventory': 120, 'safety_stock': 30, 'reorder_point': 100}
        pricing = {'current_price': 50000, 'unit_cost': 30000}
        shap = {'promo': 0.35, 'price': -0.15}
        
        f_res, i_res, p_res, s_res = MetricsValidator.validate_comprehensive(
            forecast, inventory, pricing, shap
        )
        
        assert f_res.is_valid
        assert i_res.is_valid
        assert p_res.is_valid
        assert s_res.is_valid
    
    def test_forecast_invalid_blocks_generation(self):
        """Test that invalid forecast data is caught."""
        forecast = {}  # Missing q50
        inventory = {'current_inventory': 120}
        pricing = {'current_price': 50000, 'unit_cost': 30000}
        
        f_res, _, _, _ = MetricsValidator.validate_comprehensive(
            forecast, inventory, pricing, None
        )
        
        assert not f_res.is_valid
        assert len(f_res.errors) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
