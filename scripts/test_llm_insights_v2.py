#!/usr/bin/env python3
"""
Test script for LLM Insights Module V2 with Gemini API

Tests both rule-based and LLM-powered insight generation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.llm_insights import LLMInsightGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_rule_based():
    """Test rule-based insight generation (no API required)."""
    print("\n" + "="*70)
    print("TEST 1: Rule-Based Insight Generation")
    print("="*70)
    
    generator = LLMInsightGenerator(use_llm_api=False)
    
    test_forecast = {
        'q50': 150,
        'q05': 100,
        'q95': 200,
        'vs_yesterday': 15.5,
        'vs_last_week': 8.2,
        'vs_monthly_avg': 5.0,
        'current_inventory': 120,
        'safety_stock': 30,
        'reorder_point': 100,
        'stockout_risk_pct': 45,
        'overstock_risk_pct': 20,
        'category': 'Fresh Produce',
        'date': '2025-11-16',
        'horizon': '24 hours'
    }
    
    test_shap = {
        'promo_active': 0.35,
        'price_change': -0.15,
        'day_of_week': 0.10,
        'sales_quantity_lag_24': 0.25
    }
    
    insight = generator.generate_forecast_insight(
        'P001',
        test_forecast,
        test_shap
    )
    
    print(f"\nMethod: {insight.get('method', 'unknown')}")
    print(f"Confidence: {insight.get('confidence', 0):.2f}")
    print("\n" + "-"*70)
    insight_text = insight.get('insight_text', insight.get('insight', ''))
    try:
        print(insight_text)
    except UnicodeEncodeError:
        # Fallback: remove emojis for Windows console
        import re
        text_no_emoji = re.sub(r'[^\x00-\x7F]+', '', insight_text)
        print(text_no_emoji)
    print("\n[SUCCESS] Rule-based test passed!")


def test_gemini_api():
    """Test Gemini API insight generation (requires API key)."""
    print("\n" + "="*70)
    print("TEST 2: Gemini API Insight Generation")
    print("="*70)
    
    import os
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("[WARNING] GEMINI_API_KEY not found in environment.")
        print("   Set it with: $env:GEMINI_API_KEY='your-key'")
        print("   Skipping Gemini API test...")
        return
    
    generator = LLMInsightGenerator(
        api_key=api_key,
        model="gemini-2.5-flash",
        use_llm_api=True,
        api_provider="gemini"
    )
    
    test_forecast = {
        'q50': 150,
        'q05': 100,
        'q95': 200,
        'vs_yesterday': 15.5,
        'vs_last_week': 8.2,
        'vs_monthly_avg': 5.0,
        'current_inventory': 120,
        'safety_stock': 30,
        'reorder_point': 100,
        'stockout_risk_pct': 45,
        'overstock_risk_pct': 20,
        'category': 'Fresh Produce',
        'date': '2025-11-16',
        'horizon': '24 hours'
    }
    
    test_shap = {
        'promo_active': 0.35,
        'price_change': -0.15,
        'day_of_week': 0.10,
        'sales_quantity_lag_24': 0.25
    }
    
    try:
        print("Calling Gemini API...")
        insight = generator.generate_forecast_insight(
            'P001',
            test_forecast,
            test_shap,
            use_llm=True
        )
        
        print(f"\nMethod: {insight.get('method', 'unknown')}")
        print(f"Provider: {insight.get('provider', 'unknown')}")
        print(f"Model: {insight.get('model', 'unknown')}")
        print(f"Confidence: {insight.get('confidence', 0):.2f}")
        print("\n" + "-"*70)
        insight_text = insight.get('insight_text', insight.get('insight', ''))
        try:
            print(insight_text)
        except UnicodeEncodeError:
            # Fallback: remove emojis for Windows console
            import re
            text_no_emoji = re.sub(r'[^\x00-\x7F]+', '', insight_text)
            print(text_no_emoji)
        print("\n[SUCCESS] Gemini API test passed!")
        
    except Exception as e:
        print(f"\n[ERROR] Gemini API test failed: {e}")
        print("   Falling back to rule-based mode...")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LLM INSIGHTS MODULE V2 - TEST SUITE")
    print("="*70)
    
    # Test 1: Rule-based (always works)
    test_rule_based()
    
    # Test 2: Gemini API (requires API key)
    test_gemini_api()
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)

