import pandas as pd
import logging
import json

def comprehensive_validation(df, verbose=True):
    """Validation tổng hợp toàn diện (lấy từ Notebook 2)."""
    logging.info("[Validation] Đang kiểm tra (Validate) pipeline cuối cùng...")
    validation_results = {}
    issues_found = False  # Cờ để theo dõi lỗi

    # 3.1: Thông tin cơ bản
    if verbose: logging.info("\n--- 3.1 Thông tin cơ bản DataFrame ---")
    validation_results['shape'] = df.shape

    if 'quality_score' in validation_results:
        logging.info(f"🎯 Quality Score: {validation_results['quality_score']}/100")
        if validation_results['quality_score'] >= 90:
            logging.info("✅ EXCELLENT")
        elif validation_results['quality_score'] >= 75:
            logging.info("✓ GOOD")
        else:
            logging.warning("⚠ FAIR/POOR")

    validation_results['passed'] = not issues_found
    return validation_results