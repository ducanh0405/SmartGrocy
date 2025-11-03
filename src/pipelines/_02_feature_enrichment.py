import pandas as pd
import logging
from pathlib import Path
import sys
import os

# === XÁC ĐỊNH ĐƯỜNG DẪN GỐC CỦA DỰ ÁN ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ==========================================

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- IMPORT TỪ CÁC THƯ MỤC TRONG src/ ---
try:
    # 1. Import hàm loader cho DỮ LIỆU THẬT (từ data/2_raw/)
    from src.pipelines._01_load_data import load_competition_data
    # 2. Import "THƯ VIỆN CODE" (các hàm đã refactor từ 4 PoC)
    from src.features import ws1_ecommerce_features as ws1
    from src.features import ws2_timeseries_features as ws2
    from src.features import ws3_behavior_features as ws3
    from src.features import ws4_price_features as ws4
    
    # 3. Import hàm tiện ích validation
    from src.utils.validation import comprehensive_validation

except ImportError as e:
    logging.error(f"LỖI IMPORT: {e}")
    logging.error("Hãy chắc chắn rằng bạn đã tạo các file __init__.py trong:")
    logging.error("src/, src/features/, src/pipelines/, src/utils/")
    sys.exit(1)
# ---------------------------------------------

def main():
    """
    Đây là KIẾN TRÚC SƯ PIPELINE.
    Nó tích hợp logic từ 4 Workstream (WS) để xây dựng Master Table cuối cùng.
    Nó được thiết kế để "bật/tắt" (toggle) các WS tùy theo dữ liệu có sẵn.
    """
    logging.info("========== BẮT ĐẦU PIPELINE LÀM GIÀU DỮ LIỆU (4-WS) ==========")
    
    # 1. Định nghĩa đường dẫn
    OUTPUT_PROCESSED_DIR = PROJECT_ROOT / 'data' / '3_processed'
    OUTPUT_FILE = OUTPUT_PROCESSED_DIR / 'master_feature_table.parquet'
    OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Tải Dữ liệu Thật (từ data/2_raw/)
    logging.info("--- (1/6) Tải Dữ liệu Thật (Competition Data) ---")
    dataframes = load_competition_data() # Gọi hàm từ 01_load_data.py
    
    if not dataframes:
        logging.critical("Không có dữ liệu đầu vào trong data/2_raw/. Dừng pipeline.")
        sys.exit(1)

    # 3. Khởi tạo Master Table
    # Giả sử file sales chính của cuộc thi tên là 'sales_train'
    if 'sales_train' not in dataframes:
        logging.critical("Lỗi: Không tìm thấy 'sales_train' (file sales chính) trong data/2_raw/.")
        sys.exit(1)
        
    master_df = dataframes['sales_train'].copy()
    logging.info(f"Đã khởi tạo Master Table từ 'sales_train'. Shape: {master_df.shape}")

    # 4. Tích hợp (Enrichment) theo Mô-đun (Giải quyết Rủi ro 2)
    # -----------------------------------------------------------------
    # Workstream 1: E-commerce (Reviews, Shipping, Payments...)
    # -----------------------------------------------------------------
    logging.info("--- (2/6) Tích hợp Workstream 1: E-commerce ---")
    try:
        # Giả sử hàm này thêm các cột như freight_ratio, payment_type từ dataframes
        # (Bạn cần tự viết hàm 'enrich_ecommerce_features' này trong ws1_ecommerce_features.py)
        master_df = ws1.enrich_ecommerce_features(master_df, dataframes)
        logging.info(f"-> Shape sau WS1: {master_df.shape}")
    except KeyError as e:
        logging.warning(f"⚠️ Bỏ qua WS1: Không tìm thấy dữ liệu cần thiết (ví dụ: 'reviews', 'payments'). Lỗi: {e}")
    except Exception as e:
        logging.warning(f"🚨 LỖI khi chạy WS1: {e}. Bỏ qua...")

    # -----------------------------------------------------------------
    # Workstream 2: Time-Series & Lịch (Lags, Rolling, Events)
    # -----------------------------------------------------------------
    logging.info("--- (3/6) Tích hợp Workstream 2: Time-Series ---")
    try:
        if 'calendar' in dataframes:
            # (Bạn cần tự viết các hàm này trong ws2_timeseries_features.py)
            master_df = ws2.add_lag_rolling_features(master_df)
            master_df = ws2.add_calendar_event_features(master_df, dataframes['calendar'])
            logging.info(f"-> Shape sau WS2: {master_df.shape}")
        else:
            logging.warning("⚠️ Bỏ qua WS2: Không tìm thấy file 'calendar'.")
    except Exception as e:
        logging.warning(f"🚨 LỖI khi chạy WS2: {e}. Bỏ qua...")

    # -----------------------------------------------------------------
    # Workstream 3: Hành vi (Clickstream)
    # -----------------------------------------------------------------
    logging.info("--- (4/6) Tích hợp Workstream 3: Behavior ---")
    try:
        # GỌI HÀM WS3 MÀ BẠN VỪA VIẾT
        # dataframes là dict chứa tất cả dữ liệu thô (bao gồm 'clickstream_log')
        master_df = ws3.add_behavioral_features(master_df, dataframes)
        logging.info(f"-> Shape sau WS3: {master_df.shape}")
        
    except KeyError as e:
        # Xử lý Rủi ro 2: Nếu không có file 'clickstream_log'
        logging.warning(f"⚠️ Bỏ qua WS3: Không tìm thấy dữ liệu cần thiết (ví dụ: 'clickstream_log'). Lỗi: {e}")
    except Exception as e:
        logging.warning(f"🚨 LỖI khi chạy WS3: {e}. Bỏ qua...")

    # -----------------------------------------------------------------
    # Workstream 4: Giá & Khuyến mãi (Price & Promotion)
    # -----------------------------------------------------------------
    logging.info("--- (5/6) Tích hợp Workstream 4: Price/Promotion ---")
    try:
        # (Bạn cần tự viết hàm này trong ws4_price_features.py)
        master_df = ws4.add_price_promotion_features(master_df, dataframes)
        logging.info(f"-> Shape sau WS4: {master_df.shape}")
    except Exception as e:
        logging.warning(f"🚨 LỖI khi chạy WS4: {e}. Bỏ qua...")

    # 5. Validation và Lưu trữ cuối cùng
    logging.info("--- (6/6) Kiểm tra (Validation) và Lưu Master Table ---")
    try:
        validation_report = comprehensive_validation(master_df, verbose=True)
        
        if validation_report['passed']:
            logging.info("✅ Pipeline Dữ liệu PASS. Đang lưu file...")
            master_df.to_parquet(OUTPUT_FILE, index=False)
            logging.info(f"✓ Đã lưu Master Table vào: {OUTPUT_FILE}")
            logging.info(f"Shape cuối cùng: {master_df.shape}")
        else:
            logging.warning("🚨 Pipeline Dữ liệu FAILED VALIDATION. Sẽ không lưu file.")
            
    except Exception as e:
        logging.error(f"🚨 Pipeline Dữ liệu thất bại ở bước Validation/Lưu trữ: {e}", exc_info=True)
        sys.exit(1)

    logging.info("========== HOÀN THÀNH PIPELINE LÀM GIÀU DỮ LIỆU ==========")

if __name__ == "__main__":
    main()