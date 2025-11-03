import pandas as pd
import logging
from pathlib import Path
import sys
import os

# === XÁC ĐỊNH ĐƯỜNG DẪN GỐC ===
# (file -> pipelines -> src -> E-Grocery_Forecaster)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Thêm thư mục /src vào Python path để các script khác có thể import file này
sys.path.append(str(PROJECT_ROOT / 'src'))
# ===============================

# Định nghĩa đường dẫn tới dữ liệu "THẬT" (dữ liệu cuộc thi)
RAW_DATA_DIR = PROJECT_ROOT / 'data' / '2_raw'

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_competition_data(data_dir=RAW_DATA_DIR):
    """
    Tải TẤT CẢ dữ liệu thô (của cuộc thi) từ thư mục data/2_raw.
    Nó sẽ tự động đọc các file .csv hoặc .parquet.
    
    Trả về:
        Một dictionary of DataFrames (ví dụ: {'sales': df_sales, 'calendar': df_calendar})
    """
    logging.info(f"========== [BƯỚC 1: LOAD DATA] ==========")
    logging.info(f"Bắt đầu tải dữ liệu thô từ: {data_dir}")
    
    dataframes = {}
    
    if not data_dir.exists():
        logging.error(f"🚨 LỖI: Thư mục dữ liệu thô không tồn tại: {data_dir}")
        logging.error("Vui lòng đặt dữ liệu cuộc thi (file .csv, .parquet) vào data/2_raw/")
        sys.exit(1)
        
    # Tìm tất cả các file csv hoặc parquet trong thư mục
    files = [f for f in data_dir.iterdir() if f.is_file() and (f.suffix in ['.csv', '.parquet'])]
    
    if not files:
        logging.warning(f"⚠️ CẢNH BÁO: Không tìm thấy file .csv hoặc .parquet nào trong {data_dir}")
        logging.warning("File .gitkeep là file giữ chỗ, không phải dữ liệu.")
        return {} # Trả về dict rỗng

    for file_path in files:
        try:
            # Lấy tên file (không có đuôi) làm "key" cho dictionary
            # Ví dụ: 'sales_data.csv' -> 'sales_data'
            key = file_path.stem
            
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
                
            dataframes[key] = df
            logging.info(f"✓ Đã tải thành công file: {file_path.name} (Shape: {df.shape}) -> lưu vào key: '{key}'")
            
        except Exception as e:
            logging.error(f"🚨 LỖI khi tải file {file_path.name}: {e}")
            
    logging.info(f"✓ Tải xong {len(dataframes)} file dữ liệu.")
    logging.info(f"Các khóa (keys) đã tạo: {list(dataframes.keys())}")
    logging.info(f"==========================================")
    return dataframes

if __name__ == "__main__":
    # Dùng để chạy test file này một cách độc lập
    logging.info("Chạy 01_load_data.py ở chế độ test (standalone)...")
    data = load_competition_data()
    
    if data:
        logging.info("Tải dữ liệu test thành công.")
    else:
        logging.warning("Không có dữ liệu trong data/2_raw/ để test.")