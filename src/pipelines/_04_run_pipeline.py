import subprocess # Dùng để gọi các script khác
import sys
import logging
from pathlib import Path

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Xác định Gốc
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINES_DIR = PROJECT_ROOT / 'src' / 'pipelines'

def run_script(script_name):
    """Hàm tiện ích để chạy một script pipeline và kiểm tra lỗi."""
    script_path = PIPELINES_DIR / script_name
    logging.info(f"\n--- 🚀 BẮT ĐẦU CHẠY: {script_name} ---")
    
    # Sử dụng sys.executable để đảm bảo chạy bằng chính interpreter (venv)
    # mà script này đang dùng
    process = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    if process.returncode != 0:
        logging.error(f"🚨 LỖI khi chạy {script_name}:")
        logging.error(process.stderr)
        return False
    else:
        logging.info(f"--- ✅ HOÀN THÀNH: {script_name} ---")
        logging.info("Output:\n" + process.stdout[-1000:]) # In 1000 dòng log cuối
        return True

def main():
    """
    Điều phối toàn bộ dự án E-Grocery Forecaster:
    1. Chạy pipeline xử lý dữ liệu (WS1 PoC)
    2. Chạy pipeline huấn luyện mô hình cuối cùng
    """
    logging.info("========== BẮT ĐẦU TOÀN BỘ WORKFLOW DỰ ÁN ==========")
    
    # Bước 1: Xử lý dữ liệu (Dựa trên WS1 PoC)
    # (File này sẽ tạo ra 'master_feature_table.parquet')
    if not run_script('02_feature_enrichment.py'):
        logging.critical("Pipeline xử lý dữ liệu thất bại. Dừng workflow.")
        sys.exit(1)
        
    # Bước 2: Huấn luyện mô hình
    # (File này sẽ đọc 'master_feature_table.parquet' và tạo ra 'final_forecaster.joblib')
    if not run_script('03_model_training.py'):
        logging.critical("Pipeline huấn luyện mô hình thất bại. Dừng workflow.")
        sys.exit(1)

    logging.info("\n========== 🥳 TOÀN BỘ WORKFLOW ĐÃ HOÀN THÀNH THÀNH CÔNG! ==========")

if __name__ == "__main__":
    main()