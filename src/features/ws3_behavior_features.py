import pandas as pd
import numpy as np
import logging



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _process_clickstream_logs(df_events):
    """
    Hàm nội bộ: Thực hiện logic từ 'EDA_Data_Preprocess.ipynb'.
    Làm sạch, xử lý thời gian, và chuẩn bị log.
    """
    logging.info("[WS3] Bắt đầu xử lý log hành vi (clickstream)...")
    
    # Giả sử df_events có các cột: 'timestamp', 'visitorid', 'event', 'itemid'
    if 'timestamp' in df_events.columns:
        df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='ms') # Giả sử timestamp là ms

    # Xử lý NaNs (nếu có)
    df_events = df_events.dropna(subset=['visitorid', 'itemid'])
    
    logging.info(f"[WS3] Xử lý log hoàn tất. Tổng số sự kiện: {len(df_events)}")
    return df_events

def _create_user_features(df_logs):
    """
    Hàm nội bộ: Thực hiện logic từ 'Feature_Engineering.ipynb'.
    Tạo bảng đặc trưng (feature table) ở cấp độ người dùng (user-level).
    """
    logging.info("[WS3] Bắt đầu tạo đặc trưng hành vi (Feature Engineering)...")
    
    # 1. Tạo các đặc trưng cơ bản (ví dụ từ PoC của bạn)
    # Đây là logic tính toán "phễu chuyển đổi"
    user_features = df_logs.pivot_table(
        index='visitorid', 
        columns='event', 
        aggfunc='size', 
        fill_value=0
    )
    
    # Đổi tên cột nếu cần (ví dụ: 'addtocart' -> 'total_addtocart')
    user_features = user_features.rename(columns={
        'view': 'total_views',
        'addtocart': 'total_addtocart',
        'transaction': 'total_transactions'
    })
    
    # 2. Tạo đặc trưng về tỷ lệ (Conversion Rates)
    # Tỷ lệ xem -> thêm vào giỏ
    user_features['rate_view_to_cart'] = user_features['total_addtocart'] / (user_features['total_views'] + 1e-6)
    
    # Tỷ lệ thêm vào giỏ -> mua
    user_features['rate_cart_to_buy'] = user_features['total_transactions'] / (user_features['total_addtocart'] + 1e-6)
    
    # Tỷ lệ xem -> mua (tỷ lệ chuyển đổi tổng thể)
    user_features['rate_view_to_buy'] = user_features['total_transactions'] / (user_features['total_views'] + 1e-6)

    # 3. Tạo các đặc trưng về thời gian (session-based)
    # (Đây là ví dụ, bạn sẽ thay thế bằng logic phức tạp hơn từ notebook của bạn)
    if 'timestamp' in df_logs.columns:
        time_stats = df_logs.groupby('visitorid')['timestamp'].agg(['min', 'max'])
        time_stats['session_duration_days'] = (time_stats['max'] - time_stats['min']).dt.total_seconds() / (60 * 60 * 24)
        user_features = user_features.join(time_stats['session_duration_days'], how='left')