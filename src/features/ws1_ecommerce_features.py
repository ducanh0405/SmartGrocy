import pandas as pd
import numpy as np
import logging
from haversine import haversine

def aggregate_payments(df_payments):
    """(WS1 PoC) Gộp bảng payments."""
    logging.info("[WS1] Đang gộp (Aggregate) bảng 'payments'...")
    df_payments_agg = df_payments.groupby('order_id').agg(
        payment_installments_total=('payment_installments', 'sum'),
        payment_value_total=('payment_value', 'sum'),
        payment_type_primary=('payment_type', 'first'),
        payment_sequential_count=('payment_sequential', 'max')
    ).reset_index()
    return df_payments_agg

def aggregate_geolocation(df_geo):
    """(WS1 PoC) Aggregate geolocation để tối ưu merge."""
    logging.info("[WS1] Đang gộp (Aggregate) bảng 'geolocation'...")
    df_geo_agg = df_geo.groupby('geolocation_zip_code_prefix').agg(
        geo_lat=('geolocation_lat', 'mean'),
        geo_lng=('geolocation_lng', 'mean')
    ).reset_index()
    return df_geo_agg

def merge_tables(dataframes, df_payments_agg, df_geo_agg):
    """(WS1 PoC) Thực thi pipeline hợp nhất (merge) các bảng Olist."""
    logging.info("[WS1] Đang hợp nhất (Merge) các bảng Olist...")
    df_master = dataframes['orders'].copy()

    # Merge bảng chính
    df_master = pd.merge(df_master, dataframes['customers'], on='customer_id', how='left')
    df_reviews_dedup = dataframes['reviews'].sort_values('review_creation_date', ascending=False).drop_duplicates('order_id', keep='first')
    df_master = pd.merge(df_master, df_reviews_dedup, on='order_id', how='left')
    df_master = pd.merge(df_master, df_payments_agg, on='order_id', how='left')
    df_master = pd.merge(df_master, dataframes['order_items'], on='order_id', how='left')
    df_master = pd.merge(df_master, dataframes['products'], on='product_id', how='left')
    df_master = pd.merge(df_master, dataframes['sellers'], on='seller_id', how='left')
    # Merge geolocation features for customer
    df_master = pd.merge(
        df_master, 
        df_geo_agg, 
        left_on='customer_zip_code_prefix', 
        right_on='geolocation_zip_code_prefix', 
        how='left', 
        suffixes=('', '_customer')
    )
    # Rename customer lat/lng fields for clarity
    df_master = df_master.rename(columns={
        'geo_lat': 'customer_lat',
        'geo_lng': 'customer_lng'
    })
    # Merge geolocation features for seller
    df_master = pd.merge(
        df_master, 
        df_geo_agg, 
        left_on='seller_zip_code_prefix', 
        right_on='geolocation_zip_code_prefix', 
        how='left', 
        suffixes=('', '_seller')
    )
    df_master = df_master.rename(columns={
        'geo_lat': 'seller_lat',
        'geo_lng': 'seller_lng'
    })
    logging.info(f"-> Hợp nhất (Merge) WS1 thành công. Shape: {df_master.shape}")
    return df_master

def create_features(df_merged):
    """(WS1 PoC) Tạo tất cả các đặc trưng nghiệp vụ Olist."""
    logging.info("[WS1] Đang tạo đặc trưng Olist (Feature Engineering)...")
    df_featured = df_merged.copy()

    # 1. Chuyển đổi Thời gian
    time_cols = ['order_purchase_timestamp', 'order_approved_at', ...]
    # ... (Copy toàn bộ logic từ hàm create_features của bạn) ...
    
    # 2. Đặc trưng Vận hành
    df_featured['delivery_time_days'] = ...
    
    # 4. Đặc trưng Địa lý (Khoảng cách)
    locations_available = df_featured[['customer_lat', ...]].notnull().all(axis=1)
    # ...
    
    logging.info(f"-> Tạo đặc trưng WS1 hoàn tất. Shape: {df_featured.shape}")
    return df_featured

def clean_and_impute(df_featured):
    """(WS1 PoC) Làm sạch và điền Nulls cuối cùng cho Olist."""
    logging.info("[WS1] Đang thực hiện làm sạch cuối cùng (Clean & Impute)...")
    df_clean = df_featured.copy()

    # === 1. LÀM SẠCH (Cleaning) ===
    # ... (Copy toàn bộ logic từ hàm clean_and_impute của bạn) ...
    
    # === 2. ĐIỀN NULLS (Imputation) ===
    # 2.1 Cột Review Score (0 = Chưa review)
    df_clean['review_score'] = df_clean['review_score'].fillna(0)
    
    # 2.2 Cột Vận hành (Chưa giao = -999)
    delivery_cols_to_flag = ['delivery_time_days', ...]
    # ...
    
    # 3. Làm sạch cuối cùng (loại bỏ hàng thiếu khóa chính)
    df_clean.dropna(subset=['order_id', 'order_item_id'], inplace=True)

    logging.info("-> Làm sạch cuối cùng WS1 hoàn tất.")
    return df_clean

def load_olist_data(data_dir):
    """(WS1 PoC) Tải 9 tệp Olist."""
    logging.info(f"[WS1] Đang tải dữ liệu PoC Olist từ: {data_dir}...")
    files_to_keys = {
        'olist_orders_dataset.csv': 'orders', 
        # ... (Copy toàn bộ logic từ hàm load_data của bạn) ...
    }
    dataframes = {}
    try:
        for file, key in files_to_keys.items():
            file_path = os.path.join(data_dir, file)
            dataframes[key] = pd.read_csv(file_path)
        logging.info(f"-> Tải {len(dataframes)} tệp Olist thành công.")
        return dataframes
    except FileNotFoundError as e:
        logging.error(f"🚨 LỖI (WS1): Không tìm thấy file {e.filename}.")
        sys.exit(1)