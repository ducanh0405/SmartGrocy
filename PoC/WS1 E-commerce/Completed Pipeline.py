"""
WORKSTREAM 1 (OLIST) - PIPELINE TỔNG HỢP

Mục đích:
1.  Tải (Load) các tệp .csv của Olist.
2.  Hợp nhất (Merge) chúng một cách an toàn (xử lý bẫy 'payments').
3.  Tạo (Create) các đặc trưng nghiệp vụ (features), BAO GỒM geolocation & distance.
4.  Sửa lỗi (Fix) Rò rỉ Dữ liệu (Data Leakage) trong đặc trưng review trung bình.
5.  Làm sạch (Clean) & Điền Nulls (Impute) MỘT LẦN ở cuối.
6.  Kiểm tra (Validate) tính toàn vẹn của dữ liệu cuối cùng.
7.  Xuất (Save) ra một file CSV cuối cùng đã làm sạch và tối ưu.

Cách chạy (từ Terminal):
> pip install pandas numpy haversine pyarrow
> python completed_pipeline.py
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from haversine import haversine # Cần cài đặt: pip install haversine

# --- 1. CÁC HÀM TẢI DỮ LIỆU ---

def load_data(data_dir='data/'):
    """Tải tất cả các tệp CSV cần thiết."""
    print(f"[Bước 1/8] Đang tải dữ liệu từ thư mục: {data_dir}...")
    files_to_keys = {
        'olist_orders_dataset.csv': 'orders', 'olist_order_items_dataset.csv': 'items',
        'olist_products_dataset.csv': 'products', 'olist_customers_dataset.csv': 'customers',
        'olist_order_reviews_dataset.csv': 'reviews', 'olist_order_payments_dataset.csv': 'payments',
        'olist_sellers_dataset.csv': 'sellers', 'olist_geolocation_dataset.csv': 'geolocation'
    }
    dataframes = {}
    try:
        for file, key in files_to_keys.items():
            file_path = os.path.join(data_dir, file)
            dataframes[key] = pd.read_csv(file_path)
        print(f"-> Tải {len(dataframes)} tệp dữ liệu chính thành công.")
        print(f"-> Các khóa (keys) đã tạo: {list(dataframes.keys())}")
        return dataframes
    except FileNotFoundError as e:
        print(f"🚨 LỖI: Không tìm thấy file {e.filename}. Đảm bảo các tệp CSV nằm trong thư mục '{data_dir}'.")
        sys.exit(1)

def aggregate_payments(df_payments):
    """(QUAN TRỌNG) Xử lý "Bẫy Hợp nhất" 💣. Gộp bảng payments."""
    print("[Bước 2/8] Đang gộp (Aggregate) bảng 'payments'...")
    df_payments_agg = df_payments.groupby('order_id').agg(
        payment_installments_total=('payment_installments', 'sum'),
        payment_value_total=('payment_value', 'sum'),
        payment_type_primary=('payment_type', 'first')
    ).reset_index()
    print(f"-> Đã gộp 'payments' từ {len(df_payments)} hàng xuống {len(df_payments_agg)} hàng.")
    return df_payments_agg

# --- 2. HÀM HỢP NHẤT ---

def merge_tables(dataframes, df_payments_agg):
    """Thực thi pipeline hợp nhất (merge) các bảng."""
    print("[Bước 3/8] Đang hợp nhất (Merge) các bảng...")
    df_master = dataframes['orders'].copy()
    df_master = pd.merge(df_master, dataframes['customers'], on='customer_id', how='left')
    df_reviews_dedup = dataframes['reviews'].drop_duplicates(subset='order_id', keep='last')
    df_master = pd.merge(df_master, df_reviews_dedup, on='order_id', how='left')
    df_master = pd.merge(df_master, df_payments_agg, on='order_id', how='left')
    df_master = pd.merge(df_master, dataframes['items'], on='order_id', how='left')
    df_master = pd.merge(df_master, dataframes['products'], on='product_id', how='left')
    df_master = pd.merge(df_master, dataframes['sellers'], on='seller_id', how='left')
    print(f"-> Hợp nhất (Merge) bảng lõi thành công. Kích thước bảng tổng thể: {df_master.shape}")
    return df_master

# --- 3. CÁC HÀM TẠO ĐẶC TRƯNG (CHƯA CLEAN) ---

def create_core_features(df_merged):
    """Tạo các đặc trưng cơ bản (thời gian, vận hành, thanh toán...)."""
    print("[Bước 4/8] Đang tạo các đặc trưng cơ bản (Core Features)...")
    df_featured = df_merged.copy()

    # Chuyển đổi Thời gian
    time_cols = ['order_purchase_timestamp', 'order_approved_at',
                 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in time_cols:
        df_featured[col] = pd.to_datetime(df_featured[col], errors='coerce')

    # Đặc trưng Vận hành (sẽ còn null nếu chưa giao)
    df_featured['delivery_time_days'] = (df_featured['order_delivered_customer_date'] - df_featured['order_purchase_timestamp']).dt.total_seconds() / (24 * 60 * 60)
    df_featured['delivery_vs_estimated_days'] = (df_featured['order_estimated_delivery_date'] - df_featured['order_delivered_customer_date']).dt.total_seconds() / (24 * 60 * 60)

    # Đặc trưng Vận chuyển
    df_featured['freight_ratio'] = df_featured['freight_value'] / (df_featured['price'] + 1e-6)

    # Đặc trưng Thanh toán
    df_featured['is_payment_credit_card'] = (df_featured['payment_type_primary'] == 'credit_card').astype(float) # Dùng float để chứa NaN nếu payment_type_primary là NaN
    df_featured['is_payment_boleto'] = (df_featured['payment_type_primary'] == 'boleto').astype(float)
    df_featured['is_payment_voucher'] = (df_featured['payment_type_primary'] == 'voucher').astype(float)
    df_featured['is_payment_installments'] = (df_featured['payment_installments_total'] > 1).astype(float) # Dùng float để chứa NaN

    print("-> Tạo xong core features.")
    return df_featured

def add_geolocation_features(df_featured, df_geo):
    """(TỐI ƯU 2) Tích hợp geolocation và tính khoảng cách."""
    print("[Bước 5/8] Đang thêm đặc trưng Geolocation (Tối ưu 2)...")
    df_geo_enriched = df_featured.copy()

    # Aggregate Geolocation
    df_geo_agg = df_geo.groupby('geolocation_zip_code_prefix').agg(
        geo_lat=('geolocation_lat', 'mean'),
        geo_lng=('geolocation_lng', 'mean')
    ).reset_index()

    # Merge lần 1: Cho Customer
    df_geo_enriched = pd.merge(df_geo_enriched, df_geo_agg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df_geo_enriched.rename(columns={'geo_lat': 'customer_lat', 'geo_lng': 'customer_lng'}, inplace=True)
    df_geo_enriched.drop(columns=['geolocation_zip_code_prefix'], inplace=True, errors='ignore')

    # Merge lần 2: Cho Seller
    df_geo_enriched = pd.merge(df_geo_enriched, df_geo_agg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left', suffixes=('', '_seller_geo')) # Thêm suffix để tránh trùng lặp
    df_geo_enriched.rename(columns={'geo_lat': 'seller_lat', 'geo_lng': 'seller_lng'}, inplace=True)
    df_geo_enriched.drop(columns=['geolocation_zip_code_prefix_seller_geo'], inplace=True, errors='ignore') # Sửa tên cột drop

    # Tính Khoảng cách Haversine (km) - sẽ còn null nếu thiếu lat/lng
    locations_available = df_geo_enriched[['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng']].notnull().all(axis=1)
    distances = df_geo_enriched[locations_available].apply(
        lambda row: haversine((row['customer_lat'], row['customer_lng']), (row['seller_lat'], row['seller_lng'])),
        axis=1
    )
    df_geo_enriched['distance_seller_customer'] = np.nan
    df_geo_enriched.loc[locations_available, 'distance_seller_customer'] = distances

    print("-> Đã thêm đặc trưng geolocation và distance (có thể còn nulls).")
    return df_geo_enriched

def fix_review_leakage(df_geo_enriched):
    """(TỐI ƯU 3 - V3) Sửa lỗi rò rỉ dữ liệu review và xử lý nulls."""
    print("[Bước 6/8] Đang tạo đặc trưng review 'time-safe' (Tối ưu 3)...")
    df_reviews_fixed = df_geo_enriched.sort_values('order_purchase_timestamp').copy()

    # Pre-impute review_score gốc
    mean_global_review = df_reviews_fixed['review_score'].mean()
    review_score_imputed = df_reviews_fixed['review_score'].fillna(mean_global_review)
    # print(f"-> Đã pre-impute các Nulls trong review_score gốc bằng global mean ({mean_global_review:.2f}).") # Có thể bỏ print này

    # Tính expanding mean & shift TRÊN CỘT ĐÃ IMPUTE
    df_reviews_fixed['avg_review_score_product_ts'] = df_reviews_fixed.groupby('product_id')[review_score_imputed.name].expanding().mean().shift(1).reset_index(level=0, drop=True)
    df_reviews_fixed['avg_review_score_seller_ts'] = df_reviews_fixed.groupby('seller_id')[review_score_imputed.name].expanding().mean().shift(1).reset_index(level=0, drop=True)

    # Post-impute (nulls do shift(1))
    df_reviews_fixed['avg_review_score_product_ts'] = df_reviews_fixed['avg_review_score_product_ts'].fillna(mean_global_review)
    df_reviews_fixed['avg_review_score_seller_ts'] = df_reviews_fixed['avg_review_score_seller_ts'].fillna(mean_global_review)

    print("-> Đã tạo đặc trưng review 'time-safe'.")
    return df_reviews_fixed

# --- 4. HÀM LÀM SẠCH & ĐIỀN NULLS CUỐI CÙNG ---

def final_cleaning_and_imputation(df_featured_all):
    """Làm sạch và điền TẤT CẢ nulls còn lại."""
    print("[Bước 7/8] Đang thực hiện làm sạch cuối cùng và điền Nulls...")
    df_clean = df_featured_all.copy()

    # 1. Làm sạch Cardinality
    df_clean['product_category_name'] = df_clean['product_category_name'].fillna('unknown').str.lower().str.strip()

    # 2. Xử lý Outliers
    negative_delivery_mask = (df_clean['delivery_time_days'] < 0) & (df_clean['delivery_time_days'] != -999) # Vẫn giữ lại -999 nếu có
    df_clean.loc[negative_delivery_mask, 'delivery_time_days'] = 0
    df_clean['freight_ratio'] = df_clean['freight_ratio'].clip(upper=10)

    # 3. Điền Nulls còn lại (Imputation Chiến lược)
    # 3.1 Review Score gốc (nếu còn sót - không nên)
    df_clean['review_score'] = df_clean['review_score'].fillna(0)
    # 3.2 Delivery Times (đơn chưa giao)
    df_clean['delivery_time_days'] = df_clean['delivery_time_days'].fillna(-999)
    df_clean['delivery_vs_estimated_days'] = df_clean['delivery_vs_estimated_days'].fillna(-999)
    # 3.3 Payment features
    payment_flags = ['is_payment_credit_card', 'is_payment_boleto', 'is_payment_voucher', 'is_payment_installments']
    for col in payment_flags:
        if col in df_clean.columns: df_clean[col] = df_clean[col].fillna(0).astype(int) # Chuyển về int sau khi fillna
    df_clean['payment_installments_total'] = df_clean['payment_installments_total'].fillna(0)
    df_clean['payment_value_total'] = df_clean['payment_value_total'].fillna(0)
    # 3.4 Price/Freight
    df_clean['price'] = df_clean['price'].fillna(0)
    df_clean['freight_value'] = df_clean['freight_value'].fillna(0)
    df_clean['freight_ratio'] = df_clean['freight_ratio'].fillna(0) # Nếu price=0 và freight=0
    # 3.5 Geolocation features (Điền 0 nếu thiếu)
    geo_cols = ['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng', 'distance_seller_customer']
    for col in geo_cols:
        if col in df_clean.columns: df_clean[col] = df_clean[col].fillna(0)

    # 4. Làm sạch cuối cùng (loại bỏ hàng thiếu khóa chính)
    df_clean.dropna(subset=['order_id', 'order_item_id'], inplace=True) # order_item_id có thể vẫn null nếu merge sai

    print("-> Làm sạch cuối cùng và điền Nulls hoàn tất.")
    return df_clean

# --- 5. HÀM KIỂM TRA (VALIDATION FUNCTION) ---
# (Giữ nguyên hàm validate_pipeline từ phiên bản trước - nó đã đúng)
def validate_pipeline(df_final):
    """Thực thi 4 bài kiểm tra sức khỏe 🩺."""
    print("[Bước 8/8] Đang kiểm tra (Validate) pipeline cuối cùng...")
    is_valid = True
    # 1. Nulls
    final_nulls = df_final.isnull().sum().sum()
    if final_nulls > 0:
        print(f"-> 🚨 KIỂM TRA 1 THẤT BẠI: Vẫn còn {final_nulls} giá trị Null.")
        is_valid = False
    else:
        print("-> ✅ Kiểm tra 1 (Nulls): Đạt.")
    # 2. Distribution (Delivery Time)
    delivered_mask = df_final['delivery_time_days'] != -999
    if delivered_mask.any():
        min_real = df_final[delivered_mask]['delivery_time_days'].min()
        if min_real < 0:
            print(f"-> 🚨 KIỂM TRA 2 THẤT BẠI: 'delivery_time_days' < 0 ({min_real}).")
            is_valid = False
        else:
            print(f"-> ✅ Kiểm tra 2 (Distribution): Đạt. Min delivery time là {min_real:.2f} (>= 0).")
    else:
         print("-> 🟡 Kiểm tra 2 (Distribution): Bỏ qua (Không có đơn giao?).")
    # 3. Cardinality
    nunique_categories = df_final['product_category_name'].nunique()
    print(f"-> ℹ️ Kiểm tra 3 (Cardinality): Tìm thấy {nunique_categories} danh mục.")
    # 4. Integrity Check 🚨
    key_columns = ['order_id', 'order_item_id']
    if all(col in df_final.columns for col in key_columns):
        # Quan trọng: Phải xử lý null trong khóa trước khi kiểm tra duplicated
        df_check = df_final.copy()
        df_check[key_columns[0]] = df_check[key_columns[0]].fillna('MISSING_ORDER')
        df_check[key_columns[1]] = df_check[key_columns[1]].fillna('MISSING_ITEM') # Dùng fillna khác 0
        duplicate_rows = df_check.duplicated(subset=key_columns).sum()

        if duplicate_rows > 0:
            print(f"-> 🚨 KIỂM TRA 4 THẤT BẠI: Pipeline tạo ra {duplicate_rows} hàng trùng lặp.")
            is_valid = False
        else:
            print("-> ✅ Kiểm tra 4 (Integrity): Đạt.")
    else:
        print(f"-> 🚨 KIỂM TRA 4 THẤT BẠI: Thiếu cột khóa {key_columns}.")
        is_valid = False
    return is_valid

# --- 6. HÀM CHÍNH (MAIN FUNCTION) ---

def main():
    """Điều phối toàn bộ pipeline."""
    start_time = time.time()
    DATA_DIR = 'data/'
    OUTPUT_FILE_CSV = 'olist_master_table_optimized.csv'

    # --- Chạy Pipeline ---
    dataframes = load_data(DATA_DIR)
    df_payments_agg = aggregate_payments(dataframes['payments'])
    df_merged = merge_tables(dataframes, df_payments_agg)
    df_featured_core = create_core_features(df_merged)
    df_featured_geo = add_geolocation_features(df_featured_core, dataframes['geolocation'])
    df_featured_reviews = fix_review_leakage(df_featured_geo)
    df_final = final_cleaning_and_imputation(df_featured_reviews) # Bước làm sạch cuối cùng

    # --- Kiểm tra & Lưu ---
    is_pipeline_healthy = validate_pipeline(df_final)
    if is_pipeline_healthy:
        print(f"\n[Bước 9/9] Đang lưu trữ file {OUTPUT_FILE_CSV}...")
        try:
            final_columns = [ # Danh sách cột cuối cùng (đã cập nhật)
                'order_id', 'order_item_id', 'product_id', 'customer_id', 'seller_id',
                'order_purchase_timestamp',
                'delivery_time_days', 'delivery_vs_estimated_days',
                'price', 'freight_value', 'freight_ratio',
                'is_payment_credit_card', 'is_payment_boleto', 'is_payment_installments', 'payment_value_total',
                'review_score',
                'avg_review_score_product_ts', 'avg_review_score_seller_ts',
                'distance_seller_customer',
                'product_category_name', 'customer_state', 'seller_state',
                'customer_lat', 'customer_lng', 'seller_lat', 'seller_lng'
            ]
            final_columns_exist = [col for col in final_columns if col in df_final.columns]
            df_final_output = df_final[final_columns_exist]

            df_final_output.to_csv(OUTPUT_FILE_CSV, index=False)
            print(f"\n--- 🥳 HOÀN THÀNH WORKSTREAM 1 (OPTIMIZED V3) ---")
            print(f"Output đã được lưu tại: {OUTPUT_FILE_CSV}")
            print(f"Kích thước cuối cùng: {df_final_output.shape}")
        except Exception as e:
            print(f"\n🚨 LỖI khi lưu file CSV: {e}")
    else:
        print("\n🚨 LỖI: Pipeline không vượt qua kiểm tra. Sẽ không lưu file. Vui lòng kiểm tra lại.")

    end_time = time.time()
    print(f"\nTổng thời gian chạy pipeline: {end_time - start_time:.2f} giây.")

# --- ĐIỂM BẮT ĐẦU CHẠY SCRIPT ---
if __name__ == "__main__":
    # Cấu hình Pandas để xử lý lỗi CopyWarning tốt hơn (tùy chọn)
    pd.options.mode.chained_assignment = None # Tắt cảnh báo (chỉ dùng nếu bạn hiểu rõ code)
    main()