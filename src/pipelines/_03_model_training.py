import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
import time
import sys
import joblib
import json
from pathlib import Path 

# === XÁC ĐỊNH ĐƯỜNG DẪN GỐC ===
# (file -> pipelines -> src -> E-Grocery_Forecaster)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# ===============================

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. CẤU HÌNH DỰ ÁN (ĐÃ TÍCH HỢP)
# -----------------------------------------------------------------
CONFIG = {
    # 1. File đầu vào (từ pipeline xử lý dữ liệu)
    # Đọc file Parquet (đầu ra của _02_feature_enrichment.py)
    "data_file": PROJECT_ROOT / 'data' / '3_processed' / 'master_feature_table.parquet',

    # 2. Files đầu ra (lưu vào thư mục 'models' và 'reports')
    "model_output_path": PROJECT_ROOT / 'models' / 'final_forecaster.joblib',
    "features_output_path": PROJECT_ROOT / 'models' / 'model_features.json',
    "metrics_output_path": PROJECT_ROOT / 'reports' / 'metrics' / 'final_model_metrics.json',

    "tuning_iterations": 25,
    "cv_folds": 3
}

# -----------------------------------------------------------------
# 2. CÁC HÀM CHỨC NĂNG (ĐÃ ĐIỀN ĐẦY ĐỦ)
# -----------------------------------------------------------------

def load_data(filepath):
    """Tải dữ liệu sạch từ pipeline."""
    print(f"[Hàm load_data] Đang tải dữ liệu từ: {filepath}...")
    start_time = time.time()
    try:
        # Chuyển Path object sang string để dùng .endswith()
        if str(filepath).endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif str(filepath).endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Định dạng file không được hỗ trợ: {filepath}")
            
        print(f"✓ Tải xong. Shape: {df.shape}. (Mất {time.time() - start_time:.2f}s)")
        return df
    except FileNotFoundError:
        print(f"🚨 LỖI: Không tìm thấy file {filepath}.")
        print("Vui lòng chạy pipeline xử lý dữ liệu (_02_feature_enrichment.py) trước.")
        sys.exit(1)
    except Exception as e:
        print(f"🚨 LỖI khi tải file: {e}")
        sys.exit(1)


def prepare_data(df):
    """
    Lọc, tạo biến mục tiêu (is_good_review), chọn đặc trưng (CHỈ WS1), 
    và chia dữ liệu.
    """
    print("[Hàm prepare_data] Đang chuẩn bị dữ liệu...")

    # Logic lọc (Giả sử bạn đang dự đoán 'is_good_review' từ WS1)
    if 'review_score' not in df.columns:
        print("🚨 LỖI: Thiếu cột 'review_score' trong file đã xử lý.")
        sys.exit(1)
        
    # Tìm cột trạng thái đơn hàng (ưu tiên 'order_status')
    if 'order_status' in df.columns:
        df_model = df[(df['order_status'] == 'delivered') & (df['review_score'] > 0)].copy()
    elif 'delivery_time_days' in df.columns:
         df_model = df[(df['delivery_time_days'] > -999) & (df['review_score'] > 0)].copy()
    else:
        print("🚨 LỖI: Không tìm thấy cột 'order_status' hoặc 'delivery_time_days' để lọc dữ liệu.")
        sys.exit(1)

    if df_model.empty:
        print("🚨 LỖI: Không tìm thấy dữ liệu đã giao và đã review để huấn luyện.")
        sys.exit(1)

    # Tạo biến mục tiêu (Y)
    target_col = 'is_good_review'
    df_model[target_col] = (df_model['review_score'] == 5).astype(int)
    print(f"Phân bổ biến mục tiêu (Y = is_good_review):")
    print(df_model[target_col].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))

    # === CHỈ SỬ DỤNG ĐẶC TRƯNG WS1 (OLIST) ===
    numeric_features = [
        # --- Đặc trưng WS1 (Olist PoC) ---
        'delivery_time_days', 'delivery_vs_estimated_days', 'order_processing_time_days',
        'price', 'freight_value', 'freight_ratio', 'payment_value_total',
        'payment_installments_total', 'payment_sequential_count', 'dist_cust_seller_km',
        'product_weight_g', 'product_volume_cm3', 'purchase_day_of_week', 'purchase_hour',
        
        # --- ĐẶC TRƯNG MỚI TỪ WORKSTREAM 3 (BEHAVIOR) ---
        # (Chúng ta sẽ tạm thời comment các dòng này lại)
        # 'total_views',
        # 'total_addtocart',
        # 'total_transactions',
        # 'rate_view_to_cart',
        # 'rate_cart_to_buy',
        # 'rate_view_to_buy',
        # 'session_duration_days',
        # 'days_since_last_action'
    ]
    
    categorical_features = [
        # --- Đặc trưng WS1 (Olist PoC) ---
        'product_category_name_english', 'customer_state', 'seller_state',
        'payment_type_primary', 'is_weekend'
    ]
    # === KẾT THÚC CHỈNH SỬA ===

    all_features = [col for col in (numeric_features + categorical_features) if col in df.columns]
    categorical_features = [col for col in categorical_features if col in all_features]
    
    missing_features = set(numeric_features + categorical_features) - set(df.columns)
    if missing_features:
        print(f"⚠️ Cảnh báo: Thiếu các đặc trưng sau: {missing_features}")

    if not all_features:
        print("🚨 LỖI: Không tìm thấy bất kỳ đặc trưng nào trong file.")
        sys.exit(1)
        
    print(f"Tìm thấy {len(all_features)} đặc trưng hợp lệ (WS1) để huấn luyện.")

    X = df_model[all_features]
    y = df_model[target_col]

    # Chuyển đổi dtype cho LightGBM
    print(f"Đang chuyển đổi {len(categorical_features)} cột sang 'category' dtype...")
    for col in categorical_features:
        X[col] = X[col].astype('category')

    # Chia Train/Test
    print("Đang chia Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print("✓ Chuẩn bị dữ liệu hoàn tất.")

    return X_train, X_test, y_train, y_test, all_features, categorical_features


def tune_model(X_train, y_train, categorical_features):
    """Tinh chỉnh hyperparameters bằng RandomizedSearchCV."""
    print("[Hàm tune_model] Bắt đầu tinh chỉnh siêu tham số...")
    start_train = time.time()

    try:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Mất cân bằng: Tỷ lệ (Xấu/Tốt) là {scale_pos_weight:.2f}")
    except ZeroDivisionError:
        scale_pos_weight = 1

    param_grid = {
        'n_estimators': [200, 500, 1000, 1500],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'num_leaves': [20, 31, 40, 50],
        'max_depth': [-1, 10, 15, 20],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

    kfold = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=42)

    base_model = lgb.LGBMClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=CONFIG['tuning_iterations'],
        cv=kfold,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(
        X_train,
        y_train,
        categorical_feature=categorical_features
    )

    print(f"\n✓ Tinh chỉnh hoàn tất (Mất {time.time() - start_train:.2f}s)")
    print("\n" + "=" * 50)
    print("           MÔ HÌNH TỐI ƯU NHẤT ĐÃ TÌM THẤY")
    print("=" * 50)
    print(f"Điểm (ROC AUC) tốt nhất: {random_search.best_score_:.4f}")
    print("Các tham số tốt nhất:")
    print(random_search.best_params_)
    print("=" * 50)

    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Đánh giá mô hình cuối cùng trên tập Test và trả về dict metrics."""
    print("[Hàm evaluate_model] Đang đánh giá trên tập Test...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report_dict = classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)'], output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)'])
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("      KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TRÊN TẬP TEST)")
    print("=" * 50)
    print(f"🎯 Accuracy (Độ chính xác): {accuracy:.2%}")
    print(f"🎯 ROC AUC: {roc_auc:.4f}")
    print("\n📊 Báo cáo Phân loại:")
    print(report_str)
    print("\n🔢 Ma trận nhầm lẫn:")
    print(pd.DataFrame(cm, index=['Actual: Bad', 'Actual: Good'], columns=['Predicted: Bad', 'Predicted: Good']))
    print("=" * 50)

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    return metrics


def save_artifacts(model, features_config, metrics):
    """Lưu mô hình, danh sách đặc trưng, và metrics ra file."""
    print("[Hàm save_artifacts] Đang lưu các 'artifacts' của mô hình...")

    # Tự động tạo thư mục nếu chưa có
    (PROJECT_ROOT / 'models').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'reports' / 'metrics').mkdir(parents=True, exist_ok=True)

    # 1. Lưu mô hình
    try:
        joblib.dump(model, CONFIG['model_output_path'])
        print(f"✓ Mô hình đã lưu tại: {CONFIG['model_output_path']}")
    except Exception as e:
        print(f"🚨 LỖI khi lưu mô hình: {e}")

    # 2. Lưu cấu hình đặc trưng
    try:
        with open(CONFIG['features_output_path'], 'w') as f:
            json.dump(features_config, f, indent=4)
        print(f"✓ Cấu hình đặc trưng đã lưu tại: {CONFIG['features_output_path']}")
    except Exception as e:
        print(f"🚨 LỖI khi lưu file features: {e}")

    # 3. Lưu metrics
    try:
        with open(CONFIG['metrics_output_path'], 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Metrics đã lưu tại: {CONFIG['metrics_output_path']}")
    except Exception as e:
        print(f"🚨 LỖI khi lưu file metrics: {e}")


# -----------------------------------------------------------------
# 3. HÀM CHÍNH (MAIN ORCHESTRATOR)
# -----------------------------------------------------------------

def main():
    """Điều phối toàn bộ quy trình huấn luyện."""
    print("========== BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN MÔ HÌNH (WS1 OLIST) ==========")
    total_start_time = time.time()

    # BƯỚC 1: Tải dữ liệu
    print("\n--- BƯỚC 1: TẢI DỮ LIỆU ---")
    df = load_data(CONFIG['data_file'])

    # BƯỚC 2: Chuẩn bị dữ liệu
    print("\n--- BƯỚC 2: CHUẨN BỊ DỮ LIỆU & CHIA TẬP ---")
    X_train, X_test, y_train, y_test, features, cat_features = prepare_data(df)

    # BƯỚC 3: Tinh chỉnh (Tune) mô hình
    print("\n--- BƯỚC 3: TINH CHỈNH MÔ HÌNH (TUNING) ---")
    best_model = tune_model(X_train, y_train, cat_features)

    # BƯỚC 4: Đánh giá mô hình tốt nhất
    print("\n--- BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH CUỐI CÙNG ---")
    metrics = evaluate_model(best_model, X_test, y_test)

    # BƯỚC 5: Lưu "Artifacts"
    print("\n--- BƯỚC 5: LƯU ARTIFACTS (MÔ HÌNH, FEATURES, METRICS) ---")
    features_config = {
        "all_features": features,
        "categorical_features": cat_features
    }
    save_artifacts(best_model, features_config, metrics)

    print("\n========================================================")
    print(f"🥳 HOÀN THÀNH! Tổng thời gian chạy: {time.time() - total_start_time:.2f} giây.")
    print(f"Các file kết quả đã được lưu tại: {CONFIG['model_output_path']} và các file .json liên quan.")
    print("========================================================")


# --- ĐIỂM BẮT ĐẦU CHẠY SCRIPT ---
if __name__ == "__main__":
    main()