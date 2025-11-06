# 🚀 OPTIMIZED PIPELINE - HƯỚNG DẪN SỬ DỤNG

## ✅ Các Cải Tiến Chính

### 1. **WS2 Feature Engineering - 10x Nhanh Hơn**
- **Trước**: 610 giây (~10 phút) cho 21.8M rows
- **Sau**: 60 giây (~1 phút) cho 21.8M rows  
- **Phương pháp**: Vectorized operations, native pandas rolling windows

### 2. **Hyperparameter Tuning với Optuna**
- **Mục tiêu**: Cải thiện accuracy và calibration
- **Kỹ thuật**: Time-series cross-validation (3 folds)
- **Tham số tối ưu**: `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, etc.
- **Kết quả mong đợi**:
  - Pinball loss: 0.000116 → <0.00008 (cải thiện ~30%)
  - Coverage: 99.98% → 88-92% (chính xác hơn)

### 3. **Enhanced Features**
- **Trend features**: WoW change, momentum, volatility
- **Calendar features**: Cyclical encoding, business flags
- **Better lag strategy**: Optimized lag windows [1, 4, 8, 12]

---

## 📊 So Sánh Hiệu Năng

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| WS2 Feature Time | 610s | ~60s | **10x faster** |
| Q50 Pinball Loss | 0.000116 | <0.00008 | **~30% better** |
| Coverage (90%) | 99.98% | 88-92% | **Better calibrated** |
| Total Pipeline | ~20 min | ~5-7 min | **3-4x faster** |

---

## 🔧 Cách Sử Dụng

### **Option 1: Chạy Nhanh (Không Tuning)**
```powershell
python scripts/run_optimized_pipeline.py
```
- **Thời gian**: ~5 phút
- **Dùng khi**: Testing, development, POC
- **Kết quả**: Models tốt (nhưng chưa tối ưu)

### **Option 2: Chạy Đầy Đủ (Có Tuning)**
```powershell
python scripts/run_optimized_pipeline.py --tune --trials 30
```
- **Thời gian**: ~30 phút (3 models × 30 trials + feature engineering)
- **Dùng khi**: Production deployment, final training
- **Kết quả**: Models **tối ưu nhất**

### **Option 3: Quick Test (Fast Tuning)**
```powershell
python scripts/run_optimized_pipeline.py --tune --trials 10
```
- **Thời gian**: ~15 phút
- **Dùng khi**: Kiểm tra xem tuning có giúp ích không
- **Kết quả**: Cải thiện nhẹ

### **Option 4: Only Feature Engineering**
```powershell
python scripts/run_optimized_pipeline.py --features-only
```
- **Thời gian**: ~2-3 phút
- **Dùng khi**: Chỉ muốn tạo feature table

### **Option 5: Only Model Training (từ features có sẵn)**
```powershell
python scripts/run_optimized_pipeline.py --models-only --tune --trials 30
```
- **Thời gian**: ~25 phút
- **Dùng khi**: Đã có feature table, chỉ muốn train lại models

---

## 📁 Output Files

### **Khi chạy KHÔNG tuning:**
```
models/
├── q05_forecaster.joblib         # Model Q05 (default params)
├── q50_forecaster.joblib         # Model Q50 (default params)
├── q95_forecaster.joblib         # Model Q95 (default params)
├── model_metrics_v1.json         # Metrics gốc
└── feature_config_v1.json        # Feature config
```

### **Khi chạy CÓ tuning:**
```
models/
├── q05_forecaster_tuned.joblib   # Model Q05 (tuned params)
├── q50_forecaster_tuned.joblib   # Model Q50 (tuned params)
├── q95_forecaster_tuned.joblib   # Model Q95 (tuned params)
├── best_hyperparameters.json     # Best params cho mỗi quantile
├── tuned_model_metrics.json      # Metrics sau tuning
└── tuned_feature_config.json     # Feature config
```

---

## 🔬 Kiểm Tra Kết Quả

### **1. Xem Metrics:**
```powershell
cat models/tuned_model_metrics.json
```

Expected output:
```json
{
  "q05_pinball_loss": 0.000042,
  "q50_pinball_loss": 0.000078,
  "q95_pinball_loss": 0.000045,
  "coverage_90pct": 0.895,
  "mae": 0.000123,
  "rmse": 0.000456
}
```

### **2. Xem Hyperparameters:**
```powershell
cat models/best_hyperparameters.json
```

Example:
```json
{
  "q05": {
    "n_estimators": 420,
    "learning_rate": 0.037,
    "num_leaves": 45,
    "max_depth": 7,
    "min_child_samples": 28,
    "subsample": 0.85,
    "colsample_bytree": 0.92
  }
}
```

### **3. So Sánh Original vs Tuned:**
Script tự động in ra comparison:
```
METRIC COMPARISON:
----------------------------------------------------------------------
q50_pinball_loss         : 0.000116 -> 0.000078 (BETTER, +32.8%)
coverage_90pct           : 0.999800 -> 0.895000 (BETTER, -10.5%)
----------------------------------------------------------------------
```

---

## ⚙️ Technical Details

### **WS2 Optimizations:**
1. **Vectorized Lag Creation**:
   - Sử dụng `shift()` + group boundary detection
   - Không dùng `groupby().transform()` (slow)

2. **Native Pandas Rolling**:
   - `groupby().rolling()` với `min_periods=1`
   - 8-10x nhanh hơn transform approach

3. **Memory Efficient**:
   - Process by group ID
   - Không tạo intermediate DataFrames

### **Optuna Tuning Strategy:**
1. **Time-Series CV**:
   - 3 folds với expanding window
   - Fold 1: weeks 1-54 train, 55-68 val
   - Fold 2: weeks 1-68 train, 69-81 val
   - Fold 3: weeks 1-75 train, 76-81 val

2. **Search Space**:
   ```python
   {
     'n_estimators': [100, 500],
     'learning_rate': [0.01, 0.1] (log scale),
     'num_leaves': [15, 63],
     'max_depth': [3, 10],
     'min_child_samples': [10, 50],
     'subsample': [0.6, 1.0],
     'colsample_bytree': [0.6, 1.0]
   }
   ```

3. **Separate Tuning Per Quantile**:
   - Q05, Q50, Q95 có hyperparameters riêng
   - Mỗi quantile optimize pinball loss riêng

---

## 🐛 Troubleshooting

### **Lỗi: "Optuna not available"**
```powershell
pip install optuna
```

### **Lỗi: Memory error khi tuning**
Giảm số trials:
```powershell
python scripts/run_optimized_pipeline.py --tune --trials 10
```

### **WS2 vẫn chậm?**
Kiểm tra xem đã dùng optimized version chưa:
```python
# Trong _02_feature_enrichment.py phải thấy:
# [PIPELINE] Using OPTIMIZED WS2 features (10x speedup)
```

### **Models không cải thiện sau tuning?**
- Data quá sparse (99.9% zeros) → cần zero-inflation modeling
- Try tuning với nhiều trials hơn (50-100)
- Xem xét feature selection (bỏ redundant features)

---

## 📈 Next Steps (Nếu Vẫn Muốn Cải Thiện Thêm)

1. **Feature Selection**:
   ```powershell
   # Run SHAP analysis
   python scripts/analyze_feature_importance.py
   ```

2. **Zero-Inflation Modeling**:
   - Train separate models cho high-volume vs low-volume products
   - Implement hurdle models (zero vs non-zero)

3. **Polars/DuckDB Migration**:
   - Migrate WS2 sang Polars để 50-100x speedup
   - Dùng DuckDB cho aggregations

4. **Ensemble Models**:
   - Combine LightGBM + XGBoost
   - Stack quantile predictions

---

## ✅ Checklist: Production Ready?

- [x] Pipeline chạy end-to-end không lỗi
- [x] WS2 tối ưu (10x faster)
- [x] Hyperparameter tuning implemented
- [x] Time-based split (no leakage)
- [x] Leak-safe features verified
- [x] Models saved và metrics logged
- [x] Documentation đầy đủ
- [ ] Feature selection (optional)
- [ ] Zero-inflation handling (optional)
- [ ] CI/CD setup (optional)

---

**Tác giả**: DataStorm Team  
**Ngày cập nhật**: 2025-01-24  
**Version**: 2.0 (Optimized)
