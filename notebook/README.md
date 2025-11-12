# 📓 Notebooks Guide

Hướng dẫn sử dụng các notebook trong dự án E-Grocery Forecaster.

## 📚 Danh sách Notebooks

### 1. `00_Quick_Start.ipynb` 🚀
**Mục đích**: Hướng dẫn nhanh chạy toàn bộ pipeline từ đầu đến cuối.

**Nội dung**:
- Load dữ liệu từ FreshRetail dataset
- Chạy feature engineering (WS0 → WS6)
- Train model và đánh giá
- Tạo predictions

**Thời gian**: ~5-10 phút (với sample data)

**Khi nào dùng**: Lần đầu tiên làm quen với pipeline hoặc muốn chạy nhanh end-to-end.

---

### 2. `01_EDA_Data_Exploration.ipynb` 📊
**Mục đích**: Khám phá và phân tích dữ liệu.

**Nội dung**:
- Dataset overview và cấu trúc
- Data quality checks với Great Expectations
- Sales data analysis và statistics
- Time series visualization
- Distribution analysis

**Thời gian**: ~10-15 phút

**Khi nào dùng**: 
- Trước khi chạy pipeline để hiểu dữ liệu
- Kiểm tra data quality issues
- Phát hiện patterns và anomalies

---

### 3. `02_Feature_Engineering_Guide.ipynb` 🔧
**Mục đích**: Hướng dẫn chi tiết về từng workstream trong feature engineering.

**Nội dung**:
- **WS0**: Aggregation & Master Grid
- **WS1**: Relational Features (Product, Household)
- **WS2**: Time-Series Features (Lag, Rolling, Calendar) - **LEAK-SAFE**
- **WS3**: Behavior Features (Clickstream)
- **WS4**: Price & Promotion Features
- **WS5**: Stockout Recovery Features
- **WS6**: Weather Features

**Thời gian**: ~15-20 phút

**Khi nào dùng**:
- Muốn hiểu chi tiết từng bước feature engineering
- Debug feature engineering issues
- Customize features cho dataset mới

---

### 4. `03_Model_Training.ipynb` 🤖
**Mục đích**: Training và đánh giá quantile regression models.

**Nội dung**:
- Load feature table đã được engineering
- Time-based data split (leak-safe)
- Train 7 quantile models (Q05, Q10, Q25, Q50, Q75, Q90, Q95)
- Evaluation với Pinball Loss và Prediction Interval Coverage
- Visualize training results

**Thời gian**: ~20-45 phút (tùy dataset size)

**Khi nào dùng**:
- Train models với hyperparameters mới
- Đánh giá model performance
- So sánh different model configurations

---

### 5. `04_Prediction_Forecasting.ipynb` 📈
**Mục đích**: Sử dụng trained models để tạo forecasts.

**Nội dung**:
- Load trained quantile models
- Prepare future data
- Generate predictions với uncertainty intervals
- Visualize forecasts với Plotly

**Thời gian**: ~5-10 phút

**Khi nào dùng**:
- Tạo forecasts cho tương lai
- Visualize prediction intervals
- Export predictions cho business use

---

## 🗂️ Archive Notebooks

Folder `archive/` chứa các notebook POC cũ:
- `ws1_olist_poc.ipynb`: Olist dataset POC
- `ws2_m5_poc.ipynb`: M5 dataset POC
- `ws3_retailrocket_poc.ipynb`: RetailRocket dataset POC
- `ws4_dunnhumby_poc.ipynb`: Dunnhumby dataset POC

**Lưu ý**: Các notebook này chỉ để tham khảo, không còn được maintain.

---

## 🚀 Quick Start

### Cách chạy notebook:

1. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

2. **Mở Jupyter**:
```bash
jupyter notebook
# hoặc
jupyter lab
```

3. **Chạy theo thứ tự**:
   - Bắt đầu với `00_Quick_Start.ipynb` để làm quen
   - Sau đó explore `01_EDA_Data_Exploration.ipynb` để hiểu dữ liệu
   - Tiếp theo `02_Feature_Engineering_Guide.ipynb` để hiểu features
   - Cuối cùng `03_Model_Training.ipynb` và `04_Prediction_Forecasting.ipynb`

---

## 📝 Lưu ý

1. **Path Setup**: Tất cả notebooks đều tự động setup project path, không cần config thêm.

2. **Data Requirements**: 
   - Đảm bảo data đã được load vào `data/2_raw/` hoặc `data/1_poc_data/`
   - Chạy `python scripts/load_freshretail_datasets.py` nếu cần

3. **Memory**: 
   - Sample data: ~2-4GB RAM
   - Full data: ~16GB+ RAM

4. **Execution Order**: 
   - Các notebook có thể chạy độc lập
   - Nhưng khuyến nghị chạy theo thứ tự để hiểu flow

5. **Output Files**:
   - Feature table: `data/3_processed/master_feature_table.parquet`
   - Models: `models/q{05,10,25,50,75,90,95}_forecaster.joblib`
   - Metrics: `reports/metrics/quantile_model_metrics.json`

---

## 🆘 Troubleshooting

### Import Errors
```python
# Đảm bảo project root được add vào sys.path
import sys
from pathlib import Path
project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))
```

### Data Not Found
```bash
# Load data trước
python scripts/load_freshretail_datasets.py
```

### Memory Issues
- Sử dụng sample data thay vì full data
- Giảm số lượng features trong config
- Sử dụng chunking cho large datasets

---

## 📚 Tài liệu liên quan

- **QUICKSTART.md**: Hướng dẫn setup và chạy pipeline từ command line
- **OPERATIONS.md**: Hướng dẫn deployment và production
- **TEST_README.md**: Hướng dẫn testing

---

**Happy Notebooking! 📓✨**

