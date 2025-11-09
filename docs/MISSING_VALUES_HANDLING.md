# XỬ LÝ MISSING VALUES TRONG E-GROCERY FORECASTER

## Tổng quan

Dự án E-Grocery Forecaster áp dụng chiến lược xử lý missing values một cách có hệ thống và logic, dựa trên bản chất kinh doanh của dữ liệu bán lẻ và yêu cầu của các thuật toán machine learning. Chiến lược này được triển khai qua 4 workstreams (WS0-WS4) với nguyên tắc "zero-fill for sales, sensible defaults for metadata".

## Chiến lược tổng thể

### Nguyên tắc cơ bản
1. **Missing values trong sales data = 0** (không có giao dịch)
2. **Missing values trong metadata = giá trị mặc định hợp lý**
3. **Không drop rows** để duy trì tính toàn vẹn của master grid
4. **Ưu tiên tính toàn vẹn dữ liệu** hơn việc loại bỏ outliers

### Lý do thiết kế
- **Bảo toàn master grid**: Dự án sử dụng master grid hoàn chỉnh (product × store × week) để đảm bảo tính liên tục của time-series features
- **Phù hợp với business logic**: Trong bán lẻ, không có giao dịch ≠ missing data, mà là doanh số = 0
- **Tương thích ML**: Các thuật toán forecasting có thể học từ pattern zero sales

## Chi tiết xử lý theo Workstream

### WS0: Aggregation & Master Grid Creation

#### Logic xử lý
```python
# Điền 0 cho các cột sales chính
fill_cols = ['SALES_VALUE', 'QUANTITY', 'RETAIL_DISC', 'COUPON_DISC', 'COUPON_MATCH_DISC']
master_df[fill_cols] = master_df[fill_cols].fillna(0)
```

#### Lý do
- **Master grid hoàn chỉnh**: Tạo tất cả combinations product-store-week, kể cả khi không có giao dịch
- **Sales = 0 là hợp lý**: Trong thực tế bán lẻ, không có record giao dịch = không bán được hàng
- **Discount = 0**: Không có promotion = không có discount

#### Ví dụ thực tế
Trước xử lý:
```
PRODUCT_ID | STORE_ID | WEEK_NO | SALES_VALUE | QUANTITY
-----------|----------|---------|-------------|----------
P001      | S001    | 1      | 100        | 2
P001      | S001    | 3      | 150        | 3
```

Sau xử lý:
```
PRODUCT_ID | STORE_ID | WEEK_NO | SALES_VALUE | QUANTITY
-----------|----------|---------|-------------|----------
P001      | S001    | 1      | 100        | 2
P001      | S001    | 2      | 0          | 0    # ← Điền 0
P001      | S001    | 3      | 150        | 3
```

### WS1: Relational Features (Product & Household Data)

#### Logic xử lý
```python
# Điền 'Unknown' cho categorical columns khi merge thất bại
product_cols = ['MANUFACTURER', 'DEPARTMENT', 'BRAND', 'COMMODITY_DESC']
for col in product_cols:
    master_df[col] = master_df[col].fillna('Unknown')
```

#### Lý do
- **Metadata không quan trọng bằng sales data**: Thông tin sản phẩm missing không làm hỏng tính toán sales
- **'Unknown' là placeholder hợp lý**: Các model có thể học được pattern từ category này
- **Giữ lại tất cả rows**: Không drop rows vì sales data vẫn có giá trị

#### Ví dụ thực tế
Khi merge với product data:
- **Trước**: Một số products không có thông tin manufacturer → NaN
- **Sau**: `MANUFACTURER = 'Unknown'` → có thể phân tích riêng

### WS3: Behavioral Features (Clickstream Data)

#### Logic xử lý
```python
# Drop rows thiếu thông tin cơ bản
df_events = df_events.dropna(subset=['visitorid', 'itemid'])

# Điền 0 cho numeric features
master_df[numeric_cols] = master_df[numeric_cols].fillna(0)
```

#### Lý do
- **visitorid và itemid là mandatory**: Không có thông tin này thì event vô nghĩa
- **Behavioral metrics = 0**: Không có dữ liệu hành vi = không có tương tác
- **Phù hợp với business**: User không click = engagement = 0

### WS4: Price & Promotion Features

#### Logic xử lý
```python
# Điền 0 cho các cột promotion
master_df['is_on_display'] = master_df['is_on_display'].fillna(0).astype(int)
master_df['is_on_mailer'] = master_df['is_on_mailer'].fillna(0).astype(int)

# Điền 0 cho price columns
for col in existing_price_cols:
    master_df[col] = master_df[col].fillna(0)
```

#### Lý do
- **Không promotion = 0**: Sản phẩm không được display/mailer = giá trị = 0
- **Price data missing = 0**: Giá = 0 là signal mạnh cho model (có thể là sản phẩm mới hoặc lỗi data)
- **Consistent với discount logic**: Tương tự WS0

## Xác thực và kiểm soát chất lượng

### Validation Rules
- **Không có NaN trong master table**: Tất cả missing values phải được xử lý
- **Sales range hợp lý**: SALES_VALUE ≥ 0
- **Quantity logic**: QUANTITY ≥ 0 và integer
- **Time ordering**: Dữ liệu được sort theo PRODUCT_ID, STORE_ID, WEEK_NO

### Monitoring
```python
# Kiểm tra sau mỗi workstream
zero_sales = (master_df['SALES_VALUE'] == 0).sum()
logger.info(f"Zero sales rows: {zero_sales} ({zero_sales/len(master_df)*100:.1f}%)")
```

## Lợi ích của chiến lược này

### 1. **Tính toàn vẹn dữ liệu**
- Master grid hoàn chỉnh cho time-series modeling
- Không mất thông tin do drop rows

### 2. **Business Logic**
- Missing sales = 0 sales (thực tế kinh doanh)
- Missing metadata = unknown category (có thể phân tích riêng)

### 3. **ML Compatibility**
- Các model có thể học từ zero patterns
- Không cần imputation phức tạp (có thể gây bias)
- Consistent feature engineering

### 4. **Performance**
- Zero-fill nhanh hơn các phương pháp imputation khác
- Không cần train thêm model để fill missing values

## Các trường hợp đặc biệt

### 1. **Data Quality Issues**
- Nếu >50% data missing → warning và skip workstream
- Critical columns missing → fail pipeline với error message rõ ràng

### 2. **Different Datasets**
- **Dunnhumby**: Focus trên sales + promotion data
- **E-commerce**: Có thể có behavioral data đầy đủ hơn
- **M5**: Time-series dài hơn, ít missing values

### 3. **Future Enhancements**
- Có thể thêm imputation methods cho metadata (category-based mean, etc.)
- Validation rules có thể strict hơn cho production

## Kết luận

Chiến lược xử lý missing values của E-Grocery Forecaster được thiết kế để:
- **Tối ưu cho forecasting**: Ưu tiên zero-fill cho sales data
- **Phù hợp business**: Missing = zero sales là hợp lý
- **Đơn giản và robust**: Ít risk từ imputation methods phức tạp
- **Scalable**: Hoạt động tốt với big data

Chiến lược này đã được chứng minh hiệu quả qua các PoCs và production pipeline.
