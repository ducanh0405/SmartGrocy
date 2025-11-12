# Hướng dẫn Tối ưu Hóa Bộ Nhớ

## Vấn đề

Khi chạy pipeline với dataset lớn, bạn có thể gặp lỗi bộ nhớ (MemoryError). Tài liệu này hướng dẫn cách tối ưu hóa để pipeline có thể chạy trên máy có RAM hạn chế.

## Giải pháp

### 1. Bật Memory Optimization trong Config

Mở file `src/config.py` và tìm phần `MEMORY_OPTIMIZATION`:

```python
MEMORY_OPTIMIZATION = {
    'enable_sampling': True,      # Bật sampling
    'sample_fraction': 0.1,       # Chỉ dùng 10% data
    'max_products': 10,          # Giới hạn 10 products
    'max_stores': 2,             # Giới hạn 2 stores
    'max_time_periods': 24,      # Giới hạn 24 time periods
    'use_chunking': True,        # Sử dụng chunking
    'chunk_size': 100000,        # Kích thước chunk
}
```

### 2. Sử dụng Script Helper

#### Bật memory optimization với cài đặt mặc định:
```bash
python scripts/enable_memory_optimization.py --enable
```

#### Bật với tùy chọn tùy chỉnh:
```bash
# Sample 10% data, giới hạn 10 products, 2 stores, 24 hours
python scripts/enable_memory_optimization.py --enable \
    --sample-fraction 0.1 \
    --max-products 10 \
    --max-stores 2 \
    --max-time 24
```

#### Tắt memory optimization:
```bash
python scripts/enable_memory_optimization.py --disable
```

### 3. Các Tùy Chọn Tối Ưu Hóa

#### `enable_sampling` (True/False)
- Bật/tắt việc sample data
- Mặc định: False

#### `sample_fraction` (0.0 - 1.0)
- Tỷ lệ data được sử dụng
- 0.1 = 10% data
- 0.5 = 50% data
- 1.0 = 100% data (không sample)

#### `max_products` (int hoặc None)
- Giới hạn số lượng products
- None = không giới hạn
- Ví dụ: 10 = chỉ xử lý 10 products đầu tiên

#### `max_stores` (int hoặc None)
- Giới hạn số lượng stores
- None = không giới hạn
- Ví dụ: 2 = chỉ xử lý 2 stores đầu tiên

#### `max_time_periods` (int hoặc None)
- Giới hạn số lượng time periods
- None = không giới hạn
- Ví dụ: 24 = chỉ xử lý 24 time periods đầu tiên

#### `use_chunking` (True/False)
- Tự động sử dụng chunking cho operations lớn
- Mặc định: True

#### `chunk_size` (int)
- Kích thước chunk khi xử lý
- Mặc định: 100000

## Ví dụ Cấu Hình

### Máy có RAM thấp (< 8GB)
```python
MEMORY_OPTIMIZATION = {
    'enable_sampling': True,
    'sample_fraction': 0.05,      # 5% data
    'max_products': 5,
    'max_stores': 1,
    'max_time_periods': 24,
    'use_chunking': True,
    'chunk_size': 50000,
}
```

### Máy có RAM trung bình (8-16GB)
```python
MEMORY_OPTIMIZATION = {
    'enable_sampling': True,
    'sample_fraction': 0.2,       # 20% data
    'max_products': 20,
    'max_stores': 5,
    'max_time_periods': 168,     # 1 tuần
    'use_chunking': True,
    'chunk_size': 100000,
}
```

### Máy có RAM cao (> 16GB)
```python
MEMORY_OPTIMIZATION = {
    'enable_sampling': False,     # Không sample
    'sample_fraction': 1.0,
    'max_products': None,         # Không giới hạn
    'max_stores': None,
    'max_time_periods': None,
    'use_chunking': True,
    'chunk_size': 200000,
}
```

## Lưu ý

1. **Sampling sẽ làm giảm chất lượng model**: Chỉ sử dụng khi thực sự cần thiết
2. **Test với dataset nhỏ trước**: Bắt đầu với cài đặt conservative và tăng dần
3. **Monitor memory usage**: Sử dụng Task Manager (Windows) hoặc `htop` (Linux) để theo dõi
4. **Chunking tự động**: Hệ thống sẽ tự động sử dụng chunking khi grid size > 1M rows

## Troubleshooting

### Vẫn gặp lỗi MemoryError?
1. Giảm `sample_fraction` xuống 0.05 hoặc 0.01
2. Giảm `max_products`, `max_stores`, `max_time_periods`
3. Giảm `chunk_size` xuống 50000
4. Đóng các ứng dụng khác để giải phóng RAM

### Pipeline chạy quá chậm?
1. Tăng `chunk_size` lên 200000 hoặc 500000
2. Tăng `sample_fraction` nếu có thể
3. Kiểm tra xem có process khác đang sử dụng CPU/RAM không


