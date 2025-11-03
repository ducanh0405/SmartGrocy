import sys
from pathlib import Path

# Xác định đường dẫn GỐC của thư mục /src
# Path(__file__).parent -> lấy thư mục chứa file này (tức là /src)
SRC_ROOT = Path(__file__).parent.resolve()

# Thêm đường dẫn /src vào sys.path nếu nó chưa có
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
    print(f"Đã thêm {SRC_ROOT} vào sys.path")