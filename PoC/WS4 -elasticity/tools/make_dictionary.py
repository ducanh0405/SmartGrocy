from pathlib import Path
from datetime import datetime
import duckdb, pandas as pd

# pyyaml là tùy chọn; fallback nếu không có
try:
    import yaml
except Exception:
    yaml = None

BASE = Path(__file__).resolve().parents[3]  # lên 3 cấp: tools → w4 → src → project_root
CFG  = BASE / "configs" / "paths.yaml"

def resolve_raw_dir():
    # ưu tiên đọc từ paths.yaml
    if CFG.exists() and yaml is not None:
        try:
            cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
            raw_dir = cfg.get("data", {}).get("raw_dir")
            if raw_dir:
                p = (BASE / raw_dir).resolve()
                if p.exists():
                    return p
        except Exception:
            pass
    # fallbacks phổ biến
    for cand in ["data/raw/Dunnhumby", "data/raw/dunnhumby"]:
        p = (BASE / cand).resolve()
        if p.exists():
            return p
    return (BASE / "data" / "raw").resolve()

RAW = resolve_raw_dir()
OUT_MD  = BASE / "reports" / "dictionary.md"
OUT_CSV = BASE / "reports" / "dictionary_schema.csv"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)

FILES = [
    "campaign_desc.csv",
    "campaign_table.csv",
    "coupon.csv",
    "coupon_redempt.csv",
    "hh_demographic.csv",
    "product.csv",
    "transaction_data.csv",
    "causal_data.csv",
]

con = duckdb.connect()

lines = []
lines.append("# Dunnhumby — Data Dictionary (tối thiểu)\n")
lines.append(f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_  \n")
lines.append(f"_Project root:_ `{BASE}`  \n")
lines.append(f"_RAW dir:_ `{RAW}`\n")

schema_rows = []  # thu thập cho CSV

for fn in FILES:
    path = RAW / fn
    lines.append(f"\n## {fn}\n")
    if not path.exists():
        lines.append("> **Không tìm thấy file**\n")
        continue

    size_mb = path.stat().st_size / 1_048_576
    lines.append(f"- **Kích thước:** {size_mb:.2f} MB\n")

    # Đọc an toàn: suy schema từ sample để tránh load full file
    con.execute(f"""
        CREATE OR REPLACE VIEW v AS
        SELECT * FROM read_csv_auto('{path.as_posix()}',
            SAMPLE_SIZE=200000
        );
    """)

    schema_df = con.execute("DESCRIBE v").fetch_df()
    # markdown table
    lines.append("| column_name | column_type |")
    lines.append("|---|---|")
    for _, r in schema_df.iterrows():
        col, typ = r["column_name"], r["column_type"]
        lines.append(f"| {col} | {typ} |")
        schema_rows.append({"table": fn, "column_name": col, "column_type": typ})

    head_df = con.execute("SELECT * FROM v LIMIT 3").fetch_df()
    lines.append("\n**Ví dụ 3 dòng đầu:**")
    lines.append("```")
    # to_string giữ cột đầy đủ, không index
    lines.append(head_df.to_string(index=False))
    lines.append("```")

# Gợi ý làm sạch mặc định
lines.append("\n---\n## Gợi ý làm sạch mặc định\n")
lines.append("- Chuẩn hoá text: trim khoảng trắng; thống nhất UPPER/Title; điền 'UNKNOWN' cho danh mục trống.")
lines.append("- Số tiền/giảm giá: ép float; thay NULL bằng 0.0 khi phù hợp.")
lines.append("- Ngày/giờ: `DAY` (ordinal) → thêm cột `date`; `TRANS_TIME` (hhmm) → `time` nếu cần; giữ `WEEK_NO` dạng int.")
lines.append("- Khóa: kiểm tra trùng `PRODUCT_ID` (product); kiểm tra trùng `(household_key,BASKET_ID,PRODUCT_ID,DAY)` (transaction).")
lines.append("- Join keys: `PRODUCT_ID`, `household_key`, `STORE_ID`, `WEEK_NO`, `CAMPAIGN`, `COUPON_UPC`.\n")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

if schema_rows:
    pd.DataFrame(schema_rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

print(f"[OK] Wrote {OUT_MD}")
print(f"[OK] Wrote {OUT_CSV}")