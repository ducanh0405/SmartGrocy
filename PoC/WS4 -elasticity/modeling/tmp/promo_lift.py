import os, math, logging
from pathlib import Path
import duckdb, pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
BASE = Path(__file__).resolve().parents[4]
DB   = BASE / "data" / "processed" / "dunnhumby.duckdb"
OUTD = BASE / "reports" / "w4"
PLOTD= OUTD / "plots"
OUTD.mkdir(parents=True, exist_ok=True)
PLOTD.mkdir(parents=True, exist_ok=True)

# Kết nối DB và xử lý dữ liệu
con = duckdb.connect(str(DB))
logging.info("Processing data in DuckDB...")

df = con.execute("""
    WITH filtered_data AS (
        SELECT 
            store_id, product_id, week_no,
            units, avg_net_price, avg_gross_price,
            promo_display, promo_mailer,
            LN(units + 1) as y,
            CASE WHEN promo_display > 0 THEN 1 ELSE 0 END as promo_display_binary,
            CASE WHEN promo_mailer > 0 THEN 1 ELSE 0 END as promo_mailer_binary
        FROM features.weekly_store_sku
        WHERE units >= 0 
        AND avg_net_price > 0 
        AND avg_gross_price > 0
    )
    SELECT * FROM filtered_data
""").fetch_df()
con.close()

# Đổi tên cột đã được transform trong DuckDB
df = df.rename(columns={
    'promo_display_binary': 'promo_display',
    'promo_mailer_binary': 'promo_mailer'
})

# Mô hình DID với FE store & week
logging.info("Fitting OLS with fixed effects for store and week...")
model = smf.ols("y ~ promo_display + promo_mailer + C(store_id) + C(week_no)", data=df).fit(cov_type="HC1")

coefs = model.params.filter(regex="^promo_").rename("coef").to_frame()
# %lift ≈ (exp(beta)-1)*100
coefs["pct_lift"] = np.expm1(coefs["coef"]) * 100.0

out_csv = OUTD / "promo_lift_coefs.csv"
coefs.to_csv(out_csv, index=True)
logging.info(f"Saved coefs → {out_csv}")

# Bar plot
ax = coefs["pct_lift"].plot(kind="bar")
ax.set_ylabel("% lift (approx)")
ax.set_title("Estimated Promotion Lift (%), DID with FE(store, week)")
plt.tight_layout()
plot_path = PLOTD / "promo_lift_coefbar.png"
plt.savefig(plot_path, dpi=140)
plt.close()
logging.info(f"Saved plot → {plot_path}")

# Summary MD
md = []
md.append("# Promotion Lift — DID with Store & Week Fixed Effects\n")
md.append("- Response: `log1p(units)`\n- Regressors: `promo_display`, `promo_mailer`\n- Fixed effects: `C(store_id)`, `C(week_no)` (absorbed via dummies)\n- Robust SE: HC1\n")
md.append("\n## Coefficients\n")
md.append(coefs.to_markdown())
md.append("\n\n**Interpretation**: %lift ≈ (exp(beta) − 1) × 100.\n")
md.append("\n### Notes\n- FE theo store & week giúp kiểm soát khác biệt cố hữu giữa cửa hàng và mùa vụ.\n- Nên kết hợp thêm kiểm thử robust hoặc chạy theo department nếu muốn granular hơn.\n")
(OUTD / "promo_lift_summary.md").write_text("\n".join(md), encoding="utf-8")
logging.info(f"Saved summary → {OUTD / 'promo_lift_summary.md'}")

print("\n=== PROMO LIFT SUMMARY ===")
print(model.summary().as_text())
print(f"\nCoefs CSV: {out_csv}")
print(f"Plot: {plot_path}")
print(f"MD: {OUTD / 'promo_lift_summary.md'}")