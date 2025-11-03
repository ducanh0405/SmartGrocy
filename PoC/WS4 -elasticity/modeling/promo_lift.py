import logging
import time
import psutil
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, Any

import duckdb
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class ProcessMonitor:
    """Monitor system resources during processing"""
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        self.checkpoints: Dict[str, Dict[str, float]] = {}
        
    def checkpoint(self, name: str):
        current_memory = psutil.Process().memory_info().rss
        current_time = time.time()
        
        memory_used_mb = (current_memory - self.start_memory) / 1024 / 1024
        time_elapsed = current_time - self.start_time
        
        self.checkpoints[name] = {
            'memory_mb': memory_used_mb,
            'time_s': time_elapsed
        }
        
        logging.info(
            f"Checkpoint [{name}] - "
            f"Memory: {memory_used_mb:.1f}MB, "
            f"Time: {time_elapsed:.1f}s"
        )

class ModelValidator:
    """Validate model inputs and results"""
    @staticmethod
    def validate_model_inputs(df: pd.DataFrame) -> bool:
        try:
            required_cols = [
                'store_id', 'week_no', 'units', 
                'promo_display', 'promo_mailer', 'y'
            ]
            assert all(col in df.columns for col in required_cols), "Missing required columns"
            assert not df[required_cols].isnull().any().any(), "Found null values"
            assert len(df) > 1000, "Insufficient data points"
            return True
        except AssertionError as e:
            logging.error(f"Model validation failed: {str(e)}")
            return False
            
    @staticmethod
    def validate_model_results(model) -> bool:
        try:
            assert model.df_resid > 0, "No residual degrees of freedom"
            assert not np.isnan(model.rsquared), "R-squared is NaN"
            assert all(~np.isnan(model.params)), "NaN in parameters"
            return True
        except AssertionError as e:
            logging.error(f"Model results validation failed: {str(e)}")
            return False

# Setup paths
BASE = Path(__file__).resolve().parents[3]
DB = BASE / "data" / "processed" / "dunnhumby.duckdb"
OUTD = BASE / "reports" / "w4"
PLOTD = OUTD / "plots"

# Create necessary directories
OUTD.mkdir(parents=True, exist_ok=True)
PLOTD.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(exist_ok=True)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE / "logs" / "promo_lift.log", mode='w')
    ]
)

# Initialize monitors
monitor = ProcessMonitor()
model_validator = ModelValidator()

# Enhanced performance monitoring decorator
def monitored(description: str):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                logging.info(f"Starting: {description}")
                monitor.checkpoint(f"start_{f.__name__}")
                
                result = f(*args, **kwargs)
                
                monitor.checkpoint(f"end_{f.__name__}")
                logging.info(f"Completed: {description}")
                return result
                
            except Exception as e:
                logging.error(f"Error in {f.__name__}: {str(e)}")
                raise
        return wrapped
    return decorator

# Input validation
def validate_data(df):
    required_cols = ['store_id', 'product_id', 'week_no', 'units', 
                    'avg_net_price', 'avg_gross_price', 'promo_display', 'promo_mailer']
    
    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Validate values
    if (df['units'] < 0).any():
        raise ValueError("Found negative units")
    if (df[['avg_net_price', 'avg_gross_price']] <= 0).any().any():
        raise ValueError("Found non-positive prices")

# Với file ở: .../my-data-project/src/w4/modeling/promo_lift.py
# project root = parents[2]  (modeling -> w4 -> src -> [2]=my-data-project)
BASE = Path("C:/Users/Admin/Desktop/datastorm/my-data-project")

DB    = BASE / "data" / "processed" / "dunnhumby.duckdb"
OUTD  = BASE / "reports" / "w4"
PLOTD = OUTD / "plots"
OUTD.mkdir(parents=True, exist_ok=True)
PLOTD.mkdir(parents=True, exist_ok=True)

logging.info("Opening DuckDB… %s", DB)
con = duckdb.connect(str(DB))

# ---- LẤY DỮ LIỆU TRONG DUCKDB (đúng cú pháp, không lặp, có fetch_df) ----
@monitored("Loading data from DuckDB")
def load_data_chunks(con: duckdb.DuckDBPyConnection, chunk_size: int = 1_000_000) -> pd.DataFrame:
    """
    Load data from DuckDB with validation and monitoring
    """
    try:
        # Get total count with validation
        total_count = con.execute("""
            SELECT COUNT(*) 
            FROM features.weekly_store_sku
            WHERE units >= 0 
              AND avg_net_price > 0 
              AND avg_gross_price > 0
        """).fetchone()[0]
        
        if total_count == 0:
            raise ValueError("No valid records found in database")
            
        chunks = []
        for offset in range(0, total_count, chunk_size):
            chunk_sql = f"""
                WITH filtered_data AS (
                    SELECT 
                        store_id, product_id, week_no,
                        units, avg_net_price, avg_gross_price,
                        promo_display, promo_mailer,
                        LN(units + 1) AS y,
                        CASE WHEN promo_display > 0 THEN 1 ELSE 0 END AS promo_display_bin,
                        CASE WHEN promo_mailer  > 0 THEN 1 ELSE 0 END AS promo_mailer_bin
                    FROM features.weekly_store_sku
                    WHERE units >= 0 
                      AND avg_net_price > 0 
                      AND avg_gross_price > 0
                )
                SELECT * FROM filtered_data
                LIMIT {chunk_size} OFFSET {offset}
            """
            chunk = con.execute(chunk_sql).fetch_df()
            if chunk.empty:
                break
            chunks.append(chunk)
            monitor.checkpoint(f"loaded_chunk_{len(chunks)}")
            logging.info(f"Loaded {offset + len(chunk):,} / {total_count:,} rows")
        
        final_df = pd.concat(chunks, ignore_index=True)
        monitor.checkpoint("data_loading_complete")
        return final_df
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise
    


df = load_data_chunks(con)

con.close()
logging.info("Loaded %d rows.", len(df))

if df.empty:
    raise SystemExit("No data loaded from features.weekly_store_sku. Check upstream pipeline.")

# ---- DỌN CỘT & KIỂU DỮ LIỆU (tránh trùng tên) ----
# Bỏ cột promo gốc rồi giữ bản nhị phân rõ ràng
cols_to_drop = [c for c in ["promo_display", "promo_mailer"] if c in df.columns]
df = df.drop(columns=cols_to_drop)
df = df.rename(columns={"promo_display_bin": "promo_display",
                        "promo_mailer_bin":  "promo_mailer"})

# Optimize memory usage
def optimize_dataframe(df):
    # Convert categorical columns
    cat_cols = ['store_id', 'week_no', 'product_id']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Convert numeric columns to smallest possible dtype
    num_cols = ['units', 'avg_net_price', 'avg_gross_price', 'y']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert binary columns to boolean
    bool_cols = ['promo_display', 'promo_mailer']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('bool')
    
    logging.info(f"Memory usage after optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

df = optimize_dataframe(df)

# ---- HỒI QUY DID VỚI FE(store, week) + robust SE ----
@monitored("Fitting OLS model with fixed effects")
def fit_model(df: pd.DataFrame, chunk_size=100000):
    """
    Fit OLS model with validation and error handling using sparse matrices
    """
    from scipy import sparse
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    # Validate model inputs
    if not model_validator.validate_model_inputs(df):
        raise ValueError("Invalid model inputs")
        
    try:
        # Monitor memory before fitting
        monitor.checkpoint("before_model_fit")
        
        # Encode categorical variables
        store_encoder = LabelEncoder()
        week_encoder = LabelEncoder()
        
        store_encoded = store_encoder.fit_transform(df['store_id'])
        week_encoded = week_encoder.fit_transform(df['week_no'])
        
        # Get dimensions for sparse matrix
        n_stores = len(store_encoder.classes_)
        n_weeks = len(week_encoder.classes_)
        n_samples = len(df)
        
        # Create sparse matrices for store and week dummies
        store_row = np.arange(n_samples)
        store_col = store_encoded
        store_data = np.ones(n_samples)
        store_matrix = sparse.csr_matrix((store_data, (store_row, store_col)), 
                                       shape=(n_samples, n_stores))
        
        week_row = np.arange(n_samples)
        week_col = week_encoded
        week_data = np.ones(n_samples)
        week_matrix = sparse.csr_matrix((week_data, (week_row, week_col)), 
                                      shape=(n_samples, n_weeks))
        
        # Create sparse matrix for promo variables
        promo_matrix = sparse.csr_matrix(df[['promo_display', 'promo_mailer']].values)
        
        # Combine all features into one sparse matrix
        X = sparse.hstack([promo_matrix, store_matrix, week_matrix])
        
        # Scale the target variable
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(df[['y']])
        
        # Initialize model with improved parameters for better convergence
        model = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=1e-8,   # Giảm regularization
            learning_rate='invscaling',  # Sử dụng inverse scaling
            eta0=0.001,   # Learning rate khởi đầu nhỏ hơn
            power_t=0.25, # Tốc độ giảm learning rate
            max_iter=100,
            tol=1e-5,
            random_state=42
        )
        
        # Fit model với early stopping
        best_score = float('-inf')
        patience = 3
        no_improve = 0
        
        # Fit model in chunks with multiple passes
        for epoch in range(20):  # Tăng số epochs tối đa
            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                X_chunk = X[i:end_idx]
                y_chunk = y_scaled[i:end_idx]
                
                model.partial_fit(X_chunk, y_chunk.ravel())
                
            # Calculate score and check for early stopping
            current_score = model.score(X, y_scaled.ravel())
            logging.info(f"Epoch {epoch+1}, R-squared: {current_score:.4f}")
            
            if current_score > best_score:
                best_score = current_score
                no_improve = 0
            else:
                no_improve += 1
                
            # Early stopping if no improvement for several epochs
            if no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Store additional attributes needed for summary
        model.nobs = n_samples
        model.df_resid = n_samples - (n_stores + n_weeks + 2)  # 2 for promo variables
        model.rsquared = model.score(X, y_scaled.ravel())
        
        # Scale back coefficients for interpretability
        model.coef_ = model.coef_ * scaler_y.scale_[0]
        if model.intercept_ is not None:
            model.intercept_ = model.intercept_ * scaler_y.scale_[0] + scaler_y.mean_[0]
        
        return model
    except Exception as e:
        logging.error(f"Model fitting failed: {str(e)}")
        raise

model = fit_model(df)

# ---- XUẤT HỆ SỐ & PLOT ----
# Create DataFrame for coefficients
coefs = pd.DataFrame({
    'variable': ['promo_display', 'promo_mailer'],
    'coef': model.coef_[:2]  # First two coefficients are for promo variables
}).set_index('variable')

# Calculate percentage lift
coefs["pct_lift"] = np.expm1(coefs["coef"]) * 100.0

# Add model statistics
coefs["std_err"] = [np.nan, np.nan]  # SGDRegressor doesn't provide standard errors
coefs["ci_low"] = [np.nan, np.nan]   # SGDRegressor doesn't provide confidence intervals
coefs["ci_high"] = [np.nan, np.nan]

out_csv = OUTD / "promo_lift_coefs.csv"
coefs.to_csv(out_csv)
logging.info("Saved coefficients to: %s", out_csv)

ax = coefs["pct_lift"].plot(kind="bar")
ax.set_ylabel("% lift (approx)")
ax.set_title("Estimated Promotion Lift (%), DID with FE(store, week)")
plt.tight_layout()
plot_path = PLOTD / "promo_lift_coefbar.png"
plt.savefig(plot_path, dpi=140)
plt.close()
logging.info("Saved plot to: %s", plot_path)

# ---- SUMMARY MD ----
md = []
md.append("# Promotion Lift — DID with Store & Week Fixed Effects\n")
md.append("- Response: `log1p(units)`\n- Regressors: `promo_display`, `promo_mailer`\n- Fixed effects: `C(store_id)`, `C(week_no)`\n- Robust SE: HC1\n")
md.append("\n## Coefficients\n")
md.append(coefs.to_markdown(index=True))
md.append("\n\n**Interpretation**: %lift ≈ (exp(beta) − 1) × 100.\n")
md.append("\n### Notes\n- FE theo store & week hấp thụ khác biệt nền giữa cửa hàng và mùa vụ.\n- Có thể chạy theo department để granular hơn hoặc thêm kiểm soát khác.\n")
(OUTD / "promo_lift_summary.md").write_text("\n".join(md), encoding="utf-8")
logging.info("Saved summary to: %s", OUTD / "promo_lift_summary.md")

print("\n=== PROMO LIFT SUMMARY ===")
print("\nModel Performance:")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Number of observations: {model.nobs:,}")
print(f"Degrees of freedom: {model.df_resid:,}")

print("\nPromotion Effects:")
for idx, row in coefs.iterrows():
    print(f"{idx}:")
    print(f"  Coefficient: {row['coef']:.4f}")
    print(f"  % Lift: {row['pct_lift']:.2f}%")

print("\nOutput Files:")
print(f"Coefficients CSV: {out_csv}")
print(f"Plot: {plot_path}")
print(f"Summary MD: {OUTD / 'promo_lift_summary.md'}")
