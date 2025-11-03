-- Tạo staging table cho transaction
CREATE TABLE IF NOT EXISTS stg.transaction AS
WITH 
date_bounds AS (
    SELECT 
        MIN(DAY) as min_day,
        DATE '2015-01-01' as base_date
    FROM read_csv_auto('${RAW_DIR}/transaction_data.csv', SAMPLE_SIZE=200000)
),
transactions AS (
    SELECT
        CAST(household_key AS BIGINT) as household_key,
        CAST(BASKET_ID AS BIGINT) as basket_id,
        CAST(DAY AS INTEGER) as day,
        CAST(PRODUCT_ID AS BIGINT) as product_id,
        CAST(QUANTITY AS INTEGER) as quantity,
        CAST(SALES_VALUE AS DOUBLE) as sales_value,
        CAST(STORE_ID AS INTEGER) as store_id,
        CAST(RETAIL_DISC AS DOUBLE) as retail_disc,
        CAST(TRANS_TIME AS INTEGER) as trans_time,
        CAST(WEEK_NO AS INTEGER) as week_no,
        CAST(COUPON_DISC AS DOUBLE) as coupon_disc,
        CAST(COUPON_MATCH_DISC AS DOUBLE) as coupon_match_disc
    FROM read_csv_auto('${RAW_DIR}/transaction_data.csv', SAMPLE_SIZE=200000)
)
SELECT 
    t.*,
    -- Date synthesis
    db.base_date + (t.day - db.min_day) * INTERVAL '1 day' as date_synth,
    -- Time parsing (simplified)
    CASE 
        WHEN CAST(t.trans_time / 100 AS INTEGER) >= 24 OR CAST(t.trans_time % 100 AS INTEGER) >= 60 
        THEN NULL 
        ELSE CAST(t.trans_time / 100 AS INTEGER)
    END as hour,
    CASE 
        WHEN CAST(t.trans_time / 100 AS INTEGER) >= 24 OR CAST(t.trans_time % 100 AS INTEGER) >= 60 
        THEN NULL 
        ELSE CAST(t.trans_time % 100 AS INTEGER)
    END as minute,
    -- Business metrics
    (t.QUANTITY < 0 OR t.SALES_VALUE < 0) as is_return,
    COALESCE(t.retail_disc, 0) + COALESCE(t.coupon_disc, 0) + COALESCE(t.coupon_match_disc, 0) as total_discount,
    t.SALES_VALUE + ABS(COALESCE(t.retail_disc, 0) + COALESCE(t.coupon_disc, 0) + COALESCE(t.coupon_match_disc, 0)) as gross_sales,
    CASE WHEN t.QUANTITY != 0 THEN t.SALES_VALUE / ABS(t.QUANTITY) END as unit_price_net,
    CASE WHEN t.QUANTITY != 0 
         THEN (t.SALES_VALUE + ABS(COALESCE(t.retail_disc, 0) + COALESCE(t.coupon_disc, 0) + COALESCE(t.coupon_match_disc, 0))) / ABS(t.QUANTITY)
    END as unit_price_gross
FROM transactions t
CROSS JOIN date_bounds db;

-- Tạo core table với cột tối thiểu cho join/features
CREATE TABLE IF NOT EXISTS core.transaction AS
SELECT
    household_key,
    basket_id,
    day,
    week_no,
    store_id,
    product_id,
    quantity,
    sales_value,
    total_discount,
    gross_sales,
    unit_price_net,
    unit_price_gross,
    is_return,
    date_synth,
    hour,
    minute
FROM stg.transaction;