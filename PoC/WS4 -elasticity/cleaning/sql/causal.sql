-- Tạo staging table cho causal
CREATE TABLE IF NOT EXISTS stg.causal AS
SELECT
    CAST(PRODUCT_ID AS BIGINT) as product_id,
    CAST(STORE_ID AS INTEGER) as store_id,
    CAST(WEEK_NO AS INTEGER) as week_no,
    CASE 
        WHEN LOWER(display) IN ('0','n','no','false','',' ') THEN 0 
        ELSE 1 
    END as display_ind,
    CASE 
        WHEN LOWER(mailer) IN ('0','n','no','false','',' ') THEN 0 
        ELSE 1 
    END as mailer_ind
FROM read_csv_auto('${RAW_DIR}/causal_data.csv', SAMPLE_SIZE=200000);

-- Tạo core table (dedup theo store_id, product_id, week_no)
CREATE TABLE IF NOT EXISTS core.causal AS
WITH ranked AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY store_id, product_id, week_no 
            ORDER BY display_ind DESC, mailer_ind DESC
        ) as rn
    FROM stg.causal
)
SELECT 
    store_id,
    product_id,
    week_no,
    display_ind,
    mailer_ind
FROM ranked 
WHERE rn = 1;