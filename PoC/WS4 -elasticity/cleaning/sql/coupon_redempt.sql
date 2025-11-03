-- Tạo staging table cho coupon_redempt
CREATE TABLE IF NOT EXISTS stg.coupon_redempt AS
SELECT
    CAST(household_key AS BIGINT) as household_key,
    CAST(DAY AS INTEGER) as day,
    TRIM(CAST(COUPON_UPC AS TEXT)) as coupon_upc,
    CAST(CAMPAIGN AS INTEGER) as campaign
FROM read_csv_auto('${RAW_DIR}/coupon_redempt.csv', SAMPLE_SIZE=200000);

-- Tạo core table với product_id từ coupon
CREATE TABLE IF NOT EXISTS core.coupon_redempt AS
SELECT 
    r.household_key,
    r.day,
    r.coupon_upc,
    r.campaign,
    c.product_id
FROM stg.coupon_redempt r
LEFT JOIN core.coupon c 
    ON r.coupon_upc = c.coupon_upc 
    AND r.campaign = c.campaign;