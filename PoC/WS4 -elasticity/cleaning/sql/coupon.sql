-- Tạo staging table cho coupon
CREATE TABLE IF NOT EXISTS stg.coupon AS
SELECT
    TRIM(CAST(COUPON_UPC AS TEXT)) as coupon_upc,
    CAST(PRODUCT_ID AS BIGINT) as product_id,
    CAST(CAMPAIGN AS INTEGER) as campaign
FROM read_csv_auto('${RAW_DIR}/coupon.csv', SAMPLE_SIZE=200000);

-- Tạo core table (dedup theo coupon_upc, campaign)
CREATE TABLE IF NOT EXISTS core.coupon AS
SELECT DISTINCT ON (coupon_upc, campaign)
    coupon_upc,
    product_id,
    campaign
FROM stg.coupon
ORDER BY coupon_upc, campaign;