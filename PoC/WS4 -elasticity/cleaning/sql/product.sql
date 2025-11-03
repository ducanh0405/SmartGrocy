-- Tạo staging table cho product
CREATE TABLE IF NOT EXISTS stg.product AS
SELECT 
    CAST(PRODUCT_ID AS BIGINT) AS product_id,
    CAST(MANUFACTURER AS BIGINT) AS manufacturer_id,
    TRIM(UPPER(DEPARTMENT)) AS department,
    TRIM(UPPER(BRAND)) AS brand,
    TRIM(UPPER(COMMODITY_DESC)) AS commodity,
    TRIM(UPPER(SUB_COMMODITY_DESC)) AS sub_commodity,
    TRIM(UPPER(CURR_SIZE_OF_PRODUCT)) AS size
FROM read_csv_auto('${RAW_DIR}/product.csv', SAMPLE_SIZE=200000);

-- Tạo core table (dedup theo product_id)
CREATE TABLE IF NOT EXISTS core.product AS
SELECT DISTINCT ON (product_id) *
FROM stg.product
ORDER BY product_id;