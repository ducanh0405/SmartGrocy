-- Tạo staging table cho hh_demographic
CREATE TABLE IF NOT EXISTS stg.hh_demo AS
SELECT
    CAST(household_key AS BIGINT) as household_key,
    COALESCE(TRIM(UPPER(AGE_DESC)), 'UNKNOWN') as age_desc,
    COALESCE(TRIM(UPPER(MARITAL_STATUS_CODE)), 'UNKNOWN') as marital_status_code,
    COALESCE(TRIM(UPPER(INCOME_DESC)), 'UNKNOWN') as income_desc,
    COALESCE(TRIM(UPPER(HOMEOWNER_DESC)), 'UNKNOWN') as homeowner_desc,
    COALESCE(TRIM(UPPER(HH_COMP_DESC)), 'UNKNOWN') as hh_comp_desc,
    COALESCE(TRIM(UPPER(HOUSEHOLD_SIZE_DESC)), 'UNKNOWN') as household_size_desc,
    COALESCE(TRIM(UPPER(KID_CATEGORY_DESC)), 'UNKNOWN') as kid_category_desc
FROM read_csv_auto('${RAW_DIR}/hh_demographic.csv', SAMPLE_SIZE=200000);

-- Tạo core table (giữ nguyên, đã clean)
CREATE TABLE IF NOT EXISTS core.hh_demo AS
SELECT * FROM stg.hh_demo;