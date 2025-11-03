-- Tạo staging table cho campaign_table
CREATE TABLE IF NOT EXISTS stg.campaign_table AS
SELECT
    CAST(household_key AS BIGINT) as household_key,
    CAST(CAMPAIGN AS INTEGER) as campaign,
    TRIM(UPPER(DESCRIPTION)) as group_desc
FROM read_csv_auto('${RAW_DIR}/campaign_table.csv', SAMPLE_SIZE=200000);

-- Tạo core table (dedup theo household_key, campaign)
CREATE TABLE IF NOT EXISTS core.campaign_table AS
SELECT DISTINCT ON (household_key, campaign)
    household_key,
    campaign,
    group_desc
FROM stg.campaign_hh
ORDER BY household_key, campaign;