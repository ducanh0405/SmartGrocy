-- Tạo staging table cho campaign_desc
CREATE TABLE IF NOT EXISTS stg.campaign_desc AS
SELECT
    CAST(CAMPAIGN AS INTEGER) as campaign,
    TRIM(UPPER(DESCRIPTION)) as description,
    CAST(START_DAY AS INTEGER) as start_day,
    CAST(END_DAY AS INTEGER) as end_day
FROM read_csv_auto('${RAW_DIR}/campaign_desc.csv', SAMPLE_SIZE=200000);

-- Tạo core table (giữ nguyên, đã clean)
CREATE TABLE IF NOT EXISTS core.campaign_desc AS
SELECT * FROM stg.campaign_desc;