-- Tạo feature table cho weekly_store_sku
DROP TABLE IF EXISTS features.weekly_store_sku;
CREATE TABLE features.weekly_store_sku AS
WITH transaction_agg AS (
    SELECT
        store_id,
        product_id,
        week_no,
        SUM(CASE WHEN NOT is_return THEN ABS(quantity) ELSE 0 END) as units,
        COUNT(DISTINCT CASE WHEN NOT is_return THEN basket_id END) as baskets,
        SUM(CASE WHEN NOT is_return THEN sales_value ELSE 0 END) as net_sales,
        SUM(CASE WHEN NOT is_return THEN gross_sales ELSE 0 END) as gross_sales
    FROM core.transaction 
    GROUP BY store_id, product_id, week_no
),
pricing AS (
    SELECT
        store_id,
        product_id,
        week_no,
        units,
        baskets,
        net_sales,
        gross_sales,
        -- Average prices
        net_sales / NULLIF(units, 0) as avg_net_price,
        gross_sales / NULLIF(units, 0) as avg_gross_price
    FROM transaction_agg
)
SELECT
    p.*,
    -- Discount rate
    1 - (p.avg_net_price / NULLIF(p.avg_gross_price, 0)) as avg_discount_rate,
    -- Promotions
    COALESCE(MAX(c.display_ind), 0) as promo_display,
    COALESCE(MAX(c.mailer_ind), 0) as promo_mailer,
    -- Coupon redemptions
    COUNT(DISTINCT r.coupon_upc) as coupon_redempt_ct,
    -- Product info
    prod.department,
    prod.brand,
    prod.commodity,
    prod.sub_commodity
FROM pricing p
LEFT JOIN core.causal c 
    ON p.store_id = c.store_id 
    AND p.product_id = c.product_id 
    AND p.week_no = c.week_no
LEFT JOIN core.coupon_redempt r
    ON p.product_id = r.product_id
    AND p.week_no = (
        SELECT t.week_no 
        FROM core.transaction t 
        WHERE t.day = r.day 
        LIMIT 1
    )
LEFT JOIN core.product prod
    ON p.product_id = prod.product_id
GROUP BY
    p.store_id,
    p.product_id,
    p.week_no,
    p.units,
    p.baskets,
    p.net_sales,
    p.gross_sales,
    p.avg_net_price,
    p.avg_gross_price,
    prod.department,
    prod.brand,
    prod.commodity,
    prod.sub_commodity;