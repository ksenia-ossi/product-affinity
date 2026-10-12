-- name: dataset_size
-- Overall scale of the dataset. "train" orders are each user's most recent order and serve as the target.
SELECT
    (SELECT count(*) FROM orders)                              AS orders,
    (SELECT count(DISTINCT user_id) FROM orders)               AS users,
    (SELECT count(*) FROM orders WHERE eval_set = 'train')     AS labelled_users,
    (SELECT count(*) FROM order_products)                      AS order_lines,
    (SELECT count(*) FROM products)                            AS products,
    (SELECT count(*) FROM aisles)                              AS aisles,
    (SELECT count(*) FROM departments)                         AS departments;

-- name: orders_per_user
-- Purchase history length is heavy-tailed: median 10 orders, max 100 (capped by Instacart).
SELECT
    percentile_cont(0.25) WITHIN GROUP (ORDER BY n) AS p25,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY n) AS median,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY n) AS p75,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY n) AS p95,
    max(n) AS max
FROM (SELECT user_id, count(*) AS n FROM orders WHERE eval_set = 'prior' GROUP BY user_id) t;

-- name: basket_shape
-- A basket touches ~6 distinct aisles out of 134 -> the positive rate of (user, aisle) pairs is low.
SELECT
    round(avg(n_items), 2)  AS avg_items_per_order,
    round(avg(n_aisles), 2) AS avg_aisles_per_order,
    round(avg(n_depts), 2)  AS avg_departments_per_order
FROM (
    SELECT op.order_id, count(*) AS n_items,
           count(DISTINCT p.aisle_id) AS n_aisles, count(DISTINCT p.department_id) AS n_depts
    FROM order_products op JOIN products p USING (product_id)
    GROUP BY op.order_id
) t;

-- name: top_aisles
-- Aisle popularity = share of prior orders that contain at least one product from the aisle.
SELECT a.aisle, d.department,
       round(100.0 * count(DISTINCT o.order_id) / (SELECT count(*) FROM orders WHERE eval_set = 'prior'), 1) AS pct_of_orders
FROM order_products op
JOIN orders o USING (order_id)
JOIN products p USING (product_id)
JOIN aisles a USING (aisle_id)
JOIN departments d USING (department_id)
WHERE o.eval_set = 'prior'
GROUP BY a.aisle, d.department
ORDER BY pct_of_orders DESC
LIMIT 10;

-- name: reorder_rate_by_department
-- Share of order lines that are re-purchases. Habitual categories (dairy, produce) vs. one-off (personal care).
SELECT d.department, round(avg(op.reordered)::numeric, 3) AS reorder_rate, count(*) AS order_lines
FROM order_products op
JOIN orders o USING (order_id)
JOIN products p USING (product_id)
JOIN departments d USING (department_id)
WHERE o.eval_set = 'prior'
GROUP BY d.department
ORDER BY reorder_rate DESC;

-- name: repeat_share_in_target_order
-- Key fact for modelling: what share of the aisles in a user's NEXT order did the user already buy from before?
WITH target AS (
    SELECT DISTINCT o.user_id, p.aisle_id
    FROM orders o JOIN order_products op USING (order_id) JOIN products p USING (product_id)
    WHERE o.eval_set = 'train'
), history AS (
    SELECT DISTINCT o.user_id, p.aisle_id
    FROM orders o JOIN order_products op USING (order_id) JOIN products p USING (product_id)
    WHERE o.eval_set = 'prior'
)
SELECT count(*) AS target_user_aisle_pairs,
       round(100.0 * avg(CASE WHEN h.aisle_id IS NOT NULL THEN 1 ELSE 0 END), 1) AS pct_seen_before
FROM target t LEFT JOIN history h USING (user_id, aisle_id);

-- name: days_between_orders
-- Cadence of shopping; 30 is a cap in the raw data.
SELECT
    percentile_cont(0.25) WITHIN GROUP (ORDER BY days_since_prior_order) AS p25,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY days_since_prior_order) AS median,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY days_since_prior_order) AS p75,
    round(100.0 * avg(CASE WHEN days_since_prior_order = 30 THEN 1 ELSE 0 END), 1) AS pct_capped_at_30
FROM orders WHERE days_since_prior_order IS NOT NULL;
