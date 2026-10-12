-- Each block becomes data/processed/<name>.parquet.
-- Only the 131k users whose most recent order is labelled ('train' eval_set) are kept.
-- History = all 'prior' orders; target = the single 'train' order. This is a
-- leave-last-order-out temporal split: features never see the target order.

-- name: users
-- Per-user history statistics + context of the target order (known when the user opens the app).
WITH prior AS (
    SELECT o.user_id, o.order_id, o.order_number, o.days_since_prior_order,
           count(*) AS n_items, count(DISTINCT p.aisle_id) AS n_aisles
    FROM orders o JOIN order_products op USING (order_id) JOIN products p USING (product_id)
    WHERE o.eval_set = 'prior'
    GROUP BY o.user_id, o.order_id, o.order_number, o.days_since_prior_order
)
SELECT pr.user_id,
       count(*)                                   AS u_n_orders,
       avg(pr.n_items)                            AS u_avg_items,
       avg(pr.n_aisles)                           AS u_avg_aisles,
       avg(pr.days_since_prior_order)             AS u_avg_days_between,
       t.days_since_prior_order                   AS t_days_since_prior,
       t.order_dow                                AS t_dow,
       t.order_hour_of_day                        AS t_hour
FROM prior pr
JOIN orders t ON t.user_id = pr.user_id AND t.eval_set = 'train'
GROUP BY pr.user_id, t.days_since_prior_order, t.order_dow, t.order_hour_of_day;

-- name: user_aisle
-- One row per (user, aisle) pair seen in the user's history: how often, how recently, how habitual.
WITH lines AS (
    SELECT o.user_id, o.order_id, o.order_number, p.aisle_id, op.reordered, op.add_to_cart_order
    FROM orders o JOIN order_products op USING (order_id) JOIN products p USING (product_id)
    WHERE o.eval_set = 'prior'
      AND o.user_id IN (SELECT user_id FROM orders WHERE eval_set = 'train')
), n_orders AS (
    SELECT user_id, max(order_number) AS n FROM orders WHERE eval_set = 'prior' GROUP BY user_id
)
SELECT l.user_id, l.aisle_id,
       count(DISTINCT l.order_id)                        AS ua_n_orders,
       count(*)                                          AS ua_n_items,
       sum(l.reordered)                                  AS ua_n_reordered,
       avg(l.add_to_cart_order)                          AS ua_avg_cart_pos,
       min(l.order_number)                               AS ua_first_order,
       max(l.order_number)                               AS ua_last_order,
       n.n - max(l.order_number)                         AS ua_orders_since_last,
       count(DISTINCT l.order_id)::float / n.n           AS ua_order_share,
       count(DISTINCT l.order_id)::float / (n.n - min(l.order_number) + 1) AS ua_share_since_first
FROM lines l JOIN n_orders n USING (user_id)
GROUP BY l.user_id, l.aisle_id, n.n;

-- name: aisles
-- Global aisle statistics computed on history only.
SELECT a.aisle_id, a.aisle, d.department_id, d.department,
       count(DISTINCT o.order_id)::float / (SELECT count(*) FROM orders WHERE eval_set = 'prior') AS a_order_share,
       count(DISTINCT o.user_id)                         AS a_n_users,
       avg(op.reordered)                                 AS a_reorder_rate
FROM order_products op
JOIN orders o USING (order_id)
JOIN products p USING (product_id)
JOIN aisles a USING (aisle_id)
JOIN departments d USING (department_id)
WHERE o.eval_set = 'prior'
GROUP BY a.aisle_id, a.aisle, d.department_id, d.department;

-- name: labels
-- Target: the set of aisles in each labelled user's most recent order.
SELECT DISTINCT o.user_id, p.aisle_id
FROM orders o JOIN order_products op USING (order_id) JOIN products p USING (product_id)
WHERE o.eval_set = 'train';
