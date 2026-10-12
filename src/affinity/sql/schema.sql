DROP TABLE IF EXISTS order_products, orders, products, aisles, departments CASCADE;

CREATE TABLE aisles      (aisle_id SMALLINT PRIMARY KEY, aisle TEXT NOT NULL);
CREATE TABLE departments (department_id SMALLINT PRIMARY KEY, department TEXT NOT NULL);
CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    product_name  TEXT NOT NULL,
    aisle_id      SMALLINT NOT NULL REFERENCES aisles,
    department_id SMALLINT NOT NULL REFERENCES departments
);
-- eval_set: 'prior' = purchase history, 'train' = each user's most recent order (the target),
-- 'test' = most recent order whose contents were never released (Kaggle leaderboard) -> unusable.
CREATE TABLE orders (
    order_id               INTEGER PRIMARY KEY,
    user_id                INTEGER NOT NULL,
    eval_set               TEXT NOT NULL,
    order_number           SMALLINT NOT NULL,
    order_dow              SMALLINT NOT NULL,
    order_hour_of_day      SMALLINT NOT NULL,
    days_since_prior_order REAL
);
CREATE TABLE order_products (
    order_id          INTEGER NOT NULL,
    product_id        INTEGER NOT NULL,
    add_to_cart_order SMALLINT NOT NULL,
    reordered         SMALLINT NOT NULL
);
