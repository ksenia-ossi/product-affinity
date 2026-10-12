CREATE INDEX ON orders (user_id, order_number);
CREATE INDEX ON order_products (order_id);
ANALYZE aisles; ANALYZE departments; ANALYZE products; ANALYZE orders; ANALYZE order_products;
