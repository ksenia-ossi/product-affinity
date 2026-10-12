# Results — held-out users, aisles ranked for each user's next order

| model                                     |   precision@5 |   recall@5 |   ndcg@5 |   hit_rate@5 |   precision@10 |   recall@10 |   ndcg@10 |   hit_rate@10 |   train_time_s |      auc |   log_loss |
|:------------------------------------------|--------------:|-----------:|---------:|-------------:|---------------:|------------:|----------:|--------------:|---------------:|---------:|-----------:|
| popularity                                |        0.3723 |     0.2637 |   0.4396 |       0.7864 |         0.2810 |      0.3934 |    0.4286 |        0.8848 |         0.0000 | nan      |   nan      |
| user history (order share)                |        0.5123 |     0.4130 |   0.6112 |       0.9227 |         0.3885 |      0.5795 |    0.6085 |        0.9602 |         0.0000 | nan      |   nan      |
| ALS matrix factorisation                  |        0.3339 |     0.2839 |   0.3704 |       0.8102 |         0.3072 |      0.4760 |    0.4269 |        0.9296 |         8.1000 | nan      |   nan      |
| LightGBM (history + popularity + context) |        0.5292 |     0.4281 |   0.6311 |       0.9286 |         0.3988 |      0.5939 |    0.6261 |        0.9633 |        40.1000 |   0.8315 |     0.3435 |

## LightGBM feature importance (gain)

|                      |    gain |
|:---------------------|--------:|
| ua_order_share       | 5151116 |
| ua_share_since_first |  833763 |
| aisle_id             |  429364 |
| ua_n_items           |  372542 |
| ua_orders_since_last |  243442 |
| ua_recency_ratio     |  234590 |
| t_days_since_prior   |  114776 |
| ua_n_reordered       |   95897 |
| a_order_share        |   89329 |
| u_n_orders           |   82606 |
| u_avg_aisles         |   66831 |
| u_avg_items          |   38749 |
| u_avg_days_between   |   38227 |
| a_n_users            |   34329 |
| department_id        |   23783 |