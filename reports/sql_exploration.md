# SQL exploration of the Instacart dataset

## dataset_size

Overall scale of the dataset. "train" orders are each user's most recent order and serve as the target.

|   orders |   users |   labelled_users |   order_lines |   products |   aisles |   departments |
|---------:|--------:|-----------------:|--------------:|-----------:|---------:|--------------:|
|  3421083 |  206209 |           131209 |      33819106 |      49688 |      134 |            21 |

## orders_per_user

Purchase history length is heavy-tailed: median 10 orders, max 100 (capped by Instacart).

|   p25 |   median |    p75 |    p95 |    max |
|------:|---------:|-------:|-------:|-------:|
| 5.000 |    9.000 | 19.000 | 51.000 | 99.000 |

## basket_shape

A basket touches ~6 distinct aisles out of 134 -> the positive rate of (user, aisle) pairs is low.

|   avg_items_per_order |   avg_aisles_per_order |   avg_departments_per_order |
|----------------------:|-----------------------:|----------------------------:|
|                10.110 |                  7.270 |                       4.740 |

## top_aisles

Aisle popularity = share of prior orders that contain at least one product from the aisle.

| aisle                         | department   |   pct_of_orders |
|:------------------------------|:-------------|----------------:|
| fresh fruits                  | produce      |          55.700 |
| fresh vegetables              | produce      |          44.400 |
| packaged vegetables fruits    | produce      |          36.700 |
| yogurt                        | dairy eggs   |          26.300 |
| milk                          | dairy eggs   |          24.400 |
| packaged cheese               | dairy eggs   |          23.000 |
| water seltzer sparkling water | beverages    |          19.100 |
| soy lactosefree               | dairy eggs   |          17.000 |
| chips pretzels                | snacks       |          16.700 |
| bread                         | bakery       |          16.400 |

## reorder_rate_by_department

Share of order lines that are re-purchases. Habitual categories (dairy, produce) vs. one-off (personal care).

| department      |   reorder_rate |   order_lines |
|:----------------|---------------:|--------------:|
| dairy eggs      |          0.670 |       5414016 |
| beverages       |          0.653 |       2690129 |
| produce         |          0.650 |       9479291 |
| bakery          |          0.628 |       1176787 |
| deli            |          0.608 |       1051249 |
| pets            |          0.601 |         97724 |
| babies          |          0.579 |        423802 |
| bulk            |          0.577 |         34573 |
| snacks          |          0.574 |       2887550 |
| alcohol         |          0.570 |        153696 |
| meat seafood    |          0.568 |        708931 |
| breakfast       |          0.561 |        709569 |
| frozen          |          0.542 |       2236432 |
| dry goods pasta |          0.461 |        866627 |
| canned goods    |          0.457 |       1068058 |
| other           |          0.408 |         36291 |
| household       |          0.402 |        738666 |
| missing         |          0.396 |         69145 |
| international   |          0.369 |        269253 |
| pantry          |          0.347 |       1875577 |
| personal care   |          0.321 |        447123 |

## repeat_share_in_target_order

Key fact for modelling: what share of the aisles in a user's NEXT order did the user already buy from before?

|   target_user_aisle_pairs |   pct_seen_before |
|--------------------------:|------------------:|
|                990769.000 |            80.800 |

## days_between_orders

Cadence of shopping; 30 is a cap in the raw data.

|   p25 |   median |    p75 |   pct_capped_at_30 |
|------:|---------:|-------:|-------------------:|
| 4.000 |    7.000 | 15.000 |             11.500 |
