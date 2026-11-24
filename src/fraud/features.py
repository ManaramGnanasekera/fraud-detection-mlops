"""Gold-layer fraud features.

The SQL lives here, in one place, so the Databricks notebook and the unit tests
run exactly the same logic. Every time window ends at ``1 PRECEDING``: a feature
only ever looks at the card's earlier transactions, never the current one or
future ones.
"""

GOLD_FEATURES_SQL = """
WITH base AS (
  SELECT *, unix_timestamp(transaction_ts) AS ts
  FROM {source}
)
SELECT
  transaction_id,
  card_id,
  transaction_ts,
  amount,
  category,
  gender,
  city_population,
  is_fraud,

  count(*) OVER (PARTITION BY card_id ORDER BY ts
                 RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING) AS txn_count_24h,
  count(*) OVER (PARTITION BY card_id ORDER BY ts
                 RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING)  AS txn_count_1h,

  ts - LAG(ts) OVER (PARTITION BY card_id ORDER BY ts, transaction_id) AS seconds_since_last_txn,

  avg(amount) OVER (PARTITION BY card_id ORDER BY ts
                    RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS avg_amount_30d,
  amount / avg(amount) OVER (PARTITION BY card_id ORDER BY ts
                             RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS amount_vs_avg_30d,

  hour(transaction_ts) AS hour_of_day,
  CASE WHEN hour(transaction_ts) >= 22 OR hour(transaction_ts) < 6 THEN 1 ELSE 0 END AS is_night,

  floor(datediff(transaction_ts, date_of_birth) / 365.25) AS customer_age,
  2 * 6371 * asin(sqrt(
      pow(sin(radians(merchant_lat - customer_lat) / 2), 2)
    + cos(radians(customer_lat)) * cos(radians(merchant_lat))
    * pow(sin(radians(merchant_long - customer_long) / 2), 2)
  )) AS distance_km
FROM base
"""


def gold_features_sql(source: str) -> str:
    """Return the gold feature query reading from ``source`` (a table or view name)."""
    return GOLD_FEATURES_SQL.format(source=source)


def build_gold_features(spark, source: str):
    """Run the gold feature query against ``source`` and return a Spark DataFrame."""
    return spark.sql(gold_features_sql(source))
