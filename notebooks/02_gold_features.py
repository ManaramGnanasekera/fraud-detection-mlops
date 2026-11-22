# Databricks notebook source
# DBTITLE 1,count(*) window feature
# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE TABLE workspace.fraud.gold_features AS
# MAGIC WITH base AS (
# MAGIC   SELECT *, unix_timestamp(transaction_ts) AS ts   -- time in seconds, calculated once
# MAGIC   FROM workspace.fraud.silver_transactions
# MAGIC )
# MAGIC SELECT
# MAGIC   transaction_id,
# MAGIC   card_id,
# MAGIC   transaction_ts,
# MAGIC   amount,
# MAGIC   category,
# MAGIC   gender,                 -- kept for the fairness check, not for training
# MAGIC   city_population,
# MAGIC   is_fraud,
# MAGIC
# MAGIC   -- 1. How busy has the card been?
# MAGIC   count(*) OVER (PARTITION BY card_id ORDER BY ts
# MAGIC                  RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING) AS txn_count_24h,
# MAGIC   count(*) OVER (PARTITION BY card_id ORDER BY ts
# MAGIC                  RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING)  AS txn_count_1h,
# MAGIC
# MAGIC   -- 2. How long since the card's previous transaction? (NULL for the first one)
# MAGIC   ts - LAG(ts) OVER (PARTITION BY card_id ORDER BY ts, transaction_id) AS seconds_since_last_txn,
# MAGIC
# MAGIC   -- 3. Is this amount normal for this card? (NULL when there is no 30-day history)
# MAGIC   avg(amount) OVER (PARTITION BY card_id ORDER BY ts
# MAGIC                     RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS avg_amount_30d,
# MAGIC   amount / avg(amount) OVER (PARTITION BY card_id ORDER BY ts
# MAGIC                              RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS amount_vs_avg_30d,
# MAGIC
# MAGIC   -- 4. When did it happen?
# MAGIC   hour(transaction_ts) AS hour_of_day,
# MAGIC   CASE WHEN hour(transaction_ts) >= 22 OR hour(transaction_ts) < 6 THEN 1 ELSE 0 END AS is_night,
# MAGIC
# MAGIC   -- 5. Who and where?
# MAGIC   floor(datediff(transaction_ts, date_of_birth) / 365.25) AS customer_age,
# MAGIC   2 * 6371 * asin(sqrt(
# MAGIC       pow(sin(radians(merchant_lat - customer_lat) / 2), 2)
# MAGIC     + cos(radians(customer_lat)) * cos(radians(merchant_lat))
# MAGIC     * pow(sin(radians(merchant_long - customer_long) / 2), 2)
# MAGIC   )) AS distance_km      -- haversine distance between customer and merchant
# MAGIC
# MAGIC FROM base;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   is_fraud,
# MAGIC   count(*)                              AS transactions,
# MAGIC   round(avg(txn_count_1h), 2)           AS avg_txn_1h,
# MAGIC   round(avg(txn_count_24h), 2)          AS avg_txn_24h,
# MAGIC   round(avg(seconds_since_last_txn)/60) AS avg_minutes_since_last,
# MAGIC   round(avg(amount_vs_avg_30d), 2)      AS avg_amount_vs_usual,
# MAGIC   round(100 * avg(is_night), 1)         AS pct_at_night,
# MAGIC   round(avg(distance_km), 1)            AS avg_distance_km
# MAGIC FROM workspace.fraud.gold_features
# MAGIC GROUP BY is_fraud;