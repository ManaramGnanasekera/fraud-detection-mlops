# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze and silver layers
# MAGIC
# MAGIC **Milestone M1, part 1.** Loads the raw Kaggle CSVs into a **bronze** Delta table (raw, untouched),
# MAGIC then cleans and types them into a **silver** table that later notebooks build features from.
# MAGIC
# MAGIC | Layer | Table | What it holds |
# MAGIC |---|---|---|
# MAGIC | Bronze | `workspace.fraud.bronze_transactions` | Every row exactly as it arrived, all columns as text, plus load metadata |
# MAGIC | Silver | `workspace.fraud.silver_transactions` | Typed, de-duplicated, personal details removed |
# MAGIC
# MAGIC **Run:** attach serverless compute, then *Run all*. Safe to re-run: both tables are rebuilt each time.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "fraud"
RAW_PATH = "/Volumes/workspace/default/raw"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Bronze: load the raw files as they are
# MAGIC
# MAGIC Bronze keeps everything as text (`inferSchema = False`) so nothing is lost or silently converted.
# MAGIC We add the source file name and load time, so every row can be traced back to where it came from.

# COMMAND ----------

from pyspark.sql import functions as F

raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .option("quote", '"')
    .option("escape", '"')
    .csv(f"{RAW_PATH}/*.csv")
    .withColumn("_source_file", F.col("_metadata.file_name"))
    .withColumn("_ingested_at", F.current_timestamp())
)

# The CSVs start with an unnamed pandas index column; Spark calls it _c0. It carries no information.
if "_c0" in raw.columns:
    raw = raw.drop("_c0")

(
    raw.write
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("bronze_transactions")
)

display(spark.sql("SELECT _source_file, count(*) AS rows FROM bronze_transactions GROUP BY _source_file"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Silver: clean, type and de-duplicate
# MAGIC
# MAGIC - **Types:** text becomes timestamps, dates, numbers.
# MAGIC - **Duplicates:** one row per `transaction_id`.
# MAGIC - **Privacy:** customer first name, last name and street address are dropped. The model does not need
# MAGIC   them, and leaving them out is how a bank would handle customer data.
# MAGIC - **Merchant names:** the dataset prefixes every merchant with `fraud_`; that prefix is removed so it
# MAGIC   cannot be mistaken for a label.
# MAGIC - **`unix_time` is dropped:** in this dataset it does not line up with `trans_date_trans_time`,
# MAGIC   so `transaction_ts` is the single source of truth for time.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver_transactions AS
# MAGIC SELECT
# MAGIC   trans_num                                                AS transaction_id,
# MAGIC   to_timestamp(trans_date_trans_time, 'yyyy-MM-dd HH:mm:ss') AS transaction_ts,
# MAGIC   cc_num                                                   AS card_id,
# MAGIC   regexp_replace(merchant, '^fraud_', '')                  AS merchant,
# MAGIC   category,
# MAGIC   CAST(amt AS DOUBLE)                                      AS amount,
# MAGIC   gender,
# MAGIC   city,
# MAGIC   state,
# MAGIC   zip,
# MAGIC   CAST(lat AS DOUBLE)                                      AS customer_lat,
# MAGIC   CAST(`long` AS DOUBLE)                                   AS customer_long,
# MAGIC   CAST(city_pop AS INT)                                    AS city_population,
# MAGIC   job,
# MAGIC   to_date(dob)                                             AS date_of_birth,
# MAGIC   CAST(merch_lat AS DOUBLE)                                AS merchant_lat,
# MAGIC   CAST(merch_long AS DOUBLE)                               AS merchant_long,
# MAGIC   CAST(is_fraud AS INT)                                    AS is_fraud,
# MAGIC   _source_file
# MAGIC FROM bronze_transactions
# MAGIC QUALIFY row_number() OVER (PARTITION BY trans_num ORDER BY _ingested_at) = 1

# COMMAND ----------

# MAGIC %sql
# MAGIC COMMENT ON TABLE bronze_transactions IS 'Raw card transactions from the Kaggle fraud dataset, loaded as text. Source: /Volumes/workspace/default/raw';
# MAGIC COMMENT ON TABLE silver_transactions IS 'Cleaned, typed, de-duplicated card transactions. Customer names and street addresses removed.';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data quality checks
# MAGIC
# MAGIC These stop the notebook with an error if the silver table is not fit to use, the same idea as the
# MAGIC data validation tests that will run in CI later.

# COMMAND ----------

checks = spark.sql("""
    SELECT
      count(*)                                                   AS rows,
      count(DISTINCT transaction_id)                             AS distinct_ids,
      sum(CASE WHEN transaction_ts IS NULL THEN 1 ELSE 0 END)    AS null_timestamps,
      sum(CASE WHEN amount IS NULL OR amount <= 0 THEN 1 ELSE 0 END) AS bad_amounts,
      sum(CASE WHEN is_fraud NOT IN (0, 1) OR is_fraud IS NULL THEN 1 ELSE 0 END) AS bad_labels
    FROM silver_transactions
""").first()

assert checks.rows > 0, "Silver table is empty"
assert checks.rows == checks.distinct_ids, "Duplicate transaction_id values found"
assert checks.null_timestamps == 0, f"{checks.null_timestamps} rows have no timestamp"
assert checks.bad_amounts == 0, f"{checks.bad_amounts} rows have a missing or non-positive amount"
assert checks.bad_labels == 0, f"{checks.bad_labels} rows have an invalid is_fraud label"

print("All checks passed:", checks.asDict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. First look at the data
# MAGIC
# MAGIC Expect about 1.85 million rows and a fraud rate of roughly 0.5%: about 1 fraud in 200 transactions.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   count(*)                         AS transactions,
# MAGIC   sum(is_fraud)                    AS frauds,
# MAGIC   round(100 * avg(is_fraud), 3)    AS fraud_pct,
# MAGIC   count(DISTINCT card_id)          AS cards,
# MAGIC   min(transaction_ts)              AS first_transaction,
# MAGIC   max(transaction_ts)              AS last_transaction
# MAGIC FROM silver_transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Which spending categories carry the most fraud?
# MAGIC SELECT
# MAGIC   category,
# MAGIC   count(*)                         AS transactions,
# MAGIC   sum(is_fraud)                    AS frauds,
# MAGIC   round(100 * avg(is_fraud), 2)    AS fraud_pct,
# MAGIC   round(avg(CASE WHEN is_fraud = 1 THEN amount END), 2) AS avg_fraud_amount,
# MAGIC   round(avg(CASE WHEN is_fraud = 0 THEN amount END), 2) AS avg_normal_amount
# MAGIC FROM silver_transactions
# MAGIC GROUP BY category
# MAGIC ORDER BY fraud_pct DESC
