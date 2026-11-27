# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Gold features
# MAGIC
# MAGIC **Milestone M1, part 2.** Builds `workspace.fraud.gold_features` from the silver table: one row per
# MAGIC transaction, with behavioural features for the fraud model.
# MAGIC
# MAGIC The feature logic is **not written in this notebook**. It lives in `src/fraud/features.py`, the same
# MAGIC code the unit tests in `tests/test_features.py` check. So the pipeline always runs tested logic, and any
# MAGIC change to a feature has to pass the tests.
# MAGIC
# MAGIC **Run:** from the repo's Git folder in Databricks, on serverless compute.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Make the repo's `src/` folder importable
# MAGIC
# MAGIC In a Databricks Git folder, a notebook runs from its own folder (`notebooks/`), so the repo root is one
# MAGIC level up.

# COMMAND ----------

import os
import sys

repo_root = os.path.dirname(os.getcwd())
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fraud.features import build_gold_features

print("Using feature code from:", src_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build and save the gold table

# COMMAND ----------

SOURCE = "workspace.fraud.silver_transactions"
TARGET = "workspace.fraud.gold_features"

gold = build_gold_features(spark, SOURCE)

(
    gold.write
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(TARGET)
)

spark.sql(f"""
    COMMENT ON TABLE {TARGET} IS
    'One row per transaction with point-in-time behavioural fraud features. Built by notebooks/02_gold_features from src/fraud/features.py.'
""")

print(f"Wrote {spark.table(TARGET).count():,} rows to {TARGET}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Checks
# MAGIC
# MAGIC Gold must have exactly one row per silver transaction. If not, stop the job here.

# COMMAND ----------

silver_rows = spark.table(SOURCE).count()
gold_rows = spark.table(TARGET).count()
assert gold_rows == silver_rows, f"Gold has {gold_rows:,} rows but silver has {silver_rows:,}"
print("Row count check passed:", f"{gold_rows:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Which features separate fraud from genuine transactions?

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
# MAGIC GROUP BY is_fraud
