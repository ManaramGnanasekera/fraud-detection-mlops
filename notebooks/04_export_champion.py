# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 · Export the champion model
# MAGIC
# MAGIC **Milestone M3, step 1.** Saves the champion fraud model as plain files, so it can be deployed outside
# MAGIC Databricks (on Amazon SageMaker). Databricks Free Edition cannot connect to an AWS account directly, so the
# MAGIC files are written to a Unity Catalog volume, downloaded, and uploaded to S3.
# MAGIC
# MAGIC | File | What it holds |
# MAGIC |---|---|
# MAGIC | `xgboost-model.json` | The trained XGBoost model |
# MAGIC | `model_metadata.json` | The alert threshold, the model inputs in order, and where the model came from |
# MAGIC | `sample_transactions.json` | A few prepared test transactions (genuine and fraud) to check the endpoint later |

# COMMAND ----------

# MAGIC %pip install -q mlflow xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the champion from the Unity Catalog registry

# COMMAND ----------

import json
import os

import mlflow
import mlflow.xgboost
from mlflow import MlflowClient

MODEL_NAME = "workspace.fraud.fraud_detector"
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
threshold = float(champion.tags["threshold"])

model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@champion")
booster = model.get_booster() if hasattr(model, "get_booster") else model
feature_names = list(booster.feature_names)

print(f"Champion: {MODEL_NAME} version {champion.version}, threshold {threshold}")
print(f"{len(feature_names)} model inputs:", feature_names)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Write the files to a volume

# COMMAND ----------

spark.sql("CREATE VOLUME IF NOT EXISTS workspace.fraud.exports")
EXPORT_DIR = f"/Volumes/workspace/fraud/exports/fraud_detector_v{champion.version}"
os.makedirs(EXPORT_DIR, exist_ok=True)

booster.save_model(os.path.join(EXPORT_DIR, "xgboost-model.json"))

metadata = {
    "model_name": MODEL_NAME,
    "model_version": str(champion.version),
    "alias": "champion",
    "threshold": threshold,
    "feature_names": feature_names,
    "xgboost_version": __import__("xgboost").__version__,
    "test_recall": champion.tags.get("test_recall"),
    "test_precision": champion.tags.get("test_precision"),
}
with open(os.path.join(EXPORT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Wrote model and metadata to", EXPORT_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save sample transactions for testing the endpoint
# MAGIC
# MAGIC Five genuine and five fraudulent transactions from the test months, as **gold-table rows** (the raw
# MAGIC features, before model preparation). The deployed service will do the same preparation as notebook 03, so
# MAGIC these are exactly what a caller would send. The local score is saved alongside, so the endpoint's answer
# MAGIC can be checked against it.

# COMMAND ----------

import numpy as np
import pandas as pd
from xgboost import DMatrix

GOLD_COLUMNS = ["transaction_id", "amount", "txn_count_1h", "txn_count_24h", "seconds_since_last_txn",
                "avg_amount_30d", "amount_vs_avg_30d", "hour_of_day", "is_night", "customer_age",
                "distance_km", "city_population", "category", "is_fraud"]

test_rows = (
    spark.table("workspace.fraud.gold_features")
    .where("transaction_ts >= '2020-07-01'")
    .select(*GOLD_COLUMNS)
)
sample = pd.concat([
    test_rows.where("is_fraud = 0").limit(5).toPandas(),
    test_rows.where("is_fraud = 1").limit(5).toPandas(),
], ignore_index=True)

categories = sorted(c[len("cat_"):] for c in feature_names if c.startswith("cat_"))


def prep(d):
    """Same preparation as notebook 03 (section 6)."""
    X = pd.DataFrame(index=d.index)
    X["log_amount"] = np.log1p(d["amount"])
    X["txn_count_1h"] = d["txn_count_1h"]
    X["txn_count_24h"] = d["txn_count_24h"]
    X["log_seconds_since_last_txn"] = np.log1p(d["seconds_since_last_txn"])
    X["log_avg_amount_30d"] = np.log1p(d["avg_amount_30d"])
    X["amount_vs_avg_30d"] = d["amount_vs_avg_30d"]
    X["has_history"] = d["avg_amount_30d"].notna().astype(int)
    X["hour_of_day"] = d["hour_of_day"]
    X["is_night"] = d["is_night"]
    X["customer_age"] = d["customer_age"]
    X["distance_km"] = d["distance_km"]
    X["log_city_population"] = np.log1p(d["city_population"])
    for c in categories:
        X[f"cat_{c}"] = (d["category"] == c).astype(int)
    return X[feature_names].astype(float)


sample["local_score"] = booster.predict(DMatrix(prep(sample)))
sample["local_flagged"] = (sample["local_score"] >= threshold).astype(int)

records = json.loads(sample.to_json(orient="records"))
with open(os.path.join(EXPORT_DIR, "sample_transactions.json"), "w") as f:
    json.dump(records, f, indent=2)

display(sample[["transaction_id", "amount", "category", "is_fraud", "local_score", "local_flagged"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Download the files
# MAGIC
# MAGIC In **Catalog → workspace → fraud → Volumes → exports**, open the `fraud_detector_v…` folder and download
# MAGIC all three files to your computer. They are uploaded to S3 in the next step.

# COMMAND ----------

display(dbutils.fs.ls(EXPORT_DIR))
