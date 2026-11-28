# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05 · Drift monitoring
# MAGIC
# MAGIC **Milestone M3, step 5.** Checks whether the data the model sees has moved away from the data it was trained
# MAGIC on. Fraud patterns change over time, and a model trained on last year's behaviour slowly gets worse.
# MAGIC
# MAGIC **Why here and not in SageMaker:** SageMaker Model Monitor relies on data capture, which serverless endpoints do
# MAGIC not support. The features the endpoint scores are built in this lakehouse anyway, so drift is measured here, on
# MAGIC the gold table, month by month.
# MAGIC
# MAGIC **Method: Population Stability Index (PSI).** For each input, split the training data into 10 equal-sized bins,
# MAGIC then compare what share of each month's data falls in each bin:
# MAGIC
# MAGIC `PSI = Σ (current% − reference%) × ln(current% ÷ reference%)`
# MAGIC
# MAGIC | PSI | Meaning | Action |
# MAGIC |---|---|---|
# MAGIC | below 0.10 | Stable | None |
# MAGIC | 0.10 – 0.25 | Moderate shift | Investigate |
# MAGIC | above 0.25 | Significant shift | Retrain |
# MAGIC
# MAGIC The model's **score distribution** and the monthly **fraud rate** are tracked the same way.

# COMMAND ----------

# MAGIC %pip install -q mlflow xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the gold table and the champion model

# COMMAND ----------

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from xgboost import DMatrix

MODEL_NAME = "workspace.fraud.fraud_detector"
REFERENCE_END = pd.Timestamp("2020-04-01")  # the champion's training period: Jan 2019 – Mar 2020
MONITOR_START = pd.Timestamp("2020-04-01")  # validation and test months are monitored

mlflow.set_registry_uri("databricks-uc")
champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@champion")
booster = model.get_booster() if hasattr(model, "get_booster") else model
feature_names = list(booster.feature_names)
threshold = float(champion.tags["threshold"])

df = spark.table("workspace.fraud.gold_features").toPandas()
ts = pd.to_datetime(df["transaction_ts"])
if ts.dt.tz is not None:
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
df["transaction_ts"] = ts
df["month"] = df["transaction_ts"].dt.to_period("M").astype(str)
print(f"{len(df):,} rows; champion version {champion.version}, threshold {threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Score every transaction with the champion
# MAGIC
# MAGIC Same input preparation as training (notebook 03) and the endpoint (`deploy/inference.py`).

# COMMAND ----------

categories = sorted(c[len("cat_"):] for c in feature_names if c.startswith("cat_"))


def prep(d):
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


df["fraud_score"] = booster.predict(DMatrix(prep(df)))
df["flagged"] = (df["fraud_score"] >= threshold).astype(int)

reference = df[df["transaction_ts"] < REFERENCE_END]
monitored = df[df["transaction_ts"] >= MONITOR_START]
print(f"Reference: {len(reference):,} rows · Monitored: {len(monitored):,} rows in {monitored['month'].nunique()} months")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. PSI for each input and for the model score, month by month

# COMMAND ----------

NUMERIC = ["amount", "txn_count_1h", "txn_count_24h", "seconds_since_last_txn", "amount_vs_avg_30d",
           "hour_of_day", "customer_age", "distance_km", "city_population", "fraud_score"]
EPS = 1e-6


def psi_numeric(ref, cur, bins=10):
    """PSI with bin edges from the reference deciles. Missing values get their own bin."""
    ref, cur = pd.Series(ref, dtype=float), pd.Series(cur, dtype=float)
    edges = np.unique(np.nanquantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(x):
        counts = pd.cut(x.dropna(), edges, include_lowest=True).value_counts(sort=False).to_numpy()
        counts = np.append(counts, x.isna().sum())
        return counts / max(len(x), 1)

    r, c = shares(ref) + EPS, shares(cur) + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur):
    levels = sorted(set(ref.dropna()) | set(cur.dropna()))
    r = ref.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy() + EPS
    c = cur.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy() + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def status(value):
    return "significant" if value > 0.25 else ("moderate" if value >= 0.10 else "stable")


rows = []
for month, cur in monitored.groupby("month"):
    for col in NUMERIC:
        rows.append({"month": month, "feature": col, "psi": psi_numeric(reference[col], cur[col])})
    rows.append({"month": month, "feature": "category", "psi": psi_categorical(reference["category"], cur["category"])})

drift = pd.DataFrame(rows)
drift["psi"] = drift["psi"].round(4)
drift["status"] = drift["psi"].apply(status)

display(drift.pivot(index="feature", columns="month", values="psi").reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fraud rate and alert volume by month
# MAGIC
# MAGIC Labels (`is_fraud`) arrive weeks late in real life, through chargebacks and investigations, so recall can only be
# MAGIC measured in arrears. Alert volume is available immediately and is often the first sign that something changed.

# COMMAND ----------

monthly = (
    df.groupby("month")
    .apply(lambda g: pd.Series({
        "transactions": len(g),
        "fraud_rate_pct": round(100 * g["is_fraud"].mean(), 3),
        "alert_rate_pct": round(100 * g["flagged"].mean(), 3),
        "recall": round(((g["is_fraud"] == 1) & (g["flagged"] == 1)).sum() / max(g["is_fraud"].sum(), 1), 3),
        "period": "reference" if g["transaction_ts"].max() < REFERENCE_END else "monitored",
    }), include_groups=False)
    .reset_index()
)
display(monthly)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Retraining decision
# MAGIC
# MAGIC **Rule:** retrain when, in any monitored month, the model score or one of the model's five most important inputs
# MAGIC (by SHAP, notebook 03) has PSI above 0.25, or recall falls below 0.90 once that month's labels are in.

# COMMAND ----------

TOP_SHAP_INPUTS = ["amount", "avg_amount_30d", "hour_of_day", "amount_vs_avg_30d", "is_night"]
WATCHED = [f for f in TOP_SHAP_INPUTS if f in NUMERIC] + ["fraud_score"]
RECALL_FLOOR = 0.90

watched = drift[drift["feature"].isin(WATCHED)]
breaches = watched[watched["psi"] > 0.25]
low_recall = monthly[(monthly["period"] == "monitored") & (monthly["recall"] < RECALL_FLOOR)]

if len(breaches) or len(low_recall):
    print("RETRAIN RECOMMENDED")
    if len(breaches):
        display(breaches)
    if len(low_recall):
        display(low_recall)
else:
    print(f"No retraining needed: watched inputs and the score are within PSI 0.25, "
          f"and monthly recall stays at or above {RECALL_FLOOR}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save the report
# MAGIC
# MAGIC Written to Unity Catalog so it can feed a dashboard or an alert, and so each run is kept. In production this
# MAGIC notebook would run monthly as a task in the Lakeflow Job, after the gold table refreshes.

# COMMAND ----------

drift["champion_version"] = str(champion.version)
drift["run_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
spark.createDataFrame(drift).write.mode("append").saveAsTable("workspace.fraud.drift_report")
print(f"Appended {len(drift)} rows to workspace.fraud.drift_report")
