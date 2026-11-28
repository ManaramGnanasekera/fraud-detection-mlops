# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train and compare fraud models
# MAGIC
# MAGIC **Milestone M2.** This notebook is built in steps:
# MAGIC
# MAGIC 1. **Load gold, split by time, rules baseline, MLflow** ← this version
# MAGIC 2. Logistic regression and XGBoost, cost-based threshold
# MAGIC 3. SHAP explanations, fairness check, register the champion model
# MAGIC
# MAGIC **Run:** from the repo's Git folder, on serverless compute.

# COMMAND ----------

# MAGIC %pip install -q mlflow scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the gold table into pandas
# MAGIC
# MAGIC 1.85 million rows fit comfortably in memory, so modelling uses pandas and scikit-learn.
# MAGIC `gender` is loaded only for the fairness check later. It is never a model input.

# COMMAND ----------

import pandas as pd

FEATURES = [
    "amount",
    "txn_count_1h",
    "txn_count_24h",
    "seconds_since_last_txn",
    "avg_amount_30d",
    "amount_vs_avg_30d",
    "hour_of_day",
    "is_night",
    "customer_age",
    "distance_km",
    "city_population",
    "category",
]

df = (
    spark.table("workspace.fraud.gold_features")
    .select("transaction_id", "transaction_ts", "is_fraud", "gender", *FEATURES)
    .toPandas()
)

# Work with plain (time-zone-free) timestamps so date comparisons are simple.
ts = pd.to_datetime(df["transaction_ts"])
if ts.dt.tz is not None:
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
df["transaction_ts"] = ts

print(f"{len(df):,} rows, {df['is_fraud'].sum():,} frauds")
print("From", df["transaction_ts"].min(), "to", df["transaction_ts"].max())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Split by time, not at random
# MAGIC
# MAGIC A real model is trained on the past and used on the future. A random split would let the model learn from
# MAGIC transactions that happened *after* the ones it is tested on, which flatters the results.
# MAGIC
# MAGIC | Split | Period | Used for |
# MAGIC |---|---|---|
# MAGIC | Train | Jan 2019 – Mar 2020 | Fitting the models |
# MAGIC | Validation | Apr – Jun 2020 | Choosing settings and the alert threshold |
# MAGIC | Test | Jul – Dec 2020 | The final, untouched score |

# COMMAND ----------

VALID_START = pd.Timestamp("2020-04-01")
TEST_START = pd.Timestamp("2020-07-01")

train = df[df["transaction_ts"] < VALID_START]
valid = df[(df["transaction_ts"] >= VALID_START) & (df["transaction_ts"] < TEST_START)]
test = df[df["transaction_ts"] >= TEST_START]

split_summary = pd.DataFrame({
    "rows": [len(train), len(valid), len(test)],
    "frauds": [train["is_fraud"].sum(), valid["is_fraud"].sum(), test["is_fraud"].sum()],
}, index=["train", "valid", "test"])
split_summary["fraud_pct"] = (100 * split_summary["frauds"] / split_summary["rows"]).round(3)
display(split_summary.reset_index(names="split"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. How we score a fraud system
# MAGIC
# MAGIC Accuracy is useless here: flagging nothing would be 99.5% accurate and catch no fraud. Instead:
# MAGIC
# MAGIC - **Recall:** of all the fraud, how much did we catch?
# MAGIC - **Precision:** of everything we flagged, how much was really fraud?
# MAGIC - **Cost:** money lost to missed fraud, plus the cost of reviewing false alarms.
# MAGIC
# MAGIC **Cost assumptions** (easy to change): a missed fraud loses its full amount; each false alarm costs
# MAGIC **$5** in analyst time and customer friction. "Flag nothing" is the reference point: it loses every fraud.

# COMMAND ----------

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

FALSE_ALARM_COST = 5.0


def evaluate(y_true, y_pred, amounts, false_alarm_cost=FALSE_ALARM_COST):
    """Precision, recall, F1 and dollar cost for a set of 0/1 fraud flags."""
    y_true = pd.Series(y_true).to_numpy()
    y_pred = pd.Series(y_pred).to_numpy()
    amounts = pd.Series(amounts).to_numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    missed_fraud_cost = amounts[(y_true == 1) & (y_pred == 0)].sum()
    total_fraud_amount = amounts[y_true == 1].sum()
    total_cost = missed_fraud_cost + fp * false_alarm_cost

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "alerts": int(tp + fp),
        "frauds_caught": int(tp),
        "frauds_missed": int(fn),
        "false_alarms": int(fp),
        "cost_usd": float(total_cost),
        "saving_vs_no_model_usd": float(total_fraud_amount - total_cost),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Rules baseline
# MAGIC
# MAGIC Before any machine learning, a simple rule built from the three strongest signals found in M1:
# MAGIC
# MAGIC > **Flag a transaction if it is at least 3× the card's 30-day average spend, AND it happens at night
# MAGIC > OR within a burst (2+ transactions on the card in the previous hour).**
# MAGIC
# MAGIC Any model has to beat this to be worth its complexity.

# COMMAND ----------

def rules_flag(d):
    unusual_amount = d["amount_vs_avg_30d"].fillna(0) >= 3   # no history counts as "not unusual"
    at_night = d["is_night"] == 1
    burst = d["txn_count_1h"] >= 2
    return (unusual_amount & (at_night | burst)).astype(int)


no_model = evaluate(test["is_fraud"], [0] * len(test), test["amount"])
rules = evaluate(test["is_fraud"], rules_flag(test), test["amount"])

results = pd.DataFrame([no_model, rules], index=["flag nothing", "rules baseline"])
display(results.round(3).reset_index(names="approach"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log the baseline to MLflow
# MAGIC
# MAGIC Every approach is recorded as an MLflow **run** in one experiment, so results can be compared side by side
# MAGIC in the **Experiments** page.

# COMMAND ----------

import mlflow

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fraud-detection")

with mlflow.start_run(run_name="rules_baseline"):
    mlflow.log_params({
        "approach": "rules",
        "rule": "amount_vs_avg_30d >= 3 AND (is_night OR txn_count_1h >= 2)",
        "false_alarm_cost_usd": FALSE_ALARM_COST,
        "test_period": "2020-07-01 onwards",
    })
    mlflow.log_metrics({f"test_{k}": v for k, v in rules.items()})

print("Logged the rules baseline to MLflow experiment:", f"/Users/{user}/fraud-detection")
