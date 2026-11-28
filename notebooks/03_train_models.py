# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train and compare fraud models
# MAGIC
# MAGIC **Milestone M2.** This notebook is built in steps:
# MAGIC
# MAGIC 1. Load gold, split by time, rules baseline, MLflow ✅
# MAGIC 2. **Logistic regression and XGBoost, cost-based threshold** ← this version
# MAGIC 3. SHAP explanations, fairness check, register the champion model
# MAGIC
# MAGIC **Run:** from the repo's Git folder, on serverless compute.

# COMMAND ----------

# MAGIC %pip install -q mlflow scikit-learn xgboost

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2 · Machine learning models
# MAGIC
# MAGIC ## 6. Prepare the model inputs
# MAGIC
# MAGIC - **`has_history`:** 1 if the card has any spending in the previous 30 days. The rules baseline treated
# MAGIC   "no history" as normal; this lets the models learn whether new cards are riskier.
# MAGIC - **Log transforms** for skewed values (amounts, gaps, city size), so a few huge values do not dominate.
# MAGIC - **`category`** becomes one 0/1 column per spending category.
# MAGIC - Missing values stay as `NaN`: XGBoost handles them itself; logistic regression fills them with the
# MAGIC   training median.

# COMMAND ----------

import numpy as np

CATEGORIES = sorted(train["category"].unique())


def prep(d):
    """Turn gold rows into the numeric table the models train on."""
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
    for c in CATEGORIES:
        X[f"cat_{c}"] = (d["category"] == c).astype(int)
    return X.astype(float)


X_train, y_train = prep(train), train["is_fraud"]
X_valid, y_valid = prep(valid), valid["is_fraud"]
X_test, y_test = prep(test), test["is_fraud"]

print(f"{X_train.shape[1]} model inputs:", list(X_train.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Choose the alert threshold by cost
# MAGIC
# MAGIC A model gives each transaction a **fraud score** between 0 and 1. The **threshold** decides which scores
# MAGIC become alerts. We try thresholds from 0.01 to 0.99 on the **validation** months and keep the one with the
# MAGIC lowest dollar cost. The test months are only used afterwards, so the final score stays honest.

# COMMAND ----------

from sklearn.metrics import average_precision_score

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def best_threshold(y_true, scores, amounts, false_alarm_cost=FALSE_ALARM_COST):
    """Return the threshold with the lowest total cost, and that cost."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    amounts = np.asarray(amounts)
    costs = []
    for t in THRESHOLDS:
        flagged = scores >= t
        missed = amounts[(y_true == 1) & ~flagged].sum()
        false_alarms = ((y_true == 0) & flagged).sum()
        costs.append(missed + false_alarms * false_alarm_cost)
    i = int(np.argmin(costs))
    return float(THRESHOLDS[i]), float(costs[i])


def score_model(name, model):
    """Pick the threshold on validation, then evaluate once on test."""
    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    threshold, valid_cost = best_threshold(y_valid, valid_scores, valid["amount"])
    test_result = evaluate(y_test, (test_scores >= threshold).astype(int), test["amount"])
    test_result["pr_auc"] = average_precision_score(y_test, test_scores)
    test_result["threshold"] = threshold
    print(f"{name}: threshold {threshold} (validation cost ${valid_cost:,.0f}), "
          f"test recall {test_result['recall']:.3f}, precision {test_result['precision']:.3f}, "
          f"cost ${test_result['cost_usd']:,.0f}")
    return test_result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Logistic regression
# MAGIC
# MAGIC A simple, explainable model: each input gets a weight. `class_weight="balanced"` makes the rare fraud
# MAGIC cases count as much as the common genuine ones while it learns.

# COMMAND ----------

import mlflow.sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

with mlflow.start_run(run_name="logistic_regression"):
    log_reg = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    log_reg.fit(X_train, y_train)
    log_reg_result = score_model("Logistic regression", log_reg)

    mlflow.log_params({"approach": "logistic_regression", "class_weight": "balanced",
                       "false_alarm_cost_usd": FALSE_ALARM_COST})
    mlflow.log_metrics({f"test_{k}": v for k, v in log_reg_result.items()})
    mlflow.sklearn.log_model(log_reg, artifact_path="model", input_example=X_train.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. XGBoost
# MAGIC
# MAGIC Gradient-boosted decision trees: hundreds of small trees, each correcting the previous ones. It can learn
# MAGIC combinations such as "large amount AND night AND new card" that a single weight per input cannot.
# MAGIC Training on 1.1 million rows takes a few minutes.

# COMMAND ----------

import mlflow.xgboost
from xgboost import XGBClassifier

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": float((y_train == 0).sum() / (y_train == 1).sum()),
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

with mlflow.start_run(run_name="xgboost"):
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train, y_train)
    xgb_result = score_model("XGBoost", xgb)

    mlflow.log_params({"approach": "xgboost", "false_alarm_cost_usd": FALSE_ALARM_COST, **XGB_PARAMS})
    mlflow.log_metrics({f"test_{k}": v for k, v in xgb_result.items()})
    mlflow.xgboost.log_model(xgb, artifact_path="model", input_example=X_train.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Compare everything on the test months (Jul – Dec 2020)

# COMMAND ----------

comparison = pd.DataFrame(
    [no_model, rules, log_reg_result, xgb_result],
    index=["flag nothing", "rules baseline", "logistic regression", "xgboost"],
)
columns = ["recall", "precision", "pr_auc", "threshold", "alerts", "frauds_caught",
           "false_alarms", "cost_usd", "saving_vs_no_model_usd"]
display(comparison[columns].round(3).reset_index(names="approach"))
