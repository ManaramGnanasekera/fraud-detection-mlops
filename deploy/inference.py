"""SageMaker inference handler for the fraud detector.

Runs inside the SageMaker XGBoost 3.0-5 container in framework mode. SageMaker calls the four
functions below for every request:

    model_fn   -> load the model and its metadata once, when the endpoint starts
    input_fn   -> parse the JSON request
    predict_fn -> prepare the model inputs, score, apply the alert threshold
    output_fn  -> return JSON

A request is one transaction, a list of transactions, or {"instances": [...]}, each written as a
gold-table row (the raw features, before model preparation), for example:

    {"transaction_id": "abc", "amount": 912.4, "txn_count_1h": 1, "txn_count_24h": 3,
     "seconds_since_last_txn": 300, "avg_amount_30d": 41.7, "amount_vs_avg_30d": 21.9,
     "hour_of_day": 2, "is_night": 1, "customer_age": 34, "distance_km": 60.2,
     "city_population": 15000, "category": "shopping_net"}

Each prediction carries the fraud score, the alert flag and the top_reasons: the inputs that pushed the
score furthest towards fraud, from XGBoost's own SHAP values.

The preparation in prepare_features() must stay identical to prep() in notebooks/03_train_models.py;
tests/test_inference.py checks that it does.
"""

import json
import math
import os

import numpy as np
import xgboost as xgb

MODEL_FILE = "xgboost-model.json"
METADATA_FILE = "model_metadata.json"


def _num(value):
    """A number, or NaN for a missing value (None, empty string, NaN)."""
    if value is None or value == "":
        return math.nan
    return float(value)


def _log1p(value):
    return math.log1p(value) if not math.isnan(value) else math.nan


def prepare_features(record, feature_names):
    """Turn one gold-table row into the model's inputs, in the model's order."""
    avg_30d = _num(record.get("avg_amount_30d"))
    prepared = {
        "log_amount": _log1p(_num(record.get("amount"))),
        "txn_count_1h": _num(record.get("txn_count_1h")),
        "txn_count_24h": _num(record.get("txn_count_24h")),
        "log_seconds_since_last_txn": _log1p(_num(record.get("seconds_since_last_txn"))),
        "log_avg_amount_30d": _log1p(avg_30d),
        "amount_vs_avg_30d": _num(record.get("amount_vs_avg_30d")),
        "has_history": 0.0 if math.isnan(avg_30d) else 1.0,
        "hour_of_day": _num(record.get("hour_of_day")),
        "is_night": _num(record.get("is_night")),
        "customer_age": _num(record.get("customer_age")),
        "distance_km": _num(record.get("distance_km")),
        "log_city_population": _log1p(_num(record.get("city_population"))),
    }
    category = record.get("category")
    for name in feature_names:
        if name.startswith("cat_"):
            prepared[name] = 1.0 if category == name[len("cat_"):] else 0.0

    missing = [name for name in feature_names if name not in prepared]
    if missing:
        raise ValueError(f"The model expects inputs this handler does not build: {missing}")
    return [prepared[name] for name in feature_names]


def model_fn(model_dir):
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, MODEL_FILE))
    with open(os.path.join(model_dir, METADATA_FILE)) as f:
        metadata = json.load(f)
    return {"booster": booster, "metadata": metadata}


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type {content_type!r}; send application/json")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    data = json.loads(request_body)
    if isinstance(data, dict) and "instances" in data:
        data = data["instances"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError("Send one transaction, a list of transactions, or {'instances': [...]}")
    return data


TOP_REASONS = 3


def top_reasons(contributions, values, feature_names, k=TOP_REASONS):
    """The k inputs that pushed this score furthest towards fraud (largest positive SHAP values)."""
    order = np.argsort(contributions)[::-1]
    return [
        {
            "feature": feature_names[i],
            "value": None if math.isnan(values[i]) else round(float(values[i]), 4),
            "contribution": round(float(contributions[i]), 4),
        }
        for i in order[:k]
        if contributions[i] > 0
    ]


def predict_fn(records, model):
    metadata = model["metadata"]
    feature_names = metadata["feature_names"]
    threshold = float(metadata["threshold"])

    matrix = np.array([prepare_features(r, feature_names) for r in records], dtype=float)
    dmatrix = xgb.DMatrix(matrix, feature_names=feature_names)
    scores = model["booster"].predict(dmatrix)
    contribs = model["booster"].predict(dmatrix, pred_contribs=True)[:, :-1]  # last column is the bias

    return [
        {
            "transaction_id": r.get("transaction_id"),
            "fraud_score": round(float(s), 6),
            "flagged": bool(s >= threshold),
            "threshold": threshold,
            "model_version": metadata.get("model_version"),
            "top_reasons": top_reasons(c, row, feature_names),
        }
        for r, s, c, row in zip(records, scores, contribs, matrix)
    ]


def output_fn(predictions, accept="application/json"):
    return json.dumps({"predictions": predictions}), "application/json"
