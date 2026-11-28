"""Tests for the SageMaker inference handler (deploy/inference.py).

The most important check: the handler prepares model inputs exactly as notebook 03 did at training
time. If the two drifted apart, the endpoint would score transactions differently from the model
that was evaluated, without any error.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inference", ROOT / "deploy" / "inference.py")
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)

CATEGORIES = ["food_dining", "gas_transport", "shopping_net"]


def training_prep(d):
    """prep() from notebooks/03_train_models.py, section 6, as used at training time."""
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


def gold_rows(n=400, seed=7):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "transaction_id": [f"t{i}" for i in range(n)],
        "amount": rng.gamma(2, 40, n),
        "txn_count_1h": rng.integers(0, 3, n),
        "txn_count_24h": rng.integers(0, 8, n),
        "seconds_since_last_txn": rng.integers(60, 90_000, n).astype(float),
        "avg_amount_30d": rng.gamma(2, 40, n),
        "amount_vs_avg_30d": rng.gamma(1.5, 0.8, n),
        "hour_of_day": rng.integers(0, 24, n),
        "is_night": rng.integers(0, 2, n),
        "customer_age": rng.integers(18, 90, n),
        "distance_km": rng.random(n) * 120,
        "city_population": rng.integers(100, 1_000_000, n),
        "category": rng.choice(CATEGORIES, n),
        "is_fraud": (rng.random(n) < 0.1).astype(int),
    })
    # A card with no history, as at a card's first transaction
    for col in ["avg_amount_30d", "amount_vs_avg_30d", "seconds_since_last_txn"]:
        d.loc[0, col] = np.nan
    return d


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A tiny model trained with the same input layout, saved the way notebook 04 exports it."""
    d = gold_rows()
    X = training_prep(d)
    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3).fit(X, d["is_fraud"])
    out = tmp_path_factory.mktemp("model")
    clf.get_booster().save_model(str(out / "xgboost-model.json"))
    (out / "model_metadata.json").write_text(json.dumps({
        "threshold": 0.19, "feature_names": list(X.columns), "model_version": "1",
    }))
    return out, clf, d, X


def as_request_records(d):
    """Gold rows as a caller would send them: JSON, with missing values as null."""
    return json.loads(d.drop(columns=["is_fraud"]).to_json(orient="records"))


def test_preparation_matches_training(model_dir):
    _, _, d, X = model_dir
    records = as_request_records(d)
    prepared = np.array([inference.prepare_features(r, list(X.columns)) for r in records])
    np.testing.assert_allclose(prepared, X.to_numpy(), rtol=1e-9, equal_nan=True)


def test_scores_match_the_trained_model(model_dir):
    path, clf, d, X = model_dir
    model = inference.model_fn(str(path))
    body = json.dumps({"instances": as_request_records(d)})
    predictions = inference.predict_fn(inference.input_fn(body, "application/json"), model)

    expected = clf.predict_proba(X)[:, 1]
    np.testing.assert_allclose([p["fraud_score"] for p in predictions], expected, atol=1e-6)
    assert [p["flagged"] for p in predictions] == [bool(s >= 0.19) for s in expected]
    assert predictions[0]["transaction_id"] == "t0"


def test_accepts_a_single_transaction_and_returns_json(model_dir):
    path, _, d, _ = model_dir
    model = inference.model_fn(str(path))
    one = json.dumps(as_request_records(d)[1])
    body, content_type = inference.output_fn(inference.predict_fn(inference.input_fn(one), model))

    result = json.loads(body)["predictions"]
    assert content_type == "application/json"
    assert len(result) == 1
    assert set(result[0]) == {"transaction_id", "fraud_score", "flagged", "threshold", "model_version", "top_reasons"}


def test_top_reasons_are_the_largest_positive_shap_values(model_dir):
    path, clf, d, X = model_dir
    model = inference.model_fn(str(path))
    predictions = inference.predict_fn(as_request_records(d), model)

    contribs = clf.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    for p, c in zip(predictions, contribs):
        reasons = p["top_reasons"]
        assert len(reasons) <= inference.TOP_REASONS
        assert all(r["contribution"] > 0 for r in reasons)
        assert [r["contribution"] for r in reasons] == sorted((r["contribution"] for r in reasons), reverse=True)
        expected = [X.columns[i] for i in np.argsort(c)[::-1][: len(reasons)]]
        assert [r["feature"] for r in reasons] == expected


def test_rejects_non_json(model_dir):
    with pytest.raises(ValueError):
        inference.input_fn("amount=10", "text/csv")
