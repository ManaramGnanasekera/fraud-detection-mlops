"""Unit tests for the gold-layer fraud features.

Each test builds a tiny, hand-made silver table where the right answer can be
worked out on paper, runs the real feature query on it, and checks the result.
"""

from datetime import date, datetime

import pytest

from fraud.features import build_gold_features

SILVER_COLUMNS = [
    "transaction_id", "card_id", "transaction_ts", "amount", "category", "gender",
    "city_population", "is_fraud", "date_of_birth",
    "customer_lat", "customer_long", "merchant_lat", "merchant_long",
]


def make_silver(spark, rows):
    """Create a temporary silver view from (transaction_id, card_id, timestamp, amount) tuples."""
    full_rows = [
        (txn_id, card, ts, amount, "shopping_net", "F", 1000, 0, date(1990, 1, 1),
         -33.87, 151.21, -33.87, 151.21)
        for txn_id, card, ts, amount in rows
    ]
    spark.createDataFrame(full_rows, SILVER_COLUMNS).createOrReplaceTempView("silver_test")
    return "silver_test"


def features_by_id(spark, rows):
    source = make_silver(spark, rows)
    return {r.transaction_id: r for r in build_gold_features(spark, source).collect()}


def test_worked_example_card_burst(spark):
    """The worked example: three normal purchases, then a $900 purchase five minutes later."""
    out = features_by_id(spark, [
        ("t1", "card_a", datetime(2019, 1, 7, 9, 0), 40.0),   # Mon 09:00
        ("t2", "card_a", datetime(2019, 1, 7, 18, 0), 25.0),  # Mon 18:00
        ("t3", "card_a", datetime(2019, 1, 8, 8, 0), 60.0),   # Tue 08:00
        ("t4", "card_a", datetime(2019, 1, 8, 8, 5), 900.0),  # Tue 08:05
    ])
    big = out["t4"]

    assert big.txn_count_24h == 3          # Mon 09:00 is 23h05m earlier, so all three are inside 24h
    assert big.txn_count_1h == 1           # only Tue 08:00
    assert big.seconds_since_last_txn == 300
    assert big.avg_amount_30d == pytest.approx((40 + 25 + 60) / 3)
    assert big.amount_vs_avg_30d == pytest.approx(900 / ((40 + 25 + 60) / 3))  # about 21.6


def test_windows_never_include_the_current_transaction(spark):
    """A card's first transaction has no history: counts are 0 and history-based values are NULL."""
    out = features_by_id(spark, [("t1", "card_a", datetime(2019, 1, 7, 9, 0), 40.0)])
    first = out["t1"]

    assert first.txn_count_24h == 0
    assert first.txn_count_1h == 0
    assert first.seconds_since_last_txn is None
    assert first.avg_amount_30d is None
    assert first.amount_vs_avg_30d is None


def test_cards_are_kept_separate(spark):
    """Another card's transactions must never count towards this card's features."""
    out = features_by_id(spark, [
        ("a1", "card_a", datetime(2019, 1, 7, 9, 0), 40.0),
        ("b1", "card_b", datetime(2019, 1, 7, 9, 10), 500.0),
        ("a2", "card_a", datetime(2019, 1, 7, 9, 20), 50.0),
    ])

    assert out["a2"].txn_count_1h == 1                  # only a1, not b1
    assert out["a2"].seconds_since_last_txn == 20 * 60  # gap to a1, not to b1
    assert out["b1"].txn_count_1h == 0


def test_night_flag_and_distance(spark):
    """22:00 counts as night, 21:59 does not; a purchase at home is 0 km away."""
    out = features_by_id(spark, [
        ("n1", "card_a", datetime(2019, 1, 7, 21, 59), 10.0),
        ("n2", "card_a", datetime(2019, 1, 7, 22, 0), 10.0),
    ])

    assert out["n1"].is_night == 0
    assert out["n2"].is_night == 1
    assert out["n1"].distance_km == pytest.approx(0.0, abs=1e-6)
