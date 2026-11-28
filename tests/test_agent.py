"""Tests for the fraud analyst agent (agent/fraud_agent.py), with Bedrock and SageMaker replaced by fakes.

They check the parts that must hold whatever the language model says: labels never reach the agent,
the tool loop works, and the automated checks catch policy breaches and invented reasons.
"""

import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fraud_agent", ROOT / "agent" / "fraud_agent.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

TRANSACTION = {
    "transaction_id": "t1", "amount": 912.4, "txn_count_1h": 1, "txn_count_24h": 3,
    "seconds_since_last_txn": 300.0, "avg_amount_30d": 41.7, "amount_vs_avg_30d": 21.9, "hour_of_day": 2,
    "is_night": 1, "customer_age": 34, "distance_km": 60.2, "city_population": 15000,
    "category": "shopping_net", "is_fraud": 1, "local_score": 0.99, "local_flagged": 1,
}
PREDICTION = {
    "transaction_id": "t1", "fraud_score": 0.97, "flagged": True, "threshold": 0.19, "model_version": "1",
    "top_reasons": [{"feature": "log_amount", "value": 6.8, "contribution": 4.1},
                    {"feature": "amount_vs_avg_30d", "value": 21.9, "contribution": 1.2}],
}


class FakeRuntime:
    def __init__(self):
        self.bodies = []

    def invoke_endpoint(self, **kwargs):
        self.bodies.append(json.loads(kwargs["Body"]))
        return {"Body": io.BytesIO(json.dumps({"predictions": [PREDICTION]}).encode())}


class FakeBedrock:
    """Replays a fixed conversation: ask for both tools, then give a final JSON answer."""

    def __init__(self, final_reply):
        self.final_reply = final_reply
        self.turn = 0
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        self.turn += 1
        if self.turn == 1:
            return {"stopReason": "tool_use", "output": {"message": {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "a", "name": "score_transaction", "input": {"transaction_id": "t1"}}},
                {"toolUse": {"toolUseId": "b", "name": "get_card_activity", "input": {"transaction_id": "t1"}}},
            ]}}}
        return {"stopReason": "end_turn", "output": {"message": {"role": "assistant", "content": [
            {"text": json.dumps(self.final_reply)}]}}}


def make_tools():
    runtime = FakeRuntime()
    return agent.FraudTools([TRANSACTION], runtime, "fraud-detector"), runtime


def test_labels_never_reach_the_endpoint_or_the_agent():
    tools, runtime = make_tools()
    tools.run("score_transaction", {"transaction_id": "t1"})
    activity = tools.run("get_card_activity", {"transaction_id": "t1"})
    for hidden in agent.HIDDEN_FIELDS:
        assert hidden not in runtime.bodies[0]
        assert hidden not in activity
    assert activity["minutes_since_last_txn"] == 5.0
    assert activity["has_history"] is True


def test_full_loop_with_a_faithful_policy_following_reply():
    tools, _ = make_tools()
    bedrock = FakeBedrock({"recommendation": "BLOCK",
                           "summary": "The amount is 22 times the card's usual spend, at 2am.",
                           "cited_features": ["amount_vs_avg_30d", "hour_of_day", "fraud_score"]})
    reply, seen = agent.run_agent(bedrock, "model", tools, "t1")
    checks = agent.check_reply(reply, seen)

    assert tools.calls == ["score_transaction", "get_card_activity"]
    assert checks["used_model"] and checks["follows_policy"] and checks["faithful"]
    conversation = bedrock.requests[1]["messages"]  # user question, tool requests, tool results, final answer
    tool_results = conversation[2]["content"]
    assert {r["toolResult"]["toolUseId"] for r in tool_results} == {"a", "b"}


def test_checks_catch_a_policy_breach_and_an_invented_reason():
    tools, _ = make_tools()
    bedrock = FakeBedrock({"recommendation": "ALLOW",
                           "summary": "Customer travels often.",
                           "cited_features": ["travel_history"]})
    reply, seen = agent.run_agent(bedrock, "model", tools, "t1")
    checks = agent.check_reply(reply, seen)

    assert checks["follows_policy"] is False and checks["expected_action"] == "BLOCK"
    assert checks["faithful"] is False and checks["unsupported_citations"] == ["travel_history"]


@pytest.mark.parametrize("score,expected", [(0.95, "BLOCK"), (0.5, "REVIEW"), (0.19, "REVIEW"), (0.1, "ALLOW")])
def test_policy_bands(score, expected):
    assert agent.policy_action(score, 0.19) == expected


def test_parse_reply_rejects_an_invalid_recommendation():
    with pytest.raises(ValueError):
        agent.parse_reply('{"recommendation": "MAYBE", "summary": "?"}')
    assert agent.parse_reply('Sure:\n{"recommendation": "REVIEW", "summary": "x"}')["cited_features"] == []
