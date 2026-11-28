"""Fraud analyst assistant: a Claude agent on Amazon Bedrock that explains flagged transactions.

The agent never decides on its own. For each transaction it:
  1. calls the live SageMaker endpoint (tool: score_transaction) for the fraud score and its top reasons,
  2. looks up the card's recent behaviour (tool: get_card_activity),
  3. recommends BLOCK, REVIEW or ALLOW under a fixed policy, explaining why in plain English,
and a human analyst approves or overrides the recommendation. Every case is logged.

Two modes, run from AWS CloudShell in the deploy region:

    python fraud_agent.py --bucket <bucket> review --transaction-id <id>   # one case, analyst decides
    python fraud_agent.py --bucket <bucket> evaluate                       # all sample cases, automated checks

The sample transactions come from s3://<bucket>/<prefix>/sample_transactions.json (notebook 04). Their
labels and Databricks scores are removed before anything reaches the agent.
"""

import argparse
import datetime as dt
import getpass
import json
import re
import sys

import boto3

DEFAULT_MODEL_ID = "au.anthropic.claude-haiku-4-5-20251001-v1:0"  # Claude Haiku 4.5, processed in Australia
BLOCK_SCORE = 0.90
MAX_TURNS = 6
HIDDEN_FIELDS = {"is_fraud", "local_score", "local_flagged"}  # answers: never shown to the agent

ACTIVITY_FIELDS = [
    "amount", "category", "hour_of_day", "is_night", "avg_amount_30d", "amount_vs_avg_30d",
    "txn_count_1h", "txn_count_24h", "minutes_since_last_txn", "has_history", "customer_age",
]

SYSTEM_PROMPT = f"""You are a fraud analyst assistant at a bank. You help a human analyst decide what to do with a
card transaction. You do not make the final decision; the analyst does.

For every transaction:
1. Call score_transaction to get the fraud model's score, alert flag, threshold and top reasons.
2. Call get_card_activity to see how this card normally behaves.
3. Recommend exactly one action, following this policy strictly:
   - BLOCK  if fraud_score >= {BLOCK_SCORE}
   - REVIEW if fraud_score >= threshold and below {BLOCK_SCORE}
   - ALLOW  if fraud_score < threshold
4. Explain the recommendation in two or three plain-English sentences an analyst can read in ten seconds.
   Base the explanation only on facts returned by the tools. Never invent facts, merchants or history.

Reply with JSON only, no other text:
{{"recommendation": "BLOCK" | "REVIEW" | "ALLOW",
  "summary": "<two or three sentences>",
  "cited_features": ["<names exactly as they appear in the tool results>"]}}"""

TOOLS = [
    {"toolSpec": {
        "name": "score_transaction",
        "description": "Score a card transaction with the live fraud model. Returns fraud_score (0 to 1), "
                       "flagged, threshold, model_version and top_reasons: the model inputs that pushed the "
                       "score furthest towards fraud.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"transaction_id": {"type": "string"}},
                                 "required": ["transaction_id"]}},
    }},
    {"toolSpec": {
        "name": "get_card_activity",
        "description": "The card's behaviour around this transaction: amount, category, time of day, the card's "
                       "30-day average spend, the ratio of this amount to it, recent transaction counts and the "
                       "minutes since the previous transaction.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"transaction_id": {"type": "string"}},
                                 "required": ["transaction_id"]}},
    }},
]


def policy_action(score, threshold):
    if score >= BLOCK_SCORE:
        return "BLOCK"
    return "REVIEW" if score >= threshold else "ALLOW"


class FraudTools:
    """The agent's tools, backed by the sample transactions and the SageMaker endpoint."""

    def __init__(self, transactions, runtime, endpoint_name):
        self.transactions = {t["transaction_id"]: t for t in transactions}
        self.runtime = runtime
        self.endpoint_name = endpoint_name
        self.calls = []

    def _transaction(self, transaction_id):
        if transaction_id not in self.transactions:
            raise KeyError(f"Unknown transaction_id {transaction_id!r}")
        return {k: v for k, v in self.transactions[transaction_id].items() if k not in HIDDEN_FIELDS}

    def score_transaction(self, transaction_id):
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(self._transaction(transaction_id)),
        )
        return json.loads(response["Body"].read())["predictions"][0]

    def get_card_activity(self, transaction_id):
        t = self._transaction(transaction_id)
        seconds = t.get("seconds_since_last_txn")
        activity = dict(t)
        activity["minutes_since_last_txn"] = None if seconds is None else round(seconds / 60, 1)
        activity["has_history"] = t.get("avg_amount_30d") is not None
        return {k: activity.get(k) for k in ACTIVITY_FIELDS}

    def run(self, name, tool_input):
        self.calls.append(name)
        try:
            if name == "score_transaction":
                return self.score_transaction(tool_input["transaction_id"])
            if name == "get_card_activity":
                return self.get_card_activity(tool_input["transaction_id"])
            return {"error": f"Unknown tool {name}"}
        except Exception as e:  # report tool failures to the model instead of crashing
            return {"error": str(e)}


def parse_reply(text):
    """Pull the JSON object out of the model's reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in reply: {text[:200]}")
    reply = json.loads(match.group(0))
    if reply.get("recommendation") not in {"BLOCK", "REVIEW", "ALLOW"}:
        raise ValueError(f"Invalid recommendation: {reply.get('recommendation')!r}")
    reply.setdefault("cited_features", [])
    return reply


def run_agent(bedrock, model_id, tools, transaction_id):
    """Run the tool-use loop for one transaction. Returns the parsed reply and the tool results seen."""
    messages = [{"role": "user", "content": [{"text": f"Review transaction {transaction_id}."}]}]
    seen = {}
    for _ in range(MAX_TURNS):
        response = bedrock.converse(
            modelId=model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig={"tools": TOOLS},
            inferenceConfig={"maxTokens": 800, "temperature": 0},
        )
        message = response["output"]["message"]
        messages.append(message)
        if response["stopReason"] != "tool_use":
            text = "".join(block.get("text", "") for block in message["content"])
            return parse_reply(text), seen

        results = []
        for block in message["content"]:
            if "toolUse" in block:
                use = block["toolUse"]
                result = tools.run(use["name"], use["input"])
                seen[use["name"]] = result
                results.append({"toolResult": {"toolUseId": use["toolUseId"], "content": [{"json": result}]}})
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"No final answer after {MAX_TURNS} turns")


def check_reply(reply, seen):
    """Automated checks on one agent reply: did it use the model, follow policy, and stay faithful?"""
    score = seen.get("score_transaction", {})
    activity = seen.get("get_card_activity", {})
    allowed = {r["feature"] for r in score.get("top_reasons", [])} | set(activity) | {
        "fraud_score", "threshold", "flagged"}
    cited = set(reply.get("cited_features", []))

    checks = {"used_model": "fraud_score" in score}
    if checks["used_model"]:
        expected = policy_action(score["fraud_score"], score["threshold"])
        checks["follows_policy"] = reply["recommendation"] == expected
        checks["expected_action"] = expected
    else:
        checks["follows_policy"] = False
        checks["expected_action"] = None
    checks["faithful"] = bool(cited) and cited <= allowed
    checks["unsupported_citations"] = sorted(cited - allowed)
    return checks


def log_decision(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_transactions(s3, bucket, prefix):
    body = s3.get_object(Bucket=bucket, Key=f"{prefix}/sample_transactions.json")["Body"].read()
    return json.loads(body)


def review(args, bedrock, tools, labels):
    reply, seen = run_agent(bedrock, args.model_id, tools, args.transaction_id)
    checks = check_reply(reply, seen)
    score = seen.get("score_transaction", {})

    print(f"\nTransaction {args.transaction_id}")
    print(f"  Model score: {score.get('fraud_score')} (threshold {score.get('threshold')}, "
          f"version {score.get('model_version')})")
    print(f"  Assistant recommends: {reply['recommendation']}")
    print(f"  Why: {reply['summary']}")
    print(f"  Checks: policy {'ok' if checks['follows_policy'] else 'VIOLATED'}, "
          f"faithful {'ok' if checks['faithful'] else 'NO ' + str(checks['unsupported_citations'])}")

    choice = input("\nAnalyst decision: [a]pprove, or type BLOCK / REVIEW / ALLOW to override: ").strip()
    decision = reply["recommendation"] if choice.lower() in {"a", "approve", ""} else choice.upper()
    if decision not in {"BLOCK", "REVIEW", "ALLOW"}:
        sys.exit(f"Unrecognised decision {choice!r}; nothing logged.")

    log_decision(args.log, {
        "logged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "transaction_id": args.transaction_id,
        "model_score": score.get("fraud_score"),
        "model_version": score.get("model_version"),
        "top_reasons": score.get("top_reasons"),
        "agent_recommendation": reply["recommendation"],
        "agent_summary": reply["summary"],
        "checks": checks,
        "analyst": getpass.getuser(),
        "analyst_decision": decision,
        "overridden": decision != reply["recommendation"],
        "actual_fraud": labels.get(args.transaction_id),
    })
    print(f"Logged: analyst decision {decision} -> {args.log}")


def evaluate(args, bedrock, tools, labels, transactions):
    rows = []
    for t in transactions:
        tid = t["transaction_id"]
        try:
            reply, seen = run_agent(bedrock, args.model_id, tools, tid)
            checks = check_reply(reply, seen)
            score = seen.get("score_transaction", {})
            rows.append({
                "transaction_id": tid, "actual_fraud": labels[tid],
                "fraud_score": score.get("fraud_score"), "model_flagged": score.get("flagged"),
                "recommendation": reply["recommendation"], "summary": reply["summary"],
                "valid_json": True, **checks,
            })
        except Exception as e:
            rows.append({"transaction_id": tid, "actual_fraud": labels[tid], "valid_json": False,
                         "used_model": False, "follows_policy": False, "faithful": False, "error": str(e)})

    print(f"\n{'transaction':<34} {'fraud':>5} {'score':>7} {'agent':>7} {'policy':>7} {'faithful':>8}")
    for r in rows:
        print(f"{r['transaction_id']:<34} {r['actual_fraud']:>5} {r.get('fraud_score', 0) or 0:>7.4f} "
              f"{r.get('recommendation', 'ERROR'):>7} {str(r['follows_policy']):>7} {str(r['faithful']):>8}")

    n = len(rows)
    agrees_with_model = sum(
        r.get("model_flagged") is not None and (r.get("recommendation") in {"BLOCK", "REVIEW"}) == r["model_flagged"]
        for r in rows)
    summary = {
        "cases": n,
        "valid_json": sum(r["valid_json"] for r in rows) / n,
        "used_model": sum(r["used_model"] for r in rows) / n,
        "follows_policy": sum(r["follows_policy"] for r in rows) / n,
        "faithful_explanations": sum(r["faithful"] for r in rows) / n,
        "agrees_with_model_flag": agrees_with_model / n,
        "model_id": args.model_id,
    }
    print("\n" + "\n".join(f"{k}: {v:.0%}" if isinstance(v, float) else f"{k}: {v}" for k, v in summary.items()))
    with open(args.eval_output, "w") as f:
        json.dump({"summary": summary, "cases": rows}, f, indent=2)
    print(f"Saved the full evaluation to {args.eval_output}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="models/fraud_detector_v1")
    p.add_argument("--endpoint-name", default="fraud-detector")
    p.add_argument("--region", default="ap-southeast-2")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Bedrock model or inference profile ID")
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("review", help="Review one transaction; the analyst approves or overrides")
    r.add_argument("--transaction-id", required=True)
    r.add_argument("--log", default="decisions.jsonl")
    e = sub.add_parser("evaluate", help="Run every sample transaction through the agent and check the replies")
    e.add_argument("--eval-output", default="agent_evaluation.json")
    args = p.parse_args()

    session = boto3.Session(region_name=args.region)
    transactions = load_transactions(session.client("s3"), args.bucket, args.prefix)
    labels = {t["transaction_id"]: t.get("is_fraud") for t in transactions}
    tools = FraudTools(transactions, session.client("sagemaker-runtime"), args.endpoint_name)
    bedrock = session.client("bedrock-runtime")

    if args.mode == "review":
        review(args, bedrock, tools, labels)
    else:
        evaluate(args, bedrock, tools, labels, transactions)


if __name__ == "__main__":
    main()
