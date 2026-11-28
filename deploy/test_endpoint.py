"""Send the sample transactions to the endpoint and compare with the scores from Databricks.

    python test_endpoint.py --bucket <your-bucket>

Reads sample_transactions.json from s3://<bucket>/<prefix>/ (written by notebook 04). Each sample
carries the score the model gave it in Databricks (local_score); the endpoint must return the same
score and the same alert decision. Exits with an error if any differ.
"""

import argparse
import json
import sys

import boto3

TOLERANCE = 1e-4
SAMPLE_FILE = "sample_transactions.json"
DROP_FIELDS = {"is_fraud", "local_score", "local_flagged"}  # answers, not inputs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="models/fraud_detector_v1")
    p.add_argument("--endpoint-name", default="fraud-detector")
    p.add_argument("--region", default="ap-southeast-2")
    return p.parse_args()


def main():
    args = parse_args()
    session = boto3.Session(region_name=args.region)
    body = session.client("s3").get_object(Bucket=args.bucket, Key=f"{args.prefix}/{SAMPLE_FILE}")["Body"].read()
    samples = json.loads(body)

    request = [{k: v for k, v in s.items() if k not in DROP_FIELDS} for s in samples]
    response = session.client("sagemaker-runtime").invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps({"instances": request}),
    )
    predictions = json.loads(response["Body"].read())["predictions"]

    print(f"{'transaction':<34} {'fraud':>5} {'databricks':>10} {'endpoint':>9} {'flagged':>7}  match")
    mismatches = 0
    for s, p in zip(samples, predictions):
        same = abs(p["fraud_score"] - s["local_score"]) <= TOLERANCE and p["flagged"] == bool(s["local_flagged"])
        mismatches += not same
        print(f"{str(s['transaction_id']):<34} {s['is_fraud']:>5} {s['local_score']:>10.4f} "
              f"{p['fraud_score']:>9.4f} {str(p['flagged']):>7}  {'yes' if same else 'NO'}")

    if mismatches:
        sys.exit(f"{mismatches} of {len(samples)} predictions differ from Databricks")
    print(f"All {len(samples)} predictions match Databricks.")


if __name__ == "__main__":
    main()
