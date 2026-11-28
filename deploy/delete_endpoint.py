"""Delete the endpoint, its endpoint configuration and its model, so nothing is left running.

    python delete_endpoint.py
    python delete_endpoint.py --endpoint-name fraud-detector-staging
"""

import argparse

import boto3
from botocore.exceptions import ClientError


def endpoint_exists(sm, name):
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except ClientError as e:
        if "Could not find endpoint" in str(e) or e.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def delete_endpoint_resources(sm, name, wait=True):
    """Delete an endpoint with its config and model(s). Returns False if it did not exist."""
    if not endpoint_exists(sm, name):
        return False
    config_name = sm.describe_endpoint(EndpointName=name)["EndpointConfigName"]
    variants = sm.describe_endpoint_config(EndpointConfigName=config_name)["ProductionVariants"]
    model_names = [v["ModelName"] for v in variants]

    sm.delete_endpoint(EndpointName=name)
    if wait:
        sm.get_waiter("endpoint_deleted").wait(EndpointName=name)
    sm.delete_endpoint_config(EndpointConfigName=config_name)
    for model_name in model_names:
        sm.delete_model(ModelName=model_name)
    print(f"Deleted endpoint {name}, config {config_name} and model(s) {', '.join(model_names)}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint-name", default="fraud-detector")
    p.add_argument("--region", default="ap-southeast-2")
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    if not delete_endpoint_resources(sm, args.endpoint_name):
        print(f"No endpoint named {args.endpoint_name}; nothing to delete.")


if __name__ == "__main__":
    main()
