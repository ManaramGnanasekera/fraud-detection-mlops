"""Delete the endpoint, its endpoint configuration and its model, so nothing is left running.

    python delete_endpoint.py
"""

import argparse

import boto3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint-name", default="fraud-detector")
    p.add_argument("--region", default="ap-southeast-2")
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    endpoint = sm.describe_endpoint(EndpointName=args.endpoint_name)
    config_name = endpoint["EndpointConfigName"]
    config = sm.describe_endpoint_config(EndpointConfigName=config_name)
    model_names = [v["ModelName"] for v in config["ProductionVariants"]]

    sm.delete_endpoint(EndpointName=args.endpoint_name)
    sm.delete_endpoint_config(EndpointConfigName=config_name)
    for name in model_names:
        sm.delete_model(ModelName=name)
    print(f"Deleted endpoint {args.endpoint_name}, config {config_name} and model(s) {', '.join(model_names)}")


if __name__ == "__main__":
    main()
