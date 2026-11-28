"""Deploy the champion fraud model to a serverless SageMaker endpoint.

Run in AWS CloudShell (Sydney region), from the repository's deploy/ folder:

    pip install -q "sagemaker>=2.230,<3"
    python deploy_endpoint.py --bucket <your-bucket> --role <role-arn>

It expects xgboost-model.json and model_metadata.json under s3://<bucket>/<prefix>/ (uploaded from
notebook 04). It packages them as model.tar.gz, then creates a serverless endpoint running
inference.py in the SageMaker XGBoost 3.0-5 container. A serverless endpoint costs nothing while
idle; delete it with delete_endpoint.py when finished.
"""

import argparse
import os
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.xgboost import XGBoostModel

MODEL_FILES = ["xgboost-model.json", "model_metadata.json"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True, help="S3 bucket holding the exported model files")
    p.add_argument("--role", required=True, help="ARN of the SageMaker execution role")
    p.add_argument("--prefix", default="models/fraud_detector_v1", help="S3 folder with the model files")
    p.add_argument("--endpoint-name", default="fraud-detector")
    p.add_argument("--region", default="ap-southeast-2")
    p.add_argument("--memory-mb", type=int, default=2048, help="Serverless memory: 1024 to 6144, in steps of 1024")
    p.add_argument("--max-concurrency", type=int, default=2)
    return p.parse_args()


def package_model(s3, bucket, prefix):
    """Download the model files, bundle them as model.tar.gz, and upload the bundle next to them."""
    with tempfile.TemporaryDirectory() as tmp:
        for name in MODEL_FILES:
            s3.download_file(bucket, f"{prefix}/{name}", os.path.join(tmp, name))
        archive = os.path.join(tmp, "model.tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            for name in MODEL_FILES:
                tar.add(os.path.join(tmp, name), arcname=name)
        key = f"{prefix}/model.tar.gz"
        s3.upload_file(archive, bucket, key)
    return f"s3://{bucket}/{key}"


def main():
    args = parse_args()
    boto_session = boto3.Session(region_name=args.region)
    session = sagemaker.Session(boto_session=boto_session, default_bucket=args.bucket)

    model_data = package_model(boto_session.client("s3"), args.bucket, args.prefix)
    print("Packaged model:", model_data)

    model = XGBoostModel(
        model_data=model_data,
        role=args.role,
        entry_point="inference.py",
        source_dir=os.path.dirname(os.path.abspath(__file__)),
        framework_version="3.0-5",
        py_version="py3",
        sagemaker_session=session,
    )
    print(f"Deploying serverless endpoint '{args.endpoint_name}' (this takes about 5-10 minutes)...")
    model.deploy(
        endpoint_name=args.endpoint_name,
        serverless_inference_config=ServerlessInferenceConfig(
            memory_size_in_mb=args.memory_mb, max_concurrency=args.max_concurrency
        ),
    )
    print(f"Endpoint '{args.endpoint_name}' is in service. Test it with test_endpoint.py.")


if __name__ == "__main__":
    main()
