from datetime import datetime
import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "oedi-data-lake"
PREFIX = "pvdaq/"

def discover(limit=40):
    """List a bounded sample of public PVDAQ objects through anonymous S3 access."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, retries={"max_attempts":2}))
    page = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=min(limit,1000))
    return [{"key":x["Key"],"size":x["Size"]} for x in page.get("Contents",[]) if x["Key"].endswith((".csv",".parquet",".json"))]

def acquire_sample(directory: str):
    import os
    os.makedirs(directory, exist_ok=True)
    objects = discover(100)
    # Metadata files provide site/system and metric dictionaries. Limit transfer to 3 MB total.
    selected = [x for x in objects if x["key"].endswith(".json")][:5]
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, retries={"max_attempts":2}))
    downloaded = []
    total=0
    for item in selected:
        if total + item["size"] > 3_000_000: continue
        path=os.path.join(directory, os.path.basename(item["key"]))
        s3.download_file(BUCKET,item["key"],path)
        total += item["size"]
        downloaded.append(path)
    return objects, downloaded
