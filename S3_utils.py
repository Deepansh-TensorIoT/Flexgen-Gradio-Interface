import boto3
import os
import logging
import pandas as pd
logger = logging.getLogger(__name__)

s3_bucket = "flexgen-modbus-register-extraction"

def initialize_s3_client():
    try:
        aws_access_key_id = "AKIA6KL2BTNPZ7LGCFAP"
        aws_secret_access_key = "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"
        region_name = 'us-east-1'

        s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
        )
        return s3
    except KeyError:
        print("AWS credentials not found in environment variables.")
        return None
    

def save_to_s3(vendor_name, model_input, version_input, dir_path, folder_name):
    logger.info("Save to S3 started! for %s", dir_path)
    s3 = initialize_s3_client()
    bucket_name = s3_bucket
    try:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                filepath = os.path.join(root, file)
                s3_key = f"{vendor_name}/{model_input}/{version_input}/{folder_name}/{file}"
                # Upload the file to S3
                s3.upload_file(filepath, bucket_name, s3_key)
                logger.info("file upload for \"%s\" complete!!", file)
    except Exception as e:
        logger.error(e)

def read_latest_csv_from_s3(vendor, model, version):
    logger.info("read_latest_csv_from_s3(%s,%s,%s)", vendor, model, version)
    try:
        s3 = initialize_s3_client()
        bucket_name = s3_bucket
        prefix = f"{vendor}/{model}/{version}/final_output/"

        # Get a list of objects in the prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' not in response:
            raise ValueError(f"No CSV files found in S3 bucket for {prefix}")
        
        # Filter for CSV files and sort by last modified time
        csv_files = [obj for obj in response['Contents']]
        csv_files.sort(key=lambda x: x['LastModified'], reverse=True)
        
        latest_csv_key = csv_files[0]['Key']

        # Download the latest CSV file to a temporary file
        tmp_file = '/tmp/latest_csv.csv'
        s3.download_file(bucket_name, latest_csv_key, tmp_file)

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(tmp_file)
        os.remove(tmp_file)
        return df 
    except Exception as e:
        logger.error(e)
        return None
    