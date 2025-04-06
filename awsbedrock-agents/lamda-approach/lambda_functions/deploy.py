import json
import os
import shutil
import subprocess
import time
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import (ClientError, ConnectionClosedError,
                                 ServiceNotInRegionError)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

def delete_lambda_function(function_name, aws_region):
    """Delete Lambda function if it exists"""
    lambda_client = boto3.client('lambda', region_name=aws_region)

    try:
        # Check if function exists
        lambda_client.get_function(FunctionName=function_name)

        # Delete function
        print(f"Deleting existing Lambda function: {function_name}")
        lambda_client.delete_function(FunctionName=function_name)

        # Wait for deletion to complete
        print(f"Waiting for {function_name} to be deleted...")
        time.sleep(20)  # Doubled from 10

        return True
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"Lambda function {function_name} does not exist. No need to delete.")
        return False
    except Exception as e:
        print(f"Error deleting Lambda function {function_name}: {str(e)}")
        return False



def create_lambda_function(function_name, handler, role_arn, zip_file, aws_region):
    """Create or update Lambda function with retry logic"""

    # Configure the client with increased timeouts
    config = Config(
        connect_timeout=120,  # Increased to 2 minutes
        read_timeout=300,     # Increased to 5 minutes
        retries={
            'max_attempts': 5,
            'mode': 'adaptive'  # Use adaptive retry mode for better handling of throttling
        }
    )

    lambda_client = boto3.client('lambda', region_name=aws_region, config=config)

    # Wait for role to be ready
    print(f"Waiting for role {role_arn} to be ready...")
    time.sleep(30)  # Increased to 30 seconds for IAM role propagation

    # Get zip file size
    with open(zip_file, 'rb') as f:
        zip_content = f.read()
        zip_size_mb = len(zip_content) / (1024 * 1024)
        print(f"Zip file size: {zip_size_mb:.2f} MB")

    # Always use S3 for deployment if file is larger than 10MB
    # This helps avoid connection timeouts with direct uploads
    use_s3 = zip_size_mb > 10
    
    if use_s3:
        print(f"File size ({zip_size_mb:.2f} MB) exceeds 10MB. Using S3 for deployment...")
        s3_location = upload_to_s3(zip_file, aws_region)
    
    # Retry logic with exponential backoff
    max_retries = 5
    base_delay = 5  # Start with 5 seconds delay
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Creating function {function_name} (attempt {attempt}/{max_retries})...")
            
            if use_s3:
                # Deploy from S3
                lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.9',
                    Role=role_arn,
                    Handler=handler,
                    Code=s3_location,
                    Timeout=180,  # Increased to 3 minutes
                    MemorySize=1536,  # Increased to 1.5GB
                    Environment={
                        'Variables': {
                            'CB_HOST': os.getenv('CB_HOST', 'couchbase://localhost'),
                            'CB_USERNAME': os.getenv('CB_USERNAME', 'Administrator'),
                            'CB_PASSWORD': os.getenv('CB_PASSWORD', 'password'),
                            'CB_BUCKET_NAME': os.getenv('CB_BUCKET_NAME', 'vector-search-testing'),
                            'SCOPE_NAME': os.getenv('SCOPE_NAME', 'shared'),
                            'COLLECTION_NAME': os.getenv('COLLECTION_NAME', 'bedrock'),
                            'INDEX_NAME': os.getenv('INDEX_NAME', 'vector_search_bedrock')
                        }
                    }
                )
            else:
                # Direct upload for smaller files
                lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.9',
                    Role=role_arn,
                    Handler=handler,
                    Code={'ZipFile': zip_content},
                    Timeout=180,  # Increased to 3 minutes
                    MemorySize=1536,  # Increased to 1.5GB
                    Environment={
                        'Variables': {
                            'CB_HOST': os.getenv('CB_HOST', 'couchbase://localhost'),
                            'CB_USERNAME': os.getenv('CB_USERNAME', 'Administrator'),
                            'CB_PASSWORD': os.getenv('CB_PASSWORD', 'password'),
                            'CB_BUCKET_NAME': os.getenv('CB_BUCKET_NAME', 'vector-search-testing'),
                            'SCOPE_NAME': os.getenv('SCOPE_NAME', 'shared'),
                            'COLLECTION_NAME': os.getenv('COLLECTION_NAME', 'bedrock'),
                            'INDEX_NAME': os.getenv('INDEX_NAME', 'vector_search_bedrock')
                        }
                    }
                )
            
            print(f"Successfully created function {function_name} on attempt {attempt}")
            return  # Success, exit the retry loop
            
        except ConnectionClosedError as e:
            print(f"Connection closed error on attempt {attempt}: {str(e)}")
            if attempt < max_retries:
                # Calculate exponential backoff delay with jitter
                delay = base_delay * (2 ** (attempt - 1)) + (uuid.uuid4().int % 10)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                
                # If direct upload failed, switch to S3 for next attempt
                if not use_s3:
                    print("Switching to S3 deployment for next attempt...")
                    use_s3 = True
                    s3_location = upload_to_s3(zip_file, aws_region)
            else:
                print("Maximum retry attempts reached. Deployment failed.")
                raise
                
        except (ServiceNotInRegionError, ClientError) as e:
            print(f"AWS service error on attempt {attempt}: {str(e)}")
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Maximum retry attempts reached. Deployment failed.")
                raise
                
        except Exception as e:
            print(f"Unexpected error on attempt {attempt}: {str(e)}")
            if "Connection was closed" in str(e) and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"Connection issue detected. Retrying in {delay} seconds...")
                time.sleep(delay)
                
                # Switch to S3 deployment for next attempt
                if not use_s3:
                    print("Switching to S3 deployment for next attempt...")
                    use_s3 = True
                    s3_location = upload_to_s3(zip_file, aws_region)
            else:
                print("Error creating Lambda function. Deployment failed.")
                raise

def upload_to_s3(zip_file, aws_region, bucket_name=None):
    """Upload zip file to S3 with retry logic and return S3 location"""
    # Configure the client with increased timeouts
    config = Config(
        connect_timeout=60,
        read_timeout=120,
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )
    
    s3_client = boto3.client('s3', region_name=aws_region, config=config)
    sts_client = boto3.client('sts', region_name=aws_region, config=config)

    # Create bucket if it doesn't exist
    if bucket_name is None:
        # Generate a unique bucket name
        try:
            account_id = sts_client.get_caller_identity().get('Account')
            timestamp = int(time.time())
            bucket_name = f"lambda-deployment-{account_id}-{timestamp}"
        except Exception as e:
            print(f"Error getting account ID: {str(e)}")
            # Fallback to a random name
            bucket_name = f"lambda-deployment-{uuid.uuid4().hex[:12]}"

    # Retry bucket creation with exponential backoff
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                print(f"Using existing S3 bucket: {bucket_name}")
            except Exception:
                print(f"Creating S3 bucket: {bucket_name} (attempt {attempt}/{max_retries})...")
                try:
                    if aws_region == 'us-east-1':
                        s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': aws_region}
                        )
                    print(f"Created S3 bucket: {bucket_name}")
                    
                    # Wait for bucket to be available
                    print("Waiting for bucket to be available...")
                    waiter = s3_client.get_waiter('bucket_exists')
                    waiter.wait(Bucket=bucket_name)
                except Exception as e:
                    print(f"Error creating bucket: {str(e)}")
                    if attempt < max_retries:
                        # Try with a more unique name
                        bucket_name = f"lambda-deployment-{uuid.uuid4().hex}"
                        continue
                    else:
                        raise
            
            # Upload zip file to S3 with retry logic
            key = f"lambda/{os.path.basename(zip_file)}-{uuid.uuid4().hex[:8]}"
            print(f"Uploading {zip_file} to S3 bucket {bucket_name}...")
            
            # Use multipart upload for large files
            file_size = os.path.getsize(zip_file)
            if file_size > 100 * 1024 * 1024:  # 100 MB
                print("Using multipart upload for large file...")
                transfer_config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=8 * 1024 * 1024,  # 8 MB
                    max_concurrency=10,
                    multipart_chunksize=8 * 1024 * 1024,  # 8 MB
                    use_threads=True
                )
                s3_transfer = boto3.s3.transfer.S3Transfer(client=s3_client, config=transfer_config)
                s3_transfer.upload_file(zip_file, bucket_name, key)
            else:
                s3_client.upload_file(zip_file, bucket_name, key)
                
            print(f"Successfully uploaded {zip_file} to s3://{bucket_name}/{key}")
            
            return {
                'S3Bucket': bucket_name,
                'S3Key': key
            }
            
        except Exception as e:
            print(f"S3 operation failed on attempt {attempt}: {str(e)}")
            if attempt < max_retries:
                delay = 2 ** (attempt - 1)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Maximum retry attempts reached. S3 upload failed.")
                raise

def package_function(function_name):
    """Package Lambda function with dependencies"""
    try:
        # Create build directory
        build_dir = f'build_{function_name}'
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)

        # Install dependencies
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        print(f"Installing dependencies for {function_name}...")

        # Use pip directly to install dependencies to the build directory
        subprocess.check_call([
            'pip', 'install',
            '--no-cache-dir',  # Don't use the pip cache
            '--upgrade',       # Upgrade packages if needed
            '--quiet',         # Hide output (except for errors)
            '--target', build_dir,  # Install to the build directory
            '-r', os.path.join(current_dir, 'requirements.txt')
        ])

        # Copy function code
        shutil.copy(
            os.path.join(current_dir, f'{function_name}.py'),
            os.path.join(build_dir, 'lambda_function.py')
        )

        # Create zip file
        zip_file = f'{function_name}.zip'
        shutil.make_archive(function_name, 'zip', build_dir)

        return zip_file
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"Error packaging function {function_name}: {str(e)}")
        raise

def main():
    try:
        # Load AWS credentials from environment
        aws_account_id = os.getenv('AWS_ACCOUNT_ID')
        if not aws_account_id:
            raise ValueError("AWS_ACCOUNT_ID environment variable is required")

        aws_region = os.getenv('AWS_REGION', 'us-east-1')

        # Configure the client with increased timeouts
        config = Config(
            connect_timeout=60,
            read_timeout=120,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )

        # Initialize AWS clients
        iam = boto3.client('iam', region_name=aws_region, config=config)

        # Create IAM role for Lambda
        role_name = 'bedrock_agent_lambda_role'

        try:
            role = iam.get_role(RoleName=role_name)
            print(f"Using existing IAM role: {role_name}")
        except iam.exceptions.NoSuchEntityException:
            print(f"Creating new IAM role: {role_name}")
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }

            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )

            # Attach basic Lambda execution policy
            print("Attaching Lambda execution policy...")
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )

            # Attach Bedrock invoke policy
            print("Attaching Bedrock invoke policy...")
            bedrock_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "bedrock:InvokeModel",
                            "bedrock-runtime:InvokeModel",
                            "bedrock-agent:*",
                            "bedrock-agent-runtime:*",
                            "lambda:InvokeFunction",
                            "s3:GetObject",
                            "s3:PutObject",
                            "s3:ListBucket"
                        ],
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents"
                        ],
                        "Resource": "arn:aws:logs:*:*:*"
                    }
                ]
            }

            iam.put_role_policy(
                RoleName=role_name,
                PolicyName='bedrock_invoke_policy',
                PolicyDocument=json.dumps(bedrock_policy)
            )
            
            # Wait for role policies to propagate
            print("Waiting for IAM role policies to propagate...")
            time.sleep(15)

        role_arn = role['Role']['Arn']

        # Delete existing Lambda functions
        delete_lambda_function('bedrock_agent_researcher', aws_region)
        delete_lambda_function('bedrock_agent_writer', aws_region)

        # Package and deploy researcher function
        print("\n=== Deploying researcher function ===")
        researcher_zip = package_function('bedrock_agent_researcher')
        create_lambda_function(
            'bedrock_agent_researcher',
            'lambda_function.lambda_handler',
            role_arn,
            researcher_zip,
            aws_region
        )

        # Package and deploy writer function
        print("\n=== Deploying writer function ===")
        writer_zip = package_function('bedrock_agent_writer')
        create_lambda_function(
            'bedrock_agent_writer',
            'lambda_function.lambda_handler',
            role_arn,
            writer_zip,
            aws_region
        )

        print("\nDeployment complete!")
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
