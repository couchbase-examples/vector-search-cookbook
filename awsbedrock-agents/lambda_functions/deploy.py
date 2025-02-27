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
        time.sleep(10)
        
        return True
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"Lambda function {function_name} does not exist. No need to delete.")
        return False
    except Exception as e:
        print(f"Error deleting Lambda function {function_name}: {str(e)}")
        return False

def upload_to_s3(zip_file, aws_region, bucket_name=None):
    """Upload zip file to S3 and return S3 location"""
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Create bucket if it doesn't exist
    if bucket_name is None:
        # Generate a unique bucket name
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        bucket_name = f"lambda-deployment-{account_id}-{aws_region}"
    
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Using existing S3 bucket: {bucket_name}")
    except Exception:
        try:
            if aws_region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': aws_region}
                )
            print(f"Created S3 bucket: {bucket_name}")
        except Exception as e:
            print(f"Error creating S3 bucket: {str(e)}")
            # Try with a more unique name
            bucket_name = f"lambda-deployment-{account_id}-{aws_region}-{uuid.uuid4().hex[:8]}"
            if aws_region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': aws_region}
                )
            print(f"Created S3 bucket with unique name: {bucket_name}")
    
    # Upload zip file to S3
    key = f"lambda/{os.path.basename(zip_file)}"
    s3_client.upload_file(zip_file, bucket_name, key)
    print(f"Uploaded {zip_file} to s3://{bucket_name}/{key}")
    
    return {
        'S3Bucket': bucket_name,
        'S3Key': key
    }

def create_lambda_function(function_name, handler, role_arn, zip_file, aws_region):
    """Create or update Lambda function"""

    # Configure the client with increased timeouts
    config = Config(
        connect_timeout=30,  # 30 seconds
        read_timeout=60,     # 60 seconds
        retries={'max_attempts': 3}
    )
    
    lambda_client = boto3.client('lambda', region_name=aws_region, config=config)
    
    # Wait for role to be ready
    print(f"Waiting for role {role_arn} to be ready...")
    time.sleep(10)  # IAM role propagation delay
    
    # Create new function
    print(f"Creating function {function_name}...")
    try:
        with open(zip_file, 'rb') as f:
            zip_content = f.read()
            zip_size_mb = len(zip_content) / (1024 * 1024)
            print(f"Zip file size: {zip_size_mb:.2f} MB")
            
            # If zip file is larger than 50MB, upload to S3 first
            if zip_size_mb > 50:
                print("Zip file is larger than 50MB. Uploading to S3 first...")
                s3_location = upload_to_s3(zip_file, aws_region)
                
                lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.9',
                    Role=role_arn,
                    Handler=handler,
                    Code=s3_location,
                    Timeout=60,  # Increased timeout
                    MemorySize=512,  # Increased memory
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
                # For smaller files, use direct upload
                lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.9',
                    Role=role_arn,
                    Handler=handler,
                    Code={'ZipFile': zip_content},
                    Timeout=60,  # Increased timeout
                    MemorySize=512,  # Increased memory
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
    except ConnectionClosedError as e:
        print(f"Connection closed error: {str(e)}")
        raise
    except ServiceNotInRegionError as e:
        print(f"Service not in region: {str(e)}")
        raise
    except ClientError as e:
        print(f"Client error: {str(e)}")
        raise
    except Exception as e:
        print(f"Error creating Lambda function: {str(e)}")
        if "Connection was closed" in str(e):
            print("Connection issue detected. This might be due to a large payload or network issues.")
            print("Consider uploading the Lambda code to S3 first and then creating the function from S3.")
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
        
        # # Removes Couchbase SDK files as well. Not needed for Lambda deployment.
        # # Remove unnecessary files to reduce package size
        # print("Cleaning up package to reduce size...")
        # for root, dirs, files in os.walk(build_dir):
        #     # Remove test directories
        #     for dir_name in dirs[:]:
        #         if dir_name in ['tests', 'test', '__pycache__', '.pytest_cache', 'dist-info']:
        #             shutil.rmtree(os.path.join(root, dir_name))
        #             dirs.remove(dir_name)
            
        #     # Remove unnecessary files
        #     for file_name in files:
        #         if file_name.endswith(('.pyc', '.pyo', '.pyd', '.so', '.c', '.h', '.cpp')):
        #             os.remove(os.path.join(root, file_name))
   
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
        
        # Initialize AWS clients
        iam = boto3.client('iam', region_name=aws_region)
        
        # Create IAM role for Lambda
        role_name = 'bedrock_agent_lambda_role'
        
        try:
            role = iam.get_role(RoleName=role_name)
        except iam.exceptions.NoSuchEntityException:
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
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            # Attach Bedrock invoke policy
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
                            "lambda:InvokeFunction"
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
        
        role_arn = role['Role']['Arn']
        
        # Delete existing Lambda functions
        delete_lambda_function('bedrock_agent_researcher', aws_region)
        delete_lambda_function('bedrock_agent_writer', aws_region)
        
        # Package and deploy researcher function
        researcher_zip = package_function('bedrock_agent_researcher')
        create_lambda_function(
            'bedrock_agent_researcher',
            'lambda_function.lambda_handler',
            role_arn,
            researcher_zip,
            aws_region
        )
        
        # Package and deploy writer function
        writer_zip = package_function('bedrock_agent_writer')
        create_lambda_function(
            'bedrock_agent_writer',
            'lambda_function.lambda_handler',
            role_arn,
            writer_zip,
            aws_region
        )
        
        print("Deployment complete!")
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
