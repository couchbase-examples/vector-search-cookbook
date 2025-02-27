import json
import os
import shutil
import subprocess
import time

import boto3
from botocore.config import Config
from botocore.exceptions import (ClientError, ConnectionClosedError,
                                 ServiceNotInRegionError)


def create_lambda_function(function_name, handler, role_arn, zip_file, aws_region, max_retries=3):
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
    
    for attempt in range(max_retries):
        try:
            try:
                # Try to get function
                lambda_client.get_function(FunctionName=function_name)
                
                # Update existing function
                print(f"Updating function {function_name}...")
                with open(zip_file, 'rb') as f:
                    lambda_client.update_function_code(
                        FunctionName=function_name,
                        ZipFile=f.read()
                    )
                
                # Update environment variables
                lambda_client.update_function_configuration(
                    FunctionName=function_name,
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
                break
            except lambda_client.exceptions.ResourceNotFoundException:
                # Create new function
                print(f"Lambda doesn't exist. Creating function {function_name}...")
                try:
                    with open(zip_file, 'rb') as f:
                        zip_content = f.read()
                        print(f"Zip file size: {len(zip_content) / (1024 * 1024):.2f} MB")
                        
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
                except Exception as e:
                    print(f"Error creating Lambda function: {str(e)}")
                    if "Connection was closed" in str(e):
                        print("Connection issue detected. This might be due to a large payload or network issues.")
                        print("Consider uploading the Lambda code to S3 first and then creating the function from S3.")
                    raise
                break
        except ConnectionClosedError as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(5 * (attempt + 1))  # Exponential backoff
        except ServiceNotInRegionError as e:
            print(f"Service not in region: {str(e)}")
            raise
        except ClientError as e:
            print(f"Client error: {str(e)}")
            raise
        except Exception as e:
            print(f"Unknown error: {str(e)}")
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
        subprocess.check_call([
            'pip', 'install',
            '--no-cache-dir',  # Don't use the pip cache
            '--upgrade',       # Upgrade packages if needed
            '--quiet',         # Hide output
            '-r', os.path.join(current_dir, 'requirements.txt'),
            '-t', build_dir
        ])
        
        # Remove unnecessary files to reduce package size
        print("Cleaning up package to reduce size...")
        for root, dirs, files in os.walk(build_dir):
            # Remove test directories
            for dir_name in dirs[:]:
                if dir_name in ['tests', 'test', '__pycache__', '.pytest_cache', 'dist-info']:
                    shutil.rmtree(os.path.join(root, dir_name))
                    dirs.remove(dir_name)
            
            # Remove unnecessary files
            for file_name in files:
                if file_name.endswith(('.pyc', '.pyo', '.pyd', '.so', '.c', '.h', '.cpp')):
                    os.remove(os.path.join(root, file_name))
        
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
                "Statement": [{
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock-runtime:InvokeModel",
                        "bedrock-agent:*",
                        "bedrock-agent-runtime:*"
                    ],
                    "Resource": "*"
                }]
            }
            
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName='bedrock_invoke_policy',
                PolicyDocument=json.dumps(bedrock_policy)
            )
        
        role_arn = role['Role']['Arn']
        
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
