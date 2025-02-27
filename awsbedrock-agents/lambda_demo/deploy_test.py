import json
import os
import shutil
import subprocess
import time
import boto3
from botocore.config import Config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


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
    
    # Check if function exists
    try:
        lambda_client.get_function(FunctionName=function_name)
        print(f"Function {function_name} exists. Deleting...")
        lambda_client.delete_function(FunctionName=function_name)
        print(f"Waiting for {function_name} to be deleted...")
        time.sleep(10)  # Wait for deletion to complete
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"Function {function_name} does not exist. Creating new function.")
    
    # Create new function
    print(f"Creating function {function_name}...")
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
                        'CB_BUCKET_NAME': os.getenv('CB_BUCKET_NAME', 'travel-sample')
                    }
                }
            )
            print(f"Function {function_name} created successfully")
            return True
    except Exception as e:
        print(f"Error creating Lambda function: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def package_function(function_name):
    """Package Lambda function with dependencies"""
    try:
        # Create build directory
        build_dir = f'build_{function_name}'
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create requirements.txt file
        requirements_file = os.path.join(build_dir, 'requirements.txt')
        with open(requirements_file, 'w') as f:
            f.write("couchbase==4.3.5\npython-dotenv\n")
        
        print(f"Installing dependencies for {function_name}...")
        
        # Use pip directly to install dependencies to the build directory
        subprocess.check_call([
            'pip', 'install',
            '--no-cache-dir',  # Don't use the pip cache
            '--upgrade',       # Upgrade packages if needed
            '--quiet',         # Suppress output
            '--target', build_dir,  # Install to the build directory
            '-r', requirements_file,
        ])
        print("Installed dependencies to the build directory")
        
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
        import traceback
        traceback.print_exc()
        return None

def create_or_get_role(role_name, aws_region):
    """Create or get IAM role for Lambda"""
    iam = boto3.client('iam', region_name=aws_region)
    
    try:
        # Check if role exists
        role = iam.get_role(RoleName=role_name)
        print(f"Role {role_name} exists. Using existing role.")
        return role['Role']['Arn']
    except iam.exceptions.NoSuchEntityException:
        # Create role
        print(f"Role {role_name} does not exist. Creating...")
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
        
        print(f"Role {role_name} created successfully")
        return role['Role']['Arn']
    except Exception as e:
        print(f"Error creating/getting role: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def invoke_lambda(function_name, aws_region):
    """Invoke Lambda function and return response"""
    lambda_client = boto3.client('lambda', region_name=aws_region)
    
    try:
        print(f"Invoking Lambda function {function_name}...")
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse'
        )
        
        # Parse response
        payload = response['Payload'].read().decode('utf-8')
        print(f"Lambda function {function_name} invoked successfully")
        return json.loads(payload)
    except Exception as e:
        print(f"Error invoking Lambda function: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Configuration
    function_name = 'couchbase_test'
    handler = 'lambda_function.lambda_handler'
    role_name = 'couchbase_test_role'
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    
    # Create or get role
    role_arn = create_or_get_role(role_name, aws_region)
    if not role_arn:
        print("Failed to create or get role. Exiting.")
        return
    
    # Package function
    zip_file = package_function(function_name)
    if not zip_file:
        print("Failed to package function. Exiting.")
        return
    
    # Create Lambda function
    success = create_lambda_function(function_name, handler, role_arn, zip_file, aws_region)
    if not success:
        print("Failed to create Lambda function. Exiting.")
        return
    
    # Wait for Lambda function to be ready
    print("Waiting for Lambda function to be ready...")
    time.sleep(10)
    
    # Invoke Lambda function
    response = invoke_lambda(function_name, aws_region)
    if not response:
        print("Failed to invoke Lambda function. Exiting.")
        return
    
    # Print response
    print("\nLambda function response:")
    print(json.dumps(response, indent=2))
    
    # Check if the function was successful
    if response.get('statusCode') == 200:
        body = json.loads(response.get('body', '{}'))
        airlines = body.get('airlines', [])
        print(f"\nSuccessfully retrieved {len(airlines)} airlines:")
        for i, airline in enumerate(airlines, 1):
            print(f"{i}. {airline.get('name', 'Unknown')} ({airline.get('id', 'Unknown')})")
    else:
        print("\nFunction execution failed:")
        body = json.loads(response.get('body', '{}'))
        print(f"Error: {body.get('message', 'Unknown error')}")
        print(f"Traceback: {body.get('traceback', 'No traceback available')}")

if __name__ == '__main__':
    main()
