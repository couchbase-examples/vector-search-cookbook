import json
import logging
import os
import subprocess
import time
import traceback
import uuid
from datetime import timedelta
import shutil
import sys

import boto3
from botocore.exceptions import ClientError
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import \
    SearchIndexNotFoundException  # Added more specific exceptions
from couchbase.exceptions import (BucketNotFoundException,
                                  CollectionNotFoundException,
                                  CouchbaseException,
                                  InternalServerFailureException,
                                  QueryIndexAlreadyExistsException,
                                  ScopeNotFoundException,
                                  ServiceUnavailableException)
from couchbase.management.buckets import (BucketSettings, BucketType,
                                          CreateBucketSettings)
from couchbase.management.collections import CollectionSpec
from couchbase.management.search import SearchIndex, SearchIndexManager
from couchbase.options import ClusterOptions, QueryOptions
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from botocore.config import Config
from botocore.waiter import WaiterModel, create_waiter_with_client

# --- Configuration ---
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from project root .env
dotenv_path = os.path.join(os.path.dirname(__file__),'.env') # Adjust path to root
logger.info(f"Attempting to load .env file from: {dotenv_path}")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(".env file loaded successfully.")
else:
    logger.warning(f".env file not found at {dotenv_path}. Relying on environment variables.")


# Couchbase Configuration
CB_HOST = os.getenv("CB_HOST", "couchbase://localhost")
CB_USERNAME = os.getenv("CB_USERNAME", "Administrator")
CB_PASSWORD = os.getenv("CB_PASSWORD", "password")
# Using a new bucket/scope/collection for experiments to avoid conflicts
CB_BUCKET_NAME = os.getenv("CB_BUCKET_NAME", "vector-search-exp")
SCOPE_NAME = os.getenv("SCOPE_NAME", "bedrock_exp")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs_exp")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_search_bedrock_exp")

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")

# Bedrock Model IDs
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
AGENT_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0" # Using Sonnet for the agent

# Paths (relative to this script's location initially)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMAS_DIR = os.path.join(SCRIPT_DIR, 'schemas') # New Schemas Dir
# RESEARCHER_SCHEMA_PATH = os.path.join(SCHEMAS_DIR, 'researcher_schema.json') # Removed
# WRITER_SCHEMA_PATH = os.path.join(SCHEMAS_DIR, 'writer_schema.json')     # Removed
SEARCH_FORMAT_SCHEMA_PATH = os.path.join(SCHEMAS_DIR, 'search_and_format_schema.json') # Added
INDEX_JSON_PATH = os.path.join(SCRIPT_DIR, 'aws_index.json') # Keep
DOCS_JSON_PATH = os.path.join(SCRIPT_DIR, 'documents.json') # Changed to load from script's directory


# --- Helper Functions ---

def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ACCOUNT_ID", "CB_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your environment or .env file")
        return False
    logger.info("All required environment variables are set.")
    return True

def initialize_aws_clients():
    """Initialize required AWS clients."""
    try:
        logger.info(f"Initializing AWS clients in region: {AWS_REGION}")
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        # Use a config with longer timeouts for agent operations
        agent_config = Config(
            connect_timeout=120,
            read_timeout=600, # Agent preparation can take time
            retries={'max_attempts': 5, 'mode': 'adaptive'}
        )
        bedrock_runtime = session.client('bedrock-runtime', region_name=AWS_REGION)
        iam_client = session.client('iam', region_name=AWS_REGION) 
        lambda_client = session.client('lambda', region_name=AWS_REGION)
        bedrock_agent_client = session.client('bedrock-agent', region_name=AWS_REGION, config=agent_config) # Add agent client
        bedrock_agent_runtime_client = session.client('bedrock-agent-runtime', region_name=AWS_REGION, config=agent_config) # Add agent runtime client
        logger.info("AWS clients initialized successfully.")
        return bedrock_runtime, iam_client, lambda_client, bedrock_agent_client, bedrock_agent_runtime_client # Return agent runtime client
    except Exception as e:
        logger.error(f"Error initializing AWS clients: {e}")
        raise

def connect_couchbase():
    """Connect to Couchbase cluster."""
    try:
        logger.info(f"Connecting to Couchbase cluster at {CB_HOST}...")
        auth = PasswordAuthenticator(CB_USERNAME, CB_PASSWORD)
        # Use robust options
        options = ClusterOptions(
             auth,
             # query_timeout=timedelta(seconds=75), # Example: longer timeout
             # kv_timeout=timedelta(seconds=10)
        )
        cluster = Cluster(CB_HOST, options)
        cluster.wait_until_ready(timedelta(seconds=10)) # Wait longer if needed
        logger.info("Successfully connected to Couchbase.")
        return cluster
    except CouchbaseException as e:
        logger.error(f"Couchbase connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to Couchbase: {e}")
        raise

def setup_collection(cluster, bucket_name, scope_name, collection_name):
    """Set up Couchbase collection (Original Logic from lamda-approach)"""
    logger.info(f"Setting up collection: {bucket_name}/{scope_name}/{collection_name}")
    try:
        # Check if bucket exists, create if it doesn't
        try:
            bucket = cluster.bucket(bucket_name)
            logger.info(f"Bucket '{bucket_name}' exists.")
        except BucketNotFoundException:
            logger.info(f"Bucket '{bucket_name}' does not exist. Creating it...")
            # Use BucketSettings with potentially lower RAM for experiment
            bucket_settings = BucketSettings(
                name=bucket_name,
                bucket_type=BucketType.COUCHBASE,
                ram_quota_mb=256, # Adjusted from 1024
                flush_enabled=True,
                num_replicas=0
            )
            try:
                 cluster.buckets().create_bucket(bucket_settings)
                 # Wait longer after bucket creation
                 logger.info(f"Bucket '{bucket_name}' created. Waiting for ready state (10s)...")
                 time.sleep(10) 
                 bucket = cluster.bucket(bucket_name) # Re-assign bucket object
            except Exception as create_e:
                 logger.error(f"Failed to create bucket '{bucket_name}': {create_e}")
                 raise
        except Exception as e:
             logger.error(f"Error getting bucket '{bucket_name}': {e}")
             raise

        bucket_manager = bucket.collections()

        # Check if scope exists, create if it doesn't
        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(s.name == scope_name for s in scopes)

        if not scope_exists:
            logger.info(f"Scope '{scope_name}' does not exist. Creating it...")
            try:
                 bucket_manager.create_scope(scope_name)
                 logger.info(f"Scope '{scope_name}' created. Waiting (2s)...")
                 time.sleep(2)
            except CouchbaseException as e:
                 # Handle potential race condition or already exists error more robustly
                 if "already exists" in str(e).lower() or "scope_exists" in str(e).lower():
                      logger.info(f"Scope '{scope_name}' likely already exists (caught during creation attempt).")
                 else:
                      logger.error(f"Failed to create scope '{scope_name}': {e}")
                      raise
        else:
             logger.info(f"Scope '{scope_name}' already exists.")

        # Check if collection exists, create if it doesn't
        # Re-fetch scopes in case it was just created
        scopes = bucket_manager.get_all_scopes()
        collection_exists = False
        for s in scopes:
             if s.name == scope_name:
                  if any(c.name == collection_name for c in s.collections):
                       collection_exists = True
                       break
        
        if not collection_exists:
            logger.info(f"Collection '{collection_name}' does not exist in scope '{scope_name}'. Creating it...")
            try:
                # Use CollectionSpec
                collection_spec = CollectionSpec(collection_name, scope_name)
                bucket_manager.create_collection(collection_spec)
                logger.info(f"Collection '{collection_name}' created. Waiting (2s)...")
                time.sleep(2)
            except CouchbaseException as e:
                 if "already exists" in str(e).lower() or "collection_exists" in str(e).lower():
                     logger.info(f"Collection '{collection_name}' likely already exists (caught during creation attempt).")
                 else:
                     logger.error(f"Failed to create collection '{collection_name}': {e}")
                     raise
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

        # Ensure primary index exists
        try:
            logger.info(f"Ensuring primary index exists on `{bucket_name}`.`{scope_name}`.`{collection_name}`...")
            cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`").execute()
            logger.info("Primary index present or created successfully.")
        except Exception as e:
            logger.error(f"Error creating primary index: {str(e)}")
            # Decide if this is fatal

        logger.info("Collection setup complete.")
        # Return the collection object for use
        return cluster.bucket(bucket_name).scope(scope_name).collection(collection_name)

    except Exception as e:
        logger.error(f"Error setting up collection: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def setup_search_index(cluster, index_name, bucket_name, scope_name, collection_name, index_definition_path):
    """Set up search indexes (Original Logic, adapted) """
    try:
        logger.info(f"Looking for index definition at: {index_definition_path}")
        if not os.path.exists(index_definition_path):
             logger.error(f"Index definition file not found: {index_definition_path}")
             raise FileNotFoundError(f"Index definition file not found: {index_definition_path}")

        with open(index_definition_path, 'r') as file:
            index_definition = json.load(file)
            # Update name and source based on function arguments
            index_definition['name'] = index_name
            index_definition['sourceName'] = bucket_name
            # Optional: update params to explicitly target scope.collection if needed
            # index_definition['planParams']['indexPartitions'] = 1 # Example
            # index_definition['params'] = {
            #     'mapping': {
            #         'types': {
            #             f'{scope_name}.{collection_name}': {
            #                 'enabled': True,
            #                 'dynamic': True # Or specify fields
            #             }
            #         },
            #         'default_mapping': {
            #             'enabled': False
            #         }
            #     }
            # }
            logger.info(f"Loaded index definition from {index_definition_path}, ensuring name is '{index_name}' and source is '{bucket_name}'.")

    except Exception as e:
        logger.error(f"Error loading index definition: {str(e)}")
        raise

    try:
        # Use the SearchIndexManager from the Cluster object for cluster-level indexes
        # Or use scope-level if the index JSON is structured for that
        # Assuming cluster level based on original script structure for upsert
        search_index_manager = cluster.search_indexes()

        # Create SearchIndex object from potentially modified JSON definition
        search_index = SearchIndex.from_json(index_definition)

        # Upsert the index (create if not exists, update if exists)
        logger.info(f"Upserting search index '{index_name}'...")
        search_index_manager.upsert_index(search_index)

        # Wait for indexing
        logger.info(f"Index '{index_name}' upsert operation submitted. Waiting for indexing (10s)...")
        time.sleep(10)

        logger.info(f"Search index '{index_name}' setup complete.")

    except QueryIndexAlreadyExistsException:
        # This exception might not be correct for SearchIndexManager
        # Upsert should handle exists cases, but log potential specific errors
        logger.warning(f"Search index '{index_name}' likely already existed (caught QueryIndexAlreadyExistsException, check if applicable). Upsert attempted.")
    except CouchbaseException as e:
        logger.error(f"Couchbase error during search index setup for '{index_name}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during search index setup for '{index_name}': {e}")
        raise

def clear_collection(cluster, bucket_name, scope_name, collection_name):
    """Delete all documents from the specified collection (Original Logic)."""
    try:
        logger.warning(f"Attempting to clear all documents from `{bucket_name}`.`{scope_name}`.`{collection_name}`...")
        query = f"DELETE FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
        result = cluster.query(query).execute()
        # Try to get mutation count, handle if not available
        mutation_count = 0
        try:
             metrics_data = result.meta_data().metrics()
             if metrics_data:
                  mutation_count = metrics_data.mutation_count()
        except Exception as metrics_e:
             logger.warning(f"Could not retrieve mutation count after delete: {metrics_e}")
        logger.info(f"Successfully cleared documents from the collection (approx. {mutation_count} mutations).")
    except Exception as e:
        logger.error(f"Error clearing documents from collection: {e}. Collection might be empty or index not ready.")

def create_agent_role(iam_client, role_name, aws_account_id):
    """Creates or gets the IAM role for the Bedrock Agent Lambda functions."""
    logger.info(f"Checking/Creating IAM role: {role_name}")
    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": [
                        "lambda.amazonaws.com",
                        "bedrock.amazonaws.com" 
                    ]
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }

    role_arn = None
    try:
        # Check if role exists
        get_role_response = iam_client.get_role(RoleName=role_name)
        role_arn = get_role_response['Role']['Arn']
        logger.info(f"IAM role '{role_name}' already exists with ARN: {role_arn}")
        
        # Ensure trust policy is up-to-date
        logger.info(f"Updating trust policy for existing role '{role_name}'...")
        iam_client.update_assume_role_policy(
            RoleName=role_name,
            PolicyDocument=json.dumps(assume_role_policy_document)
        )
        logger.info(f"Trust policy updated for role '{role_name}'.")

    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"IAM role '{role_name}' not found. Creating...")
        try:
            create_role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
                Description='IAM role for Bedrock Agent Lambda functions (Experiment)',
                MaxSessionDuration=3600
            )
            role_arn = create_role_response['Role']['Arn']
            logger.info(f"Successfully created IAM role '{role_name}' with ARN: {role_arn}")
            # Wait after role creation before attaching policies
            logger.info("Waiting 15s for role creation propagation...")
            time.sleep(15)
        except ClientError as e:
            logger.error(f"Error creating IAM role '{role_name}': {e}")
            raise
            
    except ClientError as e:
        logger.error(f"Error getting/updating IAM role '{role_name}': {e}")
        raise
        
    # Attach basic execution policy (idempotent)
    try:
        logger.info(f"Attaching basic Lambda execution policy to role '{role_name}'...")
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
        logger.info("Attached basic Lambda execution policy.")
    except ClientError as e:
        logger.error(f"Error attaching basic Lambda execution policy: {e}")
        # Don't necessarily raise, might already be attached or other issue
        
    # Add minimal inline policy for logging (can be expanded later if needed)
    basic_inline_policy_name = "LambdaBasicLoggingPermissions"
    basic_inline_policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": f"arn:aws:logs:{AWS_REGION}:{aws_account_id}:log-group:/aws/lambda/*:*" # Scope down logs if possible
            }
            # Add S3 permissions here ONLY if Lambda code explicitly needs it
        ]
    }
    
    # Add Bedrock permissions policy
    bedrock_policy_name = "BedrockAgentPermissions"
    bedrock_policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:*"
                ],
                "Resource": "*"  # You can scope this down to specific agents/models if needed
            }
        ]
    }
    try:
        logger.info(f"Putting basic inline policy '{basic_inline_policy_name}' for role '{role_name}'...")
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=basic_inline_policy_name,
            PolicyDocument=json.dumps(basic_inline_policy_doc)
        )
        logger.info(f"Successfully put inline policy '{basic_inline_policy_name}'.")
        
        # Add Bedrock permissions policy
        logger.info(f"Putting Bedrock permissions policy '{bedrock_policy_name}' for role '{role_name}'...")
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=bedrock_policy_name,
            PolicyDocument=json.dumps(bedrock_policy_doc)
        )
        logger.info(f"Successfully put inline policy '{bedrock_policy_name}'.")
        
        logger.info("Waiting 10s for policy changes to propagate...")
        time.sleep(10)
    except ClientError as e:
        logger.error(f"Error putting inline policy: {e}")
        # Decide if this is fatal
        
    if not role_arn:
         raise Exception(f"Failed to create or retrieve ARN for role {role_name}")
         
    return role_arn


# --- Lambda Deployment Functions (Adapted from deploy.py) ---

def delete_lambda_function(lambda_client, function_name):
    """Delete Lambda function if it exists, attempting to remove permissions first."""
    logger.info(f"Attempting to delete Lambda function: {function_name}...")
    try:
        # Use a predictable statement ID added by create_lambda_function
        statement_id = f"AllowBedrockInvokeBasic-{function_name}"
        try:
            logger.info(f"Attempting to remove permission {statement_id} from {function_name}...")
            lambda_client.remove_permission(
                FunctionName=function_name,
                StatementId=statement_id
            )
            logger.info(f"Successfully removed permission {statement_id} from {function_name}.")
            time.sleep(2) # Allow time for permission removal
        except lambda_client.exceptions.ResourceNotFoundException:
            logger.info(f"Permission {statement_id} not found on {function_name}. Skipping removal.")
        except ClientError as perm_e:
            # Log error but continue with deletion attempt
            logger.warning(f"Error removing permission {statement_id} from {function_name}: {str(perm_e)}")

        # Check if function exists before attempting deletion
        lambda_client.get_function(FunctionName=function_name)
        logger.info(f"Function {function_name} exists. Deleting...")
        lambda_client.delete_function(FunctionName=function_name)

        # Wait for deletion to complete using a waiter
        logger.info(f"Waiting for {function_name} to be deleted...")
        time.sleep(10) # Simple delay after delete call
        logger.info(f"Function {function_name} deletion initiated.")

        return True # Indicates deletion was attempted/occurred

    except lambda_client.exceptions.ResourceNotFoundException:
        logger.info(f"Lambda function '{function_name}' does not exist. No need to delete.")
        return False # Indicates function didn't exist
    except Exception as e:
        logger.error(f"Error during deletion process for Lambda function '{function_name}': {str(e)}")
        # Depending on severity, might want to raise or just return False
        return False # Indicates an error occurred beyond not found


def upload_to_s3(zip_file, region, bucket_name=None):
    """Upload zip file to S3 with retry logic and return S3 location."""
    logger.info(f"Preparing to upload {zip_file} to S3 in region {region}...")
    # Configure the client with increased timeouts
    config = Config(
        connect_timeout=60,
        read_timeout=300,
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )

    s3_client = boto3.client('s3', region_name=region, config=config)
    sts_client = boto3.client('sts', region_name=region, config=config)

    # Determine bucket name
    if bucket_name is None:
        try:
            account_id = sts_client.get_caller_identity().get('Account')
            timestamp = int(time.time())
            bucket_name = f"lambda-deployment-{account_id}-{timestamp}"
            logger.info(f"Generated unique S3 bucket name: {bucket_name}")
        except Exception as e:
            fallback_id = uuid.uuid4().hex[:12]
            bucket_name = f"lambda-deployment-{fallback_id}"
            logger.warning(f"Error getting account ID ({e}). Using fallback bucket name: {bucket_name}")

    # Create bucket if needed
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Using existing S3 bucket: {bucket_name}")
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            logger.info(f"Creating S3 bucket: {bucket_name}...")
            try:
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                logger.info(f"Created S3 bucket: {bucket_name}. Waiting for availability...")
                waiter = s3_client.get_waiter('bucket_exists')
                waiter.wait(Bucket=bucket_name, WaiterConfig={'Delay': 5, 'MaxAttempts': 12})
                logger.info(f"Bucket {bucket_name} is available.")
            except Exception as create_e:
                logger.error(f"Error creating bucket '{bucket_name}': {create_e}")
                raise
        else:
            logger.error(f"Error checking bucket '{bucket_name}': {e}")
            raise

    # Upload file
    s3_key = f"lambda/{os.path.basename(zip_file)}-{uuid.uuid4().hex[:8]}"
    try:
        logger.info(f"Uploading {zip_file} to s3://{bucket_name}/{s3_key}...")
        file_size = os.path.getsize(zip_file)
        if file_size > 100 * 1024 * 1024:  # Use multipart for files > 100MB
            logger.info("Using multipart upload for large file...")
            transfer_config = boto3.s3.transfer.TransferConfig(
                multipart_threshold=10 * 1024 * 1024, max_concurrency=10,
                multipart_chunksize=10 * 1024 * 1024, use_threads=True
            )
            s3_transfer = boto3.s3.transfer.S3Transfer(client=s3_client, config=transfer_config)
            s3_transfer.upload_file(zip_file, bucket_name, s3_key)
        else:
            with open(zip_file, 'rb') as f:
                s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=f)

        logger.info(f"Successfully uploaded to s3://{bucket_name}/{s3_key}")
        return {'S3Bucket': bucket_name, 'S3Key': s3_key}

    except Exception as upload_e:
        logger.error(f"S3 upload failed: {upload_e}")
        raise


def package_function(function_name, source_dir, build_dir):
    """Package Lambda function using Makefile found in source_dir."""
    # source_dir is where the .py, requirements.txt, Makefile live (e.g., lambda_functions)
    # build_dir is where packaging happens and final zip ends up (e.g., lambda-experiments)
    makefile_path = os.path.join(source_dir, 'Makefile')
    # Temp build dir inside source_dir, as Makefile expects relative paths
    temp_package_dir = os.path.join(source_dir, 'package_dir') 
    # Requirements file is in source_dir
    source_req_path = os.path.join(source_dir, 'requirements.txt') 
    # Target requirements path inside source_dir (needed for Makefile)
    # target_req_path = os.path.join(source_dir, 'requirements.txt') # No copy needed if running make in source_dir
    source_func_script_path = os.path.join(source_dir, f'{function_name}.py')
    # Target function script path inside source_dir, renamed for Makefile install_deps copy
    target_func_script_path = os.path.join(source_dir, 'lambda_function.py') 
    # Make output zip is created inside source_dir
    make_output_zip = os.path.join(source_dir, 'lambda_package.zip') 
    # Final zip path is in the build_dir (one level up from source_dir)
    final_zip_path = os.path.join(build_dir, f'{function_name}.zip')

    logger.info(f"--- Packaging function {function_name} --- ")
    logger.info(f"Source Dir (Makefile location & make cwd): {source_dir}")
    logger.info(f"Build Dir (Final zip location): {build_dir}")

    if not os.path.exists(source_func_script_path):
        raise FileNotFoundError(f"Source function script not found: {source_func_script_path}")
    if not os.path.exists(source_req_path):
        raise FileNotFoundError(f"Source requirements file not found: {source_req_path}")
    if not os.path.exists(makefile_path):
        raise FileNotFoundError(f"Makefile not found at: {makefile_path}")

    # Ensure no leftover target script from previous failed run
    if os.path.exists(target_func_script_path):
        logger.warning(f"Removing existing target script: {target_func_script_path}")
        os.remove(target_func_script_path)

    try:
        # 1. No need to create lambda subdir in build_dir

        # 2. Copy source function script to source_dir as lambda_function.py
        logger.info(f"Copying {source_func_script_path} to {target_func_script_path}")
        shutil.copy(source_func_script_path, target_func_script_path)
        # Requirements file is already in source_dir, no copy needed.

        # 3. Run make command (execute from source_dir where Makefile is)
        make_command = [
            'make',
            '-f', makefile_path, # Still specify Makefile path explicitly
            'clean', # Clean first
            'package',
            # 'PYTHON_VERSION=python3.9' # Let Makefile use its default or system default
        ]
        logger.info(f"Running make command: {' '.join(make_command)} (in {source_dir})")
        # Run make from source_dir; relative paths in Makefile should now work
        subprocess.check_call(make_command, cwd=source_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        logger.info("Make command completed successfully.")

        # 4. Check for output zip in source_dir and rename/move to build_dir
        if not os.path.exists(make_output_zip):
            raise FileNotFoundError(f"Makefile did not produce expected output: {make_output_zip}")

        logger.info(f"Moving and renaming {make_output_zip} to {final_zip_path}")
        if os.path.exists(final_zip_path):
             logger.warning(f"Removing existing final zip: {final_zip_path}")
             os.remove(final_zip_path)
        # Use shutil.move for cross-filesystem safety if needed, os.rename is fine here
        os.rename(make_output_zip, final_zip_path) 
        logger.info(f"Zip file ready: {final_zip_path}")

        return final_zip_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Makefile for {function_name}: {e}")
        stderr_output = "(No stderr captured)"
        if e.stderr:
             try:
                  stderr_output = e.stderr.decode()
             except Exception:
                  stderr_output = "(Could not decode stderr)"
        logger.error(f"Make stderr: {stderr_output}")
        raise
    except Exception as e:
        logger.error(f"Error packaging function {function_name} using Makefile: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # 5. Clean up intermediate files in source_dir
        if os.path.exists(target_func_script_path):
            logger.info(f"Cleaning up temporary script: {target_func_script_path}")
            os.remove(target_func_script_path)
        if os.path.exists(make_output_zip): # If rename failed
            logger.warning(f"Cleaning up intermediate zip in source dir: {make_output_zip}")
            os.remove(make_output_zip)
        # Consider cleaning temp_package_dir if make clean fails
        # if os.path.exists(temp_package_dir):
        #     logger.info(f"Cleaning up temporary package dir: {temp_package_dir}")
        #     shutil.rmtree(temp_package_dir)


def create_lambda_function(lambda_client, function_name, handler, role_arn, zip_file, region):
    """Create or update Lambda function with retry logic."""
    logger.info(f"Deploying Lambda function {function_name} from {zip_file}...")

    # Configure the client with increased timeouts for potentially long creation
    config = Config(
        connect_timeout=120,
        read_timeout=300,
        retries={'max_attempts': 5, 'mode': 'adaptive'}
    )
    lambda_client_local = boto3.client('lambda', region_name=region, config=config)

    # Check zip file size
    zip_size_mb = 0
    try:
        zip_size_bytes = os.path.getsize(zip_file)
        zip_size_mb = zip_size_bytes / (1024 * 1024)
        logger.info(f"Zip file size: {zip_size_mb:.2f} MB")
    except OSError as e:
         logger.error(f"Could not get size of zip file {zip_file}: {e}")
         raise # Cannot proceed without zip file

    use_s3 = zip_size_mb > 45 # Use S3 for packages over ~45MB
    s3_location = None
    zip_content = None

    if use_s3:
        logger.info(f"Package size ({zip_size_mb:.2f} MB) requires S3 deployment.")
        s3_location = upload_to_s3(zip_file, region)
        if not s3_location:
             raise Exception("Failed to upload Lambda package to S3.")
    else:
         logger.info("Deploying package directly.")
         try:
             with open(zip_file, 'rb') as f:
                 zip_content = f.read()
         except OSError as e:
              logger.error(f"Could not read zip file {zip_file}: {e}")
              raise

    # Define common create/update args
    common_args = {
        'FunctionName': function_name,
        'Runtime': 'python3.9',
        'Role': role_arn,
        'Handler': handler,
        'Timeout': 180,
        'MemorySize': 1536, # Adjust as needed
        # Env vars loaded from main script env or .env
        'Environment': {
            'Variables': {
                'CB_HOST': os.getenv('CB_HOST', 'couchbase://localhost'),
                'CB_USERNAME': os.getenv('CB_USERNAME', 'Administrator'),
                'CB_PASSWORD': os.getenv('CB_PASSWORD', 'password'),
                'CB_BUCKET_NAME': os.getenv('CB_BUCKET_NAME', 'vector-search-exp'),
                'SCOPE_NAME': os.getenv('SCOPE_NAME', 'bedrock_exp'),
                'COLLECTION_NAME': os.getenv('COLLECTION_NAME', 'docs_exp'),
                'INDEX_NAME': os.getenv('INDEX_NAME', 'vector_search_bedrock_exp'),
                'EMBEDDING_MODEL_ID': os.getenv('EMBEDDING_MODEL_ID', EMBEDDING_MODEL_ID),
                'AGENT_MODEL_ID': os.getenv('AGENT_MODEL_ID', AGENT_MODEL_ID)
            }
        }
    }

    if use_s3:
        code_arg = {'S3Bucket': s3_location['S3Bucket'], 'S3Key': s3_location['S3Key']}
    else:
        code_arg = {'ZipFile': zip_content}

    max_retries = 3
    base_delay = 10
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Creating function '{function_name}' (attempt {attempt}/{max_retries})...")
            create_args = common_args.copy()
            create_args['Code'] = code_arg
            create_args['Publish'] = True # Publish a version

            create_response = lambda_client_local.create_function(**create_args)
            function_arn = create_response['FunctionArn']
            logger.info(f"Successfully created function '{function_name}' with ARN: {function_arn}")

            # Add basic invoke permission after creation
            time.sleep(5) # Give function time to be fully created before adding policy
            statement_id = f"AllowBedrockInvokeBasic-{function_name}"
            try:
                logger.info(f"Adding basic invoke permission ({statement_id}) to {function_name}...")
                lambda_client_local.add_permission(
                    FunctionName=function_name,
                    StatementId=statement_id,
                    Action='lambda:InvokeFunction',
                    Principal='bedrock.amazonaws.com'
                )
                logger.info(f"Successfully added basic invoke permission {statement_id}.")
            except lambda_client_local.exceptions.ResourceConflictException:
                 logger.info(f"Permission {statement_id} already exists for {function_name}. Skipping add.")
            except ClientError as perm_e:
                logger.warning(f"Failed to add basic invoke permission {statement_id} to {function_name}: {perm_e}")

            # Wait for function to be Active
            logger.info(f"Waiting for function '{function_name}' to become active...")
            waiter = lambda_client_local.get_waiter('function_active_v2')
            waiter.wait(FunctionName=function_name, WaiterConfig={'Delay': 5, 'MaxAttempts': 24})
            logger.info(f"Function '{function_name}' is active.")

            return function_arn # Return ARN upon successful creation

        except lambda_client_local.exceptions.ResourceConflictException:
             logger.warning(f"Function '{function_name}' already exists. Attempting to update code...")
             try:
                 if use_s3:
                     update_response = lambda_client_local.update_function_code(
                         FunctionName=function_name,
                         S3Bucket=s3_location['S3Bucket'],
                         S3Key=s3_location['S3Key'],
                         Publish=True
                     )
                 else:
                     update_response = lambda_client_local.update_function_code(
                         FunctionName=function_name,
                         ZipFile=zip_content,
                         Publish=True
                     )
                 function_arn = update_response['FunctionArn']
                 logger.info(f"Successfully updated function code for '{function_name}'. New version ARN: {function_arn}")
                 
                 # Also update configuration just in case
                 try:
                      logger.info(f"Updating configuration for '{function_name}'...")
                      lambda_client_local.update_function_configuration(**common_args)
                      logger.info(f"Configuration updated for '{function_name}'.")
                 except ClientError as conf_e:
                      logger.warning(f"Could not update configuration for '{function_name}': {conf_e}")
                 
                 # Re-verify invoke permission after update
                 time.sleep(5)
                 statement_id = f"AllowBedrockInvokeBasic-{function_name}"
                 try:
                     logger.info(f"Verifying/Adding basic invoke permission ({statement_id}) after update...")
                     lambda_client_local.add_permission(
                         FunctionName=function_name,
                         StatementId=statement_id,
                         Action='lambda:InvokeFunction',
                         Principal='bedrock.amazonaws.com'
                     )
                     logger.info(f"Successfully added/verified basic invoke permission {statement_id}.")
                 except lambda_client_local.exceptions.ResourceConflictException:
                     logger.info(f"Permission {statement_id} already exists for {function_name}. Skipping add.")
                 except ClientError as perm_e:
                     logger.warning(f"Failed to add/verify basic invoke permission {statement_id} after update: {perm_e}")

                 # Wait for function to be Active after update
                 logger.info(f"Waiting for function '{function_name}' update to complete...")
                 waiter = lambda_client_local.get_waiter('function_updated_v2')
                 waiter.wait(FunctionName=function_name, WaiterConfig={'Delay': 5, 'MaxAttempts': 24})
                 logger.info(f"Function '{function_name}' update complete.")
                 
                 return function_arn # Return ARN after successful update

             except ClientError as update_e:
                 logger.error(f"Failed to update function '{function_name}': {update_e}")
                 if attempt < max_retries:
                      delay = base_delay * (2 ** (attempt - 1))
                      logger.info(f"Retrying update in {delay} seconds...")
                      time.sleep(delay)
                 else:
                      logger.error("Maximum update retries reached. Deployment failed.")
                      raise update_e
                      
        except ClientError as e:
            # Handle throttling or other retryable errors
            error_code = e.response.get('Error', {}).get('Code')
            if error_code in ['ThrottlingException', 'ProvisionedConcurrencyConfigNotFoundException', 'EC2ThrottledException'] or 'Rate exceeded' in str(e):
                logger.warning(f"Retryable error on attempt {attempt}: {e}")
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1)) + (uuid.uuid4().int % 5)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("Maximum retries reached after retryable error. Deployment failed.")
                    raise e
            else:
                logger.error(f"Error creating/updating Lambda '{function_name}': {e}")
                logger.error(traceback.format_exc()) # Log full traceback for unexpected errors
                raise e # Re-raise non-retryable or unexpected errors
        except Exception as e:
             logger.error(f"Unexpected error during Lambda deployment: {e}")
             logger.error(traceback.format_exc())
             raise e
             
    # If loop completes without returning, something went wrong
    raise Exception(f"Failed to deploy Lambda function {function_name} after {max_retries} attempts.")


# --- Agent Resource Deletion Functions ---

def get_agent_by_name(agent_client, agent_name):
    """Find an agent ID by its name using list_agents."""
    logger.info(f"Attempting to find agent by name: {agent_name}")
    try:
        paginator = agent_client.get_paginator('list_agents')
        for page in paginator.paginate():
            for agent_summary in page.get('agentSummaries', []):
                if agent_summary.get('agentName') == agent_name:
                    agent_id = agent_summary.get('agentId')
                    logger.info(f"Found agent '{agent_name}' with ID: {agent_id}")
                    return agent_id
        logger.info(f"Agent '{agent_name}' not found.")
        return None
    except ClientError as e:
        logger.error(f"Error listing agents to find '{agent_name}': {e}")
        return None # Treat as not found if error occurs

def delete_action_group(agent_client, agent_id, action_group_id):
    """Deletes a specific action group for an agent."""
    logger.info(f"Attempting to delete action group {action_group_id} for agent {agent_id}...")
    try:
        agent_client.delete_agent_action_group(
            agentId=agent_id,
            agentVersion='DRAFT', # Action groups are tied to the DRAFT version
            actionGroupId=action_group_id,
            skipResourceInUseCheck=True # Force deletion even if in use (e.g., during prepare)
        )
        logger.info(f"Successfully deleted action group {action_group_id} for agent {agent_id}.")
        time.sleep(5) # Short pause after deletion
        return True
    except agent_client.exceptions.ResourceNotFoundException:
        logger.info(f"Action group {action_group_id} not found for agent {agent_id}. Skipping deletion.")
        return False
    except ClientError as e:
        # Handle potential throttling or conflict if prepare is happening
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == 'ConflictException':
            logger.warning(f"Conflict deleting action group {action_group_id} (agent might be preparing/busy). Retrying once after delay...")
            time.sleep(15)
            try:
                agent_client.delete_agent_action_group(
                    agentId=agent_id, agentVersion='DRAFT', actionGroupId=action_group_id, skipResourceInUseCheck=True
                )
                logger.info(f"Successfully deleted action group {action_group_id} after retry.")
                return True
            except Exception as retry_e:
                 logger.error(f"Error deleting action group {action_group_id} on retry: {retry_e}")
                 return False
        else:
            logger.error(f"Error deleting action group {action_group_id} for agent {agent_id}: {e}")
            return False


def delete_agent_and_resources(agent_client, agent_name):
    """Deletes the agent and its associated action groups."""
    agent_id = get_agent_by_name(agent_client, agent_name)
    if not agent_id:
        logger.info(f"Agent '{agent_name}' not found, no deletion needed.")
        return

    logger.warning(f"--- Deleting Agent Resources for '{agent_name}' (ID: {agent_id}) ---")

    # 1. Delete Action Groups
    try:
        logger.info(f"Listing action groups for agent {agent_id}...")
        action_groups = agent_client.list_agent_action_groups(
            agentId=agent_id,
            agentVersion='DRAFT' # List groups for the DRAFT version
        ).get('actionGroupSummaries', [])

        if action_groups:
            logger.info(f"Found {len(action_groups)} action groups to delete.")
            for ag in action_groups:
                delete_action_group(agent_client, agent_id, ag['actionGroupId'])
        else:
            logger.info("No action groups found to delete.")

    except ClientError as e:
        logger.error(f"Error listing action groups for agent {agent_id}: {e}")
        # Continue to agent deletion attempt even if listing fails

    # 2. Delete the Agent
    try:
        logger.info(f"Attempting to delete agent {agent_id} ('{agent_name}')...")
        agent_client.delete_agent(agentId=agent_id, skipResourceInUseCheck=True) # Force delete

        # Wait for agent deletion (custom waiter logic might be needed if no standard waiter)
        logger.info(f"Waiting up to 2 minutes for agent {agent_id} deletion...")
        deleted = False
        for _ in range(24): # Check every 5 seconds for 2 minutes
            try:
                agent_client.get_agent(agentId=agent_id)
                time.sleep(5)
            except agent_client.exceptions.ResourceNotFoundException:
                logger.info(f"Agent {agent_id} successfully deleted.")
                deleted = True
                break
            except ClientError as e:
                 # Handle potential throttling during check
                 error_code = e.response.get('Error', {}).get('Code')
                 if error_code == 'ThrottlingException':
                     logger.warning("Throttled while checking agent deletion status, continuing wait...")
                     time.sleep(10)
                 else:
                     logger.error(f"Error checking agent deletion status: {e}")
                     # Break checking loop on unexpected error
                     break
        if not deleted:
             logger.warning(f"Agent {agent_id} deletion confirmation timed out.")

    except agent_client.exceptions.ResourceNotFoundException:
        logger.info(f"Agent {agent_id} ('{agent_name}') already deleted or not found.")
    except ClientError as e:
        logger.error(f"Error deleting agent {agent_id}: {e}")

    logger.info(f"--- Agent Resource Deletion Complete for '{agent_name}' ---")


# --- Agent Creation Functions ---

def create_agent(agent_client, agent_name, agent_role_arn, foundation_model_id):
    """Creates a new Bedrock Agent."""
    logger.info(f"--- Creating Agent: {agent_name} ---")
    try:
        # Updated Instruction for single tool
        instruction = (
            "You are a helpful research assistant. Your primary function is to use the SearchAndFormat tool "
            "to find relevant documents based on user queries and format them. " 
            "Use the user's query for the search, and specify a formatting style if requested, otherwise use the default. "
            "Present the formatted results returned by the tool directly to the user."
            "Only use the tool provided. Do not add your own knowledge."
        )

        response = agent_client.create_agent(
            agentName=agent_name,
            agentResourceRoleArn=agent_role_arn,
            foundationModel=foundation_model_id,
            instruction=instruction,
            idleSessionTTLInSeconds=1800, # 30 minutes
            description=f"Experimental agent for Couchbase search and content formatting ({foundation_model_id})"
            # promptOverrideConfiguration={} # Optional: Add later if needed
        )
        agent_info = response.get('agent')
        agent_id = agent_info.get('agentId')
        agent_arn = agent_info.get('agentArn')
        agent_status = agent_info.get('agentStatus')
        logger.info(f"Agent creation initiated. Name: {agent_name}, ID: {agent_id}, ARN: {agent_arn}, Status: {agent_status}")

        # Wait for agent to become NOT_PREPARED (initial state after creation)
        # Using custom waiter logic as there might not be a standard one for this transition
        logger.info(f"Waiting for agent {agent_id} to reach initial state...")
        for _ in range(12): # Check for up to 1 minute
             current_status = agent_client.get_agent(agentId=agent_id)['agent']['agentStatus']
             logger.info(f"Agent {agent_id} status: {current_status}")
             if current_status != 'CREATING': # Expect NOT_PREPARED or FAILED
                  break
             time.sleep(5)

        final_status = agent_client.get_agent(agentId=agent_id)['agent']['agentStatus']
        if final_status == 'FAILED':
            logger.error(f"Agent {agent_id} creation failed.")
            # Optionally retrieve failure reasons if API provides them
            raise Exception(f"Agent creation failed for {agent_name}")
        else:
             logger.info(f"Agent {agent_id} successfully created (Status: {final_status}).")
             
        return agent_id, agent_arn

    except ClientError as e:
        logger.error(f"Error creating agent '{agent_name}': {e}")
        raise

def create_action_group(agent_client, agent_id, action_group_name, function_arn, schema_path=None):
    """Creates an action group for the agent using Define with function details."""
    logger.info(f"--- Creating/Updating Action Group (Function Details): {action_group_name} for Agent: {agent_id} ---")
    logger.info(f"Lambda ARN: {function_arn}")

    # Define function schema details (for functionSchema parameter)
    function_schema_details = {
        'functions': [
            {
                'name': 'searchAndFormatDocuments', # Function name agent will call
                'description': 'Performs vector search based on query, retrieves documents, and formats results using specified style.',
                'parameters': {
                    'query': {
                        'description': 'The search query text.',
                        'type': 'string',
                        'required': True
                    },
                    'k': {
                        'description': 'The maximum number of documents to retrieve.',
                        'type': 'integer',
                        'required': False # Making optional as Lambda has default
                    },
                    'style': {
                        'description': 'The desired formatting style for the results (e.g., \'bullet points\', \'paragraph\', \'summary\').',
                        'type': 'string',
                        'required': False # Making optional as Lambda has default
                    }
                }
            }
        ]
    }

    try:
        # Check if Action Group already exists for the DRAFT version
        try:
             logger.info(f"Checking if action group '{action_group_name}' already exists for agent {agent_id} DRAFT version...")
             paginator = agent_client.get_paginator('list_agent_action_groups')
             existing_group = None
             for page in paginator.paginate(agentId=agent_id, agentVersion='DRAFT'):
                 for ag_summary in page.get('actionGroupSummaries', []):
                      if ag_summary.get('actionGroupName') == action_group_name:
                           existing_group = ag_summary
                           break
                 if existing_group:
                      break
             
             if existing_group:
                 ag_id = existing_group['actionGroupId']
                 logger.warning(f"Action Group '{action_group_name}' (ID: {ag_id}) already exists for agent {agent_id} DRAFT. Attempting update to Function Details.")
                 # Update existing action group - REMOVE apiSchema, ADD functionSchema
                 response = agent_client.update_agent_action_group(
                     agentId=agent_id,
                     agentVersion='DRAFT',
                     actionGroupId=ag_id,
                     actionGroupName=action_group_name,
                     actionGroupExecutor={'lambda': function_arn},
                     functionSchema={ # Use functionSchema
                         'functions': function_schema_details['functions'] # Pass the list with the correct key
                     },
                     actionGroupState='ENABLED'
                 )
                 ag_info = response.get('agentActionGroup')
                 logger.info(f"Successfully updated Action Group '{action_group_name}' (ID: {ag_info.get('actionGroupId')}) to use Function Details.")
                 return ag_info.get('actionGroupId')
             else:
                  logger.info(f"Action group '{action_group_name}' does not exist. Creating new with Function Details.")

        except ClientError as e:
             logger.error(f"Error checking for existing action group '{action_group_name}': {e}. Proceeding with creation attempt.")


        # Create new action group if not found or update failed implicitly
        response = agent_client.create_agent_action_group(
            agentId=agent_id,
            agentVersion='DRAFT', 
            actionGroupName=action_group_name,
            actionGroupExecutor={
                'lambda': function_arn 
            },
            functionSchema={ # Use functionSchema
                 'functions': function_schema_details['functions'] # Pass the list with the correct key
            },
            actionGroupState='ENABLED' 
        )
        ag_info = response.get('agentActionGroup')
        ag_id = ag_info.get('actionGroupId')
        logger.info(f"Successfully created Action Group '{action_group_name}' with ID: {ag_id} using Function Details.")
        time.sleep(5) # Pause after creation/update
        return ag_id

    except ClientError as e:
        logger.error(f"Error creating/updating action group '{action_group_name}' using Function Details: {e}")
        raise

def prepare_agent(agent_client, agent_id):
    """Prepares the DRAFT version of the agent."""
    logger.info(f"--- Preparing Agent: {agent_id} ---")
    try:
        response = agent_client.prepare_agent(agentId=agent_id)
        agent_version = response.get('agentVersion') # Should be DRAFT
        prepared_at = response.get('preparedAt')
        status = response.get('agentStatus') # Should be PREPARING
        logger.info(f"Agent preparation initiated for version '{agent_version}'. Status: {status}. Prepared At: {prepared_at}")

        # Wait for preparation to complete (PREPARED or FAILED)
        logger.info(f"Waiting for agent {agent_id} preparation to complete (up to 10 minutes)...")
        # Define a simple waiter config
        waiter_config = {
            'version': 2,
            'waiters': {
                'AgentPrepared': {
                    'delay': 30, # Check every 30 seconds
                    'operation': 'GetAgent',
                    'maxAttempts': 20, # Max 10 minutes
                    'acceptors': [
                        {
                            'matcher': 'path',
                            'expected': 'PREPARED',
                            'argument': 'agent.agentStatus',
                            'state': 'success'
                        },
                        {
                            'matcher': 'path',
                            'expected': 'FAILED',
                            'argument': 'agent.agentStatus',
                            'state': 'failure'
                        },
                        {
                            'matcher': 'path',
                            'expected': 'UPDATING', # Can happen during prep? Treat as retryable
                            'argument': 'agent.agentStatus',
                            'state': 'retry'
                        }
                    ]
                }
            }
        }
        waiter_model = WaiterModel(waiter_config)
        custom_waiter = create_waiter_with_client('AgentPrepared', waiter_model, agent_client)

        try: # Outer try for both preparation and alias handling
             custom_waiter.wait(agentId=agent_id)
             logger.info(f"Agent {agent_id} successfully prepared.")

             # --- Alias Creation/Update (Using Custom Alias "prod") ---
             # This logic runs only if the preparation wait succeeds
             try: # Inner try specifically for alias operations
                 logger.info(f"Checking for alias '{alias_name}' for agent {agent_id}...")
                 existing_alias = None
                 paginator = bedrock_agent_client.get_paginator('list_agent_aliases')
                 for page in paginator.paginate(agentId=agent_id):
                     for alias_summary in page.get('agentAliasSummaries', []):
                         if alias_summary.get('agentAliasName') == alias_name:
                             existing_alias = alias_summary
                             break
                     if existing_alias:
                         break
                 
                 if existing_alias:
                     agent_alias_id_to_use = existing_alias['agentAliasId']
                     logger.info(f"Using existing alias '{alias_name}' with ID: {agent_alias_id_to_use}.")
                 else:
                     logger.info(f"Alias '{alias_name}' not found. Creating new alias...")
                     create_alias_response = bedrock_agent_client.create_agent_alias(
                         agentId=agent_id,
                         agentAliasName=alias_name
                         # routingConfiguration removed - defaults to latest prepared (DRAFT)
                     )
                     # Assign the ID immediately after creation, inside the else block
                     agent_alias_id_to_use = create_alias_response.get('agentAlias', {}).get('agentAliasId')
                     logger.info(f"Successfully created alias '{alias_name}' with ID: {agent_alias_id_to_use}. (Defaults to latest prepared version - DRAFT)")

                 if not agent_alias_id_to_use:
                      # Use a more specific exception
                      raise ValueError(f"Failed to get a valid alias ID for '{alias_name}'")

                 logger.info(f"Waiting 10s for alias '{alias_name}' changes to propagate...")

             except ClientError as alias_e: # Catch only alias-specific ClientErrors here
                 logger.error(f"Failed to create/update alias '{alias_name}' for agent {agent_id}: {alias_e}")
                 logger.error(traceback.format_exc())
                 sys.exit(1) # Exit if alias setup fails
            # End of inner alias try/except

        except Exception as e: # Outer except catches prepare_agent wait errors OR unhandled alias errors
            logger.error(f"Agent {agent_id} preparation failed or timed out (or alias error): {e}")
            # Check final status if possible
            try:
                final_status = agent_client.get_agent(agentId=agent_id)['agent']['agentStatus']
                logger.error(f"Final agent status: {final_status}")
            except Exception as get_e:
                logger.error(f"Could not retrieve final agent status after wait failure: {get_e}")
            raise Exception(f"Agent preparation or alias setup failed for {agent_id}")

    except Exception as e:
        logger.error(f"Error preparing agent {agent_id}: {e}")
        # Handle error, maybe exit
        sys.exit(1)


# --- Agent Invocation Function ---

def test_agent_invocation(agent_runtime_client, agent_id, agent_alias_id, session_id, prompt):
    """Invokes the agent and prints the response."""
    logger.info(f"--- Testing Agent Invocation (Agent ID: {agent_id}, Alias: {agent_alias_id}) ---")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Prompt: \"{prompt}\"")

    try:
        response = agent_runtime_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=prompt,
            enableTrace=True # Enable trace for debugging
        )

        logger.info("Agent invocation successful. Processing response...")
        print(f"Response: {response}")
        completion_text = ""
        trace_events = []

        # The response is a stream. Iterate through the chunks.
        for event in response.get('completion', []):
            print(f"Event: {event}")
            if 'chunk' in event:
                data = event['chunk'].get('bytes', b'')
                decoded_chunk = data.decode('utf-8')
                completion_text += decoded_chunk
                # Log chunk as it arrives
                # print(f"Chunk: {decoded_chunk}") 
            elif 'trace' in event:
                trace_part = event['trace'].get('trace')
                if trace_part:
                     # Log the full trace part object for detailed debugging
                    #  print(f"Trace Event: {json.dumps(trace_part)}")
                     trace_events.append(trace_part)
            else:
                 logger.warning(f"Unhandled event type in stream: {event}")

        # Log final combined response
        logger.info(f"\n--- Agent Final Response ---\n{completion_text}")
        
        # Keep trace summary log (optional, can be removed if too verbose)
        if trace_events:
             logger.info("\n--- Invocation Trace Summary (Repeated for context) ---")
             for i, trace in enumerate(trace_events):
                  trace_type = trace.get('type')
                  step_type = trace.get('orchestration', {}).get('stepType')
                  model_invocation_input = trace.get('modelInvocationInput')
                  if model_invocation_input:
                      fm_input = model_invocation_input.get('text',
                          json.dumps(model_invocation_input.get('invocationInput',{}).get('toolConfiguration',{})) # Handle tool input
                      )
                      logger.info(f"  Model Input Snippet: {fm_input[:150]}...")
                  observation = trace.get('observation')
                  if observation:
                      logger.info(f"  Observation: {observation}")
                  log_line = f"Trace {i+1}: Type={trace_type}, Step={step_type}"
                  rationale = trace.get('rationale', {}).get('text')
                  if rationale: log_line += f", Rationale=\"{rationale[:100]}...\""
                  logger.info(log_line) # Log summary line

        return completion_text

    except ClientError as e:
        logger.error(f"Error invoking agent: {e}")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
         logger.error(f"Unexpected error during agent invocation: {e}")
         logger.error(traceback.format_exc())
         return None

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Bedrock Agent Experiment Script ---")

    if not check_environment_variables():
        exit(1)

    # Initialize all clients, including the agent client
    bedrock_runtime_client, iam_client, lambda_client, bedrock_agent_client, bedrock_agent_runtime_client = initialize_aws_clients()
    cb_cluster = connect_couchbase()

    # Use the setup functions with the script's config variables
    cb_collection = setup_collection(cb_cluster, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)

    # Pass required args to setup_search_index
    setup_search_index(cb_cluster, INDEX_NAME, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME, INDEX_JSON_PATH)

    # Clear any existing documents from previous runs
    clear_collection(cb_cluster, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)

    # --- Vector Store Initialization and Data Loading ---
    try:
        logger.info(f"Initializing Bedrock Embeddings client with model: {EMBEDDING_MODEL_ID}")
        embeddings = BedrockEmbeddings(
            client=bedrock_runtime_client,
            model_id=EMBEDDING_MODEL_ID
        )
        logger.info("Successfully created Bedrock embeddings client.")

        logger.info(f"Initializing CouchbaseSearchVectorStore with index: {INDEX_NAME}")
        vector_store = CouchbaseSearchVectorStore(
            cluster=cb_cluster,
            bucket_name=CB_BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        logger.info("Successfully created Couchbase vector store.")

        # Load documents from JSON file
        logger.info(f"Looking for documents at: {DOCS_JSON_PATH}")
        if not os.path.exists(DOCS_JSON_PATH):
             logger.error(f"Documents file not found: {DOCS_JSON_PATH}")
             raise FileNotFoundError(f"Documents file not found: {DOCS_JSON_PATH}")

        with open(DOCS_JSON_PATH, 'r') as f:
            data = json.load(f)
            documents_to_load = data.get('documents', [])
        logger.info(f"Loaded {len(documents_to_load)} documents from {DOCS_JSON_PATH}")

        # Add documents to vector store
        if documents_to_load:
            logger.info(f"Adding {len(documents_to_load)} documents to vector store...")
            texts = [doc.get('text', '') for doc in documents_to_load]
            metadatas = []
            for i, doc in enumerate(documents_to_load):
                metadata_raw = doc.get('metadata', {})
                if isinstance(metadata_raw, str):
                    try:
                        metadata = json.loads(metadata_raw)
                        if not isinstance(metadata, dict):
                             logger.warning(f"Metadata for doc {i} parsed from string is not a dict: {metadata}. Using empty dict.")
                             metadata = {}
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse metadata string for doc {i}: {metadata_raw}. Using empty dict.")
                        metadata = {}
                elif isinstance(metadata_raw, dict):
                    metadata = metadata_raw
                else:
                    logger.warning(f"Metadata for doc {i} is not a string or dict: {metadata_raw}. Using empty dict.")
                    metadata = {}
                metadatas.append(metadata)

            inserted_ids = vector_store.add_texts(texts=texts, metadatas=metadatas)
            logger.info(f"Successfully added {len(inserted_ids)} documents to the vector store.")
        else:
             logger.warning("No documents found in the JSON file to add.")

    except FileNotFoundError as e:
         logger.error(f"Setup failed: {e}")
         exit(1)
    except Exception as e:
        logger.error(f"Error during vector store setup or data loading: {e}")
        logger.error(traceback.format_exc())
        exit(1) 

    logger.info("--- Couchbase Setup and Data Loading Complete ---")


    # --- Create IAM Role ---
    agent_role_name = "bedrock_agent_lambda_exp_role"
    try:
        agent_role_arn = create_agent_role(iam_client, agent_role_name, AWS_ACCOUNT_ID)
        logger.info(f"Agent IAM Role ARN: {agent_role_arn}")
    except Exception as e:
        logger.error(f"Failed to create/verify IAM role: {e}")
        logger.error(traceback.format_exc())
        exit(1)


    # --- Deploy Lambda Function (Single Function) ---
    search_format_lambda_name = "bedrock_agent_search_format_exp"
    lambda_source_dir = os.path.join(SCRIPT_DIR, 'lambda_functions')
    lambda_build_dir = SCRIPT_DIR # Final zip ends up here

    logger.info("--- Starting Lambda Deployment (Single Function) --- ")
    search_format_lambda_arn = None
    search_format_zip_path = None

    try:
        # Delete old lambdas if they exist (optional, but good cleanup)
        delete_lambda_function(lambda_client, "bedrock_agent_researcher_exp")
        delete_lambda_function(lambda_client, "bedrock_agent_writer_exp")
        # Delete the new lambda if it exists from a previous run
        delete_lambda_function(lambda_client, search_format_lambda_name)

        search_format_zip_path = package_function("bedrock_agent_search_and_format", lambda_source_dir, lambda_build_dir)

        search_format_lambda_arn = create_lambda_function(
            lambda_client=lambda_client, function_name=search_format_lambda_name,
            handler='lambda_function.lambda_handler', role_arn=agent_role_arn,
            zip_file=search_format_zip_path, region=AWS_REGION
        )
        logger.info(f"Search/Format Lambda Deployed: {search_format_lambda_arn}")

    except FileNotFoundError as e:
         logger.error(f"Lambda packaging failed: Required file not found. {e}")
         exit(1)
    except Exception as e:
        logger.error(f"Lambda deployment failed: {e}")
        logger.error(traceback.format_exc())
        exit(1)
    finally:
        logger.info("Cleaning up deployment zip file...")
        if search_format_zip_path and os.path.exists(search_format_zip_path):
            os.remove(search_format_zip_path)

    logger.info("--- Lambda Deployment Complete --- ")


    # --- Agent Setup --- 
    agent_name = f"couchbase_search_format_agent_exp"
    agent_id = None
    agent_arn = None
    alias_name = "prod" # Define alias name here
    agent_alias_id_to_use = None # Initialize alias ID here

    # 1. Attempt to find and delete existing agent
    logger.info(f"Checking for and deleting existing agent: {agent_name}")
    try:
        found_agent_id = get_agent_by_name(bedrock_agent_client, agent_name)
        if found_agent_id:
            logger.warning(f"--- Found existing agent {found_agent_id}. Deleting Agent Resources --- ")
            delete_agent_and_resources(bedrock_agent_client, agent_name) # Deletes based on name
            logger.info(f"Deletion process completed for any existing agent named {agent_name}.")
        else:
             logger.info(f"Agent '{agent_name}' not found. No deletion needed.")
    except Exception as e:
        # Log error during find/delete but proceed to creation attempt
        logger.error(f"Error during agent finding/deletion phase: {e}. Proceeding to creation attempt.")

    # 2. Always attempt to create the agent after the delete phase
    logger.info(f"--- Creating Agent: {agent_name} ---")
    try:
        agent_id, agent_arn = create_agent(
            agent_client=bedrock_agent_client,
            agent_name=agent_name,
            agent_role_arn=agent_role_arn,
            foundation_model_id=AGENT_MODEL_ID
        )
        if not agent_id:
            raise Exception("create_agent function did not return a valid agent ID.")
    except Exception as e:
        logger.error(f"Failed to create agent '{agent_name}': {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) # Exit if creation fails

    # --- Action Group Creation/Update (Now assumes agent_id is valid) ---
    action_group_name = "SearchAndFormatActionGroup"
    action_group_id = create_action_group(
        agent_client=bedrock_agent_client,
        agent_id=agent_id,
        action_group_name=action_group_name,
        function_arn=search_format_lambda_arn,
        # schema_path=None # No longer needed explicitly if default is None
    )

    # Add a slightly longer wait after action group modification/creation
    logger.info("Waiting 30s after action group setup before preparing agent...")
    time.sleep(30)


    # --- Prepare Agent ---
    if agent_id:
        logger.info(f"--- Preparing Existing Agent: {agent_id} ---")
        preparation_successful = False
        try:
            # --- Preparation --- 
            prepare_agent(bedrock_agent_client, agent_id)
            logger.info(f"Agent {agent_id} preparation seems complete (waiter succeeded).")
            preparation_successful = True # Flag success

            # --- Alias Handling (runs only if preparation succeeded) ---
            # Moved inside the main try block
            if preparation_successful:
                try:
                    # --- Alias Creation/Update (Using Custom Alias "prod") ---
                    logger.info(f"Checking for alias '{alias_name}' for agent {agent_id}...")
                    existing_alias = None
                    paginator = bedrock_agent_client.get_paginator('list_agent_aliases')
                    for page in paginator.paginate(agentId=agent_id):
                        for alias_summary in page.get('agentAliasSummaries', []):
                            if alias_summary.get('agentAliasName') == alias_name:
                                existing_alias = alias_summary
                                break
                        if existing_alias:
                            break
                    
                    if existing_alias:
                        agent_alias_id_to_use = existing_alias['agentAliasId']
                        logger.info(f"Using existing alias '{alias_name}' with ID: {agent_alias_id_to_use}.")
                    else:
                        logger.info(f"Alias '{alias_name}' not found. Creating new alias...")
                        create_alias_response = bedrock_agent_client.create_agent_alias(
                            agentId=agent_id,
                            agentAliasName=alias_name
                            # routingConfiguration removed - defaults to latest prepared (DRAFT)
                        )
                        # Assign the ID immediately after creation, inside the else block
                        agent_alias_id_to_use = create_alias_response.get('agentAlias', {}).get('agentAliasId')
                        logger.info(f"Successfully created alias '{alias_name}' with ID: {agent_alias_id_to_use}. (Defaults to latest prepared version - DRAFT)")

                    if not agent_alias_id_to_use:
                         # Use a more specific exception
                         raise ValueError(f"Failed to get a valid alias ID for '{alias_name}'")

                    logger.info(f"Waiting 10s for alias '{alias_name}' changes to propagate...")
                    time.sleep(10)

                except ClientError as e: # Catch errors from alias logic
                    logger.error(f"Failed to create/update alias '{alias_name}' for agent {agent_id}: {e}")
                    logger.error(traceback.format_exc())
                    sys.exit(1) # Exit if alias setup fails
                # End of Alias specific try/except
            # End of if preparation_successful

        except Exception as e: # Catch errors from the outer try (preparation or unhandled alias error)
            logger.error(f"Error during agent preparation or alias setup for {agent_id}: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1) # Exit if preparation or alias setup fails

    # --- Test Invocation ---
    # Agent ID and custom alias ID should be valid here
    if agent_id and agent_alias_id_to_use: # Check both are set
         session_id = str(uuid.uuid4()) 
         test_prompt = "Search for information about Project Chimera and format the results using bullet points."
         logger.info(f"--- Invoking Agent {agent_id} using Alias '{alias_name}' ({agent_alias_id_to_use}) ---") # Updated log
         test_agent_invocation(
             agent_runtime_client=bedrock_agent_runtime_client,
             agent_id=agent_id,
             agent_alias_id=agent_alias_id_to_use, # Added missing argument
             session_id=session_id,
             prompt=test_prompt
         )
    else:
         logger.error("Agent ID or Alias ID not available, skipping invocation test.")

    logger.info("--- Script Execution Finished (Agent Setup & Test Complete) --- ")
