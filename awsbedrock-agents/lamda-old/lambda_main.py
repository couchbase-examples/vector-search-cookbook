"""## Setup and Configuration

First, let's import the necessary libraries and set up our environment:
"""

import json
import logging
import os
import time
import uuid
import subprocess
from datetime import timedelta

import boto3
from botocore.exceptions import ClientError
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import (InternalServerFailureException,
                                  QueryIndexAlreadyExistsException,
                                  ServiceUnavailableException)
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""### Ensure IAM Permissions ###"""
def ensure_invoke_model_permission(iam_client, role_name, model_id, region, account_id):
    """Ensures the IAM role has bedrock:InvokeModel permission for the specified model."""
    policy_name = "BedrockAgentModelInvokePermissions"
    model_arn = f"arn:aws:bedrock:{region}::{account_id}:foundation-model/{model_id}"
    # Also include the Titan embedding model used by the vector store
    titan_embed_arn = f"arn:aws:bedrock:{region}::{account_id}:foundation-model/amazon.titan-embed-text-v2:0"
    
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream" # Often needed too
                ],
                "Resource": [
                    model_arn, 
                    titan_embed_arn 
                    # Add other models here if needed by the role
                ]
            }
        ]
    }
    
    try:
        print(f"Ensuring inline policy '{policy_name}' for role '{role_name}' allows invocation of '{model_id}'...")
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document) # Convert dict to JSON string
        )
        print(f"Successfully attached/updated inline policy '{policy_name}' for role '{role_name}'.")
        # Allow some time for IAM changes to propagate
        print("Waiting longer (30s) for IAM changes to propagate...")
        time.sleep(30) 
    except ClientError as e:
        print(f"Error updating IAM policy '{policy_name}' for role '{role_name}': {e}")
        print("Please ensure the script has permissions to call iam:PutRolePolicy.")
        # Depending on the error, you might want to raise it or handle differently
        raise # Re-raise the exception to halt execution if permission update fails
    except Exception as e:
        print(f"An unexpected error occurred during IAM policy update: {e}")
        raise

"""### Load Environment Variables

Load environment variables from the .env file. Make sure to create a .env file with the necessary credentials before running this notebook.
"""

# Load environment variables
load_dotenv()

# Couchbase Configuration
CB_HOST = os.getenv("CB_HOST", "couchbase://localhost")
CB_USERNAME = os.getenv("CB_USERNAME", "Administrator")
CB_PASSWORD = os.getenv("CB_PASSWORD", "password")
CB_BUCKET_NAME = os.getenv("CB_BUCKET_NAME", "vector-search-testing")
SCOPE_NAME = os.getenv("SCOPE_NAME", "shared")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bedrock")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_search_bedrock")

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")

# Check if required environment variables are set
required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ACCOUNT_ID"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these variables in your .env file")
else:
    print("All required environment variables are set")

"""### Initialize AWS Clients

Set up the AWS clients for Bedrock and other services:
"""

# Initialize AWS session
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Initialize AWS clients from session
iam_client = session.client('iam')
bedrock_client = session.client('bedrock')
bedrock_agent_client = session.client('bedrock-agent')
bedrock_runtime = session.client('bedrock-runtime')
bedrock_runtime_client = session.client('bedrock-agent-runtime')
lambda_client = session.client('lambda')

print("AWS clients initialized successfully")

"""## Set Up Couchbase and Vector Store

Now let's set up the Couchbase connection, collections, and vector store:
"""

def setup_collection(cluster, bucket_name, scope_name, collection_name):
    """Set up Couchbase collection"""
    try:
        # Check if bucket exists, create if it doesn't
        try:
            bucket = cluster.bucket(bucket_name)
            print(f"Bucket '{bucket_name}' exists.")
        except Exception as e:
            print(f"Bucket '{bucket_name}' does not exist. Creating it...")
            bucket_settings = CreateBucketSettings(
                name=bucket_name,
                bucket_type='couchbase',
                ram_quota_mb=1024,
                flush_enabled=True,
                num_replicas=0
            )
            cluster.buckets().create_bucket(bucket_settings)
            bucket = cluster.bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")

        bucket_manager = bucket.collections()

        # Check if scope exists, create if it doesn't
        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(scope.name == scope_name for scope in scopes)

        if not scope_exists and scope_name != "_default":
            print(f"Scope '{scope_name}' does not exist. Creating it...")
            bucket_manager.create_scope(scope_name)
            print(f"Scope '{scope_name}' created successfully.")

        # Check if collection exists, create if it doesn't
        collections = bucket_manager.get_all_scopes()
        collection_exists = any(
            scope.name == scope_name and collection_name in [col.name for col in scope.collections]
            for scope in collections
        )

        if not collection_exists:
            print(f"Collection '{collection_name}' does not exist. Creating it...")
            bucket_manager.create_collection(scope_name, collection_name)
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists. Skipping creation.")

        # Wait for collection to be ready
        collection = bucket.scope(scope_name).collection(collection_name)
        time.sleep(2)  # Give the collection time to be ready for queries

        # Ensure primary index exists
        try:
            cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`").execute()
            print("Primary index present or created successfully.")
        except Exception as e:
            print(f"Error creating primary index: {str(e)}")

        # Clear all documents in the collection
        try:
            query = f"DELETE FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            cluster.query(query).execute()
            print("All documents cleared from the collection.")
        except Exception as e:
            print(f"Error while clearing documents: {str(e)}. The collection might be empty.")

        return collection
    except Exception as e:
        print(f"Error setting up collection: {str(e)}")
        raise

def setup_indexes(cluster):
    """Set up search indexes"""
    try:
        # Construct path relative to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        index_file_path = os.path.join(script_dir, 'aws_index.json')
        # Load index definition from file using the constructed path
        print(f"Looking for index definition at: {index_file_path}")
        with open(index_file_path, 'r') as file:
            index_definition = json.load(file)
            print(f"Loaded index definition from {index_file_path}")
    except Exception as e:
        print(f"Error loading index definition: {str(e)}")
        raise

    try:
        scope_index_manager = cluster.bucket(CB_BUCKET_NAME).scope(SCOPE_NAME).search_indexes()

        # Check if index already exists
        existing_indexes = scope_index_manager.get_all_indexes()
        index_name = index_definition["name"]

        if index_name in [index.name for index in existing_indexes]:
            print(f"Index '{index_name}' found")
        else:
            print(f"Creating new index '{index_name}'...")

        # Create SearchIndex object from JSON definition
        search_index = SearchIndex.from_json(index_definition)

        # Upsert the index (create if not exists, update if exists)
        scope_index_manager.upsert_index(search_index)
        print(f"Index '{index_name}' successfully created/updated.")

    except QueryIndexAlreadyExistsException:
        print(f"Index '{index_name}' already exists. Skipping creation/update.")
    except ServiceUnavailableException:
        print("Search service is not available. Please ensure the Search service is enabled in your Couchbase cluster.")
    except InternalServerFailureException as e:
        print(f"Internal server error: {str(e)}")
        raise

# Connect to Couchbase
auth = PasswordAuthenticator(CB_USERNAME, CB_PASSWORD)
options = ClusterOptions(auth)
cluster = Cluster(CB_HOST, options)
cluster.wait_until_ready(timedelta(seconds=5))
print("Successfully connected to Couchbase")

# Set up collections
collection = setup_collection(cluster, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)
print("Collections setup complete")

# Set up search indexes
setup_indexes(cluster)
print("Search indexes setup complete")

# Initialize Bedrock runtime client for embeddings
embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v2:0"
)
print("Successfully created Bedrock embeddings client")

# Initialize vector store
vector_store = CouchbaseSearchVectorStore(
    cluster=cluster,
    bucket_name=CB_BUCKET_NAME,
    scope_name=SCOPE_NAME,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    index_name=INDEX_NAME
)
print("Successfully created vector store")

"""## Load Documents from JSON File

Let's load the documents from the documents.json file:
"""

# Load documents from JSON file
try:
    # Construct path relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_file_path = os.path.join(script_dir, 'documents.json')
    print(f"Looking for documents at: {docs_file_path}")
    with open(docs_file_path, 'r') as f:
        data = json.load(f)
        documents = data.get('documents', [])
    print(f"Loaded {len(documents)} documents from {docs_file_path}")
except Exception as e:
    print(f"Error loading documents: {str(e)}")
    raise

# Add documents to vector store
print(f"Adding {len(documents)} documents to vector store...")
for i, doc in enumerate(documents, 1):
    text = doc.get('text', '')
    metadata = doc.get('metadata', {})

    # Add document to vector store
    metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata or {}
    doc_ids = vector_store.add_texts([text], [metadata_dict])
    doc_id = doc_ids[0] if doc_ids else None
    print(f"Added document {i}/{len(documents)} with ID: {doc_id}")

    # Add small delay between requests
    time.sleep(1)

print(f"\nProcessing complete: {len(documents)}/{len(documents)} documents added successfully")

"""## Lambda Approach Implementation

Now let's implement the Lambda approach for Bedrock agents. This approach involves deploying Lambda functions that will be invoked by the Bedrock agents.

### Deploy Lambda Functions

First, let's deploy the Lambda functions that will be invoked by our Bedrock agents. We'll create a .env file for the Lambda functions with the Couchbase configuration.
"""

# Construct path relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
lambda_env_path = os.path.join(script_dir, 'lambda_functions', '.env')
print(f"Creating .env file for Lambda functions at: {lambda_env_path}")

# Create a .env file for the Lambda functions
with open(lambda_env_path, 'w') as f:
    f.write(f"CB_HOST={CB_HOST}\n")
    f.write(f"CB_USERNAME={CB_USERNAME}\n")
    f.write(f"CB_PASSWORD={CB_PASSWORD}\n")
    f.write(f"CB_BUCKET_NAME={CB_BUCKET_NAME}\n")
    f.write(f"SCOPE_NAME={SCOPE_NAME}\n")
    f.write(f"COLLECTION_NAME={COLLECTION_NAME}\n")
    f.write(f"INDEX_NAME={INDEX_NAME}\n")

print(f"Created .env file for Lambda functions at {lambda_env_path}")

# Deploy Lambda functions
print("Deploying Lambda functions...")
# Construct path to deploy.py relative to script dir
deploy_script_path = os.path.join(script_dir, 'lambda_functions', 'deploy.py')
print(f"Running deploy script: {deploy_script_path}")
try:
    subprocess.run([
        'python3',
        deploy_script_path
    ], check=True, cwd=os.path.join(script_dir, 'lambda_functions')) # Run from lambda_functions dir
    print("Lambda functions deployed successfully")
except subprocess.CalledProcessError as e:
    print(f"Error deploying Lambda functions: {str(e)}")
    raise RuntimeError("Failed to deploy Lambda functions")

"""## Lambda Approach Helper Functions

Let's define some helper functions for the Lambda approach:
"""

def wait_for_agent_status(bedrock_agent_client, agent_id, target_statuses=['Available', 'PREPARED', 'NOT_PREPARED'], max_attempts=30, delay=2):
    """Wait for agent to reach any of the target statuses"""
    for attempt in range(max_attempts):
        try:
            response = bedrock_agent_client.get_agent(agentId=agent_id)
            current_status = response['agent']['agentStatus']

            if current_status in target_statuses:
                print(f"Agent {agent_id} reached status: {current_status}")
                return current_status
            elif current_status == 'FAILED':
                print(f"Agent {agent_id} failed")
                return 'FAILED'

            print(f"Agent status: {current_status}, waiting... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)

        except Exception as e:
            print(f"Error checking agent status: {str(e)}")
            time.sleep(delay)

    return current_status

def create_agent(bedrock_agent_client, name, instructions, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
    """Create or verify a Bedrock agent"""
    role_name = "bedrock_agent_lambda_role"
    try:
        # List existing agents
        existing_agents = bedrock_agent_client.list_agents()
        existing_agent = next(
            (agent for agent in existing_agents['agentSummaries']
             if agent['agentName'] == name),
            None
        )

        agent_id = None # Initialize agent_id

        # Handle existing agent
        if existing_agent:
            agent_id = existing_agent['agentId']
            print(f"Found existing agent '{name}' with ID: {agent_id}")

            # Check agent status and model
            response = bedrock_agent_client.get_agent(agentId=agent_id)
            status = response['agent']['agentStatus']
            current_model_id = response['agent'].get('foundationModel') # Use .get for safety

            # --- Check if model needs update OR if agent is in bad state ---
            needs_recreation = False
            if current_model_id != model_id:
                 print(f"Agent '{name}' (ID: {agent_id}) has incorrect model ID ('{current_model_id}'). Desired: '{model_id}'. Flagging for recreation.")
                 needs_recreation = True
            elif status in ['NOT_PREPARED', 'FAILED', 'UPDATING', 'PREPARING', 'CREATING']: # Added CREATING
                 print(f"Agent '{name}' (ID: {agent_id}) is in unusable state: {status}. Flagging for recreation.")
                 needs_recreation = True

            # Delete if flagged for recreation
            if needs_recreation:
                 print(f"Deleting agent '{name}' (ID: {agent_id}) for recreation.")
                 try:
                     # Clean up aliases first if possible
                     aliases = bedrock_agent_client.list_agent_aliases(agentId=agent_id).get('agentAliasSummaries', [])
                     for alias in aliases:
                         print(f"  Deleting alias {alias['agentAliasName']} ({alias['agentAliasId']})...")
                         bedrock_agent_client.delete_agent_alias(agentId=agent_id, agentAliasId=alias['agentAliasId'])
                         time.sleep(2) # Short delay after alias deletion
                 except Exception as alias_e:
                     print(f"  Warning: Could not delete aliases for {agent_id}: {alias_e}")

                 try:
                     # Also attempt to delete action groups if they exist for the DRAFT version
                     action_groups = bedrock_agent_client.list_agent_action_groups(agentId=agent_id, agentVersion='DRAFT').get('actionGroupSummaries', [])
                     for ag in action_groups:
                         print(f"  Deleting action group {ag['actionGroupName']} ({ag['actionGroupId']})...")
                         bedrock_agent_client.delete_agent_action_group(agentId=agent_id, agentVersion='DRAFT', actionGroupId=ag['actionGroupId'])
                         time.sleep(2) # Short delay
                 except Exception as ag_e:
                     print(f"  Warning: Could not delete action groups for {agent_id}: {ag_e}")


                 bedrock_agent_client.delete_agent(agentId=agent_id)
                 print(f"Waiting after deletion...")
                 time.sleep(20)  # Increase wait after deletion
                 existing_agent = None
                 agent_id = None # Reset agent_id


        # Create new agent if needed (or if old one was deleted)
        if not agent_id: # Check if agent_id is None (meaning it needs creation)
            print(f"Creating new agent '{name}'")
            # --- Get the Agent Role ARN ---
            agent_role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{role_name}"
            print(f"Using Agent Role ARN: {agent_role_arn}")
            # ----------------------------
            agent = bedrock_agent_client.create_agent(
                agentName=name,
                description=f"{name.title()} agent for document operations",
                instruction=instructions,
                idleSessionTTLInSeconds=1800,
                foundationModel=model_id,
                agentResourceRoleArn=agent_role_arn
            )
            agent_id = agent['agent']['agentId']
            print(f"Created new agent '{name}' with ID: {agent_id}")
            # Wait for it to become NOT_PREPARED before returning
            status = wait_for_agent_status(bedrock_agent_client, agent_id, target_statuses=['NOT_PREPARED'])
            if status != 'NOT_PREPARED':
                 raise Exception(f"Newly created agent {agent_id} did not reach NOT_PREPARED state, got {status}")
        else: # This else block is now only reached if agent exists, is in a good state, AND has the correct model
            # Ensure it's in a usable state before proceeding (redundant check is ok)
            print(f"Using existing agent '{name}' with ID: {agent_id}. Verifying status...")
            status = wait_for_agent_status(bedrock_agent_client, agent_id, target_statuses=['NOT_PREPARED', 'PREPARED', 'Available'])
            # If wait_for_agent_status still finds it bad after the initial check, something is wrong
            if status not in ['NOT_PREPARED', 'PREPARED', 'Available']:
                 raise Exception(f"Agent {agent_id} entered bad state {status} unexpectedly after initial check.")


        # --- Alias creation is REMOVED from here ---

        print(f"Successfully created/verified agent '{name}' with ID: {agent_id}")
        # Return only the agent ID, Alias will be created later
        return agent_id

    except Exception as e:
        print(f"Error in create_agent function for '{name}': {str(e)}")
        # Print traceback for detailed debugging
        print(traceback.format_exc())
        raise RuntimeError(f"Failed during create_agent for '{name}': {str(e)}")

def invoke_agent(agent_id, agent_alias_id, session_id, input_text):
    """Invokes the Bedrock agent and streams the response."""
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
    # Initialize variables to store the stream and final text
    response_stream = ""
    final_text_response = ""

    try:
        print(f"--- Invoking agent {agent_id} (Alias: {agent_alias_id}) ---")
        # Truncate input text for printing if too long
        truncated_input = input_text[:100] + '...' if len(input_text) > 100 else input_text
        print(f"Input Text: {repr(truncated_input)}") # Print truncated input

        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=input_text,
            enableTrace=True # Enable trace for debugging
        )
        
        print("--- Processing Response Stream ---")
        event_count = 0
        for event in response['completion']:
            event_count += 1
            event_type = list(event.keys()) # Get the key (e.g., 'chunk', 'trace')
            print(f"\n[Event {event_count}] Type: {event_type}")
            
            # Print raw event content for debugging 
            print("Event Content (raw, contains non-serializable types):")
            print(event)
            
            if 'chunk' in event:
                data = event['chunk']['bytes']
                chunk_text = data.decode('utf-8')
                response_stream += chunk_text # Concatenate chunk text
                print(f"  -> Received Chunk: {repr(chunk_text)}")
            elif 'trace' in event:
                 print("  -> Received Trace event.")
                 pass # Trace handling can be complex, logging raw above
            # Note: The structure for finalResponse seems inconsistent/unreliable 
            # for getting the text across different Bedrock versions/models.
            # Relying on concatenating chunks is more robust.

        print(f"\n--- Stream Processing Complete ({event_count} events) ---")
        final_text_response = response_stream # Assign the concatenated stream
        print("--- Returning Result --- ")
        print(repr(final_text_response)) # Print the final captured text using repr
        return final_text_response # Return the concatenated text

    except ClientError as e:
        # Extract specific error message if possible
        error_message = str(e)
        if isinstance(e, bedrock_agent_runtime.exceptions.DependencyFailedException):
            # Often contains more specific details in the message
            error_message = e.response.get('Error', {}).get('Message', str(e))
        elif isinstance(e, bedrock_agent_runtime.exceptions.ValidationException):
             error_message = e.response.get('Error', {}).get('Message', str(e))

        print(f"\n--- ERROR during agent invocation ---")
        print(traceback.format_exc())
        print(f"\n!!! Error invoking agent {agent_id}: {error_message} !!!")
        return f"ERROR: Failed to invoke agent: {error_message}"
    except Exception as e:
        print(f"\n--- ERROR during agent invocation (Unexpected) ---")
        print(traceback.format_exc())
        print(f"\n!!! Unexpected error invoking agent {agent_id}: {e} !!!")
        return f"ERROR: Unexpected error: {e}"

"""## Define Agent Instructions and Functions

Now let's define the instructions and functions for our agents:
"""

# Researcher agent instructions
researcher_instructions = """
You are a Research Assistant that helps users find relevant information in documents.
Your capabilities include:
1. Searching through documents using semantic similarity
2. Providing relevant document excerpts
3. Answering questions based on document content
"""

# Researcher agent functions
researcher_functions = [{
    "name": "search_documents",
    "description": "Search for relevant documents using semantic similarity",
    "parameters": {
        "query": {
            "type": "string",
            "description": "The search query",
            "required": True
        },
        "k": {
            "type": "integer",
            "description": "Number of results to return",
            "required": False
        }
    },
    "requireConfirmation": "DISABLED"
}]

# Writer agent instructions
writer_instructions = """
You are a Content Writer Assistant that helps format and present research findings.
Your capabilities include:
1. Formatting research findings in a user-friendly way
2. Creating clear and engaging summaries
3. Organizing information logically
4. Highlighting key insights
"""

# Writer agent functions
writer_functions = [{
    "name": "format_content",
    "description": "Format and present research findings",
    "parameters": {
        "content": {
            "type": "string",
            "description": "The research findings to format",
            "required": True
        },
        "style": {
            "type": "string",
            "description": "The desired presentation style (e.g., summary, detailed, bullet points)",
            "required": False
        }
    },
    "requireConfirmation": "DISABLED"
}]

"""## Run Lambda Approach

Now let's run the Lambda approach with our agents:
"""

# Define the new model ID
AGENT_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
print(f"Using model: {AGENT_MODEL_ID}")

# --- Ensure the agent role has invoke permission for the selected model ---
try:
    ensure_invoke_model_permission(iam_client, "bedrock_agent_lambda_role", AGENT_MODEL_ID, AWS_REGION, AWS_ACCOUNT_ID)
except Exception as e:
    print(f"Halting script due to IAM permission update failure: {e}")
    exit(1) # Stop execution if permissions can't be set
# ----------------------------------------------------------------------

# Create researcher agent core
researcher_alias = None # Initialize alias
try:
    researcher_id = create_agent(
        bedrock_agent_client,
        "researcher",
        researcher_instructions,
        model_id=AGENT_MODEL_ID # Pass the new model ID
    )
    print(f"Researcher agent core created/verified with ID: {researcher_id}")
except Exception as e:
    print(f"Failed to create researcher agent core: {str(e)}")
    researcher_id = None

# Create writer agent core
writer_alias = None # Initialize alias
try:
    writer_id = create_agent(
        bedrock_agent_client,
        "writer",
        writer_instructions,
        model_id=AGENT_MODEL_ID # Pass the new model ID
    )
    print(f"Writer agent core created/verified with ID: {writer_id}")
except Exception as e:
    print(f"Failed to create writer agent core: {str(e)}")
    writer_id = None

if not researcher_id or not writer_id:
    raise RuntimeError("Failed to create agent core(s). Cannot proceed.")

# Create action group for researcher agent with Lambda executor
try:
    bedrock_agent_client.create_agent_action_group(
        agentId=researcher_id,
        agentVersion="DRAFT",
        actionGroupExecutor={
            "lambda": f"arn:aws:lambda:{AWS_REGION}:{AWS_ACCOUNT_ID}:function:bedrock_agent_researcher"
        },  # This is the key for Lambda approach
        actionGroupName="researcher_actions",
        functionSchema={"functions": researcher_functions},
        description="Action group for researcher operations with Lambda"
    )
    print("Created researcher Lambda action group")
except bedrock_agent_client.exceptions.ConflictException:
    print("Researcher Lambda action group already exists")

# Prepare researcher agent
print("Preparing researcher agent...")
bedrock_agent_client.prepare_agent(agentId=researcher_id)
status = wait_for_agent_status(
    bedrock_agent_client,
    researcher_id,
    target_statuses=['PREPARED', 'Available']
)
print(f"Researcher agent preparation completed with status: {status}")
if status not in ['PREPARED', 'Available']:
     raise RuntimeError(f"Researcher agent failed to prepare. Status: {status}")

# Create action group for writer agent with Lambda executor
try:
    bedrock_agent_client.create_agent_action_group(
        agentId=writer_id,
        agentVersion="DRAFT",
        actionGroupExecutor={
            "lambda": f"arn:aws:lambda:{AWS_REGION}:{AWS_ACCOUNT_ID}:function:bedrock_agent_writer"
        },  # This is the key for Lambda approach
        actionGroupName="writer_actions",
        functionSchema={"functions": writer_functions},
        description="Action group for writer operations with Lambda"
    )
    print("Created writer Lambda action group")
except bedrock_agent_client.exceptions.ConflictException:
    print("Writer Lambda action group already exists")

# Prepare writer agent
print("Preparing writer agent...")
bedrock_agent_client.prepare_agent(agentId=writer_id)
status = wait_for_agent_status(
    bedrock_agent_client,
    writer_id,
    target_statuses=['PREPARED', 'Available']
)
print(f"Writer agent preparation completed with status: {status}")
if status not in ['PREPARED', 'Available']:
     raise RuntimeError(f"Writer agent failed to prepare. Status: {status}")

# --- Add Alias Creation/Retrieval AFTER Preparation ---
# Create or get alias for researcher agent AFTER preparation
try:
    researcher_alias_name = "v1"
    aliases = bedrock_agent_client.list_agent_aliases(agentId=researcher_id).get('agentAliasSummaries', [])
    researcher_alias_obj = next((a for a in aliases if a['agentAliasName'] == researcher_alias_name), None)

    if not researcher_alias_obj:
        print(f"Creating new alias '{researcher_alias_name}' for researcher agent '{researcher_id}'")
        alias_response = bedrock_agent_client.create_agent_alias(
            agentId=researcher_id,
            agentAliasName=researcher_alias_name
        )
        researcher_alias = alias_response['agentAlias']['agentAliasId']
        # Wait briefly for alias to become available if needed
        print(f"Waiting for researcher alias {researcher_alias} to be available...")
        time.sleep(10) # Increased wait time for alias propagation
    else:
        researcher_alias = researcher_alias_obj['agentAliasId']
        print(f"Using existing alias '{researcher_alias_name}' ({researcher_alias}) for researcher agent '{researcher_id}'")

except Exception as e:
    print(f"Error managing researcher alias: {e}")
    researcher_alias = None # Ensure it's None if creation failed

# Create or get alias for writer agent AFTER preparation
try:
    writer_alias_name = "v1"
    aliases = bedrock_agent_client.list_agent_aliases(agentId=writer_id).get('agentAliasSummaries', [])
    writer_alias_obj = next((a for a in aliases if a['agentAliasName'] == writer_alias_name), None)

    if not writer_alias_obj:
        print(f"Creating new alias '{writer_alias_name}' for writer agent '{writer_id}'")
        alias_response = bedrock_agent_client.create_agent_alias(
            agentId=writer_id,
            agentAliasName=writer_alias_name
        )
        writer_alias = alias_response['agentAlias']['agentAliasId']
        # Wait briefly for alias to become available if needed
        print(f"Waiting for writer alias {writer_alias} to be available...")
        time.sleep(10) # Increased wait time for alias propagation
    else:
        writer_alias = writer_alias_obj['agentAliasId']
        print(f"Using existing alias '{writer_alias_name}' ({writer_alias}) for writer agent '{writer_id}'")

except Exception as e:
    print(f"Error managing writer alias: {e}")
    writer_alias = None # Ensure it's None if creation failed

# Final check - you need valid aliases to proceed
if not researcher_alias or not writer_alias:
    raise ValueError("Failed to obtain valid Agent Alias IDs.")

print(f"\nResearcher Alias ID: {researcher_alias}")
print(f"Writer Alias ID: {writer_alias}")
# --- End of Alias Creation ---

# --- Lambda Resource-Based Permissions are now handled in deploy.py ---
# print("\nAdding/Verifying Lambda resource-based policies...")
# researcher_alias_arn = f"arn:aws:bedrock:{AWS_REGION}:{AWS_ACCOUNT_ID}:agent-alias/{researcher_id}/{researcher_alias}"
# writer_alias_arn = f"arn:aws:bedrock:{AWS_REGION}:{AWS_ACCOUNT_ID}:agent-alias/{writer_id}/{writer_alias}"
# 
# # Add permission for Researcher Agent Alias to invoke Researcher Lambda
# try:
#     # Temporarily remove SourceArn condition for debugging
#     lambda_client.add_permission(
#         FunctionName='bedrock_agent_researcher',
#         StatementId='AllowBedrockInvokeResearcherAlias', # Keep same ID for now
#         Action='lambda:InvokeFunction',
#         Principal='bedrock.amazonaws.com',
#         SourceArn=researcher_alias_arn # RE-ENABLED
#     )
#     print("Added Bedrock invoke permission to researcher Lambda (with SourceArn condition).")
# except lambda_client.exceptions.ResourceConflictException:
#     # If it conflicts, we might need to remove the old one first
#     print("Attempting to remove potentially conflicting policy before adding specific one...")
#     try:
#         lambda_client.remove_permission(
#             FunctionName='bedrock_agent_researcher',
#             StatementId='AllowBedrockInvokeResearcherAlias'
#         )
#         time.sleep(2)
#         lambda_client.add_permission(
#             FunctionName='bedrock_agent_researcher',
#             StatementId='AllowBedrockInvokeResearcherAlias',
#             Action='lambda:InvokeFunction',
#             Principal='bedrock.amazonaws.com',
#             SourceArn=researcher_alias_arn # RE-ENABLED
#         )
#         print("Re-added specific Bedrock invoke permission to researcher Lambda (with SourceArn condition).")
#     except Exception as remove_add_e:
#         print(f"Failed to replace permission for researcher: {remove_add_e}")
#         # Might need manual intervention if this fails
# except Exception as perm_e:
#     print(f"Error modifying permission for researcher Lambda: {perm_e}")
# 
# 
# # Add permission for Writer Agent Alias to invoke Writer Lambda
# try:
#     # Temporarily remove SourceArn condition for debugging
#     lambda_client.add_permission(
#         FunctionName='bedrock_agent_writer',
#         StatementId='AllowBedrockInvokeWriterAlias', # Keep same ID for now
#         Action='lambda:InvokeFunction',
#         Principal='bedrock.amazonaws.com',
#         SourceArn=writer_alias_arn # RE-ENABLED
#     )
#     print("Added Bedrock invoke permission to writer Lambda (with SourceArn condition).")
# except lambda_client.exceptions.ResourceConflictException:
#     # If it conflicts, we might need to remove the old one first
#     print("Attempting to remove potentially conflicting policy before adding specific one...")
#     try:
#         lambda_client.remove_permission(
#             FunctionName='bedrock_agent_writer',
#             StatementId='AllowBedrockInvokeWriterAlias'
#         )
#         time.sleep(2)
#         lambda_client.add_permission(
#             FunctionName='bedrock_agent_writer',
#             StatementId='AllowBedrockInvokeWriterAlias',
#             Action='lambda:InvokeFunction',
#             Principal='bedrock.amazonaws.com',
#             SourceArn=writer_alias_arn # RE-ENABLED
#         )
#         print("Re-added specific Bedrock invoke permission to writer Lambda (with SourceArn condition).")
#     except Exception as remove_add_e:
#         print(f"Failed to replace permission for writer: {remove_add_e}")
#         # Might need manual intervention if this fails
# except Exception as perm_e:
#     print(f"Error modifying permission for writer Lambda: {perm_e}")
# ----------------------------------------------------------------


"""## Test the Agents

Let's test our agents by asking the researcher agent to search for information and the writer agent to format the results:
"""

# Ensure aliases are valid before invoking
if not researcher_alias or not writer_alias:
    raise ValueError("Failed to obtain valid Agent Alias IDs.")

print("\n--- Testing Researcher Agent ---")
researcher_session_id = str(uuid.uuid4())
researcher_input = 'What is unique about the Cline AI assistant? Use the search_documents function to find relevant information.'
# Call researcher and store result
researcher_result = invoke_agent(
    researcher_id,
    researcher_alias,
    researcher_session_id,
    researcher_input
)

# Print researcher result
print("\nResearcher Response:")
print(researcher_result)

# Check if researcher succeeded before calling writer
if not researcher_result or researcher_result.startswith("ERROR:"):
    print("\nSkipping writer agent due to error or empty response from researcher.")
else:
    print("\n--- Testing Writer Agent ---")
    writer_session_id = str(uuid.uuid4())
    # Use researcher result in writer prompt
    writer_input = f"Format this research finding using the format_content function: {researcher_result}"
    # Call writer
    writer_response = invoke_agent(
        writer_id,
        writer_alias,
        writer_session_id,
        writer_input
    )
    # Print writer result
    print("\nWriter Response:")
    print(writer_response)

print("\n--- Script Execution Complete ---")
