

import json
import logging
import os
import time
import uuid
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
from langchain_couchbase.vectorstores import CouchbaseVectorStore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
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
bedrock_client = session.client('bedrock')
bedrock_agent_client = session.client('bedrock-agent')
bedrock_runtime = session.client('bedrock-runtime')
bedrock_runtime_client = session.client('bedrock-agent-runtime')
iam_client = session.client('iam')

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
        # Construct path relative to the script file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        index_file_path = os.path.join(script_dir, 'aws_index.json')
        # Load index definition from file
        with open(index_file_path, 'r') as file:
            index_definition = json.load(file)
            print(f"Loaded index definition from aws_index.json")
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
vector_store = CouchbaseVectorStore(
    cluster=cluster,
    bucket_name=CB_BUCKET_NAME,
    scope_name=SCOPE_NAME,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    index_name=INDEX_NAME
)
print("Successfully created vector store")

# Load documents from JSON file
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_file_path = os.path.join(script_dir, 'documents.json')
    with open(documents_file_path, 'r') as f:
        data = json.load(f)
        documents = data.get('documents', [])
    print(f"Loaded {len(documents)} documents from documents.json")
except Exception as e:
    print(f"Error loading documents: {str(e)}")
    raise

# Add documents to vector store
print(f"Adding {len(documents)} documents to vector store...")
for i, doc in enumerate(documents, 1):
    text = doc.get('text', '')
    metadata = doc.get('metadata', {})

    # Ensure metadata is a dictionary before adding
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse metadata for document {i}. Using empty metadata.")
            metadata = {}
    elif not isinstance(metadata, dict):
        print(f"Warning: Metadata for document {i} is not a dict or valid JSON string. Using empty metadata.")
        metadata = {}

    doc_id = vector_store.add_texts([text], [metadata])[0]
    print(f"Added document {i}/{len(documents)} with ID: {doc_id}")

    # Add small delay between requests
    time.sleep(1)

print(f"\nProcessing complete: {len(documents)}/{len(documents)} documents added successfully")

"""## Custom Control Approach Implementation

Now let's implement the Custom Control approach for Bedrock agents:
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

def get_or_create_agent_role(iam_client, role_name):
    """Gets or creates the necessary IAM role for the Bedrock agent."""
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    policy_arn_to_attach = "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"

    try:
        role_response = iam_client.get_role(RoleName=role_name)
        role_arn = role_response['Role']['Arn']
        print(f"Found existing IAM role '{role_name}' with ARN: {role_arn}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print(f"IAM role '{role_name}' not found. Creating...")
            try:
                role_response = iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description="IAM role for Bedrock Agents execution"
                )
                role_arn = role_response['Role']['Arn']
                print(f"Created IAM role '{role_name}' with ARN: {role_arn}")
                # Wait a bit for the role to be fully available before attaching policy
                time.sleep(10)
            except ClientError as create_error:
                print(f"Error creating IAM role '{role_name}': {create_error}")
                raise
        else:
            print(f"Error getting IAM role '{role_name}': {e}")
            raise

    # Attach the policy if not already attached
    try:
        attached_policies = iam_client.list_attached_role_policies(RoleName=role_name)
        if not any(p['PolicyArn'] == policy_arn_to_attach for p in attached_policies.get('AttachedPolicies', [])):
            print(f"Attaching policy '{policy_arn_to_attach}' to role '{role_name}'...")
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn_to_attach
            )
            print(f"Policy '{policy_arn_to_attach}' attached successfully.")
            # Wait a bit for the policy attachment to propagate
            time.sleep(5)
        else:
            print(f"Policy '{policy_arn_to_attach}' already attached to role '{role_name}'.")
    except ClientError as attach_error:
        print(f"Error attaching policy to role '{role_name}': {attach_error}")
        # Don't raise here, maybe the role exists but attaching failed temporarily

    return role_arn

def create_agent(bedrock_agent_client, name, instructions, functions, agent_role_arn, model_id="amazon.nova-pro-v1:0"):
    """Create a Bedrock agent with Custom Control action groups"""
    try:
        # List existing agents
        existing_agents = bedrock_agent_client.list_agents()
        existing_agent = next(
            (agent for agent in existing_agents['agentSummaries']
             if agent['agentName'] == name),
            None
        )

        # Handle existing agent
        if existing_agent:
            agent_id = existing_agent['agentId']
            print(f"Found existing agent '{name}' with ID: {agent_id}")

            # Check agent status
            response = bedrock_agent_client.get_agent(agentId=agent_id)
            status = response['agent']['agentStatus']

            if status in ['NOT_PREPARED', 'FAILED']:
                print(f"Deleting agent '{name}' with status {status}")
                bedrock_agent_client.delete_agent(agentId=agent_id)
                time.sleep(10)  # Wait after deletion
                existing_agent = None

        # Create new agent if needed
        if not existing_agent:
            print(f"Creating new agent '{name}'")
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
        else:
            agent_id = existing_agent['agentId']

        # Wait for initial creation if needed
        status = wait_for_agent_status(bedrock_agent_client, agent_id, target_statuses=['NOT_PREPARED', 'PREPARED', 'Available'])
        if status not in ['NOT_PREPARED', 'PREPARED', 'Available']:
            raise Exception(f"Agent failed to reach valid state: {status}")

        # Create action group if needed
        try:
            bedrock_agent_client.create_agent_action_group(
                agentId=agent_id,
                agentVersion="DRAFT",
                actionGroupExecutor={"customControl": "RETURN_CONTROL"},  # This is the key for Custom Control
                actionGroupName=f"{name}_actions",
                functionSchema={"functions": functions},
                description=f"Action group for {name} operations"
            )
            print(f"Created action group for agent '{name}'")
            time.sleep(5)
        except bedrock_agent_client.exceptions.ConflictException:
            print(f"Action group already exists for agent '{name}'")

        # Prepare agent if needed
        if status == 'NOT_PREPARED':
            try:
                print(f"Starting preparation for agent '{name}'")
                bedrock_agent_client.prepare_agent(agentId=agent_id)
                status = wait_for_agent_status(
                    bedrock_agent_client,
                    agent_id,
                    target_statuses=['PREPARED', 'Available']
                )
                print(f"Agent '{name}' preparation completed with status: {status}")
            except Exception as e:
                print(f"Error during preparation: {str(e)}")

        # Handle alias creation/retrieval
        try:
            aliases = bedrock_agent_client.list_agent_aliases(agentId=agent_id)
            alias = next((a for a in aliases['agentAliasSummaries'] if a['agentAliasName'] == 'v1'), None)

            if not alias:
                print(f"Creating new alias for agent '{name}'")
                alias = bedrock_agent_client.create_agent_alias(
                    agentId=agent_id,
                    agentAliasName="v1"
                )
                alias_id = alias['agentAlias']['agentAliasId']
            else:
                alias_id = alias['agentAliasId']
                print(f"Using existing alias for agent '{name}'")

            print(f"Successfully configured agent '{name}' with ID: {agent_id} and alias: {alias_id}")
            return agent_id, alias_id

        except Exception as e:
            print(f"Error managing alias: {str(e)}")
            raise

    except Exception as e:
        print(f"Error creating/updating agent: {str(e)}")
        raise

def invoke_agent(bedrock_runtime_client, agent_id, alias_id, input_text, session_id=None, vector_store=None):
    """Invoke a Bedrock agent"""
    if session_id is None:
        session_id = str(uuid.uuid4())

    try:
        print(f"Invoking agent with input: {input_text}")

        response = bedrock_runtime_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=session_id,
            inputText=input_text,
            enableTrace=True
        )

        result = ""

        for i, event in enumerate(response['completion']):
            # Process text chunks
            if 'chunk' in event:
                chunk = event['chunk']['bytes'].decode('utf-8')
                result += chunk

            # Handle custom control return
            if 'returnControl' in event:
                return_control = event['returnControl']
                invocation_inputs = return_control.get('invocationInputs', [])

                if invocation_inputs:
                    function_input = invocation_inputs[0].get('functionInvocationInput', {})
                    action_group = function_input.get('actionGroup')
                    function_name = function_input.get('function')
                    parameters = function_input.get('parameters', [])

                    # Convert parameters to a dictionary
                    param_dict = {}
                    for param in parameters:
                        param_dict[param.get('name')] = param.get('value')

                    print(f"Function call: {action_group}::{function_name}")

                    # Handle search_documents function
                    if function_name == 'search_documents':
                        query = param_dict.get('query')
                        k = int(param_dict.get('k', 3))

                        print(f"Searching for: {query}, k={k}")

                        if vector_store:
                            # Perform the search
                            docs = vector_store.similarity_search(query, k=k)

                            # Format results
                            search_results = [doc.page_content for doc in docs]
                            print(f"Found {len(search_results)} results")

                            # Format the response
                            result = f"Search results for '{query}':\n\n"
                            for i, content in enumerate(search_results):
                                result += f"Result {i+1}: {content}\n\n"
                        else:
                            print("Vector store not available")
                            result = "Error: Vector store not available"

                    # Handle format_content function
                    elif function_name == 'format_content':
                        content = param_dict.get('content')
                        style = param_dict.get('style', 'user-friendly')

                        print(f"Formatting content in {style} style")

                        # Check if content is valid
                        if content and content != '?':
                            result = f"Formatted in {style} style: {content}"
                        else:
                            result = "No content provided to format."
                    else:
                        print(f"Unknown function: {function_name}")
                        result = f"Error: Unknown function {function_name}"

        if not result.strip():
            print("Received empty response from agent")

        return result

    except Exception as e:
        print(f"Error invoking agent: {str(e)}")
        raise RuntimeError(f"Failed to invoke agent: {str(e)}")

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

"""## Run Custom Control Approach

Now let's run the Custom Control approach with our agents:
"""

# --- Get or Create IAM Role --- Start ---
agent_role_name = "BedrockExecutionRoleForAgents_CustomControl"
try:
    agent_role_arn = get_or_create_agent_role(iam_client, agent_role_name)
except Exception as e:
    print(f"Fatal error setting up IAM role: {e}")
    agent_role_arn = None # Ensure it's None if setup fails
# --- Get or Create IAM Role --- End ---

# Create researcher agent
try:
    if agent_role_arn: # Only proceed if role ARN was obtained
        researcher_id, researcher_alias = create_agent(
            bedrock_agent_client,
            "researcher",
            researcher_instructions,
            researcher_functions,
            agent_role_arn # Pass the ARN
        )
        print(f"Researcher agent created with ID: {researcher_id} and alias: {researcher_alias}")
    else:
        print("Skipping researcher agent creation due to IAM role setup failure.")
        researcher_id, researcher_alias = None, None
except Exception as e:
    print(f"Failed to create researcher agent: {str(e)}")
    researcher_id, researcher_alias = None, None

# Create writer agent
try:
    if agent_role_arn: # Only proceed if role ARN was obtained
        writer_id, writer_alias = create_agent(
            bedrock_agent_client,
            "writer",
            writer_instructions,
            writer_functions,
            agent_role_arn # Pass the ARN
        )
        print(f"Writer agent created with ID: {writer_id} and alias: {writer_alias}")
    else:
        print("Skipping writer agent creation due to IAM role setup failure.")
        writer_id, writer_alias = None, None
except Exception as e:
    print(f"Failed to create writer agent: {str(e)}")
    writer_id, writer_alias = None, None

if not any([researcher_id, writer_id]):
    # Adjust error message based on whether role setup failed
    if not agent_role_arn:
        raise RuntimeError("Failed to create agents because IAM role setup failed.")
    else:
        raise RuntimeError("Failed to create any agents despite successful IAM role setup.")

"""## Test the Agents

Let's test our agents by asking the researcher agent to search for information and the writer agent to format the results:
"""

# Test researcher agent
researcher_response = invoke_agent(
    bedrock_runtime_client,
    researcher_id,
    researcher_alias,
    'What is unique about the Cline AI assistant? Use the search_documents function to find relevant information.',
    vector_store=vector_store
)
print("\nResearcher Response:\n", researcher_response)

# Test writer agent
writer_response = invoke_agent(
    bedrock_runtime_client,
    writer_id,
    writer_alias,
    f'Format this research finding using the format_content function: {researcher_response}',
    vector_store=vector_store
)
print("\nWriter Response:\n", writer_response)