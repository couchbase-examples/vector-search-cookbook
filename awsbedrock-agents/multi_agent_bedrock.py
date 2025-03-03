import json
import logging
import os
import time
import uuid
from datetime import timedelta

import boto3
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

def setup_collection(cluster, bucket_name, scope_name, collection_name):
    try:
        # Check if bucket exists, create if it doesn't
        try:
            bucket = cluster.bucket(bucket_name)
            logging.info(f"Bucket '{bucket_name}' exists.")
        except Exception as e:
            logging.info(f"Bucket '{bucket_name}' does not exist. Creating it...")
            bucket_settings = CreateBucketSettings(
                name=bucket_name,
                bucket_type='couchbase',
                ram_quota_mb=1024,
                flush_enabled=True,
                num_replicas=0
            )
            cluster.buckets().create_bucket(bucket_settings)
            bucket = cluster.bucket(bucket_name)
            logging.info(f"Bucket '{bucket_name}' created successfully.")

        bucket_manager = bucket.collections()

        # Check if scope exists, create if it doesn't
        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(scope.name == scope_name for scope in scopes)
        
        if not scope_exists and scope_name != "_default":
            logging.info(f"Scope '{scope_name}' does not exist. Creating it...")
            bucket_manager.create_scope(scope_name)
            logging.info(f"Scope '{scope_name}' created successfully.")

        # Check if collection exists, create if it doesn't
        collections = bucket_manager.get_all_scopes()
        collection_exists = any(
            scope.name == scope_name and collection_name in [col.name for col in scope.collections]
            for scope in collections
        )

        if not collection_exists:
            logging.info(f"Collection '{collection_name}' does not exist. Creating it...")
            bucket_manager.create_collection(scope_name, collection_name)
            logging.info(f"Collection '{collection_name}' created successfully.")
        else:
            logging.info(f"Collection '{collection_name}' already exists. Skipping creation.")

        # Wait for collection to be ready
        collection = bucket.scope(scope_name).collection(collection_name)
        time.sleep(2)  # Give the collection time to be ready for queries

        # Ensure primary index exists
        try:
            cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`").execute()
            logging.info("Primary index present or created successfully.")
        except Exception as e:
            logging.warning(f"Error creating primary index: {str(e)}")

        # Clear all documents in the collection
        try:
            query = f"DELETE FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            cluster.query(query).execute()
            logging.info("All documents cleared from the collection.")
        except Exception as e:
            logging.warning(f"Error while clearing documents: {str(e)}. The collection might be empty.")

        return collection
    except Exception as e:
        raise RuntimeError(f"Error setting up collection: {str(e)}")
    
def setup_indexes(cluster):
    try:
        with open('awsbedrock-agents/aws_index.json', 'r') as file:
            index_definition = json.load(file)
    except Exception as e:
        raise ValueError(f"Error loading index definition: {str(e)}")
    
    try:
        scope_index_manager = cluster.bucket(CB_BUCKET_NAME).scope(SCOPE_NAME).search_indexes()

        # Check if index already exists
        existing_indexes = scope_index_manager.get_all_indexes()
        index_name = index_definition["name"]

        if index_name in [index.name for index in existing_indexes]:
            logging.info(f"Index '{index_name}' found")
        else:
            logging.info(f"Creating new index '{index_name}'...")

        # Create SearchIndex object from JSON definition
        search_index = SearchIndex.from_json(index_definition)

        # Upsert the index (create if not exists, update if exists)
        scope_index_manager.upsert_index(search_index)
        logging.info(f"Index '{index_name}' successfully created/updated.")

    except QueryIndexAlreadyExistsException:
        logging.info(f"Index '{index_name}' already exists. Skipping creation/update.")
    except ServiceUnavailableException:
        raise RuntimeError("Search service is not available. Please ensure the Search service is enabled in your Couchbase cluster.")
    except InternalServerFailureException as e:
        logging.error(f"Internal server error: {str(e)}")
        raise

def create_agent_role(agent_name, model_id):
    """Create IAM role and policies for the agent"""
    policy_name = f"{agent_name}-policy"
    role_name = f"AmazonBedrockExecutionRoleForAgents_{agent_name}"
    
    # Create policy
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockPermissions",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:*",
                    "bedrock-agent:*",
                    "bedrock-agent-runtime:*",
                    "bedrock-runtime:*",
                    "iam:*"
                ],
                "Resource": "*"
            }
        ]
    }
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": [
                    "bedrock.amazonaws.com",
                    "bedrock-runtime.amazonaws.com",
                    "bedrock-agent-runtime.amazonaws.com"
                ],
                "AWS": [
                    f"arn:aws:iam::{AWS_ACCOUNT_ID}:root",
                    f"arn:aws:iam::{AWS_ACCOUNT_ID}:user/{os.getenv('AWS_USER', 'default')}"
                ]
            },
            "Action": "sts:AssumeRole"
        }]
    }
    
    try:
        # Create or update policy
        try:
            policy = iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(policy_doc)
            )
        except iam_client.exceptions.EntityAlreadyExistsException:
            policy = iam_client.get_policy(PolicyArn=f"arn:aws:iam::{AWS_ACCOUNT_ID}:policy/{policy_name}")
            
            # List and delete old versions if limit reached
            versions = iam_client.list_policy_versions(
                PolicyArn=policy['Policy']['Arn']
            )['Versions']
            
            if len(versions) >= 5:
                # Delete oldest non-default versions
                for version in versions:
                    if not version['IsDefaultVersion']:
                        iam_client.delete_policy_version(
                            PolicyArn=policy['Policy']['Arn'],
                            VersionId=version['VersionId']
                        )
                        
                logging.info(f"Deleted {len(versions)} old policy versions")
            
            # Update the policy with new version
            try:
                policy_version = iam_client.create_policy_version(
                    PolicyArn=policy['Policy']['Arn'],
                    PolicyDocument=json.dumps(policy_doc),
                    SetAsDefault=True
                )
            except Exception as e:
                logging.error(f"Error updating policy version: {str(e)}")
                # Try to use existing policy without updating
                pass
        
        # Create role or get existing one
        try:
            role = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
        except iam_client.exceptions.EntityAlreadyExistsException:
            role = iam_client.get_role(RoleName=role_name)
        
        # Ensure policy is attached to role
        try:
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy['Policy']['Arn']
            )
        except iam_client.exceptions.EntityAlreadyExistsException:
            logging.info(f"Policy already attached to role {role_name}")
        
        return role
        
    except Exception as e:
        logging.error(f"Error creating/updating role and policy: {str(e)}")
        raise

def wait_for_agent_status(agent_id, target_statuses=['Available', 'PREPARED', 'NOT_PREPARED'], max_attempts=30, delay=2):
    """Wait for agent to reach any of the target statuses"""
    for attempt in range(max_attempts):
        try:
            response = bedrock_agent_client.get_agent(agentId=agent_id)
            current_status = response['agent']['agentStatus']
            
            if current_status in target_statuses:
                logging.info(f"Agent {agent_id} reached status: {current_status}")
                return current_status
            elif current_status == 'FAILED':
                logging.error(f"Agent {agent_id} failed")
                return 'FAILED'
            
            logging.info(f"Agent status: {current_status}, waiting... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
            
        except Exception as e:
            logging.warning(f"Error checking agent status: {str(e)}")
            time.sleep(delay)
    
    return current_status

def create_agent(name, instructions, functions, model_id="amazon.nova-pro-v1:0"):
    """Create a Bedrock agent with ROC action groups"""
    try:
        # Create agent role
        role = create_agent_role(name, model_id)
        
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
            logging.info(f"Found existing agent '{name}' with ID: {agent_id}")
            
            # Check agent status
            response = bedrock_agent_client.get_agent(agentId=agent_id)
            status = response['agent']['agentStatus']
            
            if status in ['NOT_PREPARED', 'FAILED']:
                logging.info(f"Deleting agent '{name}' with status {status}")
                bedrock_agent_client.delete_agent(agentId=agent_id)
                time.sleep(10)  # Wait after deletion
                existing_agent = None
        
        # Create new agent if needed
        if not existing_agent:
            logging.info(f"Creating new agent '{name}'")
            agent = bedrock_agent_client.create_agent(
                agentName=name,
                description=f"{name.title()} agent for document operations",
                instruction=instructions,
                agentResourceRoleArn=role['Role']['Arn'],
                idleSessionTTLInSeconds=1800,
                foundationModel=model_id
            )
            agent_id = agent['agent']['agentId']
            logging.info(f"Created new agent '{name}' with ID: {agent_id}")
        else:
            agent_id = existing_agent['agentId']
        
        # Wait for initial creation if needed
        status = wait_for_agent_status(agent_id, target_statuses=['NOT_PREPARED', 'PREPARED', 'Available'])
        if status not in ['NOT_PREPARED', 'PREPARED', 'Available']:
            raise Exception(f"Agent failed to reach valid state: {status}")
        
        # Create action group if needed
        try:
            bedrock_agent_client.create_agent_action_group(
                agentId=agent_id,
                agentVersion="DRAFT",
                actionGroupExecutor={"customControl": "RETURN_CONTROL"},
                actionGroupName=f"{name}_actions",
                functionSchema={"functions": functions},
                description=f"Action group for {name} operations"
            )
            logging.info(f"Created action group for agent '{name}'")
            time.sleep(5)
        except bedrock_agent_client.exceptions.ConflictException:
            logging.info(f"Action group already exists for agent '{name}'")
        
        # Prepare agent if needed
        if status == 'NOT_PREPARED':
            try:
                logging.info(f"Starting preparation for agent '{name}'")
                bedrock_agent_client.prepare_agent(agentId=agent_id)
                status = wait_for_agent_status(
                    agent_id, 
                    target_statuses=['PREPARED', 'Available']
                )
                logging.info(f"Agent '{name}' preparation completed with status: {status}")
            except Exception as e:
                logging.warning(f"Error during preparation: {str(e)}")
        
        # Handle alias creation/retrieval
        try:
            aliases = bedrock_agent_client.list_agent_aliases(agentId=agent_id)
            alias = next((a for a in aliases['agentAliasSummaries'] if a['agentAliasName'] == 'v1'), None)
            
            if not alias:
                logging.info(f"Creating new alias for agent '{name}'")
                alias = bedrock_agent_client.create_agent_alias(
                    agentId=agent_id,
                    agentAliasName="v1"
                )
                alias_id = alias['agentAlias']['agentAliasId']
            else:
                alias_id = alias['agentAliasId']
                logging.info(f"Using existing alias for agent '{name}'")
            
            logging.info(f"Successfully configured agent '{name}' with ID: {agent_id} and alias: {alias_id}")
            return agent_id, alias_id
            
        except Exception as e:
            logging.error(f"Error managing alias: {str(e)}")
            raise
        
    except Exception as e:
        logging.error(f"Error creating/updating agent: {str(e)}")
        raise RuntimeError(f"Failed to create/update agent: {str(e)}")

def invoke_agent(agent_id, alias_id, input_text, session_id=None, runtime_client=None):
    """Invoke a Bedrock agent"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Use provided runtime client or default to global one
    if runtime_client is None:
        runtime_client = bedrock_runtime_client
        
    try:
        logging.info(f"Invoking agent with input: {input_text}")
        
        response = runtime_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=session_id,
            inputText=input_text,
            enableTrace=True  # Enable tracing for debugging
        )
        
        result = ""
        
        # Process the streaming response
        for event in response['completion']:
            if 'chunk' in event:
                chunk = event['chunk']['bytes'].decode('utf-8')
                result += chunk
            
            # Handle Lambda function response in trace
            if 'trace' in event and isinstance(event['trace'], dict) and 'orchestrationTrace' in event['trace']:
                orch_trace = event['trace']['orchestrationTrace']
                if 'invocationOutput' in orch_trace:
                    invocation_output = orch_trace['invocationOutput']
                    if 'actionGroupInvocationOutput' in invocation_output:
                        action_output = invocation_output['actionGroupInvocationOutput']
                        if 'responseBody' in action_output:
                            response_body = action_output['responseBody']
                            if isinstance(response_body, dict) and 'application/json' in response_body:
                                json_body = response_body['application/json']
                                if 'body' in json_body:
                                    lambda_result = json_body['body']
                                    result = lambda_result
        
        if not result.strip():
            logging.warning("Received empty response from agent")
        
        return result
        
    except Exception as e:
        logging.error(f"Error invoking agent: {str(e)}")
        raise RuntimeError(f"Failed to invoke agent: {str(e)}")

def add_document(text, metadata=None):
    """Add a document to the vector store"""
    if metadata is None:
        metadata = {}
    elif isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    return vector_store.add_texts([text], [metadata])[0]

def search_documents(query, k=4):
    """Search for similar documents"""
    return vector_store.similarity_search(query, k=k)

def main():
    
    try:
        # Connect to Couchbase
        auth = PasswordAuthenticator(CB_USERNAME, CB_PASSWORD)
        options = ClusterOptions(auth)
        cluster = Cluster(CB_HOST, options)
        cluster.wait_until_ready(timedelta(seconds=5))
        logging.info("Successfully connected to Couchbase")
        
        # Set up collections
        setup_collection(cluster, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)
        logging.info("Collections setup complete")
        
        # Set up search indexes
        setup_indexes(cluster)
        logging.info("Search indexes setup complete")

        
        # Initialize Bedrock runtime client for embeddings
        embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id="amazon.titan-embed-text-v2:0"
        )
        logging.info("Successfully created Bedrock embeddings client")
        
        # Initialize vector store
        global vector_store
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=CB_BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        logging.info("Successfully created vector store")

        # Create agents with their instructions and functions
        embedder_instructions = """
        You are an Embedder Agent that handles document storage in the vector store.
        Your capabilities include:
        1. Adding new documents to the vector store
        2. Organizing document metadata
        3. Ensuring document quality
        """

        embedder_functions = [{
            "name": "add_document",
            "description": "Add a new document to the vector store",
            "parameters": {
                "text": {
                    "type": "string",
                    "description": "The document content to add",
                    "required": True
                },
                "metadata": {
                    "type": "string",
                    "description": "Additional metadata about the document as a JSON string",
                    "required": False
                }
            },
            "requireConfirmation": "ENABLED"
        }]

        researcher_instructions = """
        You are a Research Assistant that helps users find relevant information in documents.
        Your capabilities include:
        1. Searching through documents using semantic similarity
        2. Providing relevant document excerpts
        3. Answering questions based on document content
        """

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

        writer_instructions = """
        You are a Content Writer Assistant that helps format and present research findings.
        Your capabilities include:
        1. Formatting research findings in a user-friendly way
        2. Creating clear and engaging summaries
        3. Organizing information logically
        4. Highlighting key insights
        """

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

        # Create agents with error handling
        try:
            embedder_id, embedder_alias = create_agent("embedder", embedder_instructions, embedder_functions)
        except Exception as e:
            logging.error(f"Failed to create embedder agent: {str(e)}")
            embedder_id, embedder_alias = None, None

        try:
            researcher_id, researcher_alias = create_agent("researcher", researcher_instructions, researcher_functions)
        except Exception as e:
            logging.error(f"Failed to create researcher agent: {str(e)}")
            researcher_id, researcher_alias = None, None

        try:
            writer_id, writer_alias = create_agent("writer", writer_instructions, writer_functions)
        except Exception as e:
            logging.error(f"Failed to create writer agent: {str(e)}")
            writer_id, writer_alias = None, None

        if not any([embedder_id, researcher_id, writer_id]):
            raise RuntimeError("Failed to create any agents")

        # Load documents from JSON file
        try:
            with open('awsbedrock-agents/documents.json', 'r') as f:
                data = json.load(f)
                documents = data.get('documents', [])
                
            if not documents:
                raise ValueError("No documents found in JSON file")
                
            logging.info(f"Found {len(documents)} documents to process")
                
            # Process each document
            successful_docs = 0
            for i, doc in enumerate(documents, 1):
                try:
                    logging.info(f"Processing document {i}/{len(documents)}")
                    
                    # Extract text content and metadata from document
                    text = doc.get('text', '')
                    metadata = {k: v for k, v in doc.items() if k != 'text'}
                    
                    # Add document to vector store
                    doc_id = add_document(text, json.dumps(metadata))
                    logging.info(f"Added document {i} with ID: {doc_id}")
                    
                          

                    # Add document to vector store
                    successful_docs += 1
                    
                    # Add small delay between requests
                    time.sleep(1)
                    
                except Exception as e:
                    logging.error(f"Error processing document {i}: {str(e)}")
                    continue
            
            logging.info(f"\nProcessing complete: {successful_docs}/{len(documents)} documents added successfully")
            
            if successful_docs < len(documents):
                logging.warning(f"Failed to process {len(documents) - successful_docs} documents")
            
        except Exception as e:
            logging.error(f"Document loading failed: {str(e)}")
            raise ValueError(f"Failed to load documents: {str(e)}")

        # Run the selected approach
        try:

            # Get the approach from environment variable or default to custom
            approach = os.getenv("APPROACH", "custom").lower()
            
            if approach == "lambda":
                print("\nTrying Lambda approach...")
                # Approach: Lambda (AWS Lambda function calls)
                
                # Deploy Lambda functions first
                print("Deploying Lambda functions...")
                try:
                    # Create a .env file for the Lambda functions with the vector store configuration
                    with open('awsbedrock-agents/lambda_functions/.env', 'w') as f:
                        f.write(f"CB_HOST={os.environ.get('CB_HOST', 'couchbase://localhost')}\n")
                        f.write(f"CB_USERNAME={os.environ.get('CB_USERNAME', 'Administrator')}\n")
                        f.write(f"CB_PASSWORD={os.environ.get('CB_PASSWORD', 'password')}\n")
                        f.write(f"CB_BUCKET_NAME={os.environ.get('CB_BUCKET_NAME', 'vector-search-testing')}\n")
                        f.write(f"SCOPE_NAME={os.environ.get('SCOPE_NAME', 'shared')}\n")
                        f.write(f"COLLECTION_NAME={os.environ.get('COLLECTION_NAME', 'bedrock')}\n")
                        f.write(f"INDEX_NAME={os.environ.get('INDEX_NAME', 'vector_search_bedrock')}\n")
                    
                    import subprocess
                    subprocess.run([
                        'python3', 
                        'awsbedrock-agents/lambda_functions/deploy.py'
                    ], check=True)
                    print("Lambda functions deployed successfully")
                except Exception as e:
                    logging.error(f"Error deploying Lambda functions: {str(e)}")
                    raise RuntimeError("Failed to deploy Lambda functions")
                
                # Create action group for researcher agent with Lambda executor
                try:
                    bedrock_agent_client.create_agent_action_group(
                        agentId=researcher_id,
                        agentVersion="DRAFT",
                        actionGroupExecutor={
                            "lambda": f"arn:aws:lambda:{AWS_REGION}:{AWS_ACCOUNT_ID}:function:bedrock_agent_researcher"
                        },
                        actionGroupName="researcher_actions",
                        functionSchema={"functions": researcher_functions},
                        description="Action group for researcher operations with Lambda"
                    )
                    logging.info("Created researcher Lambda action group")
                except bedrock_agent_client.exceptions.ConflictException:
                    logging.info("Researcher Lambda action group already exists")
                    
                # Prepare researcher agent
                logging.info("Preparing researcher agent...")
                bedrock_agent_client.prepare_agent(agentId=researcher_id)
                status = wait_for_agent_status(
                    researcher_id, 
                    target_statuses=['PREPARED', 'Available']
                )
                logging.info(f"Researcher agent preparation completed with status: {status}")

                # Create action group for writer agent with Lambda executor
                try:
                    bedrock_agent_client.create_agent_action_group(
                        agentId=writer_id,
                        agentVersion="DRAFT",
                        actionGroupExecutor={
                            "lambda": f"arn:aws:lambda:{AWS_REGION}:{AWS_ACCOUNT_ID}:function:bedrock_agent_writer"
                        },
                        actionGroupName="writer_actions",
                        functionSchema={"functions": writer_functions},
                        description="Action group for writer operations with Lambda"
                    )
                    logging.info("Created writer Lambda action group")
                except bedrock_agent_client.exceptions.ConflictException:
                    logging.info("Writer Lambda action group already exists")
                    
                # Prepare writer agent
                logging.info("Preparing writer agent...")
                bedrock_agent_client.prepare_agent(agentId=writer_id)
                status = wait_for_agent_status(
                    writer_id, 
                    target_statuses=['PREPARED', 'Available']
                )
                logging.info(f"Writer agent preparation completed with status: {status}")

                # Test Lambda approach
                researcher_response = invoke_agent(
                    bedrock_runtime_client,
                    researcher_id,
                    researcher_alias,
                    'What is unique about the Cline AI assistant? Use the search_documents function to find relevant information.'
                )
                print("Lambda - Researcher Response:", researcher_response)

                writer_response = invoke_agent(
                    bedrock_runtime_client,
                    writer_id,
                    writer_alias,
                    f'Format this research finding using the format_content function: {researcher_response}'
                )
                print("Lambda - Writer Response:", writer_response)
                
            else:
                print("\nTrying Custom Control approach...")
                # Approach: Custom Control (direct function calls)
                try:
                    bedrock_agent_client.create_agent_action_group(
                        agentId=researcher_id,
                        agentVersion="DRAFT",
                        actionGroupExecutor={"customControl": "RETURN_CONTROL"},
                        actionGroupName="researcher_actions",
                        functionSchema={"functions": [{
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
                            }
                        }]},
                        description="Action group for researcher operations"
                    )
                    logging.info("Created researcher action group")
                except bedrock_agent_client.exceptions.ConflictException:
                    logging.info("Researcher action group already exists")
                    
                # Prepare researcher agent
                logging.info("Preparing researcher agent...")
                bedrock_agent_client.prepare_agent(agentId=researcher_id)
                status = wait_for_agent_status(
                    researcher_id, 
                    target_statuses=['PREPARED', 'Available']
                )
                logging.info(f"Researcher agent preparation completed with status: {status}")

                try:
                    bedrock_agent_client.create_agent_action_group(
                        agentId=writer_id,
                        agentVersion="DRAFT",
                        actionGroupExecutor={"customControl": "RETURN_CONTROL"},
                        actionGroupName="writer_actions",
                        functionSchema={"functions": [{
                            "name": "format_content",
                            "description": "Format research findings in a user-friendly way",
                            "parameters": {
                                "content": {
                                    "type": "string",
                                    "description": "The research findings to format",
                                    "required": True
                                },
                                "style": {
                                    "type": "string",
                                    "description": "The desired presentation style",
                                    "required": False
                                }
                            }
                        }]},
                        description="Action group for writer operations"
                    )
                    logging.info("Created writer action group")
                except bedrock_agent_client.exceptions.ConflictException:
                    logging.info("Writer action group already exists")
                    
                # Prepare writer agent
                logging.info("Preparing writer agent...")
                bedrock_agent_client.prepare_agent(agentId=writer_id)
                status = wait_for_agent_status(
                    writer_id, 
                    target_statuses=['PREPARED', 'Available']
                )
                logging.info(f"Writer agent preparation completed with status: {status}")

                # Test Custom Control approach
                researcher_response = invoke_agent(
                    researcher_id,
                    researcher_alias,
                    'What is unique about the Cline AI assistant? Use the search_documents function to find relevant information.'
                )
                print("Custom Control - Researcher Response:", researcher_response)

                writer_response = invoke_agent(
                    writer_id,
                    writer_alias,
                    f'Format this research finding using the format_content function: {researcher_response}'
                )
                print("Custom Control - Writer Response:", writer_response)
            
        except Exception as e:
            print(f"Error: {str(e)}")

    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
