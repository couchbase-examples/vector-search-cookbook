import json
import logging
import os
import time
import uuid
from datetime import timedelta

import boto3
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
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
    """Set up Couchbase collection for vector storage"""
    try:
        # Check if bucket exists, create if it doesn't
        try:
            bucket = cluster.bucket(bucket_name)
            logging.info(f"Bucket '{bucket_name}' exists.")
        except Exception:
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
            logging.info(f"Collection '{collection_name}' already exists.")

        # Wait for collection to be ready
        collection = bucket.scope(scope_name).collection(collection_name)
        time.sleep(2)

        # Ensure primary index exists
        try:
            cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`").execute()
            logging.info("Primary index created successfully.")
        except Exception as e:
            logging.warning(f"Error creating primary index: {str(e)}")

        return collection
    except Exception as e:
        raise RuntimeError(f"Error setting up collection: {str(e)}")

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

def invoke_agent(agent_id, alias_id, input_text, session_id=None):
    """Invoke a Bedrock agent"""
    if session_id is None:
        session_id = str(uuid.uuid4())
        
    try:
        response = bedrock_runtime_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=session_id,
            inputText=input_text
        )
        
        result = ""
        for event in response['completion']:
            if 'chunk' in event:
                chunk = event['chunk']['bytes'].decode('utf-8')
                result += chunk
        
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
        collection = setup_collection(cluster, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)
        logging.info("Collections setup complete")
        
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
                    
                          

                    # response = bedrock_runtime_client.invoke_agent(
                    #     agentId=embedder_id,
                    #     agentAliasId=embedder_alias,
                    #     sessionId=str(uuid.uuid4()),
                    #     inputText=f'Add this document: {json.dumps(doc)}'
                    # )
                    
                    # # Process streaming response
                    # result = ""
                    # for event in response['completion']:
                    #     if 'chunk' in event:
                    #         chunk = event['chunk']['bytes'].decode('utf-8')
                    #         result += chunk
                    
                    # logging.info(f"Added document {i}: {result}")
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

        # Example usage
        try:
            # # Search for information using the researcher agent
            # researcher_response = invoke_agent(
            #     researcher_id,
            #     researcher_alias,
            #     'What is unique about the Cline AI assistant?'
            # )
            # print("Researcher Agent Response:", researcher_response)

            # Try both approaches for agent action groups:
            # 1. Custom Control (direct function calls)
            # 2. Lambda (AWS Lambda function calls)
            
            # Approach 1: Custom Control
            # try:
            #     print("\nTrying Custom Control approach...")
            #     bedrock_agent_client.create_agent_action_group(
            #         agentId=researcher_id,
            #         agentVersion="DRAFT",
            #         actionGroupExecutor={"customControl": "RETURN_CONTROL"},
            #         actionGroupName="researcher_actions_custom",
            #         functionSchema={"functions": [{
            #             "name": "search_documents",
            #             "description": "Search for relevant documents using semantic similarity",
            #             "parameters": {
            #                 "query": {
            #                     "type": "string",
            #                     "description": "The search query",
            #                     "required": True
            #                 },
            #                 "k": {
            #                     "type": "integer",
            #                     "description": "Number of results to return",
            #                     "required": False
            #                 }
            #             }
            #         }]},
            #         description="Action group for researcher operations"
            #     )
            #     logging.info("Created researcher action group")
            #     # Prepare agent after updating action group
            #     bedrock_agent_client.prepare_agent(agentId=researcher_id)
            #     status = wait_for_agent_status(
            #         researcher_id, 
            #         target_statuses=['PREPARED', 'Available']
            #     )
            #     logging.info(f"Researcher agent preparation completed with status: {status}")
            # except bedrock_agent_client.exceptions.ConflictException:
            #     logging.info("Researcher action group already exists")
            #     # Prepare agent even if action group exists
            #     bedrock_agent_client.prepare_agent(agentId=researcher_id)
            #     status = wait_for_agent_status(
            #         researcher_id, 
            #         target_statuses=['PREPARED', 'Available']
            #     )
            #     logging.info(f"Researcher agent preparation completed with status: {status}")

            #     # Create action group for writer agent
            #     bedrock_agent_client.create_agent_action_group(
            #         agentId=writer_id,
            #         agentVersion="DRAFT",
            #         actionGroupExecutor={"customControl": "RETURN_CONTROL"},
            #         actionGroupName="writer_actions_custom",
            #         functionSchema={"functions": [{
            #             "name": "format_content",
            #             "description": "Format research findings in a user-friendly way",
            #             "parameters": {
            #                 "content": {
            #                     "type": "string",
            #                     "description": "The research findings to format",
            #                     "required": True
            #                 },
            #                 "style": {
            #                     "type": "string",
            #                     "description": "The desired presentation style",
            #                     "required": False
            #                 }
            #             }
            #         }]},
            #         description="Action group for writer operations"
            #     )
            #     logging.info("Created writer action group")
            #     # Prepare agent after updating action group
            #     bedrock_agent_client.prepare_agent(agentId=writer_id)
            #     status = wait_for_agent_status(
            #         writer_id, 
            #         target_statuses=['PREPARED', 'Available']
            #     )
            #     logging.info(f"Writer agent preparation completed with status: {status}")
            # except bedrock_agent_client.exceptions.ConflictException:
            #     logging.info("Writer action group already exists")
            #     # Prepare agent even if action group exists
            #     bedrock_agent_client.prepare_agent(agentId=writer_id)
            #     status = wait_for_agent_status(
            #         writer_id, 
            #         target_statuses=['PREPARED', 'Available']
            #     )
            #     logging.info(f"Writer agent preparation completed with status: {status}")

            #     # Test Custom Control approach
            #     researcher_response = invoke_agent(
            #         researcher_id,
            #         researcher_alias,
            #         'What is unique about the Cline AI assistant? Use the search_documents function to find relevant information.'
            #     )
            #     print("Custom Control - Researcher Response:", researcher_response)

            #     writer_response = invoke_agent(
            #         writer_id,
            #         writer_alias,
            #         f'Format this research finding using the format_content function: {researcher_response}'
            #     )
            #     print("Custom Control - Writer Response:", writer_response)

            print("\nTrying Lambda approach...")
            # Approach 2: Lambda Functions
            # Deploy Lambda functions first
            print("Deploying Lambda functions...")
            import subprocess
            subprocess.run([
                'python3', 
                'awsbedrock-agents/lambda_functions/deploy.py'
            ], check=True)
            print("Lambda functions deployed successfully")

            # Create action groups with Lambda executors
            bedrock_agent_client.create_agent_action_group(
                    agentId=researcher_id,
                    agentVersion="DRAFT",
                    actionGroupExecutor={
                        "lambda": f"arn:aws:lambda:{AWS_REGION}:{AWS_ACCOUNT_ID}:function:bedrock_agent_researcher"
                    },
                    actionGroupName="researcher_actions_lambda",
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
                    description="Action group for researcher operations with Lambda"
                )

            bedrock_agent_client.create_agent_action_group(
                    agentId=writer_id,
                    agentVersion="DRAFT",
                    actionGroupExecutor={
                        "lambda": f"arn:aws:lambda:{AWS_REGION}:{AWS_ACCOUNT_ID}:function:bedrock_agent_writer"
                    },
                    actionGroupName="writer_actions_lambda",
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
                    description="Action group for writer operations with Lambda"
                )

            # Test Lambda approach
            researcher_response = invoke_agent(
                    researcher_id,
                    researcher_alias,
                    'What is unique about the Cline AI assistant? Use the search_documents function to find relevant information.'
                )
            print("Lambda - Researcher Response:", researcher_response)

            writer_response = invoke_agent(
                    writer_id,
                    writer_alias,
                    f'Format this research finding using the format_content function: {researcher_response}'
                )
            print("Lambda - Writer Response:", writer_response)
            
        except Exception as e:
            print(f"Error: {str(e)}")

    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
