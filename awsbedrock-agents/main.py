import json
import logging
import os
import time
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

# Import the approach modules
from age_custom_control import run_custom_control_approach
from age_lambda import run_lambda_approach
from utils import search_documents, add_document, initialize_vector_store

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
        with open(os.path.join(os.path.dirname(__file__), 'aws_index.json'), 'r') as file:
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

# Functions moved to utils.py

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
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=CB_BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        # Initialize the vector store in utils.py
        initialize_vector_store(vector_store)
        logging.info("Successfully created vector store")

        # Load documents from JSON file
        try:
            with open(os.path.join(os.path.dirname(__file__), 'documents.json'), 'r') as f:
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
                    doc_id = add_document(vector_store, text, json.dumps(metadata))
                    logging.info(f"Added document {i} with ID: {doc_id}")
                    
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

        # Define agent instructions and functions
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

        # For Lambda approach, we need to use a different tool name
        researcher_functions_lambda = [{
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

        # For Lambda approach, we need to use a different tool name
        writer_functions_lambda = [{
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

        # Choose which approach to run
        approach = os.getenv("APPROACH", "custom_control").lower()
        
        if approach == "lambda":
            # Run Lambda approach
            run_lambda_approach(
                session=session,
                bedrock_agent_client=bedrock_agent_client,
                bedrock_runtime_client=bedrock_runtime_client,
                researcher_instructions=researcher_instructions,
                researcher_functions=researcher_functions_lambda,
                writer_instructions=writer_instructions,
                writer_functions=writer_functions_lambda,
                aws_region=AWS_REGION,
                aws_account_id=AWS_ACCOUNT_ID,
                vector_store=vector_store
            )
        else:
            # Run Custom Control approach (default)
            run_custom_control_approach(
                bedrock_agent_client=bedrock_agent_client,
                bedrock_runtime_client=bedrock_runtime_client,
                researcher_instructions=researcher_instructions,
                researcher_functions=researcher_functions,
                writer_instructions=writer_instructions,
                writer_functions=writer_functions,
                vector_store=vector_store
            )

    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
