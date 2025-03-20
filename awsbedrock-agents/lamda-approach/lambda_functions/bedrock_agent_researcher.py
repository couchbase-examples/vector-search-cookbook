import json
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_aws import BedrockEmbeddings
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
import boto3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

def lambda_handler(event, context):
    try:
        # Process the event

        # Initialize Couchbase connection
        auth = PasswordAuthenticator(
            os.environ["CB_USERNAME"],
            os.environ["CB_PASSWORD"]
        )
        cluster = Cluster(os.environ["CB_HOST"], ClusterOptions(auth))

        # Initialize Bedrock embeddings
        bedrock_runtime = boto3.client('bedrock-runtime')
        embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id="amazon.titan-embed-text-v2:0"
        )

        # Initialize vector store
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.environ["CB_BUCKET_NAME"],
            scope_name=os.environ["SCOPE_NAME"],
            collection_name=os.environ["COLLECTION_NAME"],
            embedding=embeddings,
            index_name=os.environ["INDEX_NAME"]
        )

        # Parse input parameters from the agent request
        api_path = event.get('apiPath', '')
        parameters = event.get('parameters', {})

        if api_path == '/search_documents':
            # Extract parameters
            query = parameters.get('query')
            k = int(parameters.get('k', 3))

            if not query:
                raise ValueError("Query parameter is required")

            # Perform search
            results = vector_store.similarity_search(query, k=k)

            # Format results
            formatted_results = [doc.page_content for doc in results]

            # Format the response as a simple string
            result_text = "\n\n".join([f"Result {i+1}: {content}" for i, content in enumerate(formatted_results)])
            response = {
                'messageVersion': '1.0',
                'response': {
                    'actionGroup': 'researcher_actions',
                    'apiPath': '/search_documents',
                    'httpMethod': 'POST',
                    'httpStatusCode': 200,
                    'responseBody': {
                        'application/json': {
                            'body': result_text
                        }
                    }
                }
            }

            return response
        else:
            raise ValueError(f"Unknown API path: {api_path}")

    except Exception as e:
        return {
            'messageVersion': '1.0',
            'response': {
                'actionGroup': 'researcher_actions',
                'apiPath': api_path,
                'httpMethod': 'POST',
                'httpStatusCode': 500,
                'responseBody': {
                    'application/json': {
                        'error': str(e)
                    }
                }
            }
        }
