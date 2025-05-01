from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_aws import BedrockEmbeddings
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
import boto3
import os
from dotenv import load_dotenv
import json
import traceback

# Load environment variables from .env file
# Adjusted path assumption: Assume .env is in the parent dir (lambda-experiments)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Fallback for Lambda environment where .env might not be present
    print("Lambda: .env file not found, relying on Lambda environment variables.")

def _parse_parameters(parameters_list):
    """Parses the Bedrock Agent's parameter list into a dictionary."""
    params_dict = {}
    if isinstance(parameters_list, list):
        for param in parameters_list:
            if isinstance(param, dict) and 'name' in param and 'value' in param:
                params_dict[param['name']] = param['value']
    return params_dict

def lambda_handler(event, context):
    # Log the incoming event
    print(f"--- Researcher Lambda Event: {json.dumps(event)}") 
    
    # --- Get Function Name and Action Group --- 
    api_path = event.get('apiPath', '')
    function_name = api_path.split('/')[-1] if api_path else '' # Extract from apiPath
    action_group = event.get('actionGroup', 'researcher_actions') # Keep for response
    
    try:
        print("--- Initializing Researcher Lambda ---")
        
        # --- Load Env Vars ---
        print("Loading environment variables...")
        # Use standard env var names matching the main script setup & Lambda env
        cb_username = os.environ["CB_USERNAME"]
        cb_password = os.environ["CB_PASSWORD"]
        cb_host = os.environ["CB_HOST"]
        cb_bucket = os.environ["CB_BUCKET_NAME"] # Read standard name
        cb_scope = os.environ["SCOPE_NAME"]      # Read standard name
        cb_collection = os.environ["COLLECTION_NAME"] # Read standard name
        cb_index = os.environ["INDEX_NAME"]       # Read standard name
        aws_region = os.environ.get("AWS_REGION", "us-east-1") # Get region
        embedding_model_id = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0") # Get embedding model
        print("Environment variables loaded.")

        # --- Initialize Couchbase connection ---
        print(f"Initializing Couchbase connection to {cb_host}...")
        auth = PasswordAuthenticator(cb_username, cb_password)
        options = ClusterOptions(auth)
        cluster = Cluster(cb_host, options)
        print("Couchbase cluster initialized.")

        # --- Initialize Bedrock embeddings ---
        print(f"Initializing Bedrock client and embeddings (Region: {aws_region}, Model: {embedding_model_id})...")
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region) 
        embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id=embedding_model_id
        )
        print("Bedrock embeddings initialized.")

        # --- Initialize vector store ---
        print(f"Initializing Couchbase vector store ({cb_bucket}/{cb_scope}/{cb_collection}, Index: {cb_index})...")
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=cb_bucket,
            scope_name=cb_scope,
            collection_name=cb_collection,
            embedding=embeddings,
            index_name=cb_index
        )
        print("Couchbase vector store initialized.")

        # --- Parse Agent Input (Using requestBody) --- 
        print("Parsing agent input from requestBody...")
        input_properties = event.get('requestBody', {}).get('content', {}).get('application/json', {}).get('properties', [])
        parameters = _parse_parameters(input_properties) # Use existing helper
        print(f"Function Name (from apiPath): {function_name}")
        print(f"Parsed Parameters: {parameters}")

        # Check function name (extracted from apiPath)
        if function_name == 'search_documents': 
            print("Handling search_documents function...")
            # Extract parameters from parsed dict
            query = parameters.get('query')
            k_param = parameters.get('k', '3') # k might still be string from agent
            
            try:
                 k = int(k_param)
            except (ValueError, TypeError):
                 print(f"Warning: Invalid value for k '{k_param}'. Defaulting to 3.")
                 k = 3
            
            print(f"Search Query: '{query}', k={k}")

            if not query:
                print("ERROR: Query parameter is missing.")
                raise ValueError("Query parameter is required")

            # --- Perform search (with specific try/except) ---
            try:
                print(f"Performing vector search for query: '{query}'")
                results = vector_store.similarity_search(query, k=k)
                print(f"Search completed. Found {len(results)} results.")
            except Exception as search_err:
                print(f"ERROR during similarity_search: {str(search_err)}")
                print(traceback.format_exc())
                raise # Re-raise to be caught by the outer handler

            # --- Format results ---
            print("Formatting search results...")
            formatted_results = [doc.page_content for doc in results]
            result_text = "\n\n".join([f"Result {i+1}: {content}" for i, content in enumerate(formatted_results)])
            print("Results formatted.")
            
            # --- Construct Success Response (TEXT format) --- 
            print("Constructing TEXT success response...")
            final_response = {
                "messageVersion": event.get('messageVersion', '1.0'), 
                "response": {
                    "actionGroup": event.get('actionGroup'),
                    "apiPath": event.get('apiPath'),
                    "httpMethod": event.get('httpMethod'),
                    "httpStatusCode": 200,
                    "functionResponse": {
                        "responseBody": {
                           "TEXT": { 
                               "body": result_text if result_text else "No relevant documents found."
                           }
                        }
                    }
                }
            }
            
            print(f"Success response constructed: {json.dumps(final_response)[:500]}...") # Log truncated response
            return final_response
        
        else:
            print(f"ERROR: Unknown function name received: {function_name}")
            raise ValueError(f"Unknown function name: {function_name}")

    except Exception as e:
        print(f"--- ERROR in Researcher Lambda Handler ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Construct Error Response (Using the same complex TEXT format for consistency)
        error_body_text = json.dumps({
            'error': str(e),
            'trace': traceback.format_exc()
        }) # Stringify error details
        
        error_response = {
            "messageVersion": event.get('messageVersion', '1.0'),
            "response": {
                "actionGroup": event.get('actionGroup'),
                "apiPath": event.get('apiPath'),
                "httpMethod": event.get('httpMethod'),
                "httpStatusCode": 500,
                "functionResponse": {
                    "responseBody": {
                       "TEXT": { 
                           "body": f"ERROR: {error_body_text}"
                       }
                    }
                }
            }
        }
        print("Error response constructed.")
        return error_response
