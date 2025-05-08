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
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

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
    # Use 'function' instead of 'apiPath'
    function_name = event.get('function', '') 
    action_group = event.get('actionGroup', 'researcher_actions') # Keep for response
    http_method = event.get('httpMethod', 'POST') # Keep for response, though might be irrelevant now

    try:
        print("--- Initializing Researcher Lambda ---")
        
        # --- Load Env Vars ---
        print("Loading environment variables...")
        cb_username = os.environ["CB_USERNAME"]
        cb_password = os.environ["CB_PASSWORD"]
        cb_host = os.environ["CB_HOST"]
        cb_bucket = os.environ["CB_BUCKET_NAME"]
        cb_scope = os.environ["SCOPE_NAME"]
        cb_collection = os.environ["COLLECTION_NAME"]
        cb_index = os.environ["INDEX_NAME"]
        print("Environment variables loaded.")

        # --- Initialize Couchbase connection ---
        print("Initializing Couchbase connection...")
        auth = PasswordAuthenticator(cb_username, cb_password)
        # Add timeouts
        options = ClusterOptions(auth)
        cluster = Cluster(cb_host, options)
        # Wait for cluster to be ready? Maybe add cluster.wait_until_ready(timedelta(seconds=5))
        print("Couchbase cluster initialized.")

        # --- Initialize Bedrock embeddings ---
        print("Initializing Bedrock client and embeddings...")
        # Consider adding region_name explicitly if needed
        bedrock_runtime = boto3.client('bedrock-runtime') 
        embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id="amazon.titan-embed-text-v2:0" # Make sure this model is enabled
        )
        print("Bedrock embeddings initialized.")

        # --- Initialize vector store ---
        print("Initializing Couchbase vector store...")
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=cb_bucket,
            scope_name=cb_scope,
            collection_name=cb_collection,
            embedding=embeddings,
            index_name=cb_index
        )
        print("Couchbase vector store initialized.")

        # --- Parse Agent Input (New Schema) --- 
        print("Parsing agent input (new schema)...")
        parameters_list = event.get('parameters', [])
        parameters = _parse_parameters(parameters_list)
        print(f"Function Name: {function_name}")
        print(f"Parsed Parameters: {parameters}")

        # Check function name instead of api_path
        if function_name == 'search_documents': 
            print("Handling search_documents function...")
            # Extract parameters from parsed dict
            query = parameters.get('query')
            k_param = parameters.get('k', '3') # k might still be string from agent
            k = int(k_param)
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
            
            # --- Construct Success Response (Stack Overflow TEXT format) --- 
            print("Constructing complex TEXT success response...")
            final_response = {
                "messageVersion": event.get('messageVersion', '1.0'), # Use version from event or default
                "response": {
                    "actionGroup": event.get('actionGroup'),
                    "function": event.get('function'),
                    "functionResponse": {
                        "responseBody": {
                           "TEXT": { 
                               # Ensure body is a string
                               "body": result_text if result_text else "No relevant documents found."
                           }
                        }
                    }
                }
            }
            
            print(f"Success response constructed: {json.dumps(final_response)}") # Log the final response
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
                "function": event.get('function'),
                "functionResponse": {
                    # Indicate error via responseBody content, not HTTP status equivalent here
                    "responseBody": {
                       "TEXT": { 
                           "body": f"ERROR: {error_body_text}" # Embed stringified error
                       }
                    }
                }
            }
        }
        print("Error response constructed.")
        return error_response
