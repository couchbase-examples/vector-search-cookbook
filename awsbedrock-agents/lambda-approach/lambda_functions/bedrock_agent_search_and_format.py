import json
import boto3
import traceback 
import os
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_aws import BedrockEmbeddings
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    print("Lambda: .env file not found, relying on Lambda environment variables.")

def _parse_parameters(parameters_list):
    """Parses the Bedrock Agent's parameter list into a dictionary."""
    params_dict = {}
    if isinstance(parameters_list, list):
        for param in parameters_list:
            if isinstance(param, dict) and 'name' in param and 'value' in param:
                params_dict[param['name']] = param['value']
    return params_dict

# --- Global Initializations (potentially cached between warm invocations) ---
print("--- Initializing Search and Format Lambda (Global Scope) ---")
# Load Env Vars
cb_username = os.environ["CB_USERNAME"]
cb_password = os.environ["CB_PASSWORD"]
cb_host = os.environ["CB_HOST"]
cb_bucket = os.environ["CB_BUCKET_NAME"]
cb_scope = os.environ["SCOPE_NAME"]
cb_collection = os.environ["COLLECTION_NAME"]
cb_index = os.environ["INDEX_NAME"]
aws_region = os.environ.get("AWS_REGION", "us-east-1")
embedding_model_id = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
formatting_model_id = os.environ.get("AGENT_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0") # Use agent model for formatting

# Initialize Bedrock client
print(f"Initializing Bedrock client (Region: {aws_region})...")
bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
print("Bedrock client initialized.")

# Initialize Bedrock embeddings
print(f"Initializing Bedrock embeddings (Model: {embedding_model_id})...")
embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=embedding_model_id)
print("Bedrock embeddings initialized.")

# Initialize Couchbase connection
print(f"Initializing Couchbase connection to {cb_host}...")
auth = PasswordAuthenticator(cb_username, cb_password)
options = ClusterOptions(auth)
cluster = Cluster(cb_host, options)
print("Couchbase cluster initialized.")

# Initialize vector store
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
print("--- Global Initializations Complete ---")
# --- End Global Initializations ---


def lambda_handler(event, context):
    # Log the incoming event - Structure differs for 'Define with function details'
    print(f"--- Search/Format Lambda Event (Function Details Method): {json.dumps(event)}") 

    try:
        # --- Parse Agent Input (Function Details Method) ---
        # Parameters arrive in a list under the 'parameters' key.
        print("Parsing agent input from event['parameters'] list...")
        input_properties = event.get('parameters', []) # Get the list of parameters
        parameters = _parse_parameters(input_properties) # Use helper to convert list to dict
        print(f"Parsed Parameters: {parameters}")

        # --- Extract Parameters ---
        query = parameters.get('query')
        style = parameters.get('style', 'bullet points') # Default style
        k_param = parameters.get('k', '3') 
            
        try:
            k = int(k_param)
        except (ValueError, TypeError):
            print(f"Warning: Invalid value for k '{k_param}'. Defaulting to 3.")
            k = 3
            
        print(f"Search Query: '{query}', k={k}, Style: '{style}'")

        if not query:
            print("ERROR: Query parameter is missing.")
            raise ValueError("Query parameter is required")

        # --- Perform Search ---
        try:
            print(f"Performing vector search for query: '{query}'")
            results = vector_store.similarity_search(query, k=k)
            print(f"Search completed. Found {len(results)} results.")
        except Exception as search_err:
            print(f"ERROR during similarity_search: {str(search_err)}")
            print(traceback.format_exc())
            raise # Re-raise to be caught by the outer handler

        if not results:
             search_result_text = "No relevant documents found in the knowledge base."
        else:
             # Combine results into a single block for formatting
             search_result_text = "\\n\\n".join([doc.page_content for doc in results])
             print(f"Combined search results (first 200 chars): {search_result_text[:200]}...")
             
        # --- Format Results using LLM ---
        print(f"Invoking model {formatting_model_id} to format search results...")
        
        # Construct the strict formatting prompt
        # Using triple quotes caused EOL error, reverting to concatenation
        formatting_prompt = (
            f"Strictly reformat the following text content according to the requested style: '{style}'.\\n"
            f"Do NOT add any new information, do not summarize, and do not change the meaning.\\n"
            f"Only apply the requested style ('{style}') to the provided text. If the text indicates no documents were found, just return that statement clearly.\\n\\n"
            f"Provided Text:\\n"
            f"```\\n{search_result_text}\\n```\\n\\n"
            f"Reformatted Text (in '{style}' style):"
        )

        # Prepare the request body (Assuming Claude Sonnet for now)
        # TODO: Add logic for other model types if needed later
        request_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.5, 
            "messages": [
                {
                    "role": "user",
                    "content": formatting_prompt
                }
            ]
        })
        accept = 'application/json'
        contentType = 'application/json'

        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=formatting_model_id,
            body=request_body,
            accept=accept,
            contentType=contentType
        )
        print("Model invocation complete.")

        # Extract formatted response
        print("Extracting formatted text from model response...")
        response_body_llm = json.loads(response.get('body').read().decode())
        
        formatted_text = "Error: Could not parse formatting response."
        # TODO: Add logic for other model types if needed later
        if response_body_llm.get('content') and isinstance(response_body_llm['content'], list) and len(response_body_llm['content']) > 0:
            formatted_text = response_body_llm['content'][0].get('text', formatted_text)
        
        print("Formatted text extracted.")

        # --- Construct Success Response (Function Details Method) --- 
        # Stringify the actual results payload
        actual_results = {
            "formatted_results": formatted_text 
        }
        stringified_body = json.dumps(actual_results)

        # Build the full response structure required by Bedrock Agents (Function Invocation)
        agent_response = {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event['actionGroup'], # Get from event
                "function": event['function'],       # Get from event
                "functionResponse": {
                    "responseBody": {
                        "TEXT": {                     # Changed from application/json
                            "body": stringified_body # The stringified results
                        }
                    }
                }
            }
        }

        print(f"Success response constructed (Function Details Method): {json.dumps(agent_response)}") 
        return agent_response # Return the complete structure

    except Exception as e:
        print(f"--- ERROR in Search/Format Lambda Handler ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Construct Error Response (Function Details Method) - Simple object
        error_response_object = {
            'error': f"Error during Lambda execution: {type(e).__name__}",
            'message': str(e)
        }
        print(f"Error response constructed: {json.dumps(error_response_object)}")
        # For errors, it might be necessary to raise the exception or return
        # a specific structure the agent recognizes as an error.
        # Returning the object for now, but may need adjustment.
        return error_response_object 