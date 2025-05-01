import json
import boto3
import traceback 
import os

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
    print(f"--- Writer Lambda Event: {json.dumps(event)}") 
    
    # --- Get Function Name and Action Group --- 
    api_path = event.get('apiPath', '')
    function_name = api_path.split('/')[-1] if api_path else '' # Extract from apiPath
    action_group = event.get('actionGroup', 'writer_actions')
    
    try:
        print("--- Initializing Writer Lambda ---")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")
        
        # Initialize Bedrock client
        print(f"Initializing Bedrock client (Region: {aws_region})...")
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
        print("Bedrock client initialized.")

        # --- Parse Input (Using requestBody) ---
        print("Parsing agent input from requestBody...")
        input_properties = event.get('requestBody', {}).get('content', {}).get('application/json', {}).get('properties', [])
        parameters = _parse_parameters(input_properties) # Use existing helper
        print(f"Function Name (from apiPath): {function_name}")
        print(f"Parsed Parameters: {parameters}")

        # Determine which LLM to use (Example: Use agent model if specified, else default)
        # This needs the AGENT_MODEL_ID env var passed from main script
        agent_model_id = os.environ.get("AGENT_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0") # Default to Sonnet
        print(f"Using model for formatting: {agent_model_id}")

        # Check function name
        if function_name == 'format_content':
            print("Handling format_content function...")
            # Extract parameters from parsed dict
            content = parameters.get('content')
            style = parameters.get('style', 'user-friendly')
            print(f"Content to format (first 100 chars): {content[:100]}...")
            print(f"Style: {style}")

            if not content:
                print("ERROR: Content parameter is missing.")
                raise ValueError("Content parameter is required")

            # --- Use LLM to format the content ---
            print(f"Invoking model {agent_model_id} to format content...")
            # Construct the prompt
            prompt = f"Format this research finding in a {style} way:\n\n{content}\n\nFormatted Response:"

            # Prepare the request body based on the model type
            request_body = {}
            accept = 'application/json'
            contentType = 'application/json'

            if "claude" in agent_model_id:
                request_body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "temperature": 0.5, 
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            elif "amazon.titan" in agent_model_id:
                # Example for Titan Text - adjust if using a different Titan model
                 request_body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {
                         "maxTokenCount": 1024,
                         "stopSequences": [],
                         "temperature": 0.7,
                         "topP": 0.9
                    }
                 })
            else:
                 # Add other model families if needed (e.g., Cohere, AI21)
                 print(f"Warning: Model family for {agent_model_id} not explicitly handled. Using default Claude structure.")
                 request_body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "temperature": 0.5,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                 })

            # Invoke the model
            response = bedrock_runtime.invoke_model(
                modelId=agent_model_id,
                body=request_body,
                accept=accept,
                contentType=contentType
            )
            print("Model invocation complete.")

            # Extract formatted response based on model type
            print("Extracting formatted text from model response...")
            response_body_llm = json.loads(response.get('body').read().decode())
            
            formatted_text = "Error: Could not parse response."
            if "claude" in agent_model_id:
                if response_body_llm.get('content') and isinstance(response_body_llm['content'], list) and len(response_body_llm['content']) > 0:
                    formatted_text = response_body_llm['content'][0].get('text', formatted_text)
            elif "amazon.titan" in agent_model_id:
                # Example for Titan Text - adjust if needed
                if response_body_llm.get('results') and isinstance(response_body_llm['results'], list) and len(response_body_llm['results']) > 0:
                     formatted_text = response_body_llm['results'][0].get('outputText', formatted_text)
            else:
                 # Fallback attempt for unknown models (try Claude structure)
                 if response_body_llm.get('content') and isinstance(response_body_llm['content'], list) and len(response_body_llm['content']) > 0:
                    formatted_text = response_body_llm['content'][0].get('text', formatted_text)
            
            print("Formatted text extracted.")

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
                               "body": formatted_text # Already a string
                           }
                        }
                    }
                }
            }
            
            print(f"Success response constructed: {json.dumps(final_response)[:500]}...") # Log truncated
            return final_response
            
        else:
            print(f"ERROR: Unknown function name received: {function_name}")
            raise ValueError(f"Unknown function name: {function_name}")

    except Exception as e:
        print(f"--- ERROR in Writer Lambda Handler ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Construct Error Response 
        error_body_text = json.dumps({
            'error': str(e),
            'trace': traceback.format_exc()
        }) 
        
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
