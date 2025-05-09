import json
import boto3
import traceback # Added for error logging

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
    function_name = event.get('function', '') 
    action_group = event.get('actionGroup', 'writer_actions')
    http_method = event.get('httpMethod', 'POST')
    
    try:
        print("--- Initializing Writer Lambda ---")
        # Initialize Bedrock client
        bedrock_runtime = boto3.client('bedrock-runtime')

        # --- Parse Input (New Schema) ---
        print("Parsing agent input (new schema)...")
        parameters_list = event.get('parameters', [])
        parameters = _parse_parameters(parameters_list)
        print(f"Function Name: {function_name}")
        print(f"Parsed Parameters: {parameters}")

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

            # --- Use Claude to format the content ---
            print("Invoking Claude 3 Sonnet to format content...")
            # Consider adding try/except around this call
            response = bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Format this research finding in a {style} way:\n{content}"
                        }
                    ]
                }).encode('utf-8')
            )
            print("Claude invocation complete.")

            # Extract formatted response
            print("Extracting formatted text from Claude response...")
            response_body_claude = json.loads(response['body'].read().decode())
            formatted_text = response_body_claude['content'][0]['text']
            print("Formatted text extracted.")

            # --- Construct Success Response (Stack Overflow TEXT format) --- 
            print("Constructing complex TEXT success response...")
            final_response = {
                "messageVersion": event.get('messageVersion', '1.0'),
                "response": {
                    "actionGroup": event.get('actionGroup'),
                    "function": event.get('function'),
                    "functionResponse": {
                        "responseBody": {
                           "TEXT": { 
                               "body": formatted_text # Already a string
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
        print(f"--- ERROR in Writer Lambda Handler ---")
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
