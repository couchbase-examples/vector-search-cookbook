import json
import boto3
import os

def lambda_handler(event, context):
    try:
        # Process the event

        # Initialize Bedrock client
        bedrock_runtime = boto3.client('bedrock-runtime')

        # Parse input parameters from the agent request
        api_path = event.get('apiPath', '')
        parameters = event.get('parameters', {})

        if api_path == '/format_content':
            # Extract parameters
            content = parameters.get('content')
            style = parameters.get('style', 'user-friendly')

            if not content:
                raise ValueError("Content parameter is required")

            # Use Claude to format the content
            response = bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
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

            # Extract formatted response
            response_body = json.loads(response['body'].read().decode())
            formatted_text = response_body['content'][0]['text']

            response = {
                'messageVersion': '1.0',
                'response': {
                    'actionGroup': 'writer_actions',
                    'apiPath': '/format_content',
                    'httpMethod': 'POST',
                    'httpStatusCode': 200,
                    'responseBody': {
                        'application/json': {
                            'body': formatted_text
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
                'actionGroup': 'writer_actions',
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
