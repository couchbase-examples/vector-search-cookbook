import json
import logging
import subprocess
import time
import uuid
import os

def create_agent(bedrock_agent_client, name, instructions, functions, model_id="amazon.nova-pro-v1:0"):
    """Create a Bedrock agent with Lambda action groups"""
    try:
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
                idleSessionTTLInSeconds=1800,
                foundationModel=model_id
            )
            agent_id = agent['agent']['agentId']
            logging.info(f"Created new agent '{name}' with ID: {agent_id}")
        else:
            agent_id = existing_agent['agentId']
        
        # Wait for initial creation if needed
        status = wait_for_agent_status(bedrock_agent_client, agent_id, target_statuses=['NOT_PREPARED', 'PREPARED', 'Available'])
        if status not in ['NOT_PREPARED', 'PREPARED', 'Available']:
            raise Exception(f"Agent failed to reach valid state: {status}")
        
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

def wait_for_agent_status(bedrock_agent_client, agent_id, target_statuses=['Available', 'PREPARED', 'NOT_PREPARED'], max_attempts=30, delay=2):
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

def invoke_agent(bedrock_runtime_client, agent_id, alias_id, input_text, session_id=None, vector_store=None):
    """Invoke a Bedrock agent"""
    if session_id is None:
        session_id = str(uuid.uuid4())
        
    try:
        logging.info(f"Invoking agent with input: {input_text}")
        
        # Enable trace for debugging
        response = bedrock_runtime_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=session_id,
            inputText=input_text,
            enableTrace=True  # Enable tracing for debugging
        )
        
        result = ""
        trace_info = []
        
        # Process the streaming response
        print("\n--- DEBUGGING AGENT RESPONSE ---")
        print(f"Response keys: {response.keys()}")
        
        for i, event in enumerate(response['completion']):
            print(f"Event {i} keys: {event.keys()}")
            print(f"Full event {i}: {event}")
            
            if 'chunk' in event:
                chunk = event['chunk']['bytes'].decode('utf-8')
                result += chunk
                print(f"Chunk content: {chunk}")
            
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
                                    print(f"Lambda result found: {lambda_result}")
                                    result = lambda_result
                                    print(f"Updated result to: {result}")
            
            if 'trace' in event:
                trace_info.append(event['trace'])
                print(f"Trace type: {type(event['trace'])}")
                print(f"Trace keys: {event['trace'].keys() if isinstance(event['trace'], dict) else 'Not a dict'}")
        
        print(f"Final result: '{result}'")
        print("--- END DEBUGGING ---\n")
        
        if not result.strip():
            logging.warning("Received empty response from agent")
            print("NOTE: The agent response is empty. This could be due to:")
            print("  1. The Lambda function is not properly handling the request")
            print("  2. The Lambda function is encountering an error")
            print("  3. The Lambda function is not returning data in the expected format")
            print("  4. The agent is not finding relevant information in the vector store")
            print("  5. Check CloudWatch logs for the Lambda function for more details")
            
            # Print trace information for debugging
            if trace_info:
                print("\n--- TRACE INFORMATION ---")
                for i, trace in enumerate(trace_info):
                    print(f"Trace {i}:")
                    if isinstance(trace, dict) and 'orchestrationTrace' in trace:
                        orch_trace = trace['orchestrationTrace']
                        if 'invocationInput' in orch_trace:
                            print(f"  Invocation Input: {orch_trace['invocationInput']}")
                        if 'invocationOutput' in orch_trace:
                            print(f"  Invocation Output: {orch_trace['invocationOutput']}")
                        if 'modelInvocationOutput' in orch_trace:
                            if 'rawResponse' in orch_trace['modelInvocationOutput']:
                                print(f"  Model Raw Response: {orch_trace['modelInvocationOutput']['rawResponse']}")
                print("--- END TRACE ---\n")
        
        return result
        
    except Exception as e:
        logging.error(f"Error invoking agent: {str(e)}")
        raise RuntimeError(f"Failed to invoke agent: {str(e)}")

def run_lambda_approach(
    session,
    bedrock_agent_client,
    bedrock_runtime_client,
    researcher_instructions,
    researcher_functions,
    writer_instructions,
    writer_functions,
    aws_region,
    aws_account_id,
    vector_store=None
):
    """Run the Lambda approach for Bedrock agents"""
    print("\nTrying Lambda approach...")
    
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
        
        subprocess.run([
            'python3', 
            'awsbedrock-agents/lambda_functions/deploy.py'
        ], check=True)
        print("Lambda functions deployed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error deploying Lambda functions: {str(e)}")
        raise RuntimeError("Failed to deploy Lambda functions")
    
    # Create researcher agent
    try:
        researcher_id, researcher_alias = create_agent(
            bedrock_agent_client,
            "researcher", 
            researcher_instructions, 
            researcher_functions
        )
    except Exception as e:
        logging.error(f"Failed to create researcher agent: {str(e)}")
        researcher_id, researcher_alias = None, None

    # Create writer agent
    try:
        writer_id, writer_alias = create_agent(
            bedrock_agent_client,
            "writer", 
            writer_instructions, 
            writer_functions
        )
    except Exception as e:
        logging.error(f"Failed to create writer agent: {str(e)}")
        writer_id, writer_alias = None, None

    if not any([researcher_id, writer_id]):
        raise RuntimeError("Failed to create any agents")
    
    # Create action group for researcher agent with Lambda executor
    try:
        bedrock_agent_client.create_agent_action_group(
            agentId=researcher_id,
            agentVersion="DRAFT",
            actionGroupExecutor={
                "lambda": f"arn:aws:lambda:{aws_region}:{aws_account_id}:function:bedrock_agent_researcher"
            },  # This is the key for Lambda approach
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
        bedrock_agent_client,
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
                "lambda": f"arn:aws:lambda:{aws_region}:{aws_account_id}:function:bedrock_agent_writer"
            },  # This is the key for Lambda approach
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
        bedrock_agent_client,
        writer_id, 
        target_statuses=['PREPARED', 'Available']
    )
    logging.info(f"Writer agent preparation completed with status: {status}")

    # Test Lambda approach
    researcher_response = invoke_agent(
        bedrock_runtime_client,
        researcher_id,
        researcher_alias,
        'What is unique about the Cline AI assistant? Use the search_documents function to find relevant information.',
        vector_store=vector_store
    )
    print("Lambda - Researcher Response:", researcher_response)

    writer_response = invoke_agent(
        bedrock_runtime_client,
        writer_id,
        writer_alias,
        f'Format this research finding using the format_content function: {researcher_response}',
        vector_store=vector_store
    )
    print("Lambda - Writer Response:", writer_response)
