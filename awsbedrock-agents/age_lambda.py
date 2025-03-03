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
        
        # Process the streaming response
        print("\n--- DEBUGGING AGENT RESPONSE ---")
        print(f"Response keys: {response.keys()}")
        
        # Store all trace events for later processing
        all_traces = []
        return_control_events = []
        
        for i, event in enumerate(response['completion']):
            print(f"Event {i} keys: {event.keys()}")
            
            if 'chunk' in event:
                chunk = event['chunk']['bytes'].decode('utf-8')
                result += chunk
                print(f"Chunk content: {chunk}")
            
            if 'trace' in event:
                print(f"Trace found in event {i}")
                all_traces.append(event['trace'])
            
            if 'returnControl' in event:
                print(f"Return Control Event: {event['returnControl']}")
                return_control_events.append(event['returnControl'])
        
        # After processing all events, check if we have a return control event but no result
        if not result.strip() and return_control_events:
            # Try to get the Lambda function ARN from the return control event
            for rc_event in return_control_events:
                if 'invocationInputs' in rc_event:
                    for input_item in rc_event['invocationInputs']:
                        if 'functionInvocationInput' in input_item:
                            func_input = input_item['functionInvocationInput']
                            action_group = func_input.get('actionGroup')
                            function_name = func_input.get('function')
                            
                            # Get parameters
                            parameters = {}
                            for param in func_input.get('parameters', []):
                                parameters[param.get('name')] = param.get('value')
                            
                            print(f"Function call: {action_group}::{function_name}")
                            print(f"Parameters: {parameters}")
                            
                            # For researcher agent, manually call the Lambda function
                            if action_group == 'researcher_actions' and function_name == 'search_documents':
                                query = parameters.get('query')
                                k = int(parameters.get('k', 3))
                                
                                if query and query != '?':
                                    print(f"Manually searching for: {query}, k={k}")
                                    
                                    # Import the search_documents function from utils
                                    from utils import search_documents, vector_store
                                    
                                    if vector_store:
                                        # Perform the search
                                        docs = search_documents(vector_store, query, k)
                                        
                                        # Format results
                                        search_results = [doc.page_content for doc in docs]
                                        
                                        print(f"Found {len(search_results)} results")
                                        for i, content in enumerate(search_results):
                                            print(f"Result {i+1}: {content[:100]}...")
                                        
                                        # Format the response
                                        result = f"Search results for '{query}':\n\n"
                                        for i, content in enumerate(search_results):
                                            result += f"Result {i+1}: {content}\n\n"
                                    else:
                                        print("Vector store not available")
                                        result = "Error: Vector store not available"
                            
                            # For writer agent, manually format the content
                            elif action_group == 'writer_actions' and function_name == 'format_content':
                                content = parameters.get('content')
                                style = parameters.get('style', 'user-friendly')
                                
                                if content and content != '?':
                                    print(f"Manually formatting content in {style} style")
                                    result = f"Formatted in {style} style: {content}"
                                else:
                                    result = "No content provided to format."
        
        print("--- END DEBUGGING ---\n")
            
            
        
        if not result.strip():
            logging.warning("Received empty response from agent")
        
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
