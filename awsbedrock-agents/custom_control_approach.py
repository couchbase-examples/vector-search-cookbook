import json
import logging
import time
import uuid

def create_agent(bedrock_agent_client, name, instructions, functions, model_id="amazon.nova-pro-v1:0"):
    """Create a Bedrock agent with Custom Control action groups"""
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
        
        # Create action group if needed
        try:
            bedrock_agent_client.create_agent_action_group(
                agentId=agent_id,
                agentVersion="DRAFT",
                actionGroupExecutor={"customControl": "RETURN_CONTROL"},  # This is the key for Custom Control
                actionGroupName=f"{name}_actions",
                functionSchema={"functions": functions},
                description=f"Action group for {name} operations"
            )
            logging.info(f"Created action group for agent '{name}'")
            time.sleep(5)
        except bedrock_agent_client.exceptions.ConflictException:
            logging.info(f"Action group already exists for agent '{name}'")
        
        # Prepare agent if needed
        if status == 'NOT_PREPARED':
            try:
                logging.info(f"Starting preparation for agent '{name}'")
                bedrock_agent_client.prepare_agent(agentId=agent_id)
                status = wait_for_agent_status(
                    bedrock_agent_client,
                    agent_id, 
                    target_statuses=['PREPARED', 'Available']
                )
                logging.info(f"Agent '{name}' preparation completed with status: {status}")
            except Exception as e:
                logging.warning(f"Error during preparation: {str(e)}")
        
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
            
            if 'trace' in event:
                trace_info.append(event['trace'])
                print(f"Trace type: {type(event['trace'])}")
                print(f"Trace keys: {event['trace'].keys() if isinstance(event['trace'], dict) else 'Not a dict'}")
                
            if 'returnControl' in event:
                print(f"Return Control Event: {event['returnControl']}")
                
                # Handle the returnControl event
                return_control = event['returnControl']
                invocation_inputs = return_control.get('invocationInputs', [])
                
                if invocation_inputs:
                    function_input = invocation_inputs[0].get('functionInvocationInput', {})
                    action_group = function_input.get('actionGroup')
                    function_name = function_input.get('function')
                    parameters = function_input.get('parameters', [])
                    
                    # Convert parameters to a dictionary
                    param_dict = {}
                    for param in parameters:
                        param_dict[param.get('name')] = param.get('value')
                    
                    print(f"Function call: {action_group}::{function_name}")
                    print(f"Parameters: {param_dict}")
                    
                    # Handle search_documents function
                    if function_name == 'search_documents':
                        query = param_dict.get('query')
                        k = int(param_dict.get('k', 3))
                        
                        print(f"Searching for: {query}, k={k}")
                        
                        # Import the search_documents function from main_script
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from main_script import search_documents
                        
                        # If vector_store is not provided, try to get it from main_script
                        if vector_store is None:
                            try:
                                from main_script import vector_store as vs
                                vector_store = vs
                            except ImportError:
                                print("Could not import vector_store from main_script")
                        
                        if vector_store:
                            # Perform the search
                            docs = search_documents(vector_store, query, k)
                            
                            # Format results
                            search_results = [doc.page_content for doc in docs]
                            
                            print(f"Found {len(search_results)} results")
                            for i, content in enumerate(search_results):
                                print(f"Result {i+1}: {content[:100]}...")
                            
                            # Since we can't use invoke_agent_continuation, we'll need to start a new conversation
                            # with the search results
                            result = f"Search results for '{query}':\n\n"
                            for i, content in enumerate(search_results):
                                result += f"Result {i+1}: {content}\n\n"
                        else:
                            print("Vector store not available")
                            result = "Error: Vector store not available"
                    
                    # Handle format_content function
                    elif function_name == 'format_content':
                        content = param_dict.get('content')
                        style = param_dict.get('style', 'user-friendly')
                        
                        print(f"Formatting content in {style} style")
                        
                        # Check if content is valid
                        if content and content != '?':
                            # Use a simple formatting approach
                            result = f"Formatted in {style} style: {content}"
                        else:
                            result = "No content provided to format."
                    else:
                        print(f"Unknown function: {function_name}")
                        result = f"Error: Unknown function {function_name}"
        
        print(f"Final result: '{result}'")
        print("--- END DEBUGGING ---\n")
        
        if not result.strip():
            logging.warning("Received empty response from agent")
            print("NOTE: The agent response is empty. This could be due to:")
            print("  1. The agent is not properly configured to handle the function")
            print("  2. The agent is not finding relevant information in the vector store")
            print("  3. The agent is encountering an error when executing the function")
            print("  4. The agent's response format doesn't match what we're expecting")
            
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

def run_custom_control_approach(
    bedrock_agent_client,
    bedrock_runtime_client,
    researcher_instructions,
    researcher_functions,
    writer_instructions,
    writer_functions,
    vector_store=None
):
    """Run the Custom Control approach for Bedrock agents"""
    print("\nTrying Custom Control approach...")
    
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
    
    # Create action group for researcher agent
    try:
        bedrock_agent_client.create_agent_action_group(
            agentId=researcher_id,
            agentVersion="DRAFT",
            actionGroupExecutor={"customControl": "RETURN_CONTROL"},
            actionGroupName="researcher_actions",
            functionSchema={"functions": researcher_functions},
            description="Action group for researcher operations"
        )
        logging.info("Created researcher action group")
    except bedrock_agent_client.exceptions.ConflictException:
        logging.info("Researcher action group already exists")
        
    # Prepare researcher agent
    logging.info("Preparing researcher agent...")
    bedrock_agent_client.prepare_agent(agentId=researcher_id)
    status = wait_for_agent_status(
        bedrock_agent_client,
        researcher_id, 
        target_statuses=['PREPARED', 'Available']
    )
    logging.info(f"Researcher agent preparation completed with status: {status}")

    # Create action group for writer agent
    try:
        bedrock_agent_client.create_agent_action_group(
            agentId=writer_id,
            agentVersion="DRAFT",
            actionGroupExecutor={"customControl": "RETURN_CONTROL"},
            actionGroupName="writer_actions",
            functionSchema={"functions": writer_functions},
            description="Action group for writer operations"
        )
        logging.info("Created writer action group")
    except bedrock_agent_client.exceptions.ConflictException:
        logging.info("Writer action group already exists")
        
    # Prepare writer agent
    logging.info("Preparing writer agent...")
    bedrock_agent_client.prepare_agent(agentId=writer_id)
    status = wait_for_agent_status(
        bedrock_agent_client,
        writer_id, 
        target_statuses=['PREPARED', 'Available']
    )
    logging.info(f"Writer agent preparation completed with status: {status}")

    # Test Custom Control approach
    researcher_response = invoke_agent(
        bedrock_runtime_client,
        researcher_id,
        researcher_alias,
        'What is unique about the Cline AI assistant? Use the search_documents function to find relevant information.',
        vector_store=vector_store
    )
    print("Custom Control - Researcher Response:", researcher_response)

    writer_response = invoke_agent(
        bedrock_runtime_client,
        writer_id,
        writer_alias,
        f'Format this research finding using the format_content function: {researcher_response}',
        vector_store=vector_store
    )
    print("Custom Control - Writer Response:", writer_response)
