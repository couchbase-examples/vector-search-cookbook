import logging
from crewai import Agent, Task
from .tools import create_vector_search_tool

def log_step(agent: Agent, task: Task, step: str) -> None:
    """Callback function for logging agent steps"""
    try:
        if hasattr(task, 'description'):
            task_desc = task.description[:50] + '...' if len(task.description) > 50 else task.description
        else:
            task_desc = "No description available"
            
        if hasattr(agent, 'role'):
            agent_role = agent.role
        else:
            agent_role = "Unknown agent"
            
        logging.info(f"Agent: {agent_role} | Task: {task_desc} | Step: {step}")
    except Exception as e:
        logging.warning(f"Error in log_step: {str(e)}")

def setup_agents(llm, vector_store):
    """Create CrewAI agents"""
    # Create vector search tool
    search_tool = create_vector_search_tool(vector_store)
    
    # Custom response template for better formatting
    response_template = """
    Analysis Results:
    ----------------
    {{ .Response }}
    
    Additional Notes:
    ----------------
    - Confidence Level: High
    - Sources Used: {{ len .Tools }} tools
    - Analysis Time: {{ .ExecutionTime }}
    """
    
    researcher = Agent(
        role='Research Expert',
        goal='Find and analyze the most relevant documents to answer user queries accurately',
        backstory="""You are an expert researcher with deep knowledge in information retrieval. 
        Your job is to find and analyze the most relevant documents to help answer user questions.
        You carefully examine each document to ensure it contains valuable information.""",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        memory=True,
        allow_delegation=False,
        response_template=response_template
    )
    
    writer = Agent(
        role='Technical Writer',
        goal='Generate clear, accurate, and well-structured responses based on research findings',
        backstory="""You are a skilled technical writer who excels at synthesizing information 
        from multiple sources to create clear, accurate, and engaging responses. You ensure all
        information is properly organized and presented in a user-friendly manner.""",
        llm=llm,
        verbose=True,
        memory=True,
        allow_delegation=False,
        response_template=response_template
    )
    
    logging.info("Agents created")
    return researcher, writer

def create_tasks(query, researcher, writer):
    """Create CrewAI tasks"""
    # Research task
    research_task = Task(
        description=f"""Research and analyze information relevant to: {query}
        
        Use the vector_search tool to find relevant documents.
        Examine each document carefully and identify the most relevant information.
        Focus on accuracy and completeness in your analysis.""",
        agent=researcher,
        expected_output="""A detailed analysis of the relevant information found in the documents,
        organized in a clear structure with key points and supporting details."""
    )
    
    # Writing task
    writing_task = Task(
        description="""Based on the research findings, create a comprehensive and well-structured response.
        
        Ensure the response is:
        1. Clear and easy to understand
        2. Well-organized with logical flow
        3. Accurate and supported by the research
        4. Engaging and informative""",
        agent=writer,
        expected_output="""A clear, comprehensive response that effectively answers the query,
        incorporating all relevant information from the research in a well-structured format.""",
        context=[research_task]  # Properly set task dependency
    )
    
    return [research_task, writing_task]
