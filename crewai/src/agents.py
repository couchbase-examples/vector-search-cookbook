import logging
from crewai import Agent, Task
from .tools import create_vector_search_tool

def setup_agents(llm, vector_store):
    """Create CrewAI agents"""
    # Create vector search tool
    search_tool = create_vector_search_tool(vector_store)
    
    # Custom response template for better formatting
    response_template = """
    Analysis Results:
    ----------------
    {{ .Response }}
    
    Sources Used:
    ------------
    {% for tool in .Tools %}
    - {{ tool.name }}
    {% endfor %}
    
    Confidence Level: {{ .Confidence }}
    Analysis Time: {{ .ExecutionTime }}
    """
    
    researcher = Agent(
        role='Research Expert',
        goal='Find and analyze the most relevant documents to answer user queries accurately',
        backstory="""You are an expert researcher with deep knowledge in information retrieval 
        and analysis. Your expertise lies in finding, evaluating, and synthesizing information 
        from various sources. You have a keen eye for detail and can identify key insights 
        from complex documents. You always verify information across multiple sources and 
        provide comprehensive, accurate analyses.""",
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
        backstory="""You are a skilled technical writer with expertise in making complex 
        information accessible and engaging. You excel at organizing information logically, 
        explaining technical concepts clearly, and creating well-structured documents. You 
        ensure all information is properly cited, accurate, and presented in a user-friendly 
        manner. You have a talent for maintaining the reader's interest while conveying 
        detailed technical information.""",
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
        
        Follow these steps:
        1. Use the vector_search tool to find relevant documents
        2. Search with multiple variations of the query to ensure comprehensive coverage
        3. Analyze each document carefully, noting key points and supporting evidence
        4. Cross-reference information across documents to verify accuracy
        5. Identify any conflicting information or gaps in knowledge
        6. Organize findings into clear, logical categories
        
        Focus on:
        - Accuracy and completeness of information
        - Relevance to the query
        - Quality and reliability of sources
        - Key concepts and their relationships
        - Supporting evidence and examples""",
        agent=researcher,
        expected_output="""A detailed analysis containing:
        1. Key findings organized by topic
        2. Supporting evidence from documents
        3. Any conflicting information or uncertainties
        4. Gaps in knowledge that may need further research
        5. Relevant context and background information"""
    )
    
    # Writing task
    writing_task = Task(
        description=f"""Create a comprehensive and well-structured response based on the research findings.
        
        Follow these steps:
        1. Review and analyze all research findings
        2. Organize information into a logical structure
        3. Create clear section headings and transitions
        4. Explain complex concepts in accessible language
        5. Include relevant examples and illustrations
        6. Ensure proper citation of sources
        
        The response should be:
        1. Clear and easy to understand
        2. Well-organized with logical flow
        3. Accurate and supported by research
        4. Engaging and informative
        5. Appropriate for the target audience""",
        agent=writer,
        expected_output="""A clear, comprehensive response that:
        1. Answers the query completely
        2. Is well-structured and organized
        3. Uses clear, accessible language
        4. Includes relevant examples
        5. Cites supporting evidence
        6. Maintains reader engagement""",
        context=[research_task]  # Properly set task dependency
    )
    
    return [research_task, writing_task]
