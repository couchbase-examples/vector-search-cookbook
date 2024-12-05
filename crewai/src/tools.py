import logging
from typing import Any
from langchain.tools import Tool

def vector_search(query: str, vector_store: Any) -> str:
    """Perform vector search and return formatted results"""
    try:
        # Ensure query is a string
        if isinstance(query, dict):
            query = str(query.get('query', ''))
        elif not isinstance(query, str):
            query = str(query)
            
        docs = vector_store.similarity_search(query)
        return "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
    except Exception as e:
        logging.error(f"Vector search failed: {str(e)}")
        raise

def create_vector_search_tool(vector_store: Any) -> Tool:
    """Create a vector search tool"""
    return Tool(
        name="vector_search",
        func=lambda query: vector_search(query, vector_store),
        description="""Search for relevant documents using vector similarity.
        Input should be a simple text query string.
        Returns a list of relevant document contents."""
    )
