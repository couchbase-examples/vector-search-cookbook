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
            
        # Get more results and with higher similarity threshold
        docs = vector_store.similarity_search(
            query,
            k=8,  # Increase number of results
            fetch_k=20  # Fetch more candidates for reranking
        )
        
        # Format results with more context
        results = []
        for i, doc in enumerate(docs, 1):
            # Add document number and content
            results.append(f"Document {i}:")
            results.append("-" * 40)
            results.append(doc.page_content)
            
            # Add metadata if available
            if hasattr(doc, 'metadata') and doc.metadata:
                results.append("\nMetadata:")
                for key, value in doc.metadata.items():
                    results.append(f"{key}: {value}")
            
            results.append("\n")  # Add spacing between documents
            
        return "\n".join(results)
        
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
        Returns a list of relevant document contents with metadata.
        Use this tool to find detailed information about topics."""
    )
