import json
import logging

# Global variable to store the vector store instance
vector_store = None

def search_documents(vector_store, query, k=4):
    """Search for similar documents"""
    return vector_store.similarity_search(query, k=k)

def add_document(vector_store, text, metadata=None):
    """Add a document to the vector store"""
    if metadata is None:
        metadata = {}
    elif isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    return vector_store.add_texts([text], [metadata])[0]

def initialize_vector_store(vs):
    """Initialize the global vector_store variable"""
    global vector_store
    vector_store = vs
    logging.info("Vector store initialized in utils.py")
