import os
import warnings
from datetime import timedelta

import numpy as np
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
from couchbase.options import ClusterOptions
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_couchbase.cache import CouchbaseCache
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings

# Load environment variables from .env file
load_dotenv()

def get_env_variable(var_name, default_value=None):
    value = os.getenv(var_name)
    if value is None:
        if default_value is not None:
            warnings.warn(f"Environment variable {var_name} is missing. Assigning default value: {default_value}")
            return default_value
        else:
            raise ValueError(f"Environment variable {var_name} is missing and no default value is provided.")
    return value

def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to Couchbase"""
    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    cluster = Cluster(connection_string, options)
    cluster.wait_until_ready(timedelta(seconds=5))
    return cluster

def get_vector_store(cluster, db_bucket, db_scope, db_collection, embedding, index_name):
    """Return the Couchbase vector store"""
    vector_store = CouchbaseVectorStore(
        cluster=cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        embedding=embedding,
        index_name=index_name,
    )
    return vector_store

def get_cache(cluster, db_bucket, db_scope, cache_collection):
    """Return the Couchbase cache"""
    cache = CouchbaseCache(
        cluster=cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=cache_collection,
    )
    return cache

def save_to_vector_store(vector_store, texts, embeddings):
    """Store the documents in the vector store"""
    documents = [
        Document(page_content=text, metadata={'embedding': embed})
        for embed, text in zip(embeddings, texts)
    ]
    vector_store.add_documents(documents)
    print(f"Stored {len(documents)} documents in Couchbase")

def semantic_search(vector_store, query, embeddings, top_k=10):
    """Perform semantic search"""
    try:
        query_embed = embeddings.embed_query(query)
    except Exception as e:
        print(f"Error creating query embedding: {e}")
        return []

    try:
        search_results = vector_store.similarity_search_by_vector(embedding=query_embed, k=top_k)

        results = [{'id': doc.metadata['id'], 'text': doc.page_content, 'distance': score} 
                   for doc, score in search_results]
        return results
    except CouchbaseException as e:
        print(f"Error performing semantic search: {e}")
        return []

def main():
    # Get environment variables or use default values
    JINA_API_KEY = get_env_variable('JINA_API_KEY')  # Jina API key
    OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
    CB_USERNAME = get_env_variable('CB_USERNAME', 'default-username')
    CB_PASSWORD = get_env_variable('CB_PASSWORD', 'default-password')
    CB_BUCKET_NAME = get_env_variable('CB_BUCKET_NAME', 'default-bucket-name')
    CB_HOST = get_env_variable('CB_HOST', 'couchbase://localhost')
    INDEX_NAME = get_env_variable('INDEX_NAME', 'default-index-name')
    CACHE_COLLECTION = get_env_variable('CACHE_COLLECTION', 'default-cache-collection')

    # Initialize Jina Embeddings
    try:
        embeddings = JinaEmbeddings(jina_api_key=JINA_API_KEY, model_name='jina-embeddings-v2-base-en')
        print("Jina embeddings initialized")
    except Exception as e:
        print(f"Error initializing Jina embeddings: {e}")
        return

    # Load the TREC dataset
    try:
        trec = load_dataset('trec', split='train[:500]')
        print(f"Loaded TREC dataset with {len(trec['text'])} documents")
    except Exception as e:
        print(f"Error loading TREC dataset: {e}")
        return

    # Create embeddings using Jina AI
    try:
        embeds = embeddings.embed_documents(trec['text'])
        print(f"Embedding shape: {np.array(embeds).shape}")
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        # Optionally, print the response content for more details
        response = embeddings._call_embeddings_api(trec['text'])
        print(f"API Response: {response}")
        return

    try:
        # Connect to Couchbase
        cluster = connect_to_couchbase(CB_HOST, CB_USERNAME, CB_PASSWORD)

        # Use OpenAIEmbeddings as a fallback for compatibility
        fallback_embeddings = OpenAIEmbeddings()

        # Initialize CouchbaseVectorStore
        vector_store = get_vector_store(cluster, CB_BUCKET_NAME, "shared", "docs", fallback_embeddings, INDEX_NAME)

        # Store embeddings and metadata in Couchbase
        save_to_vector_store(vector_store, trec['text'], embeds)

        # Set the LLM cache
        cache = get_cache(cluster, CB_BUCKET_NAME, "shared", CACHE_COLLECTION)
        set_llm_cache(cache)

        # Build the prompt for the RAG
        template = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
        {context}

        Question: {question}"""

        prompt = ChatPromptTemplate.from_template(template)

        # Use OpenAI GPT-4 as the LLM for the RAG
        llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", streaming=True)

        # RAG chain
        chain = (
            {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Pure OpenAI output without RAG
        template_without_rag = """You are a helpful bot. Answer the question as truthfully as possible.

        Question: {question}"""

        prompt_without_rag = ChatPromptTemplate.from_template(template_without_rag)
        llm_without_rag = ChatOpenAI(model="gpt-4-1106-preview", streaming=True)

        chain_without_rag = (
            {"question": RunnablePassthrough()}
            | prompt_without_rag
            | llm_without_rag
            | StrOutputParser()
        )

        # Sample query for testing
        query = "What caused the 1929 Great Depression?"
        results = semantic_search(vector_store, query, embeddings)

        for result in results:
            print(f"Distance: {result['distance']:.4f}, Text: {result['text']}")

        # Get the response from the RAG
        rag_response = chain.invoke(query)
        print(f"RAG Response: {rag_response}")

        # Get the response from the pure LLM
        pure_llm_response = chain_without_rag.invoke(query)
        print(f"Pure LLM Response: {pure_llm_response}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
