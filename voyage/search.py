import os
import warnings
import time
import logging
from datetime import timedelta
from uuid import uuid4
from tqdm import tqdm

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
from langchain_voyageai import VoyageAIEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

SCOPE_NAME = "shared"
COLLECTION_NAME = "docs"
CACHE_COLLECTION = "cache"

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
    try:
        auth = PasswordAuthenticator(db_username, db_password)
        options = ClusterOptions(auth)
        cluster = Cluster(connection_string, options)
        cluster.wait_until_ready(timedelta(seconds=5))
        logging.info("Successfully connected to Couchbase")
        return cluster
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Couchbase: {str(e)}")

def get_vector_store(cluster, db_bucket, db_scope, db_collection, embedding, index_name):
    try:
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=db_bucket,
            scope_name=db_scope,
            collection_name=db_collection,
            embedding=embedding,
            index_name=index_name,
        )
        logging.info("Successfully created vector store")
        return vector_store
    except Exception as e:
        raise ValueError(f"Failed to create vector store: {str(e)}")

def get_cache(cluster, db_bucket, db_scope, cache_collection):
    try:
        cache = CouchbaseCache(
            cluster=cluster,
            bucket_name=db_bucket,
            scope_name=db_scope,
            collection_name=cache_collection,
        )
        logging.info("Successfully created cache")
        return cache
    except Exception as e:
        raise ValueError(f"Failed to create cache: {str(e)}")

def load_trec_dataset(split='train[:2000]'):
    try:
        dataset = load_dataset('trec', split=split)
        logging.info(f"Successfully loaded TREC dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        raise ValueError(f"Error loading TREC dataset: {str(e)}")
    
def save_to_vector_store(vector_store, texts):
    try:
        documents = [Document(page_content=text) for text in texts]
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
        logging.info(f"Stored {len(documents)} documents in Couchbase")
    except Exception as e:
        raise RuntimeError(f"Failed to save documents to vector store: {str(e)}")
    
def save_to_vector_store_in_batches(vector_store, texts, batch_size=50):
    try:
        num_batches = (len(texts) + batch_size - 1) // batch_size  # Calculate total number of batches
        for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Processing Batches"):
            batch = texts[i:i + batch_size]
            documents = [Document(page_content=text) for text in batch]
            uuids = [str(uuid4()) for _ in range(len(documents))]
            vector_store.add_documents(documents=documents, ids=uuids)
    except Exception as e:
        raise RuntimeError(f"Failed to save documents to vector store: {str(e)}")

def create_embeddings(api_key, model='voyage-large-2'):
    try:
        os.environ["VOYAGEAI_API_KEY"] = api_key
        embeddings = VoyageAIEmbeddings(model=model)
        logging.info("Successfully created VoyageAIEmbeddings")
        return embeddings
    except Exception as e:
        raise ValueError(f"Error creating VoyageAIEmbeddings: {str(e)}")
    
def semantic_search(vector_store, query, top_k=10):
    try:
        start_time = time.time()
        search_results = vector_store.similarity_search_with_score(query, k=top_k)
        results = [{'id': doc.metadata.get('id', 'N/A'), 'text': doc.page_content, 'distance': score} 
                   for doc, score in search_results]
        elapsed_time = time.time() - start_time
        logging.info(f"Semantic search completed in {elapsed_time:.2f} seconds")
        return results, elapsed_time
    except CouchbaseException as e:
        raise RuntimeError(f"Error performing semantic search: {str(e)}")

def create_rag_chain(vector_store, llm):
    template = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
    {context}

    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Successfully created RAG chain")
    return chain

def create_pure_llm_chain(llm):
    template = """You are a helpful bot. Answer the question as truthfully as possible.

    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Successfully created pure LLM chain")
    return chain

def main():
    try:
        # Get environment variables
        VOYAGE_API_KEY = get_env_variable('VOYAGE_API_KEY')
        OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
        CB_USERNAME = get_env_variable('CB_USERNAME', 'Administrator')
        CB_PASSWORD = get_env_variable('CB_PASSWORD', 'password')
        CB_BUCKET_NAME = get_env_variable('CB_BUCKET_NAME', 'travel-sample')
        CB_HOST = get_env_variable('CB_HOST', 'couchbase://localhost')
        INDEX_NAME = get_env_variable('INDEX_NAME', 'vector_search')

        # Load dataset and create embeddings
        trec = load_trec_dataset()
        embeddings = create_embeddings(VOYAGE_API_KEY)

        # Setup Couchbase and vector store
        cluster = connect_to_couchbase(CB_HOST, CB_USERNAME, CB_PASSWORD)
        vector_store = get_vector_store(cluster, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME, embeddings, INDEX_NAME)

        # Save data to vector store in one go
        # save_to_vector_store(vector_store, trec['text'])
        
        # Save data to vector store in batches
        save_to_vector_store_in_batches(vector_store, trec['text'], batch_size=50)


        # Setup cache
        cache = get_cache(cluster, CB_BUCKET_NAME, SCOPE_NAME, CACHE_COLLECTION)
        set_llm_cache(cache)

        # Create LLMs and chains
        llm = ChatOpenAI(temperature=0, model="gpt-4o-2024-08-06", streaming=True)
        rag_chain = create_rag_chain(vector_store, llm)
        pure_llm_chain = create_pure_llm_chain(llm)

        # Sample query and search
        query = "What caused the 1929 Great Depression?"

        # Get responses
        start_time = time.time()
        rag_response = rag_chain.invoke(query)
        rag_elapsed_time = time.time() - start_time
        logging.info(f"RAG response generated in {rag_elapsed_time:.2f} seconds")

        start_time = time.time()
        pure_llm_response = pure_llm_chain.invoke(query)
        pure_llm_elapsed_time = time.time() - start_time
        logging.info(f"Pure LLM response generated in {pure_llm_elapsed_time:.2f} seconds")

        print(f"RAG Response: {rag_response}")
        print(f"Pure LLM Response: {pure_llm_response}")

        # Perform semantic search
        results, search_elapsed_time = semantic_search(vector_store, query)
        print(f"\nSemantic Search Results (completed in {search_elapsed_time:.2f} seconds):")
        for result in results:
            print(f"Distance: {result['distance']:.4f}, Text: {result['text']}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()