import json
import logging
import os
import sys
import time
from datetime import timedelta
from uuid import uuid4

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import (CouchbaseException,
                                  InternalServerFailureException,
                                  QueryIndexAlreadyExistsException)
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache

from langchain_core.prompts.chat import (ChatPromptTemplate,
                                         HumanMessagePromptTemplate,
                                         SystemMessagePromptTemplate)
from langchain_core.runnables import RunnablePassthrough
from langchain_couchbase.cache import CouchbaseCache
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

def connect_to_couchbase(connection_string, db_username, db_password):
    """
    Establishes a connection to Couchbase server.

    Args:
        connection_string (str): The Couchbase connection string.
        db_username (str): The database username.
        db_password (str): The database password.

    Returns:
        Cluster: A connected Couchbase Cluster object.

    Raises:
        ConnectionError: If connection to Couchbase fails.
    """
    try:
        auth = PasswordAuthenticator(db_username, db_password)
        options = ClusterOptions(auth)
        cluster = Cluster(connection_string, options)
        cluster.wait_until_ready(timedelta(seconds=5))
        logging.info("Successfully connected to Couchbase")
        return cluster
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Couchbase: {str(e)}")

def setup_collection(cluster, bucket_name, scope_name, collection_name):
    """
    Sets up a collection in Couchbase, creating it if it doesn't exist.

    Args:
        cluster (Cluster): The connected Couchbase Cluster object.
        bucket_name (str): The name of the bucket.
        scope_name (str): The name of the scope.
        collection_name (str): The name of the collection.

    Returns:
        Collection: The Couchbase Collection object.

    Raises:
        RuntimeError: If there's an error setting up the collection.
    """
    try:
        bucket = cluster.bucket(bucket_name)
        bucket_manager = bucket.collections()
        
        # Check if collection exists, create if it doesn't
        collections = bucket_manager.get_all_scopes()
        collection_exists = any(
            scope.name == scope_name and collection_name in [col.name for col in scope.collections]
            for scope in collections
        )
        
        if not collection_exists:
            logging.info(f"Collection '{collection_name}' does not exist. Creating it...")
            bucket_manager.create_collection(scope_name, collection_name)
            logging.info(f"Collection '{collection_name}' created successfully.")
        else:
            logging.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        
        collection = bucket.scope(scope_name).collection(collection_name)
        
        # Ensure primary index exists
        try:
            cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`").execute()
            logging.info("Primary index present or created successfully.")
        except Exception as e:
            logging.warning(f"Error creating primary index: {str(e)}")
        
        # Clear all documents in the collection
        try:
            query = f"DELETE FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            cluster.query(query).execute()
            logging.info("All documents cleared from the collection.")
        except Exception as e:
            logging.warning(f"Error while clearing documents: {str(e)}. The collection might be empty.")
        
        return collection
    except Exception as e:
        raise RuntimeError(f"Error setting up collection: {str(e)}")

def load_index_definition(file_path):
    """
    Loads a Couchbase search index definition from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the index definition.

    Returns:
        dict: The loaded index definition.

    Raises:
        ValueError: If there's an error loading the index definition.
    """
    try:
        with open(file_path, 'r') as file:
            index_definition = json.load(file)
        return index_definition
    except Exception as e:
        raise ValueError(f"Error loading index definition from {file_path}: {str(e)}")
    
def create_or_update_search_index(cluster, bucket_name, scope_name, index_definition):
    """
    Creates or updates a Couchbase search index.

    Args:
        cluster (Cluster): The connected Couchbase Cluster object.
        bucket_name (str): The name of the bucket.
        scope_name (str): The name of the scope.
        index_definition (dict): The search index definition.

    Raises:
        ValueError: If there's an error with the collection or scope names.
        RuntimeError: If there's an internal server error while creating/updating the index.
    """
    try:
        scope_index_manager = cluster.bucket(bucket_name).scope(scope_name).search_indexes()
        
        # Check if index already exists
        existing_indexes = scope_index_manager.get_all_indexes()
        index_name = index_definition["name"]
        
        if index_name in [index.name for index in existing_indexes]:
            logging.info(f"Index '{index_name}' found")
        else:
            logging.info(f"Creating new index '{index_name}'...")
        
        # Create SearchIndex object from JSON definition
        search_index = SearchIndex.from_json(index_definition)
        
        # Upsert the index (create if not exists, update if exists)
        scope_index_manager.upsert_index(search_index)
        logging.info(f"Index '{index_name}' successfully created/updated.")
    
    except QueryIndexAlreadyExistsException:
        logging.info(f"Index '{index_name}' already exists. Skipping creation/update.")
        
    except InternalServerFailureException as e:
        error_message = str(e)
        logging.error(f"InternalServerFailureException raised: {error_message}")
        
        try:
            # Accessing the response_body attribute from the context
            error_context = e.context
            response_body = error_context.response_body
            if response_body:
                error_details = json.loads(response_body)
                error_message = error_details.get('error', '')
                
                if "collection: 'claude' doesn't belong to scope: 'shared'" in error_message:
                    raise ValueError("Collection 'claude' does not belong to scope 'shared'. Please check the collection and scope names.")
        
        except ValueError as ve:
            logging.error(str(ve))
            raise
        
        except Exception as json_error:
            logging.error(f"Failed to parse the error message: {json_error}")
            raise RuntimeError(f"Internal server error while creating/updating search index: {error_message}")
    
def get_vector_store(cluster, db_bucket, db_scope, db_collection, embedding, index_name):
    """
    Creates and returns a CouchbaseVectorStore object.

    Args:
        cluster (Cluster): The connected Couchbase Cluster object.
        db_bucket (str): The name of the bucket.
        db_scope (str): The name of the scope.
        db_collection (str): The name of the collection.
        embedding (Embeddings): The embedding model to use.
        index_name (str): The name of the search index.

    Returns:
        CouchbaseVectorStore: The created vector store object.

    Raises:
        ValueError: If there's an error creating the vector store.
    """
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
    """
    Creates and returns a CouchbaseCache object.

    Args:
        cluster (Cluster): The connected Couchbase Cluster object.
        db_bucket (str): The name of the bucket.
        db_scope (str): The name of the scope.
        cache_collection (str): The name of the cache collection.

    Returns:
        CouchbaseCache: The created cache object.

    Raises:
        ValueError: If there's an error creating the cache.
    """
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

def load_trec_dataset(split='train[:1000]'):
    """
    Loads the TREC dataset.

    Args:
        split (str, optional): The dataset split to load. Defaults to 'train[:1000]'.

    Returns:
        Dataset: The loaded TREC dataset.

    Raises:
        ValueError: If there's an error loading the dataset.
    """
    try:
        dataset = load_dataset('trec', split=split)
        logging.info(f"Successfully loaded TREC dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        raise ValueError(f"Error loading TREC dataset: {str(e)}")

def save_to_vector_store_in_batches(vector_store, texts, batch_size=50):
    """
    Saves documents to the vector store in batches.

    Args:
        vector_store (VectorStore): The vector store to save documents to.
        texts (list): List of text documents to save.
        batch_size (int, optional): The batch size for processing. Defaults to 50.

    Raises:
        RuntimeError: If there's an error saving documents to the vector store.
    """
    try:
        logging.disable(sys.maxsize) # Disable logging to prevent tqdm output
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch = texts[i:i + batch_size]
            documents = [Document(page_content=text) for text in batch]
            uuids = [str(uuid4()) for _ in range(len(documents))]
            vector_store.add_documents(documents=documents, ids=uuids)
        logging.disable(logging.NOTSET) # Re-enable logging
    except Exception as e:
        raise RuntimeError(f"Failed to save documents to vector store: {str(e)}")

def create_embeddings(api_key, model='text-embedding-ada-002'):
    """
    Creates and returns an OpenAIEmbeddings object.

    Args:
        api_key (str): The OpenAI API key.
        model (str, optional): The name of the embedding model to use. Defaults to 'text-embedding-ada-002'.

    Returns:
        OpenAIEmbeddings: The created embeddings object.

    Raises:
        ValueError: If there's an error creating the embeddings.
    """
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=model)
        logging.info("Successfully created OpenAIEmbeddings")
        return embeddings
    except Exception as e:
        raise ValueError(f"Error creating OpenAIEmbeddings: {str(e)}")

def create_llm(anthropic_api_key, model='claude-3-5-sonnet-20240620'):
    """
    Creates and returns a ChatAnthropic language model object.

    Args:
        anthropic_api_key (str): The Anthropic API key.
        model (str, optional): The name of the model to use. Defaults to 'claude-3-5-sonnet-20240620'.

    Returns:
        ChatAnthropic: The created language model object.

    Raises:
        ValueError: If there's an error creating the language model.
    """
    try:
        llm = ChatAnthropic(temperature=0, anthropic_api_key=anthropic_api_key, model=model)
        logging.info("Successfully created ChatAnthropic")
        return llm
    except Exception as e:
        logging.error(f"Error creating ChatAnthropic: {str(e)}. Please check your API key and network connection.")
        raise ValueError(f"Failed to create ChatAnthropic: {str(e)}")


def semantic_search(vector_store, query, top_k=10):
    """
    Performs a semantic search using the provided vector store.

    Args:
        vector_store (VectorStore): The vector store to search in.
        query (str): The search query.
        top_k (int, optional): The number of top results to return. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - list: List of dictionaries with search results, each containing 'id', 'text', and 'distance'.
            - float: The time taken to perform the search in seconds.

    Raises:
        RuntimeError: If there's an error performing the semantic search.
    """
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
    """
    Creates and returns a Retrieval-Augmented Generation (RAG) chain.

    This function sets up a RAG chain using the provided vector store and language model.
    It creates a chat prompt template and combines it with the vector store's similarity search
    to generate responses based on the retrieved context.

    Args:
        vector_store (VectorStore): The vector store to use for similarity search.
        llm (BaseLanguageModel): The language model to use for generating responses.

    Returns:
        Chain: The created RAG chain that can be invoked with queries.

    Note:
        The created chain uses a system message template and a human message template
        to format the context and question for the language model.
    """
    system_template = "You are a helpful assistant that answers questions based on the provided context."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    human_template = "Context: {context}\n\nQuestion: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": lambda x: format_docs(vector_store.similarity_search(x)), "question": RunnablePassthrough()}
        | chat_prompt
        | llm
    )
    logging.info("Successfully created RAG chain")
    return chain

def demonstrate_cache(rag_chain):
    """
    Demonstrates the caching functionality of the RAG chain.

    This function runs a series of queries, including repeated ones,
    to show how caching affects response times. It prints out each query,
    the response, and the time taken to generate the response.

    Args:
        rag_chain (Chain): The RAG chain to use for queries.

    Note:
        This function is for demonstration purposes and does not return any value.
        The results are printed to the console.
    """
    queries = [
        "Why do heavier objects travel downhill faster?",
        "What is the capital of France?",
        "What caused the 1929 Great Depression?", # Repeated query
        "Why do heavier objects travel downhill faster?",  # Repeated query
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        start_time = time.time()
        response = rag_chain.invoke(query)
        elapsed_time = time.time() - start_time
        print(f"Response: {response.content}")
        print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    try:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        CB_HOST = os.getenv('CB_HOST', 'couchbase://localhost')
        CB_USERNAME = os.getenv('CB_USERNAME', 'Administrator')
        CB_PASSWORD = os.getenv('CB_PASSWORD', 'password')
        CB_BUCKET_NAME = os.getenv('CB_BUCKET_NAME', 'vector-search-testing')
        INDEX_NAME = os.getenv('INDEX_NAME', 'vector_search_claude')
        
        SCOPE_NAME = os.getenv('SCOPE_NAME', 'shared')
        COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'claude')
        CACHE_COLLECTION = os.getenv('CACHE_COLLECTION', 'cache')
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is missing.")

        # Setup Couchbase connection
        cluster = connect_to_couchbase(CB_HOST, CB_USERNAME, CB_PASSWORD)
        
        # Setup collection
        setup_collection(cluster, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME)
        setup_collection(cluster, CB_BUCKET_NAME, SCOPE_NAME, CACHE_COLLECTION)

        # Load and create/update search index
        index_definition = load_index_definition(os.path.join(os.path.dirname(__file__), 'claude_index.json'))
        create_or_update_search_index(cluster, CB_BUCKET_NAME, SCOPE_NAME, index_definition)

        # Load dataset and create embeddings
        trec = load_trec_dataset()
        embeddings = create_embeddings(OPENAI_API_KEY)

        # Setup vector store
        vector_store = get_vector_store(cluster, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME, embeddings, INDEX_NAME)
        
        # Save data to vector store in batches
        save_to_vector_store_in_batches(vector_store, trec['text'], batch_size=50)

        # Setup cache
        cache = get_cache(cluster, CB_BUCKET_NAME, SCOPE_NAME, CACHE_COLLECTION)
        set_llm_cache(cache)

        # Create LLM and chains
        llm = create_llm(ANTHROPIC_API_KEY)
        rag_chain = create_rag_chain(vector_store, llm)

        # Sample query and search
        query = "What caused the 1929 Great Depression?"

        # Get responses
        start_time = time.time()
        rag_response = rag_chain.invoke(query)
        rag_elapsed_time = time.time() - start_time
        logging.info(f"RAG response generated in {rag_elapsed_time:.2f} seconds")

        print(f"RAG Response: {rag_response.content}")

        # Perform semantic search
        results, search_elapsed_time = semantic_search(vector_store, query)
        print(f"\nSemantic Search Results (completed in {search_elapsed_time:.2f} seconds):")
        for result in results:
            print(f"Distance: {result['distance']:.4f}, Text: {result['text']}")
            
        # Demonstrate cache
        print("\nDemonstrating cache functionality:")
        demonstrate_cache(rag_chain)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
