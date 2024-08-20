import json
import logging
import os
import time
import warnings
from datetime import timedelta
from uuid import uuid4

import numpy as np
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import (CollectionNotFoundException,
                                  CouchbaseException,
                                  InternalServerFailureException,
                                  QueryIndexAlreadyExistsException)
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate
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
    
def setup_collection(cluster, bucket_name, scope_name, collection_name):
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
            logging.info(f"Collection '{collection_name}' already exists.")
        
        # Wait for the collection to be available
        max_retries = 3
        retry_delay = 1
        for _ in range(max_retries):
            try:
                collection = bucket.scope(scope_name).collection(collection_name)
                break
            except CollectionNotFoundException:
                time.sleep(retry_delay)
        else:
            raise RuntimeError(f"Collection '{collection_name}' not available after {max_retries} retries")
        
        # Ensure primary index exists
        try:
            cluster.query(f"CREATE PRIMARY INDEX ON `{bucket_name}`.`{scope_name}`.`{collection_name}`").execute()
            logging.info(f"Primary index created on collection '{collection_name}'")
        except QueryIndexAlreadyExistsException:
            logging.info(f"Primary index already exists on collection '{collection_name}'")
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
    try:
        with open(file_path, 'r') as file:
            index_definition = json.load(file)
        return index_definition
    except Exception as e:
        raise ValueError(f"Error loading index definition from {file_path}: {str(e)}")

def create_or_update_search_index(cluster, bucket_name, scope_name, index_definition):
    try:
        scope_index_manager = cluster.bucket(bucket_name).scope(scope_name).search_indexes()
        
        # Check if index already exists
        existing_indexes = scope_index_manager.get_all_indexes()
        index_name = index_definition["name"]
        
        if index_name in [index.name for index in existing_indexes]:
            logging.info(f"Index '{index_name}' already exists. Updating...")
        else:
            logging.info(f"Creating new index '{index_name}'...")
        
        # Create SearchIndex object
        search_index = SearchIndex(
            name=index_definition["name"],
            source_type=index_definition.get("sourceType", "couchbase"),
            idx_type=index_definition["type"],
            source_name=index_definition["sourceName"],
            params=index_definition["params"],
            source_params=index_definition.get("sourceParams", {}),
            plan_params=index_definition.get("planParams", {})
        )
        
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
                
                if "collection: 'vector_store' doesn't belong to scope: 'shared'" in error_message:
                    raise ValueError("Collection 'vector_store' does not belong to scope 'shared'. Please check the collection and scope names.")
        
        except ValueError as ve:
            logging.error(str(ve))
            raise
        
        except Exception as json_error:
            logging.error(f"Failed to parse the error message: {json_error}")
            raise RuntimeError(f"Internal server error while creating/updating search index: {error_message}")
    
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating/updating search index: {str(e)}")

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

def load_trec_dataset(split='train[:1000]'):
    try:
        dataset = load_dataset('trec', split=split)
        logging.info(f"Successfully loaded TREC dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        raise ValueError(f"Error loading TREC dataset: {str(e)}")

def save_to_vector_store_in_batches(vector_store, texts, batch_size=50):
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch = texts[i:i + batch_size]
            documents = [Document(page_content=text) for text in batch]
            uuids = [str(uuid4()) for _ in range(len(documents))]
            vector_store.add_documents(documents=documents, ids=uuids)
    except Exception as e:
        raise RuntimeError(f"Failed to save documents to vector store: {str(e)}")

def create_embeddings(api_key,model='text-embedding-ada-002'):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=model)
        
        logging.info("Successfully created OpenAIEmbeddings")
        return embeddings
    except Exception as e:
        raise ValueError(f"Error creating OpenAIEmbeddings: {str(e)}")

def create_llm(anthropic_api_key, model='claude-3-5-sonnet-20240620'):
    try:
        llm = ChatAnthropic(temperature=0, anthropic_api_key=anthropic_api_key, model_name=model)
        logging.info("Successfully created ChatAnthropic")
        return llm
    except Exception as e:
        logging.error(f"Error creating ChatAnthropic: {str(e)}. Please check your API key and network connection.")
        raise

def create_rag_chain(vector_store, llm):
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

def main():
    try:
        # Get environment variables
        OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
        ANTHROPIC_API_KEY = get_env_variable('ANTHROPIC_API_KEY')
        CB_HOST = get_env_variable('CB_HOST', 'couchbase://localhost')
        CB_USERNAME = get_env_variable('CB_USERNAME', 'Administrator')
        CB_PASSWORD = get_env_variable('CB_PASSWORD', 'password')
        CB_BUCKET_NAME = get_env_variable('CB_BUCKET_NAME', 'vector-search-testing')
        INDEX_NAME = get_env_variable('INDEX_NAME', 'vector_search_claude')
        
        SCOPE_NAME = get_env_variable('SCOPE_NAME', 'shared')
        COLLECTION_NAME = get_env_variable('COLLECTION_NAME', 'claude')
        CACHE_COLLECTION = get_env_variable('CACHE_COLLECTION', 'cache')

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

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        

if __name__ == "__main__":
    main()