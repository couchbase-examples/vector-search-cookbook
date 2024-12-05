import json
import logging
from uuid import uuid4
from datasets import load_dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_couchbase.cache import CouchbaseCache
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from couchbase.management.search import SearchIndex
from tqdm import tqdm
from . import config

def setup_vector_store(cluster):
    """Initialize vector store and embeddings"""
    try:
        # Load index definition
        with open('crewai/crew_index.json', 'r') as file:
            index_definition = json.load(file)
        
        # Setup vector search index
        scope_index_manager = cluster.bucket(config.CB_BUCKET_NAME).scope(config.SCOPE_NAME).search_indexes()
        
        # Check if index exists
        try:
            existing_indexes = scope_index_manager.get_all_indexes()
            index_exists = any(index.name == config.INDEX_NAME for index in existing_indexes)
            
            if index_exists:
                logging.info(f"Index '{config.INDEX_NAME}' already exists")
            else:
                search_index = SearchIndex.from_json(index_definition)
                scope_index_manager.upsert_index(search_index)
                logging.info(f"Index '{config.INDEX_NAME}' created")
                
        except Exception as e:
            logging.warning(f"Error handling index: {str(e)}")
            # Continue anyway since the index might exist
        
        # Initialize OpenAI components
        embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        
        llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0
        )
        
        # Setup vector store
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=config.CB_BUCKET_NAME,
            scope_name=config.SCOPE_NAME,
            collection_name=config.COLLECTION_NAME,
            embedding=embeddings,
            index_name=config.INDEX_NAME,
        )
        logging.info("Vector store initialized")
        
        # Setup cache
        cache = CouchbaseCache(
            cluster=cluster,
            bucket_name=config.CB_BUCKET_NAME,
            scope_name=config.SCOPE_NAME,
            collection_name=config.CACHE_COLLECTION,
        )
        set_llm_cache(cache)
        logging.info("Cache initialized")
        
        return vector_store, llm
        
    except Exception as e:
        logging.error(f"Failed to setup vector store: {str(e)}")
        raise

def load_sample_data(vector_store):
    """Load sample data into vector store"""
    try:
        # Load TREC dataset
        trec = load_dataset('trec', split='train[:1000]')
        logging.info(f"Loaded {len(trec)} samples from TREC dataset")
        
        # Disable logging during data loading
        logging.disable(logging.INFO)
        
        # Add documents in batches
        batch_size = 50
        for i in tqdm(range(0, len(trec['text']), batch_size), desc="Loading data"):
            batch = trec['text'][i:i + batch_size]
            documents = [Document(page_content=text) for text in batch]
            uuids = [str(uuid4()) for _ in range(len(documents))]
            vector_store.add_documents(documents=documents, ids=uuids)
            
        # Re-enable logging
        logging.disable(logging.NOTSET)
        logging.info("Sample data loaded into vector store")
            
    except Exception as e:
        # Re-enable logging in case of error
        logging.disable(logging.NOTSET)
        logging.error(f"Failed to load sample data: {str(e)}")
        raise
