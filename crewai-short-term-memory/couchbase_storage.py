import os
import logging
from typing import Any, Dict, List, Optional
from crewai.memory.storage.rag_storage import RAGStorage
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.management.search import SearchIndex
from couchbase.exceptions import (
    CouchbaseException,
    DocumentNotFoundException,
    QueryIndexNotFoundException,
    TimeoutException
)
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CouchbaseStorage(RAGStorage):
    """
    Extends Storage to handle embeddings for memory entries using Couchbase.
    """

    def __init__(self, type, allow_reset=True, embedder_config=None, crew=None):
        try:
            super().__init__(type, allow_reset, embedder_config, crew)
            self._initialize_app()
            logger.info(f"CouchbaseStorage initialized for type: {type}")
        except Exception as e:
            logger.error(f"Failed to initialize CouchbaseStorage: {str(e)}")
            raise

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0,
    ) -> List[Any]:
        """Search memory entries using vector similarity."""
        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter=filter,
                score_threshold=score_threshold
            )
            
            return [{
                "id": str(i),
                "metadata": doc.metadata,
                "context": doc.page_content,
                "score": score
            } for i, (doc, score) in enumerate(results)]
        except TimeoutException as e:
            logger.error(f"Search operation timed out: {str(e)}")
            raise
        except CouchbaseException as e:
            logger.error(f"Couchbase error during search: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset the memory storage."""
        if self.allow_reset:
            try:
                # Delete all documents in the collection using N1QL
                self.cluster.query(
                    f"DELETE FROM `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`"
                ).execute()
                logger.info(f"Successfully reset collection: {self.collection_name}")
            except QueryIndexNotFoundException:
                logger.error("Primary index not found. Attempting to create...")
                try:
                    self.cluster.query(
                        f"CREATE PRIMARY INDEX ON `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`"
                    ).execute()
                    # Retry delete after creating index
                    self.cluster.query(
                        f"DELETE FROM `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`"
                    ).execute()
                    logger.info("Primary index created and collection reset successfully")
                except Exception as e:
                    logger.error(f"Failed to create primary index: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Failed to reset collection: {str(e)}")
                raise

    def _initialize_app(self):
        """Initialize Couchbase client and vector store."""
        try:
            # Check for required environment variables
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY environment variable is required")

            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                model="text-embedding-ada-002"
            )
            logger.info("OpenAI embeddings initialized")

            # Connect to Couchbase
            auth = PasswordAuthenticator(
                os.getenv('CB_USERNAME', 'Administrator'),
                os.getenv('CB_PASSWORD', 'password')
            )
            self.cluster = Cluster(
                os.getenv('CB_HOST', 'couchbase://localhost'),
                ClusterOptions(auth)
            )
            logger.info("Connected to Couchbase cluster")
            
            # Set up bucket, scope, and collection names
            self.bucket_name = os.getenv('CB_BUCKET_NAME', 'vector-search-testing')
            self.scope_name = os.getenv('SCOPE_NAME', 'shared')
            self.collection_name = self.type  # Use the type parameter as collection name
            self.index_name = os.getenv('INDEX_NAME', 'vector_search_crew')

            # Create primary index if it doesn't exist
            try:
                self.cluster.query(
                    f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`"
                ).execute()
                logger.info(f"Primary index ensured for collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Could not create primary index: {str(e)}")

            # Initialize vector store
            self.vector_store = CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=self.scope_name,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                index_name=self.index_name,
            )
            logger.info("Vector store initialized successfully")

        except ValueError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise
        except CouchbaseException as e:
            logger.error(f"Couchbase error during initialization: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {str(e)}")
            raise

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """Save a memory entry with metadata."""
        try:
            # Add text to vector store
            self.vector_store.add_texts(
                texts=[value],
                metadatas=[metadata or {}],
                ids=[f"{self.type}_{metadata.get('id', len(self.search('', limit=1)) + 1)}"]
            )
            logger.info(f"Successfully saved entry with metadata: {metadata}")
        except TimeoutException as e:
            logger.error(f"Save operation timed out: {str(e)}")
            raise
        except CouchbaseException as e:
            logger.error(f"Couchbase error during save: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during save: {str(e)}")
            raise

def main():
    """
    Demo function to test CouchbaseStorage functionality.
    """
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Initialize storage
        logger.info("Initializing CouchbaseStorage...")
        storage = CouchbaseStorage("crew_stm_demo")
        
        # Clear existing data
        logger.info("Clearing existing data...")
        storage.reset()
        
        # Test saving entries
        logger.info("\nSaving test entries...")
        test_entries = [
            ("This is a test document about AI", {"category": "technology"}),
            ("Couchbase provides excellent vector search capabilities", {"category": "database"}),
            ("Vector embeddings help with semantic search", {"category": "search"})
        ]
        
        for text, metadata in test_entries:
            try:
                storage.save(text, metadata)
                logger.info(f"Saved entry with metadata: {metadata}")
            except Exception as e:
                logger.error(f"Failed to save entry: {str(e)}")
                raise
        
        # Test searching
        logger.info("\nTesting search functionality...")
        query = "Tell me about vector search"
        try:
            results = storage.search(query, limit=2)
            
            logger.info(f"\nSearch results for query: '{query}'")
            print("-"*80)
            for result in results:
                print("\nResult:")
                print(f"Context: {result['context']}")
                print(f"Metadata: {result['metadata']}")
                print(f"Score: {result['score']}")
                print("-"*80)
        except Exception as e:
            logger.error(f"Search operation failed: {str(e)}")
            raise
        
        # Test reset
        if input("\nWould you like to reset the storage? (y/n): ").lower() == 'y':
            try:
                logger.info("Resetting storage...")
                storage.reset()
                logger.info("Storage reset complete")
            except Exception as e:
                logger.error(f"Reset operation failed: {str(e)}")
                raise
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
