import os
import logging
from typing import Any, Dict, List, Optional
from crewai.memory.storage.rag_storage import RAGStorage
from crewai import Agent, Crew, Task, Process
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.management.search import SearchIndex
from couchbase.exceptions import CouchbaseException
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Disable all logging except our own
for name in logging.root.manager.loggerDict:
    if name != __name__:
        logging.getLogger(name).setLevel(logging.WARNING)

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
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset the memory storage."""
        if self.allow_reset:
            try:
                # Create primary index if it doesn't exist
                self.cluster.query(
                    f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`"
                ).execute()
                
                # Delete all documents
                self.cluster.query(
                    f"DELETE FROM `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`"
                ).execute()
                logger.info(f"Successfully reset collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Reset failed: {str(e)}")
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

            # Connect to Couchbase
            auth = PasswordAuthenticator(
                os.getenv('CB_USERNAME', 'Administrator'),
                os.getenv('CB_PASSWORD', 'password')
            )
            self.cluster = Cluster(
                os.getenv('CB_HOST', 'couchbase://localhost'),
                ClusterOptions(auth)
            )
            
            # Set up bucket, scope, and collection names
            self.bucket_name = os.getenv('CB_BUCKET_NAME', 'vector-search-testing')
            self.scope_name = os.getenv('SCOPE_NAME', 'shared')
            self.collection_name = self.type  # Use the type parameter as collection name
            self.index_name = os.getenv('INDEX_NAME', 'vector_search_crew')

            # Create primary index if it doesn't exist
            self.cluster.query(
                f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`"
            ).execute()

            # Initialize vector store
            self.vector_store = CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=self.scope_name,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                index_name=self.index_name,
            )
            logger.info("Storage initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
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
        except Exception as e:
            logger.error(f"Save failed: {str(e)}")
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
            ("Vector search uses mathematical vectors to find similar items by converting data into high-dimensional vector space", 
             {"category": "technology", "type": "concept"}),
            ("Couchbase vector search enables semantic similarity matching by storing and comparing vector embeddings", 
             {"category": "database", "type": "implementation"}),
            ("Vector embeddings represent text, images, and other data as numerical vectors for efficient similarity search", 
             {"category": "search", "type": "technique"})
        ]
        
        for text, metadata in test_entries:
            storage.save(text, metadata)
            logger.info(f"Saved entry with metadata: {metadata}")
        
        # Test searching
        logger.info("\nTesting search functionality...")
        query = "Tell me about vector search"
        results = storage.search(query, limit=2)
        
        logger.info(f"\nSearch results for query: '{query}'")
        print("-"*80)
        for result in results:
            print("\nResult:")
            print(f"Context: {result['context']}")
            print(f"Metadata: {result['metadata']}")
            print(f"Score: {result['score']}")
            print("-"*80)
        
        # Test CrewAI integration
        logger.info("\nTesting CrewAI integration...")
        
        # Initialize language model
        llm = ChatOpenAI(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model="gpt-4",
            temperature=0.7
        )
        
        # Create agents
        researcher = Agent(
            role='Research Expert',
            goal='Find relevant information',
            backstory='Expert at finding and analyzing information',
            llm=llm,
            memory=True,
            memory_storage=storage
        )
        
        writer = Agent(
            role='Technical Writer',
            goal='Create clear documentation',
            backstory='Expert at technical writing and documentation',
            llm=llm,
            memory=True,
            memory_storage=storage
        )
        
        # Create tasks
        research_task = Task(
            description='Research vector search capabilities',
            agent=researcher,
            expected_output="Detailed findings about vector search technology and implementations"
        )
        
        writing_task = Task(
            description='Document the findings',
            agent=writer,
            expected_output="Clear and comprehensive documentation of the research findings",
            context=[research_task]
        )
        
        # Create and run crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=False
        )
        
        logger.info("Starting crew tasks...")
        result = crew.kickoff()
        logger.info("Crew tasks completed")
        print("\nCrew Result:")
        print("-"*80)
        print(result)
        print("-"*80)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
