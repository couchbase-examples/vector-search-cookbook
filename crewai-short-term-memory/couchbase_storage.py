from typing import Any, Dict, List, Optional
import os
import logging
from datetime import timedelta
from dotenv import load_dotenv
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai import Agent, Crew, Task, Process
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.diagnostics import PingState, ServiceType
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import time
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CouchbaseStorage(RAGStorage):
    """
    Extends RAGStorage to handle embeddings for memory entries using Couchbase.
    """

    def __init__(self, type: str, allow_reset: bool = True, embedder_config: Optional[Dict[str, Any]] = None, crew: Optional[Any] = None):
        """Initialize CouchbaseStorage with configuration."""
        super().__init__(type, allow_reset, embedder_config, crew)
        self._initialize_app()

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search memory entries using vector similarity.
        """
        try:
            # Add type filter
            search_filter = {"memory_type": self.type}
            if filter:
                search_filter.update(filter)

            # Execute search
            results = self.vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter=search_filter
            )
            
            # Format results and deduplicate by content
            seen_contents = set()
            formatted_results = []
            
            for i, (doc, score) in enumerate(results):
                if score >= score_threshold:
                    content = doc.page_content
                    if content not in seen_contents:
                        seen_contents.add(content)
                        formatted_results.append({
                            "id": doc.metadata.get("memory_id", str(i)),
                            "metadata": doc.metadata,
                            "context": content,
                            "score": float(score)
                        })
            
            logger.info(f"Found {len(formatted_results)} unique results for query: {query}")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """
        Save a memory entry with metadata.
        """
        try:
            # Generate unique ID
            memory_id = str(uuid.uuid4())
            timestamp = int(time.time() * 1000)
            
            # Prepare metadata
            if not metadata:
                metadata = {}
            metadata.update({
                "memory_id": memory_id,
                "memory_type": self.type,
                "timestamp": timestamp,
                "source": "crewai"
            })

            # Convert value to string if needed
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, str):
                value = str(value)

            # Save to vector store
            self.vector_store.add_texts(
                texts=[value],
                metadatas=[metadata],
                ids=[memory_id]
            )
            logger.info(f"Saved memory {memory_id}: {value[:100]}...")

        except Exception as e:
            logger.error(f"Save failed: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset the memory storage if allowed."""
        if not self.allow_reset:
            return

        try:
            # Delete documents of this memory type
            self.cluster.query(
                f"DELETE FROM `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}` WHERE memory_type = $type",
                type=self.type
            ).execute()
            logger.info(f"Reset memory type: {self.type}")
        except Exception as e:
            logger.error(f"Reset failed: {str(e)}")
            raise

    def _initialize_app(self):
        """Initialize Couchbase connection and vector store."""
        try:
            load_dotenv()

            # Initialize embeddings
            if self.embedder_config and self.embedder_config.get("provider") == "openai":
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    model=self.embedder_config.get("config", {}).get("model", "text-embedding-3-small")
                )
            else:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    model="text-embedding-3-small"
                )

            # Connect to Couchbase
            auth = PasswordAuthenticator(
                os.getenv('CB_USERNAME', ''),
                os.getenv('CB_PASSWORD', '')
            )
            options = ClusterOptions(auth)
            
            # Initialize cluster connection
            self.cluster = Cluster(os.getenv('CB_HOST', ''), options)
            self.cluster.wait_until_ready(timedelta(seconds=5))

            # Check search service
            ping_result = self.cluster.ping()
            search_available = False
            for service_type, endpoints in ping_result.endpoints.items():
                if service_type == ServiceType.Search:
                    for endpoint in endpoints:
                        if endpoint.state == PingState.OK:
                            search_available = True
                            logger.info(f"Search service is responding at: {endpoint.remote}")
                            break
                    break
            if not search_available:
                raise RuntimeError("Search/FTS service not found or not responding")
            
            # Set up storage configuration
            self.bucket_name = os.getenv('CB_BUCKET_NAME', 'vector-search-testing')
            self.scope_name = os.getenv('SCOPE_NAME', 'shared')
            self.collection_name = os.getenv('COLLECTION_NAME', 'crew')
            self.index_name = os.getenv('INDEX_NAME', 'vector_search_crew')

            # Initialize vector store
            self.vector_store = CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=self.scope_name,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                index_name=self.index_name
            )
            logger.info(f"Initialized CouchbaseStorage for type: {self.type}")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

def main():
    """Demo function to test CouchbaseStorage with CrewAI."""
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")

        # Configure embeddings
        embedder_config = {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        }

        # Initialize storage
        storage = CouchbaseStorage(
            type="short_term",
            embedder_config=embedder_config
        )

        # Reset storage for clean demo
        storage.reset()

        # Test basic storage functionality
        logger.info("\nTesting basic storage...")
        test_memory = "Vector search enables semantic similarity matching by converting data into high-dimensional vectors"
        test_metadata = {"category": "technology"}
        storage.save(test_memory, test_metadata)

        # Test search functionality
        logger.info("\nTesting search functionality...")
        search_results = storage.search(
            query="What is vector search?",
            limit=1,
            score_threshold=0.0
        )
        
        if search_results:
            logger.info("Search test successful!")
            for result in search_results:
                print(f"Found: {result['context']}")
                print(f"Score: {result['score']}")
                print(f"Metadata: {result['metadata']}")
        else:
            logger.warning("No search results found")

        # Initialize language model
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7
        )

        # Create agents with memory
        researcher = Agent(
            role='Research Expert',
            goal='Research vector search capabilities',
            backstory='Expert at finding and analyzing information about vector search technology',
            llm=llm,
            memory=True,
            memory_storage=ShortTermMemory(storage=storage)
        )

        writer = Agent(
            role='Technical Writer',
            goal='Create clear documentation',
            backstory='Expert at technical documentation',
            llm=llm,
            memory=True,
            memory_storage=ShortTermMemory(storage=storage)
        )

        # Create tasks
        research_task = Task(
            description='Research vector search capabilities in modern databases. Focus on Couchbase vector search features.',
            agent=researcher,
            expected_output="A comprehensive analysis of vector search capabilities in modern databases, with emphasis on Couchbase implementation."
        )

        writing_task = Task(
            description='Create documentation about vector search findings, focusing on practical implementation details.',
            agent=writer,
            context=[research_task],
            expected_output="A well-structured technical document explaining vector search implementation in Couchbase."
        )

        # Create crew with memory
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            memory=True,
            verbose=True
        )

        # Run the crew
        logger.info("\nStarting crew tasks...")
        result = crew.kickoff()
        
        print("\nCrew Result:")
        print("-" * 80)
        print(result)
        print("-" * 80)

        # Wait for memories to be stored
        time.sleep(2)

        # Test memory retention
        logger.info("\nTesting memory retention...")
        memory_query = "What are the key features of vector search in Couchbase?"
        memory_results = storage.search(
            query=memory_query,
            limit=2,
            score_threshold=0.0  # Lower threshold to see all results
        )
        
        print("\nMemory Search Results:")
        print("-" * 80)
        for result in memory_results:
            print(f"Context: {result['context']}")
            print(f"Score: {result['score']}")
            print(f"Metadata: {result['metadata']}")
            print("-" * 80)

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
