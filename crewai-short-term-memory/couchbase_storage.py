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

        # Initialize storage with debug logging
        logger.info("Initializing CouchbaseStorage for CrewAI integration")
        storage = CouchbaseStorage(
            type="short_term",
            embedder_config=embedder_config,
            allow_reset=True
        )

        # Reset storage for clean demo
        storage.reset()
        logger.info("Storage reset completed")

        # Test basic storage functionality
        logger.info("\nTesting basic storage...")
        test_memory = "Pep Guardiola praised Manchester City's current form, saying 'The team is playing well, we are in a good moment. The way we are training, the way we are playing - I am really pleased.'"
        test_metadata = {"category": "sports", "test": "initial_memory"}
        storage.save(test_memory, test_metadata)

        # Test search functionality
        logger.info("\nTesting search functionality...")
        search_results = storage.search(
            query="What did Guardiola say about Manchester City?",
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

        # Initialize ShortTermMemory with our storage
        logger.info("\nInitializing ShortTermMemory with CouchbaseStorage")
        memory = ShortTermMemory(storage=storage)
        
        # Test memory save via ShortTermMemory
        logger.info("Testing ShortTermMemory direct save...")
        memory.save(
            value="Test memory via ShortTermMemory: Manchester City has been dominant in recent matches.",
            metadata={"test": "stm_direct"},
            agent="test_agent"
        )

        # Initialize language model
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7
        )

        logger.info("\nCreating agents with CouchbaseStorage-backed memory")
        # Create agents with memory, ensuring CouchbaseStorage is properly connected
        sports_analyst = Agent(
            role='Sports Analyst',
            goal='Analyze Manchester City performance',
            backstory='Expert at analyzing football teams and providing insights on their performance',
            llm=llm,
            memory=True,
            memory_storage=memory
        )
        
        # Verify memory storage is properly set
        if hasattr(sports_analyst, 'memory') and sports_analyst.memory:
            logger.info(f"Sports Analyst memory type: {type(sports_analyst.memory).__name__}")
            if hasattr(sports_analyst.memory, 'storage'):
                logger.info(f"Sports Analyst memory storage type: {type(sports_analyst.memory.storage).__name__}")
        else:
            logger.warning("Sports Analyst has no memory configured!")

        journalist = Agent(
            role='Sports Journalist',
            goal='Create engaging football articles',
            backstory='Experienced sports journalist who specializes in Premier League coverage',
            llm=llm,
            memory=True,
            memory_storage=memory
        )

        # Create tasks
        analysis_task = Task(
            description='Analyze Manchester City\'s recent performance based on Pep Guardiola\'s comments: "The team is playing well, we are in a good moment. The way we are training, the way we are playing - I am really pleased."',
            agent=sports_analyst,
            expected_output="A comprehensive analysis of Manchester City's current form based on Guardiola's comments."
        )

        writing_task = Task(
            description='Write a sports article about Manchester City\'s form using the analysis and Guardiola\'s comments.',
            agent=journalist,
            context=[analysis_task],
            expected_output="An engaging sports article about Manchester City's current form and Guardiola's perspective."
        )

        # Create crew with memory - explicitly setting short_term_memory directly 
        logger.info("\nCreating crew with memory enabled")
        crew = Crew(
            agents=[sports_analyst, journalist],
            tasks=[analysis_task, writing_task],
            process=Process.sequential,
            memory=True,
            short_term_memory=memory,  # Explicitly pass our memory implementation
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
        
        # List all documents in the collection
        logger.info("\nListing all memory entries:")
        try:
            # Query to fetch all documents of this memory type
            query_str = f"SELECT META().id, * FROM `{storage.bucket_name}`.`{storage.scope_name}`.`{storage.collection_name}` WHERE memory_type = $type"
            query_result = storage.cluster.query(query_str, type=storage.type)
            
            print(f"\nAll memory entries in Couchbase:")
            print("-" * 80)
            for i, row in enumerate(query_result, 1):
                doc_id = row.get('id')
                memory_id = row.get(storage.collection_name, {}).get('memory_id', 'unknown')
                content = row.get(storage.collection_name, {}).get('text', '')[:100] + "..."  # Truncate for readability
                source = row.get(storage.collection_name, {}).get('source', 'unknown')
                agent = row.get(storage.collection_name, {}).get('agent', 'unknown')
                
                print(f"Entry {i}:")
                print(f"ID: {doc_id}")
                print(f"Memory ID: {memory_id}")
                print(f"Content: {content}")
                print(f"Source: {source}")
                print(f"Agent: {agent}")
                print("-" * 80)
        except Exception as e:
            logger.error(f"Failed to list memory entries: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        # Test memory retention
        logger.info("\nTesting memory retention...")
        memory_query = "What is Manchester City's current form according to Guardiola?"
        memory_results = storage.search(
            query=memory_query,
            limit=5,  # Increased to see more results
            score_threshold=0.0  # Lower threshold to see all results
        )
        
        print("\nMemory Search Results:")
        print("-" * 80)
        for result in memory_results:
            print(f"Context: {result['context']}")
            print(f"Score: {result['score']}")
            print(f"Metadata: {result['metadata']}")
            print("-" * 80)

        # Try a more specific query to find agent interactions
        logger.info("\nSearching for agent interactions in memory...")
        interaction_query = "Manchester City playing style analysis tactical"
        interaction_results = storage.search(
            query=interaction_query,
            limit=5,
            score_threshold=0.0
        )
        
        print("\nAgent Interaction Memory Results:")
        print("-" * 80)
        for result in interaction_results:
            print(f"Context: {result['context'][:200]}...")  # Limit output size
            print(f"Score: {result['score']}")
            print(f"Agent: {result['metadata'].get('agent', 'unknown')}")
            print(f"Source: {result['metadata'].get('source', 'unknown')}")
            print("-" * 80)

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
