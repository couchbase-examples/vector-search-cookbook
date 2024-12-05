import logging
import time
import sys
from crewai import Crew, Process

from .agents import setup_agents, create_tasks
from .couchbase_setup import setup_couchbase
from .vector_store import setup_vector_store, load_sample_data

def search(query, vector_store, researcher, writer):
    """Perform search and generate response"""
    try:
        # Create and execute crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=create_tasks(query, researcher, writer),
            process=Process.sequential,  # Execute tasks in order
            verbose=True,
            cache=True,  # Enable caching
            planning=True  # Enable planning capability
        )
        
        result = crew.kickoff()
        return result
        
    except Exception as e:
        logging.error(f"Search failed: {str(e)}")
        raise

def main():
    """Main function"""
    try:
        # Setup components
        logging.info("Setting up components...")
        cluster = setup_couchbase()
        vector_store, llm = setup_vector_store(cluster)
        load_sample_data(vector_store)
        researcher, writer = setup_agents(llm, vector_store)
        logging.info("Setup complete")
        
        # Interactive search loop
        while True:
            try:
                query = input("\nEnter your query (or 'quit' to exit): ").strip()
                if query.lower() == 'quit':
                    break
                
                if not query:
                    print("Please enter a valid query")
                    continue
                    
                print("\nProcessing your query...")
                start_time = time.time()
                result = search(query, vector_store, researcher, writer)
                elapsed_time = time.time() - start_time
                
                print(f"\nResponse (completed in {elapsed_time:.2f} seconds):")
                print("-" * 80)
                print(result)
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\nSearch interrupted")
                break
                
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                print("Please try a different query or check the logs for more details")
                logging.exception("Error details:")
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.exception("Error details:")
        sys.exit(1)
    
    finally:
        print("\nThank you for using the search system!")

if __name__ == "__main__":
    main()
