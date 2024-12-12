import logging
import time
import sys
from typing import Any, Optional
from crewai import Crew, Process

from .agents import setup_agents, create_tasks
from .couchbase_setup import setup_couchbase
from .vector_store import setup_vector_store, load_sample_data

def format_response(result: Any) -> str:
    """Format the response for better readability"""
    if not result:
        return "No response generated"
        
    # Format the main response
    formatted = []
    formatted.append("=" * 80)
    formatted.append("RESPONSE")
    formatted.append("=" * 80)
    formatted.append(str(result))
    
    # Add task outputs if available
    if hasattr(result, 'tasks_output'):
        formatted.append("\n" + "=" * 80)
        formatted.append("DETAILED TASK OUTPUTS")
        formatted.append("=" * 80)
        for task_output in result.tasks_output:
            formatted.append(f"\nTask: {task_output.description[:100]}...")
            formatted.append("-" * 40)
            formatted.append(f"Output: {task_output.raw}")
            formatted.append("-" * 40)
    
    return "\n".join(formatted)

def search(query: str, vector_store: Any, researcher: Any, writer: Any) -> Optional[str]:
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
        logging.exception("Error details:")
        raise

def main():
    """Main function"""
    try:
        # Setup logging with more detail
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup components
        print("\nInitializing search system...")
        print("This may take a few minutes for initial setup.")
        print("=" * 80)
        
        cluster = setup_couchbase()
        vector_store, llm = setup_vector_store(cluster)
        load_sample_data(vector_store)
        researcher, writer = setup_agents(llm, vector_store)
        
        print("\nSetup complete! You can now enter your queries.")
        print("=" * 80)
        
        # Interactive search loop
        while True:
            try:
                # Get query
                query = input("\nEnter your query (or 'quit' to exit): ").strip()
                if query.lower() == 'quit':
                    break
                
                if not query:
                    print("Please enter a valid query")
                    continue
                
                # Process query
                print("\nProcessing your query...")
                print("This may take a moment as the AI agents work on your request.")
                print("-" * 80)
                
                start_time = time.time()
                result = search(query, vector_store, researcher, writer)
                elapsed_time = time.time() - start_time
                
                # Format and display results
                print(f"\nQuery completed in {elapsed_time:.2f} seconds")
                print(format_response(result))
                
            except KeyboardInterrupt:
                print("\nSearch interrupted by user")
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
        print("=" * 80)

if __name__ == "__main__":
    main()
