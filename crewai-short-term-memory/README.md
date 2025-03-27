# CrewAI with Couchbase Short-Term Memory

This implementation demonstrates how to use Couchbase as a storage backend for CrewAI's memory system, particularly for short-term memory and entity memory using vector search capabilities.

## Features

- Custom Couchbase storage implementation for CrewAI's memory system
- Vector similarity search using Couchbase's vector search feature
- Support for both short-term and entity memory
- Configurable embedding providers (defaults to OpenAI)
- Uses existing vector search index for efficient similarity search

## Prerequisites

1. Couchbase Server running with vector search capability
2. Python 3.8+
3. OpenAI API key
4. Required Python packages:
   ```bash
   pip install crewai[tools] langchain-couchbase langchain-openai python-dotenv couchbase
   ```

## Setup

1. Copy the environment variables template:
   ```bash
   cp .env.sample .env
   ```

2. Update the .env file with your credentials:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   CB_USERNAME=Administrator
   CB_PASSWORD=password
   CB_HOST=couchbase://localhost
   CB_BUCKET_NAME=vector-search-testing
   CB_SCOPE_NAME=shared
   ```

3. Ensure the vector search index exists in Couchbase:
   - Index name: vector_search_crew
   - Collection: crew
   - Scope: shared
   - Bucket: vector-search-testing

## Usage

### Basic Usage

```python
from couchbase_storage import CouchbaseStorage
from crewai.memory.short_term.short_term_memory import ShortTermMemory

# Initialize storage
storage = ShortTermMemory(
    storage=CouchbaseStorage(
        type="short_term",
        embedder_config={
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        }
    )
)
```

### With CrewAI

```python
from crewai import Agent, Crew, Task

# Create agent with memory
agent = Agent(
    role='Researcher',
    goal='Research topic',
    memory=True,
    memory_storage=storage
)

# Create crew with memory
crew = Crew(
    agents=[agent],
    tasks=[...],
    memory=True
)
```

## Running the Demo

The implementation includes a demo that showcases:
- Storage initialization
- Agent creation with memory
- Task execution
- Memory retention and retrieval

Run the demo:
```bash
python couchbase_storage.py
```

## Vector Search Index

The implementation uses an existing vector search index with the following configuration:

```json
{
  "type": "fulltext-index",
  "name": "vector_search_crew",
  "sourceName": "vector-search-testing",
  "params": {
    "mapping": {
      "types": {
        "shared.crew": {
          "properties": {
            "embedding": {
              "dims": 1536,
              "similarity": "dot_product",
              "type": "vector"
            }
          }
        }
      }
    }
  }
}
```

## Implementation Details

The CouchbaseStorage class:
- Extends CrewAI's RAGStorage
- Implements vector similarity search
- Handles document storage and retrieval
- Manages memory reset functionality
- Supports configurable embeddings

Key methods:
- `search()`: Vector similarity search with filtering
- `save()`: Store memory entries with metadata
- `reset()`: Clear memory storage
- `_initialize_app()`: Setup Couchbase connection and vector store

## Error Handling

The implementation includes comprehensive error handling:
- Connection errors
- Search failures
- Storage issues
- Configuration problems

All errors are logged with descriptive messages for debugging.
