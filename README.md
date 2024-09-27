# Semantic Search with Couchbase Vector Store and LLM Integration

This repository demonstrates a robust semantic search system using Couchbase's vector store capabilities, multiple embedding models (Cohere, Voyage AI, and Jina AI), and a Retrieval-Augmented Generation (RAG) pipeline with OpenAI's GPT-4.

## Features

- Support for multiple embedding models: Cohere, Voyage AI, and Jina AI
- Flexible Couchbase index structure for multi-model vector search
- Retrieval-Augmented Generation (RAG) with OpenAI's GPT-4
- Efficient semantic search across different embedding types
- Easy switching between embedding models
- Caching functionality for improved performance

## Prerequisites

- Python 3.8+
- Couchbase Server
- Required Python packages (listed in `requirements.txt` for each model)
- API keys for Cohere, Voyage AI, Jina AI, and OpenAI

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```plaintext
COHERE_API_KEY=your-cohere-api-key
VOYAGE_API_KEY=your-voyage-api-key
JINA_API_KEY=your-jina-api-key
JINACHAT_API_KEY=your-jinachat-api-key
OPENAI_API_KEY=your-openai-api-key
CB_USERNAME=your-couchbase-username
CB_PASSWORD=your-couchbase-password
CB_BUCKET_NAME=your-couchbase-bucket-name
CB_HOST=your-couchbase-host
INDEX_NAME=vector_search_xyz (replace xyz with cohere, voyage, or jina)
SCOPE_NAME=shared
COLLECTION_NAME=xyz (replace xyz with cohere, voyage, or jina)
CACHE_COLLECTION=cache
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vector-search-cookbook.git
   cd vector-search-cookbook
   ```

2. Install dependencies for the desired embedding model:
   ```bash
   cd cohere  # or voyage or jinaai
   pip install -r requirements.txt
   ```

3. Set up the Couchbase index:
   - Use the provided `vector_search_xyz.json` index definition in each model's directory to create a new index in your Couchbase cluster.
   - The index supports separate properties for each embedding model: `shared.cohere`, `shared.voyage`, and `shared.jina`.

4. Run the script for the desired embedding model:
   ```bash
   python search.py
   ```

## Components

### 1. Multi-Model Embeddings
The system supports embeddings from Cohere, Voyage AI, and Jina AI. Each model has its own implementation in separate directories.

### 2. Couchbase Vector Store
Couchbase is used to store document embeddings and metadata. The index structure allows for efficient retrieval across different embedding types.

### 3. Retrieval-Augmented Generation (RAG)
The RAG pipeline integrates OpenAI's GPT-4 for Cohere and Voyage AI implementations, and JinaChat for the Jina AI implementation, to generate contextually relevant answers based on retrieved documents.

### 4. Semantic Search
Each `search.py` file implements a semantic search function that performs similarity searches using the appropriate embedding type and retrieves the top-k most similar documents.

### 5. Caching
The system implements caching functionality using CouchbaseCache to improve performance for repeated queries.

## Usage

Each `search.py` file demonstrates:
1. Loading the TREC dataset
2. Generating embeddings using the selected model
3. Storing documents and embeddings in Couchbase
4. Performing semantic search
5. Generating responses using the RAG pipeline
6. Demonstrating caching functionality

To use a specific embedding model, navigate to its directory and run the `search.py` file:

```bash
cd cohere  # or voyage or jinaai
python search.py
```

## Couchbase Index Structure

The Couchbase index is structured to support multiple embedding types:

```json
{
  "types": {
    "shared.cohere": {
      "dynamic": false,
      "fields": {
        "embedding": {
          "dims": 1024,
          "type": "vector"
        }
      }
    },
    "shared.voyage": {
      "dynamic": false,
      "fields": {
        "embedding": {
          "dims": 1536,
          "type": "vector"
        }
      }
    },
    "shared.jina": {
      "dynamic": false,
      "fields": {
        "embedding": {
          "dims": 768,
          "type": "vector"
        }
      }
    }
  }
}
```

Each type has its own `embedding` field with the appropriate dimensions for the respective model.

## Future Work

- Implement a user interface for easy model switching and query input
- Add support for more embedding models
- Optimize RAG pipeline for better response generation
- Implement advanced query preprocessing and result post-processing
- Conduct performance comparisons between different embedding models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.