# Semantic Search with Couchbase Vector Store and LLM Integration

This repository demonstrates a robust semantic search system using Couchbase's vector store capabilities, multiple embedding models (Cohere, Voyage, and Jina), and a Retrieval-Augmented Generation (RAG) pipeline with OpenAI's GPT-4.

## Features

- Support for multiple embedding models: Cohere, Voyage, and Jina
- Flexible Couchbase index structure for multi-model vector search
- Retrieval-Augmented Generation (RAG) with OpenAI's GPT-4
- Efficient semantic search across different embedding types
- Easy switching between embedding models

## Prerequisites

- Python 3.8+
- Couchbase Server
- Required Python packages (listed in `requirements.txt`)
- API keys for Cohere, Voyage, Jina, and OpenAI

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```plaintext
COHERE_API_KEY=your-cohere-api-key
VOYAGE_API_KEY=your-voyage-api-key
JINA_API_KEY=your-jina-api-key
OPENAI_API_KEY=your-openai-api-key
CB_USERNAME=your-couchbase-username
CB_PASSWORD=your-couchbase-password
CB_BUCKET_NAME=your-couchbase-bucket-name
CB_HOST=your-couchbase-host
INDEX_NAME=vector_search_xyz
CACHE_COLLECTION=cache
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vector-search-cookbook.git
   cd vector-search-cookbook
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the Couchbase index:
   - Use the provided `vector_search_xyz` index definition to create a new index in your Couchbase cluster.
   - The index supports separate properties for each embedding model: `shared.cohere`, `shared.voyage`, and `shared.jina`.

4. Run the script:
   ```bash
   python3 search.py
   ```

## Components

### 1. Multi-Model Embeddings
The system supports embeddings from Cohere, Voyage, and Jina. You can easily switch between these models in the code.

### 2. Couchbase Vector Store
Couchbase is used to store document embeddings and metadata. The index structure allows for efficient retrieval across different embedding types.

### 3. Retrieval-Augmented Generation (RAG)
The RAG pipeline integrates OpenAI's GPT-4 to generate contextually relevant answers based on retrieved documents.

### 4. Semantic Search
The semantic search function performs similarity searches using the appropriate embedding type and retrieves the top-k most similar documents.

### 5. Model Switching
A function is provided to switch between embedding models, allowing for easy comparison and flexibility in your search pipeline.

## Usage

The main script demonstrates:
1. Loading a dataset
2. Generating embeddings using the selected model
3. Storing documents and embeddings in Couchbase
4. Performing semantic search
5. Generating responses using the RAG pipeline

You can modify the `current_model` variable in the main function to switch between 'cohere', 'voyage', and 'jina' embedding models.

## Couchbase Index Structure

The Couchbase index is structured to support multiple embedding types:

```json
{
  "types": {
    "shared.cohere": { ... },
    "shared.voyage": { ... },
    "shared.jina": { ... }
  }
}
```

Each type has its own `embedding` field with the appropriate dimensions for the respective model.

## Future Work

- Implement a user interface for easy model switching and query input
- Add support for more embedding models
- Optimize RAG pipeline for better response generation
- Implement advanced query preprocessing and result post-processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.