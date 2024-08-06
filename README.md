# Semantic Search with Couchbase Vector Store and LLM Integration

This repository contains a Python script that demonstrates the integration of a semantic search system using Couchbase's vector store capabilities, Cohere's embedding model, and a Retrieval-Augmented Generation (RAG) pipeline with OpenAI's GPT-4.

## Prerequisites

Before running the script, ensure that you have the following installed:

- Python 3.8+
- Couchbase Server
- Required Python packages (listed in `requirements.txt`)
- A `.env` file containing necessary environment variables

## Environment Variables

The script uses several environment variables, which should be defined in a `.env` file in the root directory. Below are the required variables:

```plaintext
COHERE_API_KEY=your-cohere-api-key
OPENAI_API_KEY=your-openai-api-key
CB_USERNAME=your-couchbase-username
CB_PASSWORD=your-couchbase-password
CB_BUCKET_NAME=your-couchbase-bucket-name
CB_HOST=your-couchbase-host
INDEX_NAME=your-index-name
CACHE_COLLECTION=your-cache-collection-name
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the necessary environment variables.

4. Run the script:

   ```bash
   python main.py
   ```

## Components

### 1. Cohere Embeddings

The script uses Cohere's small model to generate embeddings for the documents loaded from the TREC dataset. These embeddings are then stored in Couchbase's vector store for efficient similarity search.

### 2. Couchbase Integration

Couchbase is used to store document embeddings and their metadata, allowing for fast retrieval during the semantic search. The script demonstrates connecting to Couchbase, storing embeddings, and retrieving similar documents based on a query.

### 3. Retrieval-Augmented Generation (RAG)

The RAG pipeline integrates OpenAI's GPT-4 model to generate answers based on retrieved documents. This allows the system to provide more contextually relevant answers.

### 4. Semantic Search

A semantic search function is provided, which performs a similarity search using the vector store. The top-k most similar documents are retrieved and used for generating responses.

### 5. Error Handling

The script includes basic error handling for connectivity issues, missing environment variables, and embedding generation failures.

## Future Work

This script serves as a foundation for integrating other embedding models and vector stores. Planned expansions include:

- **Voyage Integration:** Extend the script to support embedding generation and retrieval using the Voyage model.
- **Additional Models:** Integrate other models for generating embeddings and performing semantic search.
- **Advanced Query Handling:** Implement more sophisticated query parsing and handling strategies.

## Contribution

Contributions are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This template covers the core functionality of the current task and leaves room for future enhancements and integrations. Feel free to adjust the sections as you see fit!