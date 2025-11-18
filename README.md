# Semantic Search with Couchbase Vector Store and LLM Integration

This repository demonstrates how to build a powerful semantic search engine using Couchbase as the backend database, combined with various AI-powered embedding and language model providers such as OpenAI, Azure OpenAI, Anthropic (Claude), Cohere, Hugging Face, Jina AI, Mistral AI, and Voyage AI.

Each example provides two distinct approaches:
- **Couchbase Vector Search with Search**: Uses Couchbase's Full-Text Search (FTS) based vector search capabilities with pre-created search indices
- **Couchbase Vector Search**: Leverages Couchbase's native SQL++ queries with vector similarity functions

For guidance on choosing the right vector index for your use case, see the [Couchbase documentation](https://docs.couchbase.com/server/current/vector-search/choose-the-right-vector-index.html).

Semantic search goes beyond simple keyword matching by understanding the context and meaning behind the words in a query, making it essential for applications that require intelligent information retrieval.

## Features

- **Multiple Embedding Models**: Support for embeddings from OpenAI, Azure OpenAI, Anthropic (Claude), Cohere, Hugging Face, Jina AI, Mistral AI, and Voyage AI.
- **Couchbase Vector Store**: Utilizes Couchbase's vector storage capabilities for efficient similarity search.
- **Retrieval-Augmented Generation (RAG)**: Integrates with advanced language models like GPT-4 for generating contextually relevant responses.
- **Scalable and Flexible**: Easy to switch between different embedding models and adjust the index structure accordingly.
- **Caching Mechanism**: Implements `CouchbaseCache` for improved performance on repeated queries.

## Prerequisites

- Python 3.8+
- Couchbase Cluster (Self Managed or Capella) version 7.6+ with [Search Service](https://docs.couchbase.com/server/current/search/search.html)

- API keys for the respective AI providers (e.g., OpenAI, Azure OpenAI, Anthropic, Cohere, etc.)

## Setup

### 1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vector-search-cookbook.git
   cd vector-search-cookbook
   ```

### 2. Choose Your Approach:

#### For Couchbase Vector Search with Search Examples:
Use the provided `{model}_index.json` index definition file in each model's `fts/` directory to create a new vector search index in your Couchbase cluster.

#### For Couchbase Vector Search Examples:
No additional setup required. Vector search index will be created in each model's example.

### 3. Run the notebook file

You can either run the notebook file on [Google Colab](https://colab.research.google.com/) or run it on your system by setting up the Python environment.


## Components

### 1. Multiple Embedding Models

The system supports embeddings from various AI providers:

* OpenAI
* Azure OpenAI
* Anthropic (Claude)
* Cohere
* Hugging Face
* Jina AI
* Mistral AI
* Voyage AI

### 2. Couchbase Vector Store

Couchbase is used to store document embeddings and metadata. The index structure allows for efficient retrieval across different embedding types.

### 3. Retrieval-Augmented Generation (RAG)

The RAG pipeline integrates with language models like GPT-4 to generate contextually relevant answers based on retrieved documents.

### 4. Semantic Search

Each notebook implements a semantic search function that performs similarity searche using the appropriate embedding type and retrieves the top-k most similar documents.

### 5. Caching

The system implements caching functionality using `CouchbaseCache` to improve performance for repeated queries.

## Couchbase Vector Search Index (Search Approach Only)

For Couchbase Vector Search with Search examples, you'll need to create a vector search index using the provided JSON configuration files. For more information on creating a vector search index, please follow the [instructions](https://docs.couchbase.com/cloud/vector-search/create-vector-search-index-ui.html). The following is an example for Azure OpenAI Model.

```json
{
    "type": "fulltext-index",
    "name": "vector_search_azure",
    "uuid": "",
    "sourceType": "gocbcore",
    "sourceName": "vector-search-testing",
    "planParams": {
      "maxPartitionsPerPIndex": 64,
      "indexPartitions": 16
    },
    "params": {
      "doc_config": {
        "docid_prefix_delim": "",
        "docid_regexp": "",
        "mode": "scope.collection.type_field",
        "type_field": "type"
      },
      "mapping": {
        "analysis": {},
        "default_analyzer": "standard",
        "default_datetime_parser": "dateTimeOptional",
        "default_field": "_all",
        "default_mapping": {
          "dynamic": true,
          "enabled": false
        },
        "default_type": "_default",
        "docvalues_dynamic": false,
        "index_dynamic": true,
        "store_dynamic": false,
        "type_field": "_type",
        "types": {
          "shared.azure": {
            "dynamic": true,
            "enabled": true,
            "properties": {
              "embedding": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "dims": 1536,
                    "index": true,
                    "name": "embedding",
                    "similarity": "dot_product",
                    "type": "vector",
                    "vector_index_optimized_for": "recall"
                  }
                ]
              },
              "text": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "index": true,
                    "name": "text",
                    "store": true,
                    "type": "text"
                  }
                ]
              }
            }
          }
        }
      },
      "store": {
        "indexType": "scorch",
        "segmentVersion": 16
      }
    },
    "sourceParams": {}
  }
```