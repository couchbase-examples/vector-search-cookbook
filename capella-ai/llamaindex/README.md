# LlamaIndex RAG with Couchbase Capella

This directory contains an example of building a Retrieval Augmented Generation (RAG) system using [LlamaIndex](https://www.llamaindex.ai/) and [Couchbase Capella](https://www.couchbase.com/products/capella/).

## Overview

The `llamaindex_rag.py` script demonstrates how to:

1. Connect to a Couchbase Capella cluster
2. Set up a collection and vector search index
3. Load and process the HotpotQA dataset from RAGBench
4. Generate embeddings using Capella AI Services
5. Store documents and their embeddings in Couchbase
6. Perform semantic search using vector similarity
7. Generate responses using a large language model with retrieved context

## Key Differences from LangChain Implementation

This implementation differs from the LangChain version (`../langchain/lachain_rag.py`) in several ways:

1. **Framework**: Uses LlamaIndex instead of LangChain as the RAG framework
2. **Dataset**: Uses the HotpotQA dataset from RAGBench instead of BBC News articles
3. **Architecture**: 
   - LlamaIndex uses a node-based architecture with documents split into nodes
   - Different retrieval and query engine APIs
   - Different approach to document chunking and indexing

4. **Features**:
   - Demonstrates multi-hop question answering (questions that require connecting information from multiple sources)
   - Includes comparison with ground truth answers from the dataset

## Requirements

- Python 3.8+
- Couchbase Capella account with a deployed cluster
- Capella AI Services with deployed models
- Required Python packages:
  ```
  datasets
  llama-index-vector-stores-couchbase
  llama-index-embeddings-openai
  llama-index-llms-openai
  ```

## Running the Example

1. Install the required packages:
   ```
   pip install datasets llama-index-vector-stores-couchbase llama-index-embeddings-openai llama-index-llms-openai
   ```

2. Run the script:
   ```
   python llamaindex_rag.py
   ```
   
   Or open the notebook version in Jupyter or Google Colab.

3. When prompted, enter your Couchbase Capella credentials and configuration.

## Dataset

The HotpotQA dataset is a question-answering dataset that features multi-hop questions, requiring the system to gather information from multiple sources to provide a complete answer. This makes it an excellent test case for RAG systems, as it demonstrates the ability to connect information across different contexts.

## Additional Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Couchbase Capella Documentation](https://docs.couchbase.com/cloud/index.html)
- [RAGBench Dataset](https://huggingface.co/datasets/rungalileo/ragbench) 