# Multi-Agent System with Couchbase Vector Store and Amazon Bedrock

This application demonstrates how to create a multi-agent system using Amazon Bedrock agents and Couchbase as a vector store. The system consists of three specialized agents:

1. **Embedder Agent**: Handles document storage and vector embeddings
2. **Researcher Agent**: Searches and retrieves information from documents
3. **Content Writer Agent**: Formats and presents research findings in a user-friendly way

## Prerequisites

- Python 3.9+
- Couchbase Server running locally or in the cloud
- AWS account with access to Amazon Bedrock
- AWS credentials with appropriate permissions

## Environment Setup

1. Copy `.env.example` to `.env` and fill in your configuration:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Couchbase Configuration
CB_HOST=couchbase://localhost
CB_USERNAME=Administrator
CB_PASSWORD=password
CB_BUCKET_NAME=vector-search-testing
SCOPE_NAME=shared
COLLECTION_NAME=bedrock
INDEX_NAME=vector_search_bedrock
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `couchbase_agents.ipynb` and run through the cells

## Architecture

The system uses:
- Amazon Bedrock for:
  - Agent infrastructure and orchestration
  - LLM capabilities (Claude 3 Sonnet)
  - Text embeddings (Titan)
- Couchbase for:
  - Vector storage
  - Similarity search
  - Document management
- LangChain for:
  - Vector store integration
  - Bedrock integration

All agents use the Return of Control (ROC) pattern to interact directly with the system, eliminating the need for Lambda functions.

## Agent Roles

### Embedder Agent
- Handles document storage in the vector store
- Validates document content
- Manages metadata
- Ensures document quality

### Researcher Agent
- Performs semantic similarity searches
- Retrieves relevant documents
- Provides accurate citations
- Maintains search quality

### Content Writer Agent
- Formats research findings
- Creates clear summaries
- Organizes information logically
- Makes complex information accessible

## Example Usage

### Adding Documents
Use the Embedder Agent to add new documents:

```python
embedder_response = invoke_agent(
    embedder_id,
    embedder_alias,
    'Add this document: "Your document text here"'
)
```

### Searching Documents
Use the Researcher Agent to find information:

```python
researcher_response = invoke_agent(
    researcher_id,
    researcher_alias,
    'Your search query here'
)
```

### Formatting Results
Use the Content Writer Agent to present findings:

```python
writer_response = invoke_agent(
    writer_id,
    writer_alias,
    f'Format this research finding: {researcher_response}'
)
```

## Important Notes

1. Ensure your Couchbase cluster has the vector search index configured (see aws_index.json)
2. The Embedder Agent requires confirmation before adding documents
3. The Researcher Agent provides citations from source documents
4. The Content Writer Agent focuses on presentation, not document storage
5. All vector operations use semantic similarity with dot product

## Vector Search Configuration

The system uses a vector search index optimized for:
- 1024-dimensional vectors (Titan embeddings)
- Dot product similarity
- Recall optimization
- Text field storage

See `aws_index.json` for the complete index configuration.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
