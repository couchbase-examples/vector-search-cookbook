# Couchbase Capella AI Services Auto-Vectorization with LangChain

This guide is a comprehensive tutorial demonstrating how to use Couchbase Capella's AI Services auto-vectorization feature to automatically convert your data into vector embeddings and perform semantic search using LangChain.

## 📋 Overview

The main tutorial is contained in the Jupyter notebook `autovec_langchain.ipynb`, which walks you through:

1. **Couchbase Capella Setup** - Creating account, cluster, and access controls
2. **Data Upload & Processing** - Using sample data
3. **Model Deployment** - Deploying embedding models for vectorization
4. **Auto-Vectorization Workflow** - Setting up automated embedding generation
5. **LangChain Integration** - Building semantic search applications with vector similarity

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A Couchbase Capella account
- Basic understanding of vector databases and embeddings

### Installation Steps

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd vector-search-cookbook/autovec-tutorial
   ```

2. **Install Python dependencies**
   ```bash
   pip install jupyter
   pip install couchbase
   pip install langchain-couchbase
   pip install langchain-nvidia-ai-endpoints
   ```

3. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

4. **Open the tutorial notebook**
   - Navigate to `autovec_langchain.ipynb` in the Jupyter interface
   - Follow the step-by-step instructions in the notebook
```

**Note**: This tutorial is designed for educational purposes. For production deployments, ensure proper security configurations and SSL/TLS verification.
