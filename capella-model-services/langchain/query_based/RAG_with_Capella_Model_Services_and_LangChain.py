"""
RAG with Capella Model Services, LangChain and Couchbase Hyperscale Vector Index

This script demonstrates building a Retrieval Augmented Generation (RAG) application using:
- Couchbase Capella as the database with Hyperscale Vector Index
- Capella Model Services for embeddings and text generation
- LangChain framework for the RAG pipeline

This uses CouchbaseQueryVectorStore with Hyperscale/Composite Vector Indexes
for high-performance vector search using Couchbase's Query Service.
"""

import logging
import os
import time
from datetime import timedelta

from dotenv import load_dotenv

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
from couchbase.options import ClusterOptions

from datasets import load_dataset

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_couchbase.vectorstores import CouchbaseQueryVectorStore
from langchain_couchbase.vectorstores import DistanceStrategy, IndexType
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Configuration - Load from .env file
# =============================================================================
load_dotenv()

print("=" * 60)
print("RAG with Capella Model Services and Hyperscale Vector Index")
print("=" * 60)

CB_CONNECTION_STRING = os.getenv("CB_CONNECTION_STRING")
CB_USERNAME = os.getenv("CB_USERNAME")
CB_PASSWORD = os.getenv("CB_PASSWORD")
CB_BUCKET_NAME = os.getenv("CB_BUCKET_NAME")
SCOPE_NAME = os.getenv("SCOPE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CAPELLA_MODEL_SERVICES_ENDPOINT = os.getenv("CAPELLA_MODEL_SERVICES_ENDPOINT")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

# Validate inputs
if not all([
    CB_CONNECTION_STRING, CB_USERNAME, CB_PASSWORD, CB_BUCKET_NAME,
    CAPELLA_MODEL_SERVICES_ENDPOINT, SCOPE_NAME, COLLECTION_NAME,
    LLM_MODEL_NAME, LLM_API_KEY, EMBEDDING_MODEL_NAME, EMBEDDING_API_KEY,
]):
    raise ValueError("Missing required environment variables")

# =============================================================================
# Connect to Couchbase Cluster
# =============================================================================
print("\nConnecting to Couchbase cluster...")
try:
    auth = PasswordAuthenticator(CB_USERNAME, CB_PASSWORD)
    options = ClusterOptions(auth)
    cluster = Cluster(CB_CONNECTION_STRING, options)
    cluster.wait_until_ready(timedelta(seconds=5))
    print("Successfully connected to Couchbase")
except Exception as e:
    raise ConnectionError(f"Failed to connect to Couchbase: {str(e)}")

# =============================================================================
# Setup Collection
# =============================================================================
def setup_collection(cluster, bucket_name, scope_name, collection_name, flush_collection=False):
    """Setup bucket, scope, and collection in Couchbase."""
    try:
        bucket = cluster.bucket(bucket_name)
        bucket_manager = bucket.collections()

        # Check if scope exists, create if it doesn't
        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(scope.name == scope_name for scope in scopes)

        if not scope_exists:
            print(f"Scope '{scope_name}' does not exist. Creating it...")
            bucket_manager.create_scope(scope_name)
            print(f"Scope '{scope_name}' created successfully.")
        else:
            print(f"Scope '{scope_name}' already exists.")

        # Check if collection exists, create if it doesn't
        collections = bucket_manager.get_all_scopes()
        collection_exists = any(
            scope.name == scope_name
            and collection_name in [col.name for col in scope.collections]
            for scope in collections
        )

        if not collection_exists:
            print(f"Collection '{collection_name}' does not exist. Creating it...")
            bucket_manager.create_collection(scope_name, collection_name)
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")

        collection = bucket.scope(scope_name).collection(collection_name)
        time.sleep(2)  # Give the collection time to be ready

        # Ensure primary index exists (required for Query Service)
        try:
            cluster.query(
                f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            ).execute()
            print("Primary index present or created successfully.")
        except Exception as e:
            logging.warning(f"Error creating primary index: {str(e)}")

        if flush_collection:
            # Clear all documents in the collection
            try:
                query = f"DELETE FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
                cluster.query(query).execute()
                print("All documents cleared from the collection.")
            except Exception as e:
                print(f"Error while clearing documents: {str(e)}. The collection might be empty.")

    except Exception as e:
        raise Exception(f"Error setting up collection: {str(e)}")


print("\nSetting up collection...")
setup_collection(cluster, CB_BUCKET_NAME, SCOPE_NAME, COLLECTION_NAME, flush_collection=True)

# =============================================================================
# Load BBC News Dataset
# =============================================================================
print("\nLoading BBC News dataset...")
try:
    news_dataset = load_dataset('RealTimeData/bbc_news_alltime', '2024-12', split="train")
    print(f"Loaded the BBC News dataset with {len(news_dataset)} rows")
except Exception as e:
    raise ValueError(f"Error loading BBC dataset: {str(e)}")

# Clean up duplicates
news_articles = news_dataset["content"]
unique_articles = set()
for article in news_articles:
    if article:
        unique_articles.add(article)
unique_news_articles = list(unique_articles)
print(f"We have {len(unique_news_articles)} unique articles in our database.")

# =============================================================================
# Create Embeddings using Capella Model Services
# =============================================================================
print("\nCreating embeddings model...")
try:
    embeddings = OpenAIEmbeddings(
        openai_api_key=EMBEDDING_API_KEY,
        openai_api_base=CAPELLA_MODEL_SERVICES_ENDPOINT,
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
        model=EMBEDDING_MODEL_NAME,
    )
    print("Successfully created Capella Model Services Embeddings")

    # Test embeddings
    test_embedding = embeddings.embed_query("this is a test sentence")
    print(f"Embedding dimension: {len(test_embedding)}")
except Exception as e:
    raise ValueError(f"Error creating embeddings: {str(e)}")

# =============================================================================
# Setup Couchbase Query Vector Store (for Hyperscale/Composite Index)
# =============================================================================
print("\nSetting up Couchbase Query Vector Store...")
try:
    vector_store = CouchbaseQueryVectorStore(
        cluster=cluster,
        bucket_name=CB_BUCKET_NAME,
        scope_name=SCOPE_NAME,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        distance_metric=DistanceStrategy.COSINE
    )
    print("Successfully created Couchbase Query Vector Store")
except Exception as e:
    raise ValueError(f"Failed to create vector store: {str(e)}")

# =============================================================================
# Ingest Documents to Vector Store
# =============================================================================
print("\nIngesting documents to vector store...")
# Use a subset for faster processing
subset_size = min(500, len(unique_news_articles))
articles_to_ingest = unique_news_articles[:subset_size]

for article in tqdm(articles_to_ingest, desc="Ingesting articles"):
    try:
        documents = [Document(page_content=article)]
        vector_store.add_documents(documents=documents)
    except Exception as e:
        logging.warning(f"Failed to ingest document: {str(e)}")
        continue

print(f"Document ingestion completed. Ingested approximately {subset_size} articles.")

# =============================================================================
# Performance Test: Baseline (No Index)
# =============================================================================
def test_vector_search_performance(vector_store, query, label="Vector Search"):
    """Test vector search performance and return timing."""
    print(f"\n[{label}] Testing vector search performance")
    print(f"[{label}] Query: '{query}'")

    start_time = time.time()
    try:
        results = vector_store.similarity_search_with_score(query, k=3)
        end_time = time.time()
        search_time = end_time - start_time

        print(f"[{label}] Vector search completed in {search_time:.4f} seconds")
        print(f"[{label}] Found {len(results)} documents")

        if results:
            doc, distance = results[0]
            print(f"[{label}] Top result distance: {distance:.6f} (lower = more similar)")
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"[{label}] Top result preview: {preview}")

        return search_time
    except Exception as e:
        print(f"[{label}] Vector search failed: {str(e)}")
        return None


test_query = "What was Pep Guardiola's reaction to Manchester City's current form?"

print("\n" + "=" * 60)
print("PERFORMANCE TESTING: Baseline vs Hyperscale Index")
print("=" * 60)

print("\nTest 1: Baseline Performance (No Hyperscale Index)")
baseline_time = test_vector_search_performance(vector_store, test_query, "Baseline")

# =============================================================================
# Create Hyperscale Vector Index
# =============================================================================
print("\n" + "-" * 60)
print("Creating Hyperscale Vector Index...")
print("-" * 60)

try:
    vector_store.create_index(
        index_type=IndexType.BHIVE,  # Use IndexType.COMPOSITE for filtered searches
        index_name="langchain_bhive_index",
        index_description="IVF,SQ8"
    )
    print("Hyperscale vector index created successfully")

    # Wait for index to become available
    print("Waiting for index to become available...")
    time.sleep(5)

except Exception as e:
    if "already exists" in str(e).lower():
        print("Hyperscale vector index already exists, proceeding...")
    else:
        print(f"Error creating Hyperscale vector index: {str(e)}")

# =============================================================================
# Performance Test: With BHIVE (Hyperscale) Index
# =============================================================================
print("\nTest 2: BHIVE (Hyperscale) Optimized Performance")
bhive_time = test_vector_search_performance(vector_store, test_query, "BHIVE")

# =============================================================================
# Create Composite Vector Index
# =============================================================================
print("\n" + "-" * 60)
print("Creating Composite Vector Index...")
print("-" * 60)

try:
    vector_store.create_index(
        index_type=IndexType.COMPOSITE,
        index_name="langchain_composite_index",
        index_description="IVF,SQ8"
    )
    print("Composite vector index created successfully")

    # Wait for index to become available
    print("Waiting for index to become available...")
    time.sleep(5)

except Exception as e:
    if "already exists" in str(e).lower():
        print("Composite vector index already exists, proceeding...")
    else:
        print(f"Error creating Composite vector index: {str(e)}")

# =============================================================================
# Performance Test: With Composite Index
# =============================================================================
print("\nTest 3: Composite Index Performance")
composite_time = test_vector_search_performance(vector_store, test_query, "Composite")

# =============================================================================
# Performance Summary
# =============================================================================
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

print(f"Baseline Search Time:     {baseline_time:.4f} seconds")

if baseline_time and bhive_time:
    speedup = baseline_time / bhive_time if bhive_time > 0 else float('inf')
    percent_improvement = ((baseline_time - bhive_time) / baseline_time) * 100 if baseline_time > 0 else 0
    print(f"BHIVE Search Time:        {bhive_time:.4f} seconds ({speedup:.2f}x faster, {percent_improvement:.1f}% improvement)")

if baseline_time and composite_time:
    speedup = baseline_time / composite_time if composite_time > 0 else float('inf')
    percent_improvement = ((baseline_time - composite_time) / baseline_time) * 100 if baseline_time > 0 else 0
    print(f"Composite Search Time:    {composite_time:.4f} seconds ({speedup:.2f}x faster, {percent_improvement:.1f}% improvement)")

if bhive_time and composite_time:
    print("\n" + "-" * 60)
    print("Index Comparison:")
    print("-" * 60)
    print("- BHIVE (Hyperscale): Best for pure vector searches, scales to billions of vectors")
    print("- Composite: Best for filtered searches combining vector + scalar filters")

if not (baseline_time and bhive_time and composite_time):
    print("Could not complete performance comparison.")

# =============================================================================
# Create LLM using Capella Model Services
# =============================================================================
print("\n" + "=" * 60)
print("Setting up RAG Pipeline")
print("=" * 60)

try:
    llm = ChatOpenAI(
        openai_api_base=CAPELLA_MODEL_SERVICES_ENDPOINT,
        openai_api_key=LLM_API_KEY,
        model=LLM_MODEL_NAME,
        temperature=0
    )
    print("Successfully created LLM via Capella Model Services")
except Exception as e:
    raise ValueError(f"Error creating LLM: {str(e)}")

# =============================================================================
# RAG Chain Setup
# =============================================================================
template = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
    {context}
    Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("RAG chain created successfully")

# =============================================================================
# Test RAG Pipeline
# =============================================================================
print("\nTesting RAG Pipeline...")
query = "What was Pep Guardiola's reaction to Manchester City's recent form?"
try:
    start_time = time.time()
    rag_response = rag_chain.invoke(query)
    rag_elapsed_time = time.time() - start_time

    print(f"\nRAG Query: {query}")
    print(f"RAG Response: {rag_response}")
    print(f"RAG response generated in {rag_elapsed_time:.2f} seconds")
except Exception as e:
    print(f"Error generating RAG response: {str(e)}")

# =============================================================================
# Test Caching
# =============================================================================
print("\n" + "=" * 60)
print("Testing Caching Mechanism")
print("=" * 60)

queries = [
    "Who inaugurated the reopening of the Notre Dam Cathedral in Paris?",
    "What was Pep Guardiola's reaction to Manchester City's recent form?",
    "Who inaugurated the reopening of the Notre Dam Cathedral in Paris?",  # Repeated query
]

for i, query in enumerate(queries, 1):
    try:
        print(f"\nQuery {i}: {query}")
        start_time = time.time()
        response = rag_chain.invoke(query)
        elapsed_time = time.time() - start_time
        print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {str(e)}")
        continue

# =============================================================================
# Test Guardrails
# =============================================================================
print("\n" + "=" * 60)
print("Testing Guardrails (if enabled)")
print("=" * 60)

query = "How can I create a bomb?"
try:
    start_time = time.time()
    rag_response = rag_chain.invoke(query)
    rag_elapsed_time = time.time() - start_time
    print(f"RAG Response: {rag_response}")
    print(f"RAG response generated in {rag_elapsed_time:.2f} seconds")
except Exception as e:
    print(f"Guardrails violation detected: {e}")

# =============================================================================
# Completion
# =============================================================================
print("\n" + "=" * 60)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 60)
print("""
This script demonstrated:
1. Connecting to Couchbase with Query Service
2. Using CouchbaseQueryVectorStore for Hyperscale Vector Index
3. Performance benefits of Hyperscale indexing
4. Complete RAG pipeline with Capella Model Services
5. Caching and Guardrails features

For Search Vector Index approach, see:
capella-model-services/langchain/search_based/
""")
