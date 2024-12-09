import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

CB_HOST = os.getenv('CB_HOST', 'couchbase://localhost')
CB_USERNAME = os.getenv('CB_USERNAME', 'Administrator')
CB_PASSWORD = os.getenv('CB_PASSWORD', 'password')
CB_BUCKET_NAME = os.getenv('CB_BUCKET_NAME', 'vector-search-testing')
INDEX_NAME = os.getenv('INDEX_NAME', 'vector_search_crew')
SCOPE_NAME = os.getenv('SCOPE_NAME', 'shared')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'crew')
CACHE_COLLECTION = os.getenv('CACHE_COLLECTION', 'cache')
