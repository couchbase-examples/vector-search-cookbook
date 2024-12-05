import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

from .config import *
from .couchbase_setup import setup_couchbase
from .vector_store import setup_vector_store, load_sample_data
from .tools import create_vector_search_tool
from .agents import setup_agents, create_tasks
