import logging
from datetime import timedelta
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from . import config

def setup_couchbase():
    """Initialize Couchbase connection and setup collections"""
    try:
        # Connect to Couchbase
        auth = PasswordAuthenticator(config.CB_USERNAME, config.CB_PASSWORD)
        options = ClusterOptions(auth)
        cluster = Cluster(config.CB_HOST, options)
        cluster.wait_until_ready(timedelta(seconds=5))
        logging.info("Successfully connected to Couchbase")
        
        # Setup collections
        def setup_collection(collection_name):
            bucket = cluster.bucket(config.CB_BUCKET_NAME)
            bucket_manager = bucket.collections()
            
            # Check if collection exists
            collections = bucket_manager.get_all_scopes()
            collection_exists = any(
                scope.name == config.SCOPE_NAME and collection_name in [col.name for col in scope.collections]
                for scope in collections
            )
            
            if not collection_exists:
                bucket_manager.create_collection(config.SCOPE_NAME, collection_name)
                logging.info(f"Collection '{collection_name}' created")
            else:
                logging.info(f"Collection '{collection_name}' already exists")
            
            # Create primary index
            cluster.query(
                f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{config.CB_BUCKET_NAME}`.`{config.SCOPE_NAME}`.`{collection_name}`"
            ).execute()
            logging.info(f"Primary index created for '{collection_name}'")
            
            # Clear collection
            cluster.query(
                f"DELETE FROM `{config.CB_BUCKET_NAME}`.`{config.SCOPE_NAME}`.`{collection_name}`"
            ).execute()
            logging.info(f"Collection '{collection_name}' cleared")
            
            return bucket.scope(config.SCOPE_NAME).collection(collection_name)
        
        # Setup main and cache collections
        setup_collection(config.COLLECTION_NAME)
        setup_collection(config.CACHE_COLLECTION)
        
        return cluster
        
    except Exception as e:
        logging.error(f"Failed to setup Couchbase: {str(e)}")
        raise
