import json
import os
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("../.env")

def lambda_handler(event, context):
    try:
        print("Starting Lambda test with Couchbase...")
        
        # Print environment variables for debugging
        print("Environment variables:")
        for key, value in os.environ.items():
            print(f"  {key}: {value}")
        
        # Connect to Couchbase
        auth = PasswordAuthenticator(
            os.environ.get("CB_USERNAME", "Administrator"),
            os.environ.get("CB_PASSWORD", "password")
        )
        options = ClusterOptions(auth)
        
        # Connect to the cluster
        cb_host = os.environ.get("CB_HOST", "couchbase://localhost")
        print(f"Connecting to Couchbase at {cb_host}...")
        cluster = Cluster(cb_host, options)
        print("Connected to Couchbase cluster")
        
        # Get the bucket
        bucket_name = os.environ.get("CB_BUCKET_NAME", "travel-sample")
        print(f"Opening bucket {bucket_name}...")
        bucket = cluster.bucket(bucket_name)
        print(f"Connected to {bucket_name} bucket")
        
        # Create a query to get all airlines
        query = """
        SELECT airline.* 
        FROM `travel-sample`.inventory.airline AS airline 
        LIMIT 10
        """
        
        print(f"Executing query: {query}")
        result = cluster.query(query)
        
        # Process the results
        airlines = []
        for row in result:
            airlines.append(row)
            
        print(f"Found {len(airlines)} airlines")
        
        # Return the results
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully connected to Couchbase and retrieved airlines',
                'airlines': airlines
            }, indent=2)
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error: {str(e)}',
                'traceback': traceback.format_exc()
            }, indent=2)
        }
