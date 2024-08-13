from langchain_anthropic import AnthropicEmbeddings
import os

# Ensure you have set the ANTHROPIC_API_KEY environment variable
api_key = os.getenv('ANTHROPIC_API_KEY')

def get_embedding_dimensions(api_key, model='claude-3-sonnet-20240229'):
    embeddings = AnthropicEmbeddings(
        anthropic_api_key=api_key,
        model=model
    )
    
    # Generate an embedding for a sample text
    sample_text = "This is a sample text to check embedding dimensions."
    embedding = embeddings.embed_query(sample_text)
    
    # Get the dimensions of the embedding
    dimensions = len(embedding)
    
    print(f"The embedding dimensions for model {model} are: {dimensions}")
    return dimensions

# Run the function
dimensions = get_embedding_dimensions(api_key)