import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from langchain_couchbase.vectorstores import CouchbaseVectorStore

# Load environment variables
load_dotenv()

# Initialize Couchbase connection
auth = PasswordAuthenticator(
    os.getenv("CB_USERNAME"),
    os.getenv("CB_PASSWORD")
)
cluster = Cluster(f'couchbase://{os.getenv("CB_HOST")}', ClusterOptions(auth))
bucket = cluster.bucket(os.getenv("CB_BUCKET"))
collection = bucket.scope(os.getenv("CB_SCOPE")).collection(os.getenv("CB_COLLECTION"))

# Initialize embeddings
try:
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
        openai_api_type="azure",
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
except Exception as e:
    print(f"Failed to initialize Azure embeddings: {e}")
    print("Falling back to OpenAI embeddings")
    embeddings = OpenAIEmbeddings()

# Initialize Couchbase as vector store
vector_store = CouchbaseVectorStore(
    collection=collection,
    embedding=embeddings,
    index_name="azure_index"
)

# Initialize the language model
try:
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_GPT_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
        openai_api_type="azure",
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
except Exception as e:
    print(f"Failed to initialize Azure ChatGPT: {e}")
    print("Falling back to OpenAI ChatGPT")
    llm = ChatOpenAI()

# Create a conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create a prompt template
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# Create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT, "memory": memory}
)

def semantic_search(query: str, k: int = 4):
    """
    Perform a semantic search and return the top k results.
    
    :param query: The search query string
    :param k: The number of top results to return (default is 4)
    :return: List of top k semantically similar documents
    """
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def rag_search(query: str):
    """
    Perform a RAG (Retrieval-Augmented Generation) search and return the result.
    
    :param query: The search query string
    :return: The generated answer and source documents
    """
    with get_openai_callback() as cb:
        result = qa_chain({"query": query})
    
    answer = result["result"]
    source_documents = result["source_documents"]
    
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
    
    return answer, source_documents

if __name__ == "__main__":
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        print("\nSemantic Search Results:")
        semantic_results = semantic_search(query)
        for i, result in enumerate(semantic_results, 1):
            print(f"{i}. {result}\n")
        
        print("\nRAG Search Result:")
        rag_answer, sources = rag_search(query)
        print(f"Answer: {rag_answer}\n")
        print("Sources:")
        for i, doc in enumerate(sources, 1):
            print(f"{i}. {doc.page_content}\n")
        
        print("-" * 50)