# Bedrock Agent with Couchbase using a Lambda Function

This application demonstrates how to build a generative AI agent using Amazon Bedrock that leverages a Couchbase Vector Store via an AWS Lambda function. The agent can search for information within documents stored in Couchbase and format the findings.

This example uses a single agent with a tool implemented as a Lambda function. This provides a robust, serverless backend for the agent's capabilities.

## Architecture

The system uses:
- **Amazon Bedrock** for:
  - The core agent functionality.
  - A powerful foundation model (`anthropic.claude-3-sonnet-20240229-v1:0`) to reason and orchestrate tasks.
  - A text embedding model (`amazon.titan-embed-text-v2:0`) to create vector representations of documents.
- **AWS Lambda** to:
  - Host the business logic for the agent's tool.
  - The Lambda function performs a vector search in Couchbase and then uses a Bedrock model to format the results before returning them to the agent.
- **Couchbase Server** as:
  - A high-performance vector store.
  - The primary database for document storage and retrieval.
- **LangChain** (`langchain-couchbase` and `langchain-aws`) to:
  - Simplify integration between the application, Couchbase, and Bedrock.

## How It Works

The entire process is orchestrated by the `lambda-approach/Bedrock_Agents_Lambda.ipynb` notebook. When you run this notebook, it performs the following actions:

1.  **Environment Setup**: It reads configuration from a `.env` file, including your AWS and Couchbase credentials.
2.  **Couchbase Initialization**: It connects to your Couchbase cluster, creates the specified bucket, scope, and collection if they don't exist, and sets up a vector search index based on the `lambda-approach/aws_index.json` definition.
3.  **Data Ingestion**: It reads documents from `lambda-approach/documents.json`, generates vector embeddings for them using Amazon Titan, and stores them in the Couchbase collection.
4.  **IAM & Lambda Deployment**:
    *   It creates the necessary IAM role with permissions for Bedrock and Lambda.
    *   It packages the Lambda function code located in `lambda-approach/lambda_functions/` (including its Python dependencies) into a `.zip` file using the provided `Makefile`.
    *   It deploys this package to AWS Lambda, creating the function that the agent will use.
5.  **Agent Creation & Configuration**:
    *   It creates a new Bedrock Agent. If an agent with the same name already exists, it is deleted and recreated to ensure a clean state.
    *   It defines an "Action Group" for the agent. This action group contains a single tool that points to the Lambda function. The tool's capabilities (`searchAndFormatDocuments`) are defined directly in the script using the "Function Schema" method, which simplifies the process by avoiding the need for a separate OpenAPI schema.
6.  **Agent Preparation**: It "prepares" the agent, which compiles the agent's configuration and makes it ready for use. It also creates a `prod` alias to point to this prepared version.
7.  **Invocation**: Finally, the script invokes the agent with a sample prompt to demonstrate its ability to search for information and format the results.

## Prerequisites

- Python 3.9+
- The `make` utility installed on your system (for packaging the Lambda function).
- A running Couchbase Server instance (7.6+ required for vector search) with the Search service enabled.
- An AWS account with programmatic access (credentials configured).

## Setup and Execution

1.  **Navigate to the project directory**:
    ```bash
    cd awsbedrock-agents/lambda-approach
    ```

2.  **Create your environment file**:
    Copy `.env.example` from the parent directory to this directory as `.env` and fill in your configuration. **Crucially, you must add your `AWS_ACCOUNT_ID`**.
    ```bash
    cp ../.env.example .env
    ```
    Your `.env` file should look like this:
    ```
    # AWS Configuration
    AWS_REGION=us-east-1
    AWS_ACCESS_KEY_ID=your-access-key
    AWS_SECRET_ACCESS_KEY=your-secret-key
    AWS_ACCOUNT_ID=your-12-digit-account-id # <-- Add this line

    # Couchbase Configuration
    CB_HOST=couchbase://localhost
    CB_USERNAME=Administrator
    CB_PASSWORD=password
    CB_BUCKET_NAME=vector-search-exp
    SCOPE_NAME=bedrock_exp
    COLLECTION_NAME=docs_exp
    INDEX_NAME=vector_search_bedrock_exp
    ```
    *Note: The script uses experimental bucket/scope/collection names (`...-exp`) to avoid conflicts with other examples.*

3.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install dependencies**:
    A `requirements.txt` is provided for the deployment script.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Notebook**:
    Start Jupyter and open the `Bedrock_Agents_Lambda.ipynb` notebook.
    ```bash
    jupyter notebook
    ```
    Execute the cells in the notebook sequentially. The process will take several minutes as it involves creating and preparing AWS resources. You will see detailed output under each cell as it sets up Couchbase, deploys the Lambda, creates the agent, and finally invokes it.

## Customization

-   **Documents**: To use your own data, modify the `lambda-approach/documents.json` file. Each object in the `documents` array should have `text` and `metadata` fields.
-   **Vector Index**: The Couchbase vector search index can be customized by editing `lambda-approach/aws_index.json`.
-   **Agent Behavior**: The agent's core instructions and the models it uses (`AGENT_MODEL_ID`, `EMBEDDING_MODEL_ID`) can be modified in the `lambda-approach/Bedrock_Agents_Lambda.ipynb` notebook.

## Cleanup

The script is designed to clean up the agent and Lambda function on subsequent runs. However, the IAM role and the S3 bucket used for deployment are not automatically deleted. If you want to remove all resources, you will need to manually delete the following from the AWS Console:
- The IAM Role (`bedrock_agent_lambda_exp_role`).
- The S3 bucket created for the Lambda deployment (the name is dynamically generated and logged by the script, e.g., `lambda-deployment-your-account-id-timestamp`).
- The CloudWatch log groups for the Lambda function.
