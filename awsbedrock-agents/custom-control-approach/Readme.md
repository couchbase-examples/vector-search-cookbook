# AWS Bedrock Agents - Custom Control Approach

This project demonstrates how to implement AWS Bedrock Agents using the **Custom Control** approach, where agents return control to your application instead of executing functions directly. This approach gives you full control over function execution while leveraging Bedrock's AI capabilities for natural language understanding and function calling.

## 🏗️ Architecture Overview

The project implements a multi-agent system with the following components:

- **Researcher Agent**: Searches through documents using semantic similarity
- **Writer Agent**: Formats and presents research findings
- **Couchbase Vector Store**: Stores and indexes documents for semantic search
- **AWS Bedrock**: Provides LLM capabilities and agent orchestration

## 🚀 What is Custom Control?

In the Custom Control approach, Bedrock agents don't execute functions directly. Instead, they:

1. Analyze user requests and determine which functions to call
2. Return control to your application with function parameters
3. Allow your application to execute the function
4. Accept the results and continue the conversation

This approach provides:
- ✅ **Full Control**: Execute functions in your own environment
- ✅ **Security**: No direct access to your systems from Bedrock
- ✅ **Flexibility**: Implement complex business logic
- ✅ **Debugging**: Full visibility into function execution

## 📋 Prerequisites

### AWS Requirements
- AWS Account with Bedrock access
- IAM permissions for:
  - Bedrock (agents, models)
  - IAM (role creation)
- Access to Bedrock foundation models (Nova Pro, Titan embeddings)

### Couchbase Requirements
- Couchbase Server 7.0+ with Search Service enabled
- Or Couchbase Cloud account

### Python Requirements
- Python 3.8+
- Required packages (see `requirements.txt`)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd awsbedrock-agents/custom-control-approach
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with the following variables:

   ```env
   # AWS Configuration (Required)
   AWS_ACCESS_KEY_ID=your_aws_access_key_id
   AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
   AWS_REGION=us-east-1
   AWS_ACCOUNT_ID=your_aws_account_id

   # Couchbase Configuration (Optional - defaults provided)
   CB_HOST=couchbase://localhost
   CB_USERNAME=Administrator
   CB_PASSWORD=password
   CB_BUCKET_NAME=vector-search-testing
   SCOPE_NAME=shared
   COLLECTION_NAME=bedrock
   INDEX_NAME=vector_search_bedrock
   ```

## 📁 File Structure

```
awsbedrock-agents/custom-control-approach/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── bedrock_agents_custom_control.py      # Main implementation
├── Bedrock_Agents_Custom_Control.ipynb   # Jupyter notebook version
├── documents.json                         # Sample documents for indexing
├── aws_index.json                        # Couchbase search index definition
└── .env                                   # Environment variables
```

## 🏃‍♂️ Running the Project

### Method 1: Python Script
```bash
python bedrock_agents_custom_control.py
```

### Method 2: Jupyter Notebook
```bash
jupyter notebook Bedrock_Agents_Custom_Control.ipynb
```

## 📊 What the Script Does

### 1. **Environment Setup**
- Loads environment variables
- Initializes AWS clients (Bedrock, IAM, etc.)
- Connects to Couchbase cluster

### 2. **Couchbase Configuration**
- Creates bucket, scope, and collection if they don't exist
- Sets up search indexes from `aws_index.json`
- Loads and indexes documents from `documents.json`

### 3. **Vector Store Setup**
- Creates embeddings using Amazon Titan Embed Text v2
- Configures Couchbase as the vector store backend
- Indexes sample documents for semantic search

### 4. **Agent Creation**
- Creates IAM roles for Bedrock agents
- Configures two agents with custom control functions:
  
  **Researcher Agent**:
  - Function: `search_documents(query, k)`
  - Capability: Semantic search through document collection
  
  **Writer Agent**:
  - Function: `format_content(content, style)`
  - Capability: Format and present research findings

### 5. **Testing**
- Tests the researcher agent with a sample query
- Tests the writer agent to format the results
- Demonstrates the custom control workflow

## 🔧 Key Functions

### Agent Management
- `get_or_create_agent_role()`: Creates IAM roles for Bedrock agents
- `create_agent()`: Creates or updates Bedrock agents with custom control
- `wait_for_agent_status()`: Waits for agent deployment

### Agent Invocation
- `invoke_agent()`: Invokes agents and handles custom control responses
- Processes function calls and executes them locally
- Returns formatted results back to the conversation

### Couchbase Operations
- `setup_collection()`: Creates Couchbase collections
- `setup_indexes()`: Configures search indexes
- Vector similarity search using LangChain integration

## 🎯 Example Usage

The script demonstrates a complete workflow:

1. **User Query**: "What is unique about the Cline AI assistant?"
2. **Researcher Agent**: Uses `search_documents` to find relevant information
3. **Writer Agent**: Uses `format_content` to present findings clearly
4. **Result**: Formatted, comprehensive answer

## 🔍 Custom Control Implementation Details

### Function Definition
```python
researcher_functions = [{
    "name": "search_documents",
    "description": "Search for relevant documents using semantic similarity",
    "parameters": {
        "query": {"type": "string", "description": "The search query", "required": True},
        "k": {"type": "integer", "description": "Number of results to return", "required": False}
    },
    "requireConfirmation": "DISABLED"
}]
```

### Agent Creation with Custom Control
```python
create_agent_action_group(
    agentId=agent_id,
    agentVersion="DRAFT",
    actionGroupExecutor={"customControl": "RETURN_CONTROL"},  # Key setting
    actionGroupName=f"{name}_actions",
    functionSchema={"functions": functions},
    description=f"Action group for {name} operations"
)
```

### Handling Function Calls
```python
if 'returnControl' in event:
    return_control = event['returnControl']
    invocation_inputs = return_control.get('invocationInputs', [])
    # Extract function name and parameters
    # Execute function locally
    # Return results to continue conversation
```

## 🛡️ Security Considerations

- **IAM Roles**: Agents use minimal IAM permissions
- **Local Execution**: Functions execute in your controlled environment
- **Credential Management**: AWS credentials stored securely in environment variables
- **Network Security**: Couchbase connections can be secured with TLS

## 🐛 Troubleshooting

### Common Issues

1. **AWS Credentials**:
   - Ensure AWS credentials are valid and have necessary permissions
   - Check region availability for Bedrock services

2. **Couchbase Connection**:
   - Verify Couchbase server is running and accessible
   - Ensure Search service is enabled in Couchbase

3. **Agent Creation Failures**:
   - Check IAM role creation permissions
   - Verify Bedrock model access

4. **Empty Responses**:
   - Check if documents are properly indexed
   - Verify search index configuration

### Debug Tips
- Enable detailed logging by setting log level to DEBUG
- Check AWS CloudWatch logs for Bedrock agent execution
- Use Couchbase web console to verify document indexing

## 📚 Learn More

- [AWS Bedrock Agents Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [Couchbase Vector Search](https://docs.couchbase.com/server/current/fts/fts-introduction.html)
- [LangChain Couchbase Integration](https://python.langchain.com/docs/integrations/vectorstores/couchbase)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this example.

## 📄 License

This project is provided as-is for educational and demonstration purposes.

