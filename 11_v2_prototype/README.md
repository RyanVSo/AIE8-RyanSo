# ⚓ Kubernetes Copilot - Agentic RAG System

An intelligent Kubernetes management assistant that provides natural language interactions with your cluster data, cost analysis, and actionable recommendations.

## 🌟 Features

- **🗣️ Natural Language Queries**: Ask questions like "What are the costs of my Kubernetes deployments?" or "How many GPUs does this deployment use?"
- **🤖 Agentic RAG**: Multi-agent system with specialized tools for different K8s operations
- **💰 Cost Analysis**: Comprehensive cost tracking and optimization recommendations
- **⚡ Resource Optimization**: Suggestions for improving resource utilization
- **📄 YAML Manifest Analysis**: Direct analysis of Kubernetes manifests
- **🖥️ Interactive UI**: Web-based interface for easy interaction
- **📊 Evaluation Framework**: RAGAS-based evaluation for system performance
- **🔍 Advanced Retrieval**: Multiple retrieval strategies with performance comparison

## 🏗️ Architecture

- **LangGraph**: Multi-agent orchestration and workflow management
- **Vector Database**: Qdrant for storing and retrieving K8s manifests, cost data, and kubectl outputs
- **LLM**: OpenAI GPT-4o-mini for reasoning and response generation
- **Embedding Model**: OpenAI text-embedding-3-small for semantic search
- **Advanced Retrieval**: BM25, Multi-Query, Parent-Document, Contextual Compression, Ensemble methods
- **UI**: Streamlit-based web interface with retrieval method comparison
- **Evaluation**: RAGAS for comprehensive system evaluation

## 📁 Project Structure

```
k8s_copilot/
├── agents/              # LangGraph agents and orchestration
├── data/               # Mock K8s data (manifests, costs, kubectl outputs)
├── tools/              # K8s-specific tools and functions
├── vector_db/          # Vector database utilities
├── retrieval/          # Advanced retrieval methods and performance evaluation
├── ui/                 # Streamlit web interface
├── evaluation/         # RAGAS evaluation framework
└── utils/             # Utility functions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd 11_certification_challenge
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

4. **Generate mock Kubernetes data:**
   ```bash
   python k8s_copilot/data/generate_mock_data.py
   ```

5. **Run the comprehensive demo:**
   ```bash
   python demo.py
   ```

6. **Or run a quick test:**
   ```bash
   python demo.py --quick
   ```

7. **Launch the web UI:**
   ```bash
   # Standard UI
   python run_ui.py
   # OR
   streamlit run k8s_copilot/ui/app.py
   
   # Enhanced UI with retrieval comparison
   python run_retrieval_ui.py
   # OR
   streamlit run k8s_copilot/ui/retrieval_comparison_ui.py
   ```

8. **Run retrieval method comparison:**
   ```bash
   # Quick test
   python retrieval_comparison_demo.py --quick
   
   # Full comparison demo
   python retrieval_comparison_demo.py
   ```

## 💬 Example Queries

### Cost Analysis
- "What are the costs of my Kubernetes deployments?"
- "Which deployment is the most expensive?"
- "Show me the total cluster cost for the last 30 days"
- "What are the potential cost savings?"

### Resource Analysis
- "How many GPUs does the ml-training deployment use?"
- "Which deployments are using the most memory?"
- "What's the GPU utilization across my cluster?"
- "Analyze the nginx-deployment resources"

### Optimization
- "How can I improve resource utilization?"
- "Show me optimization opportunities"
- "Generate YAML optimization for the api-server deployment"
- "Suggest ways to reduce costs"

### General Queries
- "What deployments are running in my cluster?"
- "How many total pods do I have?"
- "What's the resource utilization of my cluster?"

## 🎯 Key Components

### 1. Vector Database (`vector_db/`)
- **K8sVectorStore**: Specialized vector store for Kubernetes data
- **K8sDataLoader**: Loads manifests, cost data, and kubectl outputs
- Semantic search across all K8s resources

### 2. Agents (`agents/`)
- **K8sCopilotAgent**: Full-featured agent with specialized tools
- **K8sRAGAgent**: Simple RAG agent for basic Q&A
- LangGraph orchestration for complex workflows

### 3. Tools (`tools/`)
- **K8sManifestAnalyzer**: Analyzes deployment resources and configurations
- **K8sCostAnalyzer**: Provides cost insights and optimization opportunities
- **K8sResourceOptimizer**: Suggests resource optimizations
- **K8sQueryTool**: General-purpose search across K8s data

### 4. UI (`ui/`)
- **Streamlit Web Interface**: Interactive chat interface
- **Cost Dashboard**: Visual cost analysis and trends
- **Resource Dashboard**: Cluster resource utilization metrics
- **Agent Comparison**: Compare copilot vs RAG agent responses

### 5. Advanced Retrieval (`retrieval/`)
- **K8sRetrieverFactory**: Factory for creating different retrieval methods
- **Performance Evaluator**: RAGAS-based comparison of retrieval strategies
- **Multiple Methods**: BM25, Multi-Query, Parent-Document, Contextual Compression, Ensemble
- **Configurable Parameters**: Fine-tune each retrieval method

### 6. Evaluation (`evaluation/`)
- **RAGAS Integration**: Comprehensive evaluation framework
- **Custom Metrics**: K8s-specific evaluation criteria
- **Agent Comparison**: Performance benchmarking

## 📊 Demo Features

The comprehensive demo (`demo.py`) showcases:

1. **Data Loading**: How K8s data is ingested and processed
2. **Basic Queries**: Simple cluster information retrieval
3. **Cost Analysis**: Detailed cost breakdowns and insights
4. **Resource Analysis**: GPU usage, memory consumption, etc.
5. **Optimization**: Actionable recommendations for improvements
6. **Agent Comparison**: Copilot vs RAG agent capabilities
7. **Vector Search**: Semantic search across K8s data
8. **Evaluation**: RAGAS-based performance assessment
9. **Advanced Retrieval**: Compare BM25, Multi-Query, Parent-Document, and Ensemble methods

## 🧪 Testing & Evaluation

### Quick Test
```bash
python demo.py --quick
```

### Advanced Retrieval Testing
```bash
# Quick retrieval test
python retrieval_comparison_demo.py --quick

# Full retrieval comparison (includes RAGAS evaluation)
python retrieval_comparison_demo.py
```

### Full Evaluation
```bash
python -c "
from k8s_copilot.evaluation.evaluator import K8sEvaluator
from k8s_copilot.vector_db.vector_store import K8sVectorStore
from k8s_copilot.vector_db.data_loader import K8sDataLoader
from pathlib import Path

# Initialize system
vector_store = K8sVectorStore()
data_loader = K8sDataLoader(Path('k8s_copilot/data'))
data_loader.load_all_data(vector_store)

# Run evaluation
evaluator = K8sEvaluator(vector_store)
results = evaluator.run_evaluation('copilot')
print(results)
"
```

### Retrieval Performance Comparison
```bash
python -c "
from k8s_copilot.retrieval import RetrievalPerformanceEvaluator
from k8s_copilot.vector_db.vector_store import K8sVectorStore
from k8s_copilot.vector_db.data_loader import K8sDataLoader
from pathlib import Path

# Initialize system
vector_store = K8sVectorStore()
data_loader = K8sDataLoader(Path('k8s_copilot/data'))
data_loader.load_all_data(vector_store)

# Compare retrieval methods
evaluator = RetrievalPerformanceEvaluator(vector_store)
results = evaluator.compare_all_retrievers()
report = evaluator.generate_comparison_report(results)
print(report)
"
```

## 🔧 Configuration

Environment variables:
- `OPENAI_API_KEY`: Required - Your OpenAI API key
- `COHERE_API_KEY`: Optional - For contextual compression/reranking (recommended)
- `EMBEDDING_MODEL`: Optional - Embedding model (default: text-embedding-3-small)
- `LLM_MODEL`: Optional - LLM model (default: gpt-4o-mini)
- `DEBUG`: Optional - Enable debug mode (default: false)

## 📈 Performance

The system demonstrates:
- **Comprehensive K8s Knowledge**: Understands manifests, costs, and optimizations
- **Tool Integration**: Specialized tools for different K8s operations
- **Cost Awareness**: Specific dollar amounts and savings opportunities
- **Actionable Insights**: Concrete YAML optimizations and recommendations
- **Multi-Agent Architecture**: Flexible agent selection based on query type

## 🛠️ Extending the System

### Adding New Tools
1. Create tool functions in `tools/k8s_tools.py`
2. Register tools in `create_k8s_tools()`
3. Update agent initialization

### Adding New Data Sources
1. Extend `K8sDataLoader` in `vector_db/data_loader.py`
2. Update vector store document types
3. Modify search filters as needed

### Custom Evaluation
1. Add test cases in `evaluation/evaluator.py`
2. Define custom metrics
3. Run comparative evaluations

## 🤝 Contributing

This is a demonstration project showcasing agentic RAG patterns for Kubernetes management. Key patterns demonstrated:

- **Multi-Agent Architecture**: LangGraph orchestration
- **Specialized Tools**: Domain-specific K8s operations
- **Vector RAG**: Semantic search across structured K8s data
- **Cost-Aware AI**: Financial optimization recommendations
- **Evaluation Framework**: RAGAS-based performance assessment

## 📄 License

This project is for educational purposes as part of the AI Engineering course.
