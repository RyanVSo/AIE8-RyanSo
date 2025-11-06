# ⚓ Kubernetes Copilot - Vercel Deployment

An intelligent Kubernetes management assistant that provides natural language interactions with your cluster data, cost analysis, and actionable recommendations. **Now optimized for Vercel deployment!**

## 🌟 Features

- **🗣️ Natural Language Queries**: Ask questions like "What are the costs of my Kubernetes deployments?" or "How many GPUs does this deployment use?"
- **🤖 Agentic RAG**: Multi-agent system with specialized tools for different K8s operations
- **💰 Cost Analysis**: Interactive cost tracking and optimization recommendations with charts
- **⚡ Resource Optimization**: Suggestions for improving resource utilization
- **📄 YAML Manifest Analysis**: Direct analysis of Kubernetes manifests
- **🖥️ Modern Web UI**: Responsive web interface optimized for all devices
- **☁️ Serverless Ready**: Optimized for Vercel's serverless platform
- **🚀 Fast Deployment**: One-click deployment to Vercel

## 🏗️ Architecture

- **FastAPI Backend**: RESTful API optimized for serverless deployment
- **LangGraph**: Multi-agent orchestration and workflow management
- **Vector Database**: Qdrant for storing and retrieving K8s manifests, cost data, and kubectl outputs
- **LLM**: OpenAI GPT-4o-mini for reasoning and response generation
- **Embedding Model**: OpenAI text-embedding-3-small for semantic search
- **Web Frontend**: Modern HTML/CSS/JavaScript with Chart.js visualizations
- **Vercel Platform**: Serverless functions with global CDN

## 📁 Project Structure

```
11_v2_deployment/
├── api/
│   └── main.py              # FastAPI backend (Vercel serverless function)
├── public/
│   ├── index.html           # Frontend HTML
│   ├── styles.css           # CSS styles
│   └── app.js               # JavaScript frontend logic
├── k8s_copilot/             # Core application logic
│   ├── agents/              # LangGraph agents and orchestration
│   ├── data/                # Mock K8s data (manifests, costs, kubectl outputs)
│   ├── tools/               # K8s-specific tools and functions
│   ├── vector_db/           # Vector database utilities
│   ├── retrieval/           # Advanced retrieval methods
│   ├── evaluation/          # RAGAS evaluation framework
│   └── utils/               # Utility functions
├── vercel.json              # Vercel configuration
├── requirements.txt         # Python dependencies (optimized)
├── DEPLOYMENT.md            # Detailed deployment guide
└── README.md                # This file
```

## 🚀 Quick Start - Vercel Deployment

### Prerequisites

- [Vercel Account](https://vercel.com) (free tier available)
- OpenAI API key
- GitHub repository (for automatic deployments)

### One-Click Deployment

1. **Fork this repository** to your GitHub account

2. **Deploy to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your forked repository
   - Add environment variables:
     - `OPENAI_API_KEY` (required)
     - `COHERE_API_KEY` (optional, for enhanced retrieval)
   - Click "Deploy"

3. **Access your deployed app** at `https://your-project-name.vercel.app`

### Local Development

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd 11_v2_deployment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   export COHERE_API_KEY="your-cohere-api-key-here"  # Optional, for enhanced retrieval
   ```

4. **Run locally:**
   ```bash
   # Start the FastAPI server
   cd api
   uvicorn main:app --reload --port 8000
   
   # Serve the frontend (in another terminal)
   cd public
   python -m http.server 3000
   ```

5. **Access locally** at `http://localhost:3000`

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

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

## 📈 Performance & Features

The Vercel deployment offers:
- **⚡ Fast Cold Starts**: Optimized dependencies for quick serverless function initialization
- **🌍 Global CDN**: Static assets served from Vercel's edge network
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices 
- **🔄 Real-time Updates**: Interactive chat interface with live responses
- **📊 Interactive Visualizations**: Cost charts and resource metrics using Chart.js
- **🛡️ Security**: Environment variables protect API keys, CORS configured properly

### Core Capabilities
- **Comprehensive K8s Knowledge**: Understands manifests, costs, and optimizations
- **Tool Integration**: Specialized tools for different K8s operations
- **Cost Awareness**: Specific dollar amounts and savings opportunities
- **Actionable Insights**: Concrete YAML optimizations and recommendations
- **Multi-Agent Architecture**: Flexible agent selection based on query type

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required - Your OpenAI API key
- `COHERE_API_KEY`: Optional - For enhanced retrieval methods (can improve response quality)

### Vercel Settings
- **Function Timeout**: 30 seconds (configured in vercel.json)
- **Memory Limit**: 1GB (Vercel default)
- **Runtime**: Python 3.9+

## 🛠️ Extending the System

### Adding New API Endpoints
1. Add new routes in `api/main.py`
2. Update frontend JavaScript in `public/app.js`
3. Test locally before deploying

### Adding New Features
1. **Backend**: Modify FastAPI routes and K8s components
2. **Frontend**: Update HTML/CSS/JS in the `public/` directory
3. **Deploy**: Push to GitHub for automatic Vercel deployment

### Custom Data Sources
1. Extend `K8sDataLoader` in `k8s_copilot/vector_db/data_loader.py`
2. Update vector store document types
3. Modify search filters as needed

## 🚀 Live Demo

Once deployed, your app will be available at:
- **Live URL**: `https://your-project-name.vercel.app`
- **API Health Check**: `https://your-project-name.vercel.app/api/health`
- **API Documentation**: `https://your-project-name.vercel.app/docs` (FastAPI auto-generated docs)

## 🤝 Contributing

This project demonstrates modern deployment patterns for AI applications:

- **Serverless Architecture**: Vercel Functions for scalable backend
- **JAMstack Frontend**: Static HTML/CSS/JS with API integration
- **Multi-Agent RAG**: LangGraph orchestration in serverless environment
- **Vector Database**: Qdrant integration for semantic search
- **Cost-Aware AI**: Financial optimization recommendations

## 📚 Additional Resources

- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Detailed deployment guide
- **[Vercel Documentation](https://vercel.com/docs)**: Platform documentation
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)**: Backend framework docs
- **[LangChain Documentation](https://python.langchain.com/)**: AI framework docs

## 📄 License

This project is for educational purposes as part of the AI Engineering course. Feel free to fork and adapt for your own Kubernetes management needs!
