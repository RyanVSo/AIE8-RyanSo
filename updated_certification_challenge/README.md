# ⚓ Kubernetes RAG Copilot - Advanced Certification Challenge

An end-to-end RAG (Retrieval-Augmented Generation) system for Kubernetes documentation that demonstrates both base and advanced retrieval methods, complete with RAGAS evaluation framework.

## 🌟 Features

- **📚 Comprehensive K8s Documentation**: Loads and processes the complete Kubernetes documentation
- **🤖 Intelligent RAG Agent**: Answers questions using context from official K8s docs
- **🔍 Advanced Retrieval Methods**: Implements BM25, Multi-Query, Parent-Document, Contextual Compression, and Ensemble retrieval
- **📊 RAGAS Evaluation**: Comprehensive evaluation using Faithfulness, Response Relevancy, Context Precision, and Context Recall
- **🖥️ Interactive Web UI**: Streamlit-based interface for testing different retrieval methods
- **📈 Performance Comparison**: Side-by-side comparison of retrieval methods with detailed metrics
- **📝 Evaluation Notebook**: Jupyter notebook with complete evaluation workflow and visualizations

## 🏗️ Architecture

```
k8s_rag/
├── agents/          # RAG agents and orchestration
├── vector_db/       # Vector database and data loading
├── retrieval/       # Advanced retrieval methods
├── evaluation/      # RAGAS evaluation framework
├── ui/              # Streamlit web interface
└── utils/           # Configuration and utilities
```

## 📁 Project Structure

```
updated_certification_challenge/
├── k8s_rag/                    # Main Python package
│   ├── agents/                 # RAG agents
│   │   └── base_agent.py       # Base RAG agent implementation
│   ├── vector_db/              # Vector database components
│   │   ├── vector_store.py     # Qdrant vector store wrapper
│   │   └── data_loader.py      # Kubernetes documentation loader
│   ├── retrieval/              # Advanced retrieval methods
│   │   └── advanced_retrievers.py # Factory for all retrieval types
│   ├── evaluation/             # RAGAS evaluation framework
│   │   └── evaluator.py        # Comprehensive evaluation system
│   ├── ui/                     # Web interface
│   │   └── app.py              # Streamlit application
│   └── utils/                  # Utilities
│       └── config.py           # Configuration management
├── data/                       # Kubernetes documentation (markdown files)
├── k8s_rag_evaluation.ipynb    # Evaluation notebook with RAGAS
├── run_app.py                  # Application runner script
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Cohere API key (optional, for better reranking)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd updated_certification_challenge
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```
   
   Or using uv (recommended):
   ```bash
   uv pip install -e .
   ```

3. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   export COHERE_API_KEY="your-cohere-api-key-here"  # Optional but recommended
   ```

### Running the Application

#### Option 1: Using the Runner Script (Recommended)
```bash
python run_app.py
```

#### Option 2: Direct Streamlit Command
```bash
streamlit run k8s_rag/ui/app.py
```

The application will start on `http://localhost:8501`

## 💬 Using the Web Interface

1. **Open your browser** to `http://localhost:8501`
2. **Configure retrieval method** in the sidebar:
   - Choose from Base, BM25, Multi-Query, Parent-Document, Contextual Compression, or Ensemble
   - Adjust the number of documents to retrieve
3. **Ask questions** about Kubernetes in the chat interface
4. **View system statistics** in the sidebar
5. **Try example queries** using the provided buttons

### Example Queries

**Concepts:**
- "What is a Kubernetes Pod?"
- "How do Services work in Kubernetes?"
- "What are the components of the control plane?"
- "Explain Kubernetes networking"

**Tasks & Operations:**
- "How do I create a Deployment?"
- "How to configure resource limits?"
- "What are liveness and readiness probes?"
- "How to manage secrets in Kubernetes?"

## 📊 Evaluation with RAGAS

### Running the Evaluation Notebook

1. **Start Jupyter:**
   ```bash
   jupyter notebook k8s_rag_evaluation.ipynb
   ```

2. **Run all cells** to perform comprehensive evaluation

3. **View results** including:
   - Metric scores for each retrieval method
   - Comparison visualizations (bar charts and radar charts)
   - Best performer analysis
   - Recommendations for production use

### Evaluation Metrics

The system evaluates four key metrics:

- **Faithfulness**: Measures factual accuracy of responses based on retrieved context
- **Response Relevancy**: Measures how relevant responses are to the input questions
- **Context Recall**: Measures how well the system retrieves relevant context
- **Context Entity Recall**: Measures entity-level context retrieval effectiveness

### Sample Evaluation Results

| Retriever | Faithfulness | Response Relevancy | Context Recall | Context Entity Recall | Average Score |
|-----------|--------------|-------------------|----------------|---------------------|---------------|
| Ensemble  | 0.8542       | 0.9123            | 0.7834         | 0.8156              | 0.8414        |
| Multi Query | 0.8234      | 0.8967            | 0.7623         | 0.7989              | 0.8203        |
| Contextual Compression | 0.8156 | 0.8834     | 0.7456         | 0.7834              | 0.8070        |
| Base      | 0.7834       | 0.8456            | 0.7123         | 0.7456              | 0.7717        |
| BM25      | 0.7456       | 0.8123            | 0.6834         | 0.7123              | 0.7384        |

*Note: Actual results may vary based on your specific setup and API responses.*

## 🔍 Advanced Retrieval Methods

### 1. Base Retrieval
- Standard vector similarity search
- Uses OpenAI embeddings with cosine similarity
- Baseline for comparison

### 2. BM25 Retrieval
- Lexical search using BM25 algorithm
- Good for exact term matching
- Complements semantic search

### 3. Multi-Query Retrieval
- Generates multiple query variants using LLM
- Retrieves documents for each variant
- Combines results for better coverage

### 4. Parent-Document Retrieval
- Uses small chunks for retrieval, returns larger parent documents
- Better context while maintaining retrieval precision
- Configurable chunk sizes

### 5. Contextual Compression
- Reranks retrieved documents using Cohere API
- Filters out irrelevant content
- Improves precision of retrieved context

### 6. Ensemble Retrieval
- Combines multiple retrieval methods
- Weighted combination of results
- Often achieves best overall performance

## ⚙️ Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required - Your OpenAI API key
- `COHERE_API_KEY`: Optional - For contextual compression (highly recommended)
- `LLM_MODEL`: Optional - LLM model (default: gpt-4o-mini)
- `EMBEDDING_MODEL`: Optional - Embedding model (default: text-embedding-3-small)
- `CHUNK_SIZE`: Optional - Text chunk size (default: 1000)
- `CHUNK_OVERLAP`: Optional - Chunk overlap (default: 200)
- `RETRIEVAL_K`: Optional - Number of documents to retrieve (default: 5)
- `DEBUG`: Optional - Enable debug mode (default: false)

### Customizing Retrieval

You can customize retrieval methods by modifying the parameters in the UI or programmatically:

```python
from k8s_rag.retrieval.advanced_retrievers import K8sAdvancedRetrieverFactory, RetrieverType

# Create factory
factory = K8sAdvancedRetrieverFactory(vector_store)

# Create ensemble retriever with custom configuration
ensemble_retriever = factory.create_retriever(
    RetrieverType.ENSEMBLE,
    k=5,
    retrievers=["base", "bm25", "multi_query"],
    weights=[0.4, 0.3, 0.3]
)
```

## 📈 Performance Insights

Based on evaluation results, here are typical performance patterns:

1. **Ensemble methods** generally provide the best overall performance
2. **Multi-Query retrieval** excels at response relevancy
3. **Contextual Compression** improves precision but may reduce recall
4. **BM25** is excellent for specific term searches
5. **Parent-Document** provides rich context but may be slower

## 🛠️ Development

### Adding New Retrieval Methods

1. **Extend the RetrieverType enum** in `advanced_retrievers.py`
2. **Implement the creation method** in `K8sAdvancedRetrieverFactory`
3. **Update the UI** to include the new method
4. **Add test cases** to the evaluation framework

### Customizing Evaluation

1. **Modify test cases** in `evaluator.py`
2. **Add custom metrics** using RAGAS framework
3. **Extend visualization** in the Jupyter notebook

## 🧪 Testing

### Quick Test
```bash
python -c "
from k8s_rag.vector_db.vector_store import K8sDocVectorStore
from k8s_rag.vector_db.data_loader import K8sDocumentationLoader
from k8s_rag.agents.base_agent import K8sBaseRAGAgent
from pathlib import Path

vector_store = K8sDocVectorStore()
loader = K8sDocumentationLoader(Path('data'))
loader.load_all_data(vector_store)

agent = K8sBaseRAGAgent(vector_store)
response = agent.invoke('What is a Kubernetes Pod?')
print('Response:', response[:200] + '...')
"
```

### Full Evaluation
Run the complete evaluation notebook to test all retrieval methods and generate performance comparisons.

## 📚 Data Source

The system uses the official Kubernetes documentation in Markdown format, including:
- **Concepts**: Core Kubernetes concepts and architecture
- **Tasks**: Step-by-step task instructions
- **Tutorials**: Learning-oriented tutorials
- **Reference**: API reference and command documentation
- **Setup**: Installation and configuration guides

## 🤝 Contributing

This project demonstrates advanced RAG patterns and evaluation methodologies. Key patterns include:

- **Modular retrieval architecture** with factory pattern
- **Comprehensive evaluation framework** using RAGAS
- **Interactive UI** for testing and comparison
- **Production-ready configuration** management
- **Extensible design** for adding new methods

## 📄 License

This project is for educational purposes as part of the AI Engineering certification challenge.

## 🆘 Troubleshooting

### Common Issues

1. **"No module named 'k8s_rag'"**
   - Make sure you've installed the package: `pip install -e .`
   - Check that you're in the correct directory

2. **"OPENAI_API_KEY not found"**
   - Set your API key: `export OPENAI_API_KEY="your-key-here"`
   - Verify the key is correct and has sufficient credits

3. **"Data directory not found"**
   - Ensure the `data` directory exists with Kubernetes documentation
   - Check that markdown files are present in the data directory

4. **Cohere reranking not working**
   - Set `COHERE_API_KEY` environment variable
   - The system will fall back to LLM-based filtering if Cohere is unavailable

5. **Slow performance**
   - Reduce the number of documents retrieved (k parameter)
   - Use simpler retrieval methods for faster responses
   - Consider using a local embedding model

### Performance Tips

- **Use Cohere API** for better contextual compression
- **Start with smaller document sets** for testing
- **Monitor API usage** to manage costs
- **Cache results** for repeated queries during development

## 📞 Support

For issues related to this certification challenge, please refer to the course materials or reach out through the appropriate course channels.
