# ⚓ Kubernetes RAG Copilot - Advanced Certification Challenge

An end-to-end RAG (Retrieval-Augmented Generation) system for Kubernetes documentation that demonstrates both base and advanced retrieval methods, complete with RAGAS evaluation framework.

## 🎥 Loom Video

[Watch the project demonstration video](https://www.loom.com/share/c31225b74a8346e4bc5e4c9aa8d21eb8?sid=dc8dd914-ae7f-4c46-a19c-8d9ce749956a)

## 📄 Written Document

[View the detailed project documentation (PDF)](https://github.com/RyanVSo/AIE8-RyanSo/blob/ryan-so-certification-challenge-submission/11_certification_challenge_v1/Certification%20Challenge.pdf)

## 🌟 Features

- **📚 Comprehensive K8s Documentation**: Loads and processes the complete Kubernetes documentation
- **🤖 Intelligent RAG Agent**: Answers questions using context from official K8s docs
- **🔍 Advanced Retrieval Methods**: Implements BM25, Multi-Query, and Ensemble retrieval
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
   - Choose from Base, BM25, Multi-Query, or Ensemble
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


### Full Evaluation
Run the complete evaluation notebook to test all retrieval methods and generate performance comparisons.

## 📚 Data Source

The system uses the official Kubernetes documentation in Markdown format, including:
- **Concepts**: Core Kubernetes concepts and architecture
- **Tasks**: Step-by-step task instructions
- **Tutorials**: Learning-oriented tutorials
- **Reference**: API reference and command documentation
- **Setup**: Installation and configuration guides

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


### Performance Tips

- **Use Cohere API** for better contextual compression
- **Start with smaller document sets** for testing
- **Monitor API usage** to manage costs
- **Cache results** for repeated queries during development

