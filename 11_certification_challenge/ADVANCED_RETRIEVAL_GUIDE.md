# 🔍 Advanced Retrieval Methods Guide

This guide explains how to use and compare different retrieval methods in the Kubernetes Copilot system.

## 🚀 Quick Start

### 1. Install Additional Dependencies
```bash
pip install rank-bm25 langchain-text-splitters cohere
```

### 2. Set Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export COHERE_API_KEY="your-cohere-api-key"  # Optional, for contextual compression
```

### 3. Run Quick Test
```bash
python retrieval_comparison_demo.py --quick
```

### 4. Launch Enhanced UI
```bash
python run_retrieval_ui.py
```

## 📋 Available Retrieval Methods

### 1. **Base Vector Search** (Default)
- **Description**: Standard semantic similarity search using embeddings
- **Best For**: General queries, semantic understanding
- **Speed**: ⚡ Fast
- **Configuration**: None required

### 2. **BM25 (Lexical Search)**
- **Description**: Keyword-based search using term frequency
- **Best For**: Exact term matching, keyword-heavy queries
- **Speed**: ⚡ Fast
- **Configuration**:
  ```python
  config = {
      "bm25_params": {
          "k1": 1.5,  # Term frequency saturation
          "b": 0.75   # Field length normalization
      }
  }
  ```

### 3. **Multi-Query Retrieval**
- **Description**: Generates multiple query variants for better coverage
- **Best For**: Complex or ambiguous queries
- **Speed**: 🐌 Slower (multiple LLM calls)
- **Configuration**:
  ```python
  config = {
      "num_queries": 3  # Number of query variants
  }
  ```

### 4. **Parent-Document Retrieval**
- **Description**: Small-to-big strategy: search small chunks, return large documents
- **Best For**: Queries needing broader context
- **Speed**: 🐌 Slower (document processing)
- **Configuration**:
  ```python
  config = {
      "parent_chunk_size": 2000,
      "child_chunk_size": 400,
      "chunk_overlap": 50
  }
  ```

### 5. **Contextual Compression**
- **Description**: Post-processes retrieved documents using reranking
- **Best For**: Improving precision, reducing noise
- **Speed**: 🐌 Slower (reranking step)
- **Requirements**: Cohere API key for optimal performance
- **Configuration**:
  ```python
  config = {
      "top_k": 5  # Documents after compression
  }
  ```

### 6. **Ensemble Retrieval**
- **Description**: Combines multiple retrieval methods using rank fusion
- **Best For**: Balanced performance across query types
- **Speed**: 🐌 Slowest (multiple methods)
- **Configuration**:
  ```python
  config = {
      "retrievers": ["base", "bm25", "multi_query"],
      "weights": [0.4, 0.3, 0.3]
  }
  ```

## 💻 Usage Examples

### Programmatic Usage

```python
from k8s_copilot.retrieval import K8sRetrieverFactory, RetrieverType
from k8s_copilot.agents.enhanced_agents import EnhancedK8sRAGAgent
from k8s_copilot.vector_db.vector_store import K8sVectorStore

# Initialize system
vector_store = K8sVectorStore()
# ... load data ...

# Create agent with BM25 retriever
agent = EnhancedK8sRAGAgent(
    vector_store=vector_store,
    retriever_type=RetrieverType.BM25,
    retriever_config={"bm25_params": {"k1": 1.5, "b": 0.75}}
)

# Ask a question
response = agent.invoke("What are the costs of my deployments?")
print(response)

# Switch to ensemble retriever
agent.switch_retriever(
    RetrieverType.ENSEMBLE,
    {"retrievers": ["base", "bm25"], "weights": [0.6, 0.4]}
)

response = agent.invoke("How many GPUs does my cluster use?")
print(response)
```

### Performance Comparison

```python
from k8s_copilot.retrieval import RetrievalPerformanceEvaluator

# Initialize evaluator
evaluator = RetrievalPerformanceEvaluator(vector_store)

# Compare all methods
results = evaluator.compare_all_retrievers(k=5)

# Generate report
report = evaluator.generate_comparison_report(results)
print(report)

# Save detailed metrics
evaluator.save_metrics_csv(results, Path("retrieval_metrics.csv"))
```

## 📊 Performance Metrics

The evaluation framework measures:

- **Context Precision**: How relevant are the retrieved documents?
- **Context Recall**: How well do retrieved documents cover the reference information?
- **Faithfulness**: Are responses grounded in the retrieved context?
- **Response Relevancy**: How relevant are responses to the queries?
- **Factual Correctness**: Are the factual claims in responses accurate?
- **Retrieval Time**: How fast is each method?

## 🎯 Choosing the Right Method

### Query Type Recommendations

| Query Type | Recommended Method | Reason |
|------------|-------------------|---------|
| **Exact term search** | BM25 | Excellent keyword matching |
| **Semantic understanding** | Base Vector Search | Best semantic similarity |
| **Complex/ambiguous queries** | Multi-Query | Better coverage through variants |
| **Need broader context** | Parent-Document | Returns larger context windows |
| **High precision required** | Contextual Compression | Filters irrelevant content |
| **Mixed query types** | Ensemble | Balanced performance |

### Performance vs. Speed Trade-offs

| Method | Speed | Precision | Recall | Best Use Case |
|--------|-------|-----------|---------|---------------|
| Base | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | General purpose |
| BM25 | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐ | Keyword queries |
| Multi-Query | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Complex queries |
| Parent-Document | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Context-heavy |
| Contextual Compression | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | High precision |
| Ensemble | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Production use |

## 🛠️ Advanced Configuration

### Custom Retriever Creation

```python
from k8s_copilot.retrieval import K8sRetrieverFactory

factory = K8sRetrieverFactory(vector_store)

# Create custom BM25 with specific parameters
bm25_retriever = factory.create_retriever(
    RetrieverType.BM25,
    k=10,
    bm25_params={"k1": 2.0, "b": 0.5}
)

# Create ensemble with custom weights
ensemble_retriever = factory.create_retriever(
    RetrieverType.ENSEMBLE,
    k=5,
    retrievers=["base", "bm25", "multi_query"],
    weights=[0.5, 0.3, 0.2]
)
```

### Batch Evaluation

```python
# Define custom test queries
custom_queries = [
    "Show me all GPU deployments",
    "Which services cost the most?",
    "Optimize my nginx configuration"
]

# Run targeted evaluation
evaluator = RetrievalPerformanceEvaluator(vector_store)
evaluator.test_queries = custom_queries

# Compare specific methods
results = []
for method in [RetrieverType.BASE, RetrieverType.BM25, RetrieverType.ENSEMBLE]:
    metrics = evaluator.evaluate_retriever(method, k=5)
    results.append(metrics)

# Generate comparison
report = evaluator.generate_comparison_report(results)
```

## 🚨 Troubleshooting

### Common Issues

1. **BM25 Error**: "No documents available for BM25 indexing"
   - **Solution**: Ensure vector store is loaded with documents before creating BM25 retriever

2. **Cohere API Error**: "API key required"
   - **Solution**: Set `COHERE_API_KEY` environment variable or use LLM-based compression fallback

3. **Slow Performance**: Multi-query or ensemble methods taking too long
   - **Solution**: Reduce `num_queries` or use fewer methods in ensemble

4. **Memory Issues**: Parent-document retrieval using too much memory
   - **Solution**: Reduce `parent_chunk_size` and `child_chunk_size`

### Performance Optimization

1. **Use caching**: Retrievers are automatically cached by configuration
2. **Tune parameters**: Start with defaults, then optimize based on your queries
3. **Profile methods**: Use the performance evaluator to identify bottlenecks
4. **Consider trade-offs**: Balance speed vs. accuracy based on your use case

## 📈 Best Practices

1. **Start Simple**: Begin with base vector search, then experiment with advanced methods
2. **Measure Performance**: Always evaluate methods on your specific queries
3. **Consider Context**: Choose methods based on your data characteristics and query types
4. **Optimize Iteratively**: Fine-tune parameters based on evaluation results
5. **Monitor Production**: Track retrieval performance in production environments

## 🔗 Related Resources

- [LangChain Retrievers Documentation](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)
- [BM25 Algorithm Explanation](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Cohere Rerank API](https://docs.cohere.com/reference/rerank)
