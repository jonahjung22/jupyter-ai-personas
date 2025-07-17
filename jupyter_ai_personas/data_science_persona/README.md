# PocketFlow Context Persona

A streamlined, high-performance context retrieval specialist built with PocketFlow framework. This persona provides the same functionality as the original Agno-based system but with significant improvements in performance, maintainability, and simplicity.

## ✨ Key Features

- **⚡ High Performance**: 1 LLM call vs 3 in the original system
- **🏗️ Simplified Architecture**: 4 nodes vs 3 agents + team coordination
- **🔧 Easy Debugging**: Clear node pipeline with deterministic logic
- **📊 Smart Context Detection**: Automatically detects notebook analysis needs
- **🔍 Intelligent RAG Search**: Multiple search strategies for comprehensive results
- **🎯 Focused Responses**: Actionable recommendations based on context

## 📁 Architecture Overview

```
PocketFlow Context Retrieval Pipeline
├── InputAnalysisNode     → Parse request, detect context type
├── NotebookExtractionNode → Extract notebook content (conditional)
├── RAGSearchNode         → Search knowledge base intelligently
└── ResponseGeneratorNode → Generate comprehensive response
```

## 🚀 Quick Start

### Basic Usage

```python
from pocketflow_persona import PocketFlowContextPersona

# Create persona instance
persona = PocketFlowContextPersona()

# Simple query
result = persona.process_simple_query("How do I use pandas for data analysis?")

# Notebook analysis
result = persona.process_notebook_query(
    "notebook: /path/to/notebook.ipynb - help me with data cleaning"
)
```

### Flow-Level Usage

```python
from pocketflow_persona.flow import process_context_query

# Process any query
result = process_context_query("What is the best way to visualize data?")

# Process notebook query
result = process_context_query(
    "notebook: /Users/me/analysis.ipynb - optimize my machine learning model"
)
```

## 📊 Performance Comparison

| Metric | Original (Agno) | PocketFlow | Improvement |
|--------|-----------------|------------|-------------|
| LLM Calls | 3 per request | 1 per request | **3x faster** |
| Lines of Code | 284 lines | ~150 lines | **47% reduction** |
| Complexity | High (agent coordination) | Low (simple pipeline) | **Much simpler** |
| Debugging | Difficult | Easy | **Clear flow** |
| Memory Usage | High (3 agents) | Low (4 nodes) | **Lower overhead** |

## 🔧 Configuration

### Default Configuration

```python
from pocketflow_persona.config import PocketFlowConfig

config = PocketFlowConfig()

# Customize settings
config = PocketFlowConfig(
    rag={"search_results_limit": 15},
    features={"enable_detailed_logging": True}
)
```

### Environment Variables

```bash
# Optional: Override default notebook path
export POCKETFLOW_DEFAULT_NOTEBOOK="/path/to/default.ipynb"

# Optional: Enable debug logging
export POCKETFLOW_DEBUG=true
```

## 📝 Usage Examples

### 1. Simple Data Science Questions

```python
query = "What's the difference between pandas merge and join?"
result = process_context_query(query)
print(result["response"])
```

### 2. Notebook Analysis

```python
query = """
notebook: /Users/me/data_analysis.ipynb
I'm having trouble with my data cleaning process. 
Can you help me optimize it?
"""
result = process_context_query(query)
print(result["response"])
```

### 3. Library-Specific Help

```python
query = "How do I create interactive plots with plotly?"
result = process_context_query(query)
print(result["response"])
```

## 🔍 Node Details

### 1. InputAnalysisNode
**Purpose**: Parse user input and detect context type
- Detects notebook context indicators
- Extracts notebook paths
- Sets up shared state for pipeline

**Key Features**:
- Fast string parsing (no LLM needed)
- Automatic default path assignment
- Context flag setting

### 2. NotebookExtractionNode
**Purpose**: Extract notebook content when needed
- Only executes if notebook context detected
- Comprehensive content extraction
- Library and analysis stage detection

**Key Features**:
- Conditional execution
- Reuses existing notebook reader tool
- Structured content parsing

### 3. RAGSearchNode
**Purpose**: Search knowledge base intelligently
- Multiple search strategies
- Context-aware query enhancement
- Duplicate removal and result ranking

**Search Strategies**:
- Primary query search
- Library-specific searches
- Analysis stage searches
- Code example searches

### 4. ResponseGeneratorNode
**Purpose**: Generate comprehensive response
- LLM-powered response generation
- Context-aware formatting
- Fallback response handling

**Key Features**:
- Single LLM call for efficiency
- Rich context integration
- Actionable recommendations

## 🧪 Testing

### Run Tests

```bash
cd pocketflow_persona
python test_pocketflow.py
```

### Test Coverage

- ✅ Node-level unit tests
- ✅ Flow integration tests
- ✅ Configuration validation
- ✅ Error handling tests
- ✅ Performance benchmarks

## 🔧 Development

### Project Structure

```
pocketflow_persona/
├── __init__.py          # Package initialization
├── persona.py           # Main persona class
├── nodes.py             # PocketFlow nodes
├── flow.py              # Flow orchestration
├── utils.py             # Utility functions
├── config.py            # Configuration management
├── test_pocketflow.py   # Test suite
└── README.md            # Documentation
```

### Adding New Nodes

```python
from pocketflow import Node

class NewNode(Node):
    def exec(self, shared, **kwargs):
        # Your node logic here
        return {"success": True}
```

### Extending Search Strategies

```python
from pocketflow_persona.utils import RAGSearcher

class CustomRAGSearcher(RAGSearcher):
    def custom_search(self, query, **kwargs):
        # Custom search logic
        return {"success": True, "results": []}
```

## 📈 Performance Tips

1. **Enable Result Caching**: Set `cache_results=True` in config
2. **Limit Search Results**: Adjust `search_results_limit` based on needs
3. **Use Notebook Context**: More context = better results
4. **Optimize Queries**: Be specific about what you need

## 🐛 Troubleshooting

### Common Issues

**RAG System Not Available**
```python
# Check RAG initialization
from pocketflow_persona.utils import rag_searcher
print(rag_searcher.rag_tool is not None)
```

**Notebook Path Issues**
```python
# Verify notebook path format
query = "notebook: /full/path/to/notebook.ipynb - your question"
```

**Performance Issues**
```python
# Enable performance monitoring
config = PocketFlowConfig(features={"enable_detailed_logging": True})
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is part of the Jupyter AI Personas package and follows the same license terms.

## 🙏 Acknowledgments

- Built on the excellent [PocketFlow](https://github.com/the-pocket/PocketFlow) framework
- Integrates existing RAG system from data_science_persona
- Inspired by the original Agno-based context retrieval system

---

**🎯 Ready to use? Start with the Quick Start guide above!**