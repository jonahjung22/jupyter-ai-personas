# 🧠 Advanced Data Science Agent

An intelligent PocketFlow-powered data science persona that uses sophisticated reasoning to provide targeted, context-aware analysis and recommendations. This persona combines AI decision-making with deep notebook understanding to deliver actionable insights for data science projects.

## ✨ Key Features

- **🤖 Intelligent Decision-Making**: Uses LLM reasoning to choose optimal analysis approaches
- **🔄 Iterative Analysis**: Can perform multiple analysis rounds based on findings
- **📊 Context Integration**: Combines repo context, notebook content, and conversation history
- **🎯 Targeted Responses**: Provides focused analysis based on specific user needs
- **📓 Smart Notebook Reading**: Automatically detects and analyzes notebook files
- **⚡ Adaptive Workflows**: Routes between focused analysis and comprehensive reviews
- **🛡️ Robust Error Handling**: Graceful fallbacks with detailed logging

## 🏗️ Architecture Overview

### **Agent-Based Architecture**
```
                    DataScienceAgent (Orchestrator)
                            |
                    ┌───────┴───────┐
                    │   Load Context │
                    │ • repo_context.md
                    │ • notebook files
                    └───────┬───────┘
                            |
                    ┌───────▼───────┐
                    │ DecideAction  │ ◄──┐
                    │ (Decision Node)│    │
                    └───┬───────┬───┘    │
                        │       │        │
            ┌───────────▼───┐   │        │
            │ DataAnalysis   │   │        │
            │ (Focused)      │───┘        │
            └───────────────┘            │
                        │                │
            ┌───────────▼───────────────┐│
            │ CompleteAnalysis          ││
            │ (Comprehensive)           ││
            └───────────┬───────────────┘│
                        │                │
                ┌───────▼────────┐       │
                │ Final Response │       │
                └────────────────┘       │
                                        │
                    Loop Back ──────────┘
```

### **Intelligent Decision Flow**
1. **Context Loading**: Reads `repo_context.md` and notebook files
2. **Decision Making**: LLM analyzes context and chooses action via YAML
3. **Targeted Execution**: Routes to appropriate analysis node
4. **Iterative Refinement**: Can loop back for additional analysis
5. **Comprehensive Response**: Delivers actionable insights and code

## 🚀 Quick Start

### **Installation & Setup**

```python
# The persona is automatically available in Jupyter AI
# Just ensure your environment has the required dependencies:
pip install agno boto3 pyyaml
```

### **Basic Usage**

```python
# In Jupyter AI chat:
@DataSciencePersona analyze my sales data

# With specific notebook:
@DataSciencePersona notebook: path/to/analysis.ipynb help me improve my model

# For code generation:
@DataSciencePersona generate code for feature engineering on my dataset
```

### **Direct API Usage**

```python
from jupyter_ai_personas.data_science_persona import DataSciencePersona

# Create persona
persona = DataSciencePersona()

# Process analysis request
result = await persona.process_message(message)
```

## 🧠 Agent Components

### **1. DecideAction Node**
**Purpose**: AI-powered decision making

**Capabilities**:
- Analyzes user intent and available context
- Uses YAML-structured LLM responses for reliable parsing
- Routes to appropriate analysis approaches
- Tracks reasoning and action history

**Decision Types**:
- `analyze_data` → Focused data analysis
- `generate_code` → Code generation and examples  
- `explain_concept` → Conceptual explanations
- `find_issues` → Problem identification and debugging
- `create_visualization` → Visualization recommendations
- `optimize_model` → Model improvement suggestions
- `debug_code` → Code debugging assistance
- `complete_analysis` → Comprehensive analysis

### **2. DataAnalysis Node**
**Purpose**: Targeted, focused analysis

**Features**:
- Performs specific analysis based on agent decisions
- Provides targeted recommendations
- Focuses on user's immediate questions
- Can route back to decision node for iterative analysis

**Output Format**:
- **📊 Data Analysis**: Current state and quality assessment
- **🔍 Specific Findings**: Direct answers to user questions
- **💡 Recommendations**: Actionable next steps

### **3. CompleteAnalysis Node**
**Purpose**: Comprehensive data science analysis

**Features**:
- Full analysis combining all available context
- Detailed code implementations
- Strategic recommendations and roadmaps
- Testing and validation approaches

**Output Format**:
- **📊 Current State Analysis**: Thorough assessment
- **🎯 Targeted Recommendations**: Priority-ordered suggestions
- **💻 Implementation Code**: Ready-to-use code snippets
- **🔄 Next Steps Roadmap**: Strategic development plan
- **🧪 Testing & Validation**: Quality assurance recommendations

### **4. Context Loading System**
**Purpose**: Intelligent file reading and context preparation

**Features**:
- **Automatic notebook detection**: Finds `.ipynb` files intelligently
- **Explicit path support**: Handles `notebook: path/to/file.ipynb` syntax
- **Recursive search**: Searches subdirectories when needed
- **Repository context**: Reads `repo_context.md` for project understanding
- **Conversation history**: Integrates chat history for context

## 📊 Usage Examples

### **1. Data Analysis Request**
```python
# User message:
"@DataSciencePersona My sales model has poor accuracy. What's wrong?"

# Agent process:
1. DecideAction: Analyzes context → action: find_issues
2. DataAnalysis: Examines notebook for model issues
3. Response: Specific problems and solutions
```

### **2. Code Generation Request**
```python
# User message:
"@DataSciencePersona generate feature engineering code for my dataset"

# Agent process:
1. DecideAction: Analyzes intent → action: generate_code
2. CompleteAnalysis: Creates comprehensive implementation
3. Response: Ready-to-use code with explanations
```

### **3. Comprehensive Analysis**
```python
# User message:
"@DataSciencePersona notebook: analysis.ipynb review my entire approach"

# Agent process:
1. Load Context: Reads analysis.ipynb + repo_context.md
2. DecideAction: Comprehensive scope → action: complete_analysis
3. CompleteAnalysis: Full review with strategic recommendations
4. Response: Complete analysis with roadmap
```

## 🔧 Configuration

### **AWS Bedrock Setup**
```python
# Configure in Jupyter AI settings
{
  "model_provider": "bedrock",
  "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
  "api_keys": {
    "AWS_ACCESS_KEY_ID": "your-key",
    "AWS_SECRET_ACCESS_KEY": "your-secret"
  }
}
```

### **Repository Context**
Create a `repo_context.md` file in your working directory:

```markdown
# Project Context
## Overview
Sales prediction project using linear regression

## Goals
- Predict monthly sales revenue
- Identify key factors affecting sales
- Optimize marketing spend allocation

## Current Status
- Basic model implemented
- Accuracy: 65% (needs improvement)
- Next: Feature engineering and model selection
```

### **Notebook Path Formats**
```python
# Supported formats:
"notebook: /absolute/path/to/file.ipynb"
"notebook: relative/path/to/file.ipynb"  
"/direct/path/to/notebook.ipynb"  # Direct path in message
# Auto-detection: Searches current directory and subdirectories
```

## 🧪 Advanced Features

### **Iterative Analysis**
The agent can perform multiple analysis rounds:

```
User Request → DecideAction → DataAnalysis → DecideAction → CompleteAnalysis → Final Response
```

### **YAML Decision Parsing**
Robust parsing with multiple fallback strategies:
- Primary: YAML parsing of LLM response
- Fallback 1: Text extraction for common patterns
- Fallback 2: Default comprehensive analysis

### **Error Recovery**
- **Model unavailable**: Falls back to structured templates
- **File not found**: Provides guidance and continues with available context
- **YAML parsing errors**: Uses text extraction fallbacks
- **Configuration issues**: Detailed error messages with troubleshooting

## 📈 Performance & Monitoring

### **Logging Levels**
```python
# Debug logging shows:
- Notebook path detection process
- Decision reasoning from LLM  
- Action routing decisions
- Context loading details
- YAML parsing attempts

# Info logging shows:
- Agent initialization status
- Processing summary
- Success/failure status
- Action history
```

### **Processing Summary**
Every response includes:
```
**Agent Processing Summary:**
- Repo Context: ✅ Loaded / ❌ Not found
- Notebook Analysis: ✅ Loaded / ❌ Not found  
- AI Analysis: ✅ Generated / ❌ Failed
- Actions Taken: 2
- Agent Actions: analyze_data → complete_analysis
- Notebook: `/path/to/notebook.ipynb`
```

## 🐛 Troubleshooting

### **Common Issues**

**"No notebook files found"**
```python
# Solutions:
1. Use explicit path: "notebook: /full/path/to/file.ipynb"
2. Check working directory
3. Ensure .ipynb file exists
4. Check file permissions
```

**"YAML parsing error"**
```python
# The agent automatically handles this with fallbacks
# Check logs for details, but it should continue working
```

**"AI model not available"**  
```python
# Check AWS Bedrock configuration
# Agent will work in fallback mode with templates
```

**"Configuration error"**
```python
# Verify Jupyter AI model configuration
# Check AWS credentials and permissions
```

## 🔬 Technical Details

### **Dependencies**
- `agno`: AWS Bedrock integration and message handling
- `pyyaml`: YAML parsing for decision responses  
- `boto3`: AWS SDK for Bedrock client
- `pathlib`: File path handling
- `jupyter_ai`: Base persona framework

### **File Structure**
```
data_science_persona/
├── __init__.py           # Package exports
├── persona.py            # Jupyter AI integration layer
├── agent.py              # Core agent implementation
├── pocketflow.py         # PocketFlow base classes
├── file_reader_tool.py   # Notebook reading utilities
└── README.md             # This documentation
```

### **System Requirements**
- Python 3.9+
- Jupyter AI 3.0+
- AWS Bedrock access (or compatible model provider)
- Sufficient memory for notebook content processing

## 🚀 Performance Characteristics

| Metric | Value | Description |
|--------|--------|-------------|
| **Decision Latency** | ~2-5s | Time for agent to choose action |
| **Analysis Latency** | ~5-15s | Time for complete analysis |
| **Memory Usage** | Low | Efficient context loading |
| **Notebook Size Limit** | ~1MB | Recommended maximum notebook size |
| **Context Window** | 200K+ tokens | With modern LLMs |

## 🤝 Contributing

### **Adding New Actions**
1. Update `DecideAction._create_decision_prompt()` with new action
2. Add routing logic in `DecideAction.post()`
3. Create corresponding analysis logic
4. Add tests and documentation

### **Extending Analysis Nodes**
1. Inherit from `Node` base class
2. Implement `prep()`, `exec()`, `post()` methods
3. Add to agent flow connections
4. Test with various inputs

### **Improving Decision Making**
1. Enhance the decision prompt with better context
2. Add more sophisticated YAML parsing
3. Include additional context sources
4. Refine action categorization

## 📄 License

Part of the Jupyter AI Personas package. See project license for details.

## 🙏 Acknowledgments

- **PocketFlow**: Excellent flow orchestration framework
- **agno**: Powerful agent framework for LLM integration  
- **Jupyter AI**: Extensible AI platform for notebooks
- **AWS Bedrock**: Reliable foundation model access

---

**🎯 Ready to analyze your data science projects with AI-powered insights? Get started with `@DataSciencePersona` in Jupyter AI!**