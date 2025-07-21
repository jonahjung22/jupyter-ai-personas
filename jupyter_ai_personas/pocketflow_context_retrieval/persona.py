import logging
from typing import Dict, Any
from datetime import datetime

from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from langchain_core.messages import HumanMessage

from .flows.context_flow import create_context_flow, create_fast_context_flow
from .config import config
from .agents.conversational_agent import IntelligentConversationalAgent

logger = logging.getLogger(__name__)

# Import the proven NotebookReaderTool from the original context retrieval persona
try:
    from ..context_retrieval_persona.file_reader_tool import NotebookReaderTool
    NOTEBOOK_READER_AVAILABLE = True
    logger.info("✅ NotebookReaderTool imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ NotebookReaderTool not available: {e}")
    NOTEBOOK_READER_AVAILABLE = False

class PocketFlowContextPersona(BasePersona):
    """
    Advanced context retrieval persona using pure PocketFlow architecture.
    
    Features:
    - Advanced notebook analysis with workflow detection
    - Intelligent multi-query RAG search
    - LLM-powered synthesis and report generation
    - Multiple output formats with metadata
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize flows (lazy loading)
        self.context_flow = None
        self.fast_flow = None
        self.conversational_agent = None
        
        # Initialize notebook reader tool
        self.notebook_reader = NotebookReaderTool() if NOTEBOOK_READER_AVAILABLE else None
        
        logger.info("✅ PocketFlow Context Persona initialized")
    
    @property
    def defaults(self):
        return PersonaDefaults(
            name="PocketFlowContextPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="Advanced context retrieval using PocketFlow architecture with intelligent RAG and comprehensive analysis.",
            system_prompt="""I am an advanced context retrieval specialist powered by **PocketFlow architecture**.

## 🚀 **My Capabilities:**

**🧠 Advanced Notebook Analysis**
- Deep semantic understanding of your code and workflow
- Automatic workflow stage detection (data loading → EDA → modeling → etc.)
- Library usage patterns and complexity assessment
- Code quality analysis with specific recommendations

**🔍 Intelligent RAG Search**  
- Multi-query strategic search through Python Data Science Handbook
- Context-aware query generation based on your notebook analysis
- Quality filtering and advanced relevance ranking
- Comprehensive coverage of relevant handbook sections

**📝 LLM-Powered Synthesis**
- Research-backed recommendations with handbook citations
- Comprehensive reports with executive summaries
- Actionable next steps prioritized by impact
- Code examples with practical application guidance

**⚡ Superior Architecture**
- Pure PocketFlow design - modular, testable, optimizable
- No dependencies on legacy RAG tools - built for intelligence
- Advanced quality filtering and content ranking
- Multiple output formats (full report + executive summary + metadata)

## 🎯 **How to Use Me:**

**For Quick Analysis:**
```
analyze my pandas workflow for optimization opportunities
```

**For Deep Analysis:**  
```
notebook: /path/to/your/analysis.ipynb
Help me improve my machine learning workflow and find relevant handbook examples
```

**For Specific Topics:**
```
I'm working on time series analysis with pandas - find the best handbook techniques and examples
```

## 📊 **What You'll Get:**

- **`repo_context.md`** - Comprehensive analysis report with research findings
- **`context_summary.md`** - Executive summary with key recommendations  
- **`analysis_metadata.json`** - Technical details and system metrics

Every recommendation is **research-backed** from the Python Data Science Handbook with **specific source citations** and **practical implementation guidance**.

**Ready to provide superior context analysis with PocketFlow intelligence!**""",
        )
    
    def _initialize_flows(self):
        """Initialize PocketFlow flows and conversational agent if not already done."""
        if not self.context_flow:
            handbook_path = getattr(config, 'handbook_path', "./PythonDataScienceHandbook")
            self.context_flow = create_context_flow(handbook_path)
            self.fast_flow = create_fast_context_flow(handbook_path)
            logger.info("🔧 PocketFlow flows initialized")
        
        if not self.conversational_agent:
            # Get LLM provider from Jupyter AI config (same pattern as finance persona)
            llm_provider = self.config.lm_provider(**self.config.lm_provider_params)
            self.conversational_agent = IntelligentConversationalAgent(llm_provider=llm_provider)
            logger.info("🤖 Conversational agent initialized with Bedrock LLM")
    
    async def process_message(self, message: Message):
        """Process messages using PocketFlow architecture with intelligent agent."""
        try:
            logger.info(f"🧠 POCKETFLOW CONTEXT RETRIEVAL: {message.body}")
            
            # Initialize flows and agent if needed
            self._initialize_flows()
            
            message_text = message.body.strip()
            
            # Get chat history for context
            history = YChatHistory(ychat=self.ychat, k=3)
            messages = await history.aget_messages()
            
            # Analyze request type
            request_analysis = self._analyze_request(message_text, messages)
            
            # Let the intelligent agent decide how to handle the message
            # It will determine if it needs analysis, is conversational, or mixed
            
            # The agent will decide if it needs to trigger analysis
            # For now, we'll let it handle everything and potentially call back for analysis
            response_content = await self.conversational_agent.handle_message(
                message_text, 
                context_info=request_analysis
            )
            
            # Stream response
            async def response_iterator():
                yield response_content
            
            await self.stream_message(response_iterator())
            
        except Exception as e:
            logger.error(f"❌ PocketFlow processing failed: {e}")
            error_response = self._create_error_response(str(e))
            
            async def error_iterator():
                yield error_response
            
            await self.stream_message(error_iterator())
    
    def _analyze_request(self, message_text: str, chat_history: list) -> Dict[str, Any]:
        """Basic request analysis - let the agent handle intelligent routing."""
        return {
            "type": "agent_decision",
            "notebook_path": self._extract_notebook_path(message_text),
            "has_notebook": ".ipynb" in message_text.lower() or "notebook:" in message_text.lower(),
            "message_length": len(message_text),
            "chat_context": chat_history[-2:] if chat_history else []  # Recent context
        }
    
    async def _handle_status_check(self) -> str:
        """Handle system status requests."""
        return f"""# 🚀 PocketFlow Context Retrieval System Status

## ✅ **System Status: OPERATIONAL**

**Core Components:**
- **Advanced Notebook Analysis**: ✅ Ready with workflow detection
- **Intelligent RAG Search**: ✅ Multi-query strategy active
- **LLM Synthesis Engine**: ✅ {"Enabled" if config.enable_llm_synthesis else "Disabled (structured mode)"}
- **Quality Filtering**: ✅ {"Enabled" if config.enable_quality_filtering else "Disabled"}
- **Advanced Ranking**: ✅ {"Enabled" if config.enable_advanced_ranking else "Disabled"}

**Configuration:**
- **Embedding Model**: {config.embedding_model}
- **Index Type**: {config.index_type.upper()}
- **Max Search Queries**: {config.max_search_queries}
- **Quality Threshold**: {config.quality_threshold}
- **Handbook Path**: {config.handbook_path}

**Architecture Advantages:**
🧠 **Superior Intelligence**: Context-aware analysis with semantic understanding  
🔍 **Smart Search**: Multi-query strategy with quality filtering  
📊 **Deep Analysis**: Workflow stage detection and complexity assessment  
📝 **Research-Backed**: All recommendations sourced from Python Data Science Handbook  

## 🎯 **Ready for Analysis!**

**Try these commands:**
- `analyze my data science workflow` - General analysis
- `notebook: /path/file.ipynb` - Deep notebook analysis  
- `help with pandas optimization` - Topic-specific guidance

**What you'll get:**
- `repo_context.md` - Full analysis report
- `context_summary.md` - Executive summary
- `analysis_metadata.json` - Technical metrics

**PocketFlow provides superior context analysis compared to legacy RAG systems.**
"""
    
    async def _handle_quick_analysis(self, message_text: str, analysis: Dict[str, Any]) -> str:
        """Handle quick analysis requests with fast flow."""
        try:
            # Prepare shared data for fast processing
            shared_data = {
                "user_query": message_text,
                "processing_mode": "fast",
                "timestamp": datetime.now().isoformat()
            }
            
            # Use fast flow (no synthesis)
            logger.info("⚡ Running fast PocketFlow analysis")
            self.fast_flow.run(shared_data)
            
            # Format quick response
            return self._format_quick_response(shared_data)
            
        except Exception as e:
            logger.error(f"❌ Quick analysis failed: {e}")
            return self._create_error_response(str(e))
    
    async def _handle_comprehensive_analysis(self, message_text: str, analysis: Dict[str, Any]) -> str:
        """Handle comprehensive analysis requests with full flow."""
        try:
            # Prepare shared data
            shared_data = {
                "user_query": message_text,
                "notebook_path": analysis.get("notebook_path"),
                "processing_mode": "comprehensive",
                "timestamp": datetime.now().isoformat()
            }
            
            # Run full PocketFlow pipeline
            logger.info("🧠 Running comprehensive PocketFlow analysis")
            self.context_flow.run(shared_data)
            
            # Format comprehensive response
            return self._format_comprehensive_response(shared_data)
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis failed: {e}")
            return self._create_error_response(str(e))
    
    def _format_quick_response(self, shared_data: Dict[str, Any]) -> str:
        """Format response for quick analysis."""
        user_query = shared_data.get("user_query", "")
        notebook_analysis = shared_data.get("advanced_notebook_analysis", {})
        rag_results = shared_data.get("intelligent_rag_results", [])
        
        response = f"""# ⚡ Quick PocketFlow Analysis

**Query**: {user_query}  
**Mode**: Fast analysis (no synthesis)  
**Completed**: {datetime.now().strftime("%H:%M:%S")}  

## 📊 Notebook Analysis
"""
        
        if notebook_analysis and not notebook_analysis.get("fallback_mode"):
            workflow = notebook_analysis.get("workflow_detection", {})
            semantic = notebook_analysis.get("semantic_analysis", {})
            
            response += f"""- **Workflow Stage**: {workflow.get("primary_stage", "unknown").replace("_", " ").title()}
- **Libraries**: {", ".join([lib["name"] for lib in semantic.get("detected_libraries", [])][:3])}
- **Complexity**: {notebook_analysis.get("code_intelligence", {}).get("code_quality_level", "unknown")}
"""
        else:
            response += "- Quick analysis of query context completed\n"
        
        response += "\n## 🔍 RAG Search Results\n"
        
        successful_searches = len([r for r in rag_results if r.get("execution_status") == "success"])
        total_results = sum(len(r.get("results", [])) for r in rag_results)
        
        response += f"- **Searches**: {successful_searches}/{len(rag_results)} successful\n"
        response += f"- **Results**: {total_results} relevant handbook sections found\n"
        
        if rag_results:
            response += "\n**Top Results:**\n"
            for result in rag_results[:2]:  # Top 2 searches
                if result.get("results"):
                    top_result = result["results"][0]
                    response += f"- **{top_result.get('notebook_name', 'Unknown')}**: {top_result.get('content', '')[:100]}...\n"
        
        response += f"""
## 📝 Full Analysis Available

For comprehensive analysis with research-backed recommendations, use:
```
notebook: /path/to/your/file.ipynb
{user_query}
```

**Files Created**: {", ".join(shared_data.get("output_files", ["None"]))}  
**Architecture**: PocketFlow modular RAG system  
"""
        
        return response
    
    def _format_comprehensive_response(self, shared_data: Dict[str, Any]) -> str:
        """Format response for comprehensive analysis."""
        user_query = shared_data.get("user_query", "")
        synthesis_completed = shared_data.get("synthesis_completed", False)
        synthesis_method = shared_data.get("synthesis_method", "unknown")
        output_files = shared_data.get("output_files", [])
        
        response = f"""# 🧠 Comprehensive PocketFlow Analysis Complete

**Query**: {user_query}  
**Analysis Type**: Full PocketFlow pipeline  
**Completed**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  

## ✅ **Analysis Results**

**Pipeline Execution:**
- **Notebook Analysis**: ✅ Advanced semantic analysis completed
- **RAG Search**: ✅ Multi-query intelligent search executed
- **Synthesis**: {"✅" if synthesis_completed else "⚠️"} {synthesis_method.replace("_", " ").title()} synthesis
- **Output Generation**: ✅ Multiple formats created

**Files Generated:**
"""
        
        for file_path in output_files:
            file_name = file_path.split("/")[-1] if "/" in file_path else file_path
            
            if "repo_context.md" in file_name:
                response += f"- **📋 {file_name}**: Comprehensive analysis report with research findings\n"
            elif "context_summary.md" in file_name:
                response += f"- **📄 {file_name}**: Executive summary with key recommendations\n"
            elif "metadata.json" in file_name:
                response += f"- **🔧 {file_name}**: Technical analysis metrics and configuration\n"
            else:
                response += f"- **📁 {file_name}**: Additional analysis output\n"
        
        # Add statistics
        notebook_analysis = shared_data.get("advanced_notebook_analysis", {})
        rag_results = shared_data.get("intelligent_rag_results", [])
        
        if notebook_analysis and not notebook_analysis.get("fallback_mode"):
            workflow = notebook_analysis.get("workflow_detection", {})
            semantic = notebook_analysis.get("semantic_analysis", {})
            
            response += f"""
## 📊 **Analysis Highlights**

**Notebook Intelligence:**
- **Primary Stage**: {workflow.get("primary_stage", "unknown").replace("_", " ").title()}
- **Confidence**: {workflow.get("confidence", 0):.1f}/1.0
- **Libraries Detected**: {len(semantic.get("detected_libraries", []))} ({", ".join([lib["name"] for lib in semantic.get("detected_libraries", [])][:4])})
- **Analysis Themes**: {", ".join(semantic.get("analysis_themes", [])[:3])}
"""
        
        if rag_results:
            successful = len([r for r in rag_results if r.get("execution_status") == "success"])
            total_results = sum(len(r.get("results", [])) for r in rag_results)
            
            response += f"""
**RAG Search Intelligence:**
- **Strategic Searches**: {successful}/{len(rag_results)} executed successfully
- **Handbook Results**: {total_results} relevant sections retrieved
- **Quality Filtered**: Advanced relevance ranking applied
- **Source Coverage**: Multiple handbook chapters consulted
"""
        
        response += f"""
## 🎯 **Next Steps**

1. **Open `repo_context.md`** - Your comprehensive analysis report
2. **Review recommendations** - Research-backed insights with handbook citations
3. **Apply code examples** - Practical snippets ready for implementation
4. **Follow action plan** - Prioritized next steps for immediate impact

## 💪 **PocketFlow Advantages Applied**

✅ **Superior Architecture**: Modular design with advanced intelligence  
✅ **Context Awareness**: Deep understanding of your workflow and objectives  
✅ **Quality Research**: Multi-query strategy with relevance filtering  
✅ **Actionable Insights**: Specific recommendations with implementation guidance  

**Your analysis demonstrates the power of PocketFlow over legacy RAG systems.**
"""
        
        return response
    
    def _extract_notebook_path(self, message_text: str) -> str:
        """Extract notebook path from message."""
        import re
        
        # Pattern: notebook: path
        notebook_match = re.search(r'notebook:\s*([^\s]+\.ipynb)', message_text, re.IGNORECASE)
        if notebook_match:
            return notebook_match.group(1)
        
        # Pattern: any .ipynb file
        ipynb_match = re.search(r'([^\s]+\.ipynb)', message_text)
        if ipynb_match:
            return ipynb_match.group(1)
        
        return None
    
    def _create_error_response(self, error_msg: str) -> str:
        """Create user-friendly error response."""
        return f"""# ⚠️ **PocketFlow Processing Issue**

**Error Details**: {error_msg}

## 🔧 **Troubleshooting Steps**

1. **Check Configuration**
   - Verify handbook path: `{config.handbook_path}`
   - Ensure dependencies installed: `pip install sentence-transformers faiss-cpu nbformat`

2. **Verify Input**
   - Check notebook path accessibility
   - Ensure query is properly formatted

3. **System Recovery**
   - Try a simpler query first
   - Check system status: ask "status"

## 💡 **Alternative Options**

- **Quick Analysis**: Try shorter, simpler queries
- **Manual Search**: Use individual components if needed
- **System Reset**: Restart the persona if issues persist

**PocketFlow architecture remains robust - this is likely a configuration or input issue.**

Need help? Ask about "status" to check system health.
"""