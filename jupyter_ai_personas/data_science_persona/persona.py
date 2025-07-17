"""
Simplified PocketFlow Data Science Persona

A streamlined implementation using proper PocketFlow architecture:
1. Reads repo_context.md for project context
2. Reads notebook content based on user query  
3. Generates comprehensive analysis and code recommendations

Follows PocketFlow best practices with clean node separation and shared state.
"""

import logging
from typing import Dict, Any, AsyncGenerator
from datetime import datetime

from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from langchain_core.messages import HumanMessage
from agno.models.aws import AwsBedrock
import boto3

from .flow import DataScienceFlow

logger = logging.getLogger(__name__)

# Session for AWS Bedrock
session = boto3.Session()


class DataSciencePersona(BasePersona):
    """
    Simplified PocketFlow Data Science Persona
    
    Uses proper PocketFlow architecture to:
    - Read repo_context.md for project understanding
    - Analyze notebook content (specify path or auto-detect)
    - Generate actionable analysis and code recommendations
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow = None
        self._initialization_attempted = False
    
    @property
    def defaults(self):
        return PersonaDefaults(
            name="DataSciencePersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="PocketFlow-powered data science analyst. Reads repo context and notebook content to provide actionable analysis and code recommendations.",
            system_prompt="""I am a data science analyst powered by PocketFlow.

I help with data science projects by:
1. Reading your repo_context.md for project understanding
2. Analyzing your notebook content (specify 'notebook: path/to/file.ipynb' or I'll find .ipynb files)
3. Providing comprehensive analysis and actionable code recommendations

My analysis includes:
- Current state assessment of your work
- Specific recommendations for improvement
- Ready-to-use code snippets you can implement
- Clear next steps for continued development

Just describe what you need help with, and I'll analyze your current work to provide targeted guidance!""",
        )
    
    def _ensure_flow_initialized(self):
        """Initialize the PocketFlow if not already done"""
        if not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                # Create AWS Bedrock client
                model_id = self.config_manager.lm_provider_params["model_id"]
                model_client = AwsBedrock(id=model_id, session=session)
                
                # Create PocketFlow
                self.flow = DataScienceFlow(model_client=model_client)
                
                logger.info("✅ DataSciencePersona initialized with AWS Bedrock")
                
            except Exception as e:
                logger.error(f"❌ Initialization failed: {e}")
                # Create flow without model client for fallback
                self.flow = DataScienceFlow(model_client=None)
                logger.info("⚠️ DataSciencePersona initialized in fallback mode")
    
    async def process_message(self, message: Message):
        """Process messages using PocketFlow data science analysis"""
        logger.info(f"🚀 DATA SCIENCE REQUEST: {message.body}")
        
        try:
            # Ensure flow is initialized
            self._ensure_flow_initialized()
            
            # Get context information
            context_info = await self._prepare_context_info(message)
            
            # Run PocketFlow analysis
            result = self.flow.run_analysis(
                user_query=message.body,
                **context_info
            )
            
            response_content = result.get("response", "Error: No response generated")
            
            # Add processing summary to response
            if result.get("processing_summary"):
                summary = result["processing_summary"]
                status_info = f"""

---
**Processing Summary:**
- Repo Context: {'✅ Loaded' if summary['markdown_read'] else '❌ Not found'} 
- Notebook Analysis: {'✅ Loaded' if summary['notebook_read'] else '❌ Not found'}
- AI Analysis: {'✅ Generated' if summary['response_generated'] else '❌ Failed'}
"""
                if result.get("notebook_path"):
                    status_info += f"- Notebook: `{result['notebook_path']}`\n"
                
                response_content += status_info
            
            # Log processing results
            self._log_processing_summary(result)
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            response_content = f"""# Data Science Analysis Error

An error occurred: {str(e)}

## Troubleshooting:
1. Ensure `repo_context.md` exists in your current directory
2. Check that your notebook path is correct (use 'notebook: path/to/file.ipynb')
3. Verify AWS Bedrock configuration
4. Make sure you're in the correct working directory

## Quick Fix:
Create a `repo_context.md` file in your current directory with:
```markdown
# Project Context
Brief description of your data science project, goals, and current status.
```

Please try again with a simpler query."""
        
        # Stream response back to user
        await self.stream_message(self._create_response_iterator(response_content))
    
    async def _prepare_context_info(self, message: Message) -> Dict[str, Any]:
        """Prepare context information for the flow"""
        try:
            # Get chat history
            history = YChatHistory(ychat=self.ychat, k=2)
            messages = await history.aget_messages()
            
            history_text = ""
            if messages:
                history_text = "\nPrevious conversation:\n"
                for msg in messages:
                    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    history_text += f"{role}: {msg.content[:100]}...\n"
            
            return {
                "history": history_text,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Context preparation error: {e}")
            return {}
    
    def _log_processing_summary(self, result: Dict[str, Any]):
        """Log processing summary for debugging"""
        try:
            logger.info(f"📊 Processing Summary:")
            logger.info(f"   Success: {result.get('success', False)}")
            logger.info(f"   Markdown Loaded: {result.get('markdown_loaded', False)}")
            logger.info(f"   Notebook Loaded: {result.get('notebook_loaded', False)}")
            logger.info(f"   Notebook Path: {result.get('notebook_path', 'None')}")
            logger.info(f"   Clean Query: {result.get('clean_query', 'None')}")
            
            if result.get("error"):
                logger.error(f"   Error: {result['error']}")
                
        except Exception as e:
            logger.error(f"Logging error: {e}")
    
    async def _create_response_iterator(self, content: str) -> AsyncGenerator[str, None]:
        """Create response iterator for streaming"""
        yield content
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for debugging"""
        self._ensure_flow_initialized()
        return {
            "persona_type": "DataSciencePersona",
            "flow_initialized": self.flow is not None,
            "architecture": "PocketFlow 3-node pipeline",
            "nodes": ["MarkdownReaderNode", "NotebookReaderNode", "ResponseGeneratorNode"],
            "timestamp": datetime.now().isoformat()
        }