"""
Advanced PocketFlow Data Science Agent

A sophisticated implementation using PocketFlow agent architecture with AI reasoning:
1. Analyzes user intent and makes intelligent decisions
2. Reads repo_context.md and notebook content automatically
3. Chooses targeted analysis approach based on context
4. Provides iterative, context-aware data science guidance

Features intelligent decision-making, reasoning capabilities, and adaptive responses.
"""

import logging
from typing import Dict, Any, AsyncGenerator, Optional
from datetime import datetime

from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from langchain_core.messages import HumanMessage
from agno.models.aws import AwsBedrock
import boto3

from .agent import DataScienceAgent

logger = logging.getLogger(__name__)

# Session for AWS Bedrock
session = boto3.Session()


class DataSciencePersona(BasePersona):
    """
    Advanced PocketFlow Data Science Persona
    
    Uses sophisticated PocketFlow agent with reasoning capabilities to:
    - Analyze user queries and make intelligent decisions
    - Read repo_context.md for project understanding
    - Analyze notebook content with targeted approach
    - Generate actionable analysis and code recommendations
    - Provide iterative, context-aware assistance
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = None
        self._initialization_attempted = False
    
    @property
    def defaults(self):
        return PersonaDefaults(
            name="DataSciencePersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="Advanced PocketFlow agent for data science analysis. Uses AI reasoning to provide targeted, context-aware recommendations.",
            system_prompt="""I am an advanced data science agent powered by PocketFlow with sophisticated reasoning capabilities.

                        I intelligently analyze your requests and choose the most appropriate approach:

                        🤖 **Agent Capabilities:**
                        - **Smart Decision Making**: I analyze your query and context to decide on the best action
                        - **Iterative Analysis**: I can perform multiple analysis rounds based on findings
                        - **Context Integration**: I combine repo context, notebook content, and conversation history
                        - **Targeted Responses**: I provide focused analysis based on your specific needs

                        🔧 **What I Do:**
                        1. **Analyze Intent**: Understand what you're really asking for
                        2. **Load Context**: Read your repo_context.md and notebook content automatically
                        3. **Choose Action**: Decide between focused analysis, code generation, or comprehensive review
                        4. **Provide Results**: Deliver actionable insights and ready-to-use code
                        5. **Iterate**: Continue analysis if needed based on results

                        📊 **Analysis Types:**
                        - **Focused Data Analysis**: Targeted insights on specific questions
                        - **Code Generation**: Ready-to-implement solutions
                        - **Comprehensive Review**: Complete analysis with recommendations
                        - **Issue Detection**: Identify problems and provide fixes

                        Just describe what you need help with, and I'll intelligently analyze your situation to provide the most relevant guidance!""",
                                )
    
    def _ensure_agent_initialized(self):
        """Initialize the PocketFlow agent if not already done"""
        if not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                # Create AWS Bedrock client
                model_id = self.config_manager.lm_provider_params["model_id"]
                logger.info(f"🔧 Using model_id: {model_id}")
                model_client = AwsBedrock(id=model_id, session=session)
                
                # Create PocketFlow Agent
                self.agent = DataScienceAgent(model_client=model_client)
                
                logger.info("✅ DataSciencePersona agent initialized with AWS Bedrock")
                
            except KeyError as e:
                logger.error(f"❌ Configuration error - missing key: {e}")
                logger.error(f"Available config_manager attributes: {dir(self.config_manager)}")
                if hasattr(self.config_manager, 'lm_provider_params'):
                    logger.error(f"Available lm_provider_params keys: {list(self.config_manager.lm_provider_params.keys())}")
                # Create agent without model client for fallback
                self.agent = DataScienceAgent(model_client=None)
                logger.info("⚠️ DataSciencePersona agent initialized in fallback mode")
            except Exception as e:
                logger.error(f"❌ Initialization failed: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                # Create agent without model client for fallback
                self.agent = DataScienceAgent(model_client=None)
                logger.info("⚠️ DataSciencePersona agent initialized in fallback mode")
    
    async def process_message(self, message: Message):
        """Process messages using PocketFlow data science agent"""
        logger.info(f"🤖 DATA SCIENCE AGENT REQUEST: {message.body}")
        
        try:
            # Ensure agent is initialized
            self._ensure_agent_initialized()
            
            # Get context information
            context_info = await self._prepare_context_info(message)
            
            # Run PocketFlow agent analysis
            result = self.agent.run_analysis(
                user_query=message.body,
                **context_info
            )
            
            response_content = result.get("response", "Error: No response generated")
            
            # Add processing summary to response
            if result.get("processing_summary"):
                summary = result["processing_summary"]
                status_info = f"""
                            ---
                            **Agent Processing Summary:**
                            - Repo Context: {'✅ Loaded' if summary['repo_context_loaded'] else '❌ Not found'} 
                            - Notebook Analysis: {'✅ Loaded' if summary['notebook_loaded'] else '❌ Not found'}
                            - AI Analysis: {'✅ Generated' if summary['analysis_complete'] else '❌ Failed'}
                            - Actions Taken: {summary.get('actions_taken', 0)}
                            """
                if result.get("notebook_path"):
                    status_info += f"- Notebook: `{result['notebook_path']}`\n"
                
                if result.get("action_history"):
                    status_info += f"- Agent Actions: {' → '.join(result['action_history'])}\n"
                
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
        """Prepare context information for the agent"""
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
                "timestamp": datetime.now().isoformat(),
                "current_message": message.body  # Now using the message parameter
            }
            
        except Exception as e:
            logger.error(f"Context preparation error: {e}")
            return {}
    
    def _log_processing_summary(self, result: Dict[str, Any]):
        """Log processing summary for debugging"""
        try:
            logger.info(f"🤖 Agent Processing Summary:")
            logger.info(f"   Success: {result.get('success', False)}")
            logger.info(f"   Context Loaded: {result.get('context_loaded', False)}")
            logger.info(f"   Notebook Loaded: {result.get('notebook_loaded', False)}")
            logger.info(f"   Notebook Path: {result.get('notebook_path', 'None')}")
            logger.info(f"   Actions Taken: {len(result.get('action_history', []))}")
            logger.info(f"   Action History: {result.get('action_history', [])}")
            
            if result.get("error"):
                logger.error(f"   Error: {result['error']}")
                
        except Exception as e:
            logger.error(f"Logging error: {e}")
    
    async def _create_response_iterator(self, content: str) -> AsyncGenerator[str, None]:
        """Create response iterator for streaming"""
        yield content
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for debugging"""
        self._ensure_agent_initialized()
        return {
            "persona_type": "DataSciencePersona",
            "agent_initialized": self.agent is not None,
            "architecture": "PocketFlow Agent with Decision-Making",
            "nodes": ["DecideAction", "DataAnalysisNode", "CompleteAnalysisNode"],
            "capabilities": ["reasoning", "decision_making", "iterative_analysis"],
            "timestamp": datetime.now().isoformat()
        }