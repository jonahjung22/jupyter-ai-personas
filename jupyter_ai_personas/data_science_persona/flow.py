"""
PocketFlow Flow for Data Science Analysis

Orchestrates the three-node pipeline following PocketFlow best practices:
MarkdownReaderNode -> NotebookReaderNode -> ResponseGeneratorNode

Implements the "Graph + Shared Store" pattern for efficient data science analysis.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .pocketflow import Flow
from .nodes import MarkdownReaderNode, NotebookReaderNode, ResponseGeneratorNode

logger = logging.getLogger(__name__)


class DataScienceFlow(Flow):
    """
    PocketFlow implementation for data science analysis
    
    Orchestrates the three nodes in sequence:
    1. MarkdownReaderNode - Reads repo_context.md for project context
    2. NotebookReaderNode - Reads and analyzes notebook content
    3. ResponseGeneratorNode - Generates comprehensive analysis and recommendations
    """
    
    def __init__(self, model_client=None):
        super().__init__()
        
        # Initialize nodes following PocketFlow pattern
        self.markdown_reader = MarkdownReaderNode()
        self.notebook_reader = NotebookReaderNode()
        self.response_generator = ResponseGeneratorNode(model_client=model_client)
        
        # Set up flow: markdown -> notebook -> response
        self.start(self.markdown_reader)
        self.markdown_reader >> self.notebook_reader
        self.notebook_reader >> self.response_generator
        
        logger.info("✅ DataScienceFlow initialized")
    
    def run_analysis(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """
        Run complete data science analysis using PocketFlow
        
        Args:
            user_query: User's question/request
            **kwargs: Additional context parameters
            
        Returns:
            Dict containing analysis results and processing summary
        """
        try:
            # Initialize shared state (PocketFlow's shared store)
            shared = {
                "user_query": user_query,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            
            logger.info(f"🚀 Starting data science analysis for: {user_query[:50]}...")
            
            # Run the PocketFlow pipeline
            result = self.run(shared)
            
            # Extract results from shared state
            return {
                "success": shared.get("generation_success", False),
                "response": shared.get("final_response", "No response generated"),
                "markdown_loaded": shared.get("markdown_success", False),
                "notebook_loaded": shared.get("notebook_success", False),
                "notebook_path": shared.get("notebook_path", ""),
                "markdown_path": shared.get("markdown_path", ""),
                "clean_query": shared.get("clean_query", user_query),
                "processing_summary": {
                    "markdown_read": shared.get("markdown_success", False),
                    "notebook_read": shared.get("notebook_success", False),
                    "response_generated": shared.get("generation_success", False)
                },
                "shared_state": shared  # For debugging
            }
            
        except Exception as e:
            logger.error(f"❌ DataScienceFlow error: {e}")
            return {
                "success": False,
                "response": f"Error in analysis flow: {str(e)}",
                "error": str(e),
                "processing_summary": {
                    "markdown_read": False,
                    "notebook_read": False,
                    "response_generated": False
                }
            }
    
    def prep(self, shared):
        """Flow preparation - set up initial context"""
        return {
            "flow_start_time": datetime.now().isoformat(),
            "user_query": shared.get("user_query", "")
        }
    
    def post(self, shared, prep_res, exec_res):
        """Flow completion - finalize results"""
        shared["flow_completed"] = True
        shared["flow_end_time"] = datetime.now().isoformat()
        
        # Log completion summary
        logger.info(f"📊 Flow completed - Success: {shared.get('generation_success', False)}")
        
        return exec_res