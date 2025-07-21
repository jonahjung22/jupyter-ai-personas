import logging
from pocketflow import Flow
from ..nodes.notebook_analysis import AdvancedNotebookAnalysisNode
from ..nodes.rag_search import IntelligentRAGSearchNode
from ..nodes.synthesis import LLMSynthesisNode
from ..nodes.output import AdvancedOutputNode
from ..config import config

logger = logging.getLogger(__name__)

def create_context_flow(handbook_path: str = None) -> Flow:
    """
    Create the main PocketFlow context retrieval flow.
    
    Args:
        handbook_path: Path to Python Data Science Handbook
    
    Returns:
        Configured PocketFlow flow
    """
    
    # Initialize all nodes
    notebook_node = AdvancedNotebookAnalysisNode()
    rag_node = IntelligentRAGSearchNode(handbook_path=handbook_path)
    synthesis_node = LLMSynthesisNode()
    output_node = AdvancedOutputNode()
    
    # Create linear pipeline
    notebook_node >> rag_node >> synthesis_node >> output_node
    
    # Create flow
    flow = Flow(start=notebook_node)
    
    logger.info("🔧 PocketFlow context retrieval flow created")
    logger.info(f"   Components: Notebook → RAG → Synthesis → Output")
    logger.info(f"   Handbook path: {handbook_path or config.handbook_path}")
    
    return flow

def create_fast_context_flow(handbook_path: str = None) -> Flow:
    """
    Create a faster flow that skips synthesis for quick results.
    
    Args:
        handbook_path: Path to Python Data Science Handbook
    
    Returns:
        Fast PocketFlow flow (without synthesis)
    """
    
    notebook_node = AdvancedNotebookAnalysisNode()
    rag_node = IntelligentRAGSearchNode(handbook_path=handbook_path)
    output_node = AdvancedOutputNode()
    
    # Direct pipeline without synthesis
    notebook_node >> rag_node >> output_node
    
    flow = Flow(start=notebook_node)
    
    logger.info("⚡ Fast PocketFlow context flow created (no synthesis)")
    
    return flow