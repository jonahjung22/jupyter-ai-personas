"""
config.py - Centralized configuration for PocketFlow RAG system
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

@dataclass
class PocketFlowConfig:
    """Configuration for PocketFlow RAG system."""
    
    # Core paths
    handbook_path: str = "./PythonDataScienceHandbook"
    vector_store_path: str = "./data/vector_stores/handbook_index"
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 30
    
    # Search settings
    max_search_queries: int = 8
    default_search_k: int = 5
    quality_threshold: float = 0.3
    
    # Index settings
    index_type: str = "faiss"  # Options: "faiss", "simple"
    enable_metadata_indexing: bool = True
    
    # Analysis settings
    enable_deep_analysis: bool = True
    enable_quality_filtering: bool = True
    enable_advanced_ranking: bool = True
    
    # LLM settings
    llm_provider: str = "aws_bedrock"  # Will be set dynamically
    enable_llm_synthesis: bool = True
    synthesis_fallback: bool = True
    
    # Performance settings
    batch_size: int = 50
    enable_caching: bool = True
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not Path(self.handbook_path).exists():
            return False
        
        if self.chunk_size < self.min_chunk_size:
            return False
        
        if self.quality_threshold < 0 or self.quality_threshold > 1:
            return False
        
        return True

# Global config instance
config = PocketFlowConfig()