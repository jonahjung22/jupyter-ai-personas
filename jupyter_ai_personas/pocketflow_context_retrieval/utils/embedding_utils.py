"""
utils/embedding_utils.py - Embedding generation and management
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from ..config import config

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding generation with caching and optimization."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model
        self._model = None
        self._model_cache = {}
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text with caching."""
        try:
            if not self._model:
                self._load_model()
            
            # Simple caching based on text hash
            text_hash = hash(text)
            if text_hash in self._model_cache:
                return self._model_cache[text_hash]
            
            embedding = self._generate_embedding(text)
            
            # Cache if reasonable size
            if len(self._model_cache) < 1000:
                self._model_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._get_fallback_embedding(text)
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []
        
        try:
            if not self._model:
                self._load_model()
            
            embeddings = []
            for text in texts:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [self._get_fallback_embedding(text) for text in texts]
    
    def _load_model(self):
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
            
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            self._model = "fallback"
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate actual embedding."""
        if self._model == "fallback":
            return self._get_fallback_embedding(text)
        
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """Generate fallback embedding for testing."""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        # Create deterministic embedding from hash
        hex_digits = hash_obj.hexdigest()
        embedding = []
        for i in range(0, min(len(hex_digits), 32), 2):
            value = int(hex_digits[i:i+2], 16) / 255.0
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < config.embedding_dimension:
            embedding.extend(embedding[:config.embedding_dimension - len(embedding)])
        
        return embedding[:config.embedding_dimension]

# Global embedding manager
embedding_manager = EmbeddingManager()
