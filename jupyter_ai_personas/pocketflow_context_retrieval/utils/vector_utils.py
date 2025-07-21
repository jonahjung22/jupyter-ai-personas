"""
utils/vector_utils.py - Vector index creation and search operations
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Any, Dict
import numpy as np
from ..config import config

logger = logging.getLogger(__name__)

class VectorIndexManager:
    """Manages vector index operations with persistence."""
    
    def __init__(self, index_path: str = None):
        self.index_path = Path(index_path or config.vector_store_path)
        self.index = None
        self.index_metadata = {}
    
    def create_index(self, embeddings: List[List[float]], metadata: List[Dict] = None) -> bool:
        """Create vector index from embeddings."""
        try:
            if not embeddings:
                raise ValueError("No embeddings provided")
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            if config.index_type == "faiss":
                self.index = self._create_faiss_index(embeddings_array)
            else:
                self.index = self._create_simple_index(embeddings_array)
            
            # Store metadata
            if metadata:
                self.index_metadata = {
                    "document_count": len(embeddings),
                    "dimension": embeddings_array.shape[1],
                    "index_type": config.index_type,
                    "documents_metadata": metadata
                }
            
            logger.info(f"Created {config.index_type} index with {len(embeddings)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return False
    
    def search(self, query_embedding: List[float], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search vector index for similar embeddings."""
        if not self.index:
            raise ValueError("Index not initialized")
        
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            
            if hasattr(self.index, 'search'):  # FAISS index
                distances, indices = self.index.search(query_array, k)
                return indices, distances
            else:  # Simple index
                return self._search_simple_index(query_array, k)
                
        except Exception as e:
            logger.error(f"Index search failed: {e}")
            return np.array([[0]]), np.array([[0.0]])
    
    def save_index(self) -> bool:
        """Save index to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config.index_type == "faiss":
                return self._save_faiss_index()
            else:
                return self._save_simple_index()
                
        except Exception as e:
            logger.error(f"Index saving failed: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load index from disk."""
        try:
            if not self.index_path.exists():
                return False
            
            if config.index_type == "faiss":
                return self._load_faiss_index()
            else:
                return self._load_simple_index()
                
        except Exception as e:
            logger.error(f"Index loading failed: {e}")
            return False
    
    def _create_faiss_index(self, embeddings: np.ndarray):
        """Create FAISS index."""
        try:
            import faiss
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            index.add(embeddings)
            return index
        except ImportError:
            logger.warning("FAISS not available, falling back to simple index")
            return self._create_simple_index(embeddings)
    
    def _create_simple_index(self, embeddings: np.ndarray):
        """Create simple in-memory index."""
        return {
            "embeddings": embeddings,
            "type": "simple"
        }
    
    def _search_simple_index(self, query_array: np.ndarray, k: int):
        """Search simple index."""
        embeddings = self.index["embeddings"]
        
        # Calculate cosine similarities
        query_norm = np.linalg.norm(query_array)
        similarities = np.dot(embeddings, query_array.T).flatten()
        similarities = similarities / (np.linalg.norm(embeddings, axis=1) * query_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        
        return np.array([top_indices]), np.array([top_similarities])
    
    def _save_faiss_index(self) -> bool:
        """Save FAISS index."""
        try:
            import faiss
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))
            
            # Save metadata separately
            with open(self.index_path / "metadata.pkl", "wb") as f:
                pickle.dump(self.index_metadata, f)
            
            return True
        except ImportError:
            return self._save_simple_index()
    
    def _save_simple_index(self) -> bool:
        """Save simple index."""
        index_data = {
            "index": self.index,
            "metadata": self.index_metadata
        }
        
        with open(self.index_path / "simple_index.pkl", "wb") as f:
            pickle.dump(index_data, f)
        
        return True
    
    def _load_faiss_index(self) -> bool:
        """Load FAISS index."""
        try:
            import faiss
            self.index = faiss.read_index(str(self.index_path / "faiss.index"))
            
            # Load metadata
            metadata_path = self.index_path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    self.index_metadata = pickle.load(f)
            
            return True
        except ImportError:
            return self._load_simple_index()
    
    def _load_simple_index(self) -> bool:
        """Load simple index."""
        index_path = self.index_path / "simple_index.pkl"
        if not index_path.exists():
            return False
        
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)
        
        self.index = index_data["index"]
        self.index_metadata = index_data.get("metadata", {})
        
        return True

# Global vector index manager
vector_manager = VectorIndexManager()