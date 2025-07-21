import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path

from pocketflow import Node
from ..utils.embedding_utils import embedding_manager
from ..utils.vector_utils import vector_manager
from ..utils.notebook_utils import extract_notebook_content
from ..utils.content_utils import chunk_text_intelligently, filter_low_quality_content
from ..config import config

logger = logging.getLogger(__name__)

class IntelligentRAGSearchNode(Node):
    """Intelligent RAG search with multi-query strategy and quality filtering."""
    
    def __init__(self, handbook_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.handbook_path = Path(handbook_path or config.handbook_path)
        self.index_ready = False
        self.indexed_documents = []
        
        # Initialize RAG system
        self._initialize_rag_system()
    
    def _initialize_rag_system(self):
        """Initialize the RAG system with index building."""
        try:
            logger.info("🚀 Initializing PocketFlow RAG system")
            
            if not self.handbook_path.exists():
                logger.error(f"❌ Handbook path not found: {self.handbook_path}")
                return
            
            # Try to load existing index
            if vector_manager.load_index():
                logger.info("✅ Loaded existing vector index")
                self.index_ready = True
                self._load_indexed_documents()
            else:
                # Build new index
                logger.info("🔨 Building new vector index...")
                if self._build_comprehensive_index():
                    self.index_ready = True
                    logger.info("✅ PocketFlow RAG system ready")
                else:
                    logger.error("❌ Failed to build RAG index")
        
        except Exception as e:
            logger.error(f"❌ RAG system initialization failed: {e}")
    
    def _build_comprehensive_index(self) -> bool:
        """Build comprehensive vector index from handbook."""
        try:
            # Find notebook files
            notebook_files = list(self.handbook_path.glob("**/*.ipynb"))
            
            if not notebook_files:
                logger.error("📚 No notebook files found")
                return False
            
            logger.info(f"📚 Processing {len(notebook_files)} notebooks")
            
            # Extract all documents
            all_documents = []
            for nb_file in notebook_files:
                try:
                    docs = extract_notebook_content(str(nb_file))
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process {nb_file}: {e}")
            
            if not all_documents:
                logger.error("📄 No documents extracted")
                return False
            
            logger.info(f"📄 Extracted {len(all_documents)} documents")
            
            # Chunk documents intelligently
            chunked_documents = []
            for doc in all_documents:
                chunks = chunk_text_intelligently(doc["content"], doc["metadata"]["cell_type"])
                
                for i, chunk in enumerate(chunks):
                    chunked_doc = doc.copy()
                    chunked_doc["content"] = chunk
                    chunked_doc["metadata"]["chunk_id"] = i
                    chunked_doc["metadata"]["chunk_count"] = len(chunks)
                    chunked_documents.append(chunked_doc)
            
            logger.info(f"🧩 Created {len(chunked_documents)} chunks")
            
            # Filter for quality
            if config.enable_quality_filtering:
                filtered_documents = filter_low_quality_content(chunked_documents)
                logger.info(f"✨ Quality filtered to {len(filtered_documents)} high-value chunks")
            else:
                filtered_documents = chunked_documents
            
            # Generate embeddings
            embeddings = []
            document_metadata = []
            
            for i, doc in enumerate(filtered_documents):
                if i % 100 == 0:
                    logger.info(f"🔢 Generating embeddings: {i}/{len(filtered_documents)}")
                
                try:
                    embedding = embedding_manager.get_embedding(doc["content"])
                    embeddings.append(embedding)
                    document_metadata.append(doc["metadata"])
                except Exception as e:
                    logger.warning(f"⚠️ Embedding failed for document {i}: {e}")
                    continue
            
            logger.info(f"🔢 Generated {len(embeddings)} embeddings")
            
            # Create vector index
            success = vector_manager.create_index(embeddings, document_metadata)
            if not success:
                return False
            
            # Save index
            if not vector_manager.save_index():
                logger.warning("⚠️ Failed to save index to disk")
            
            # Store documents for retrieval
            self.indexed_documents = filtered_documents
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Index building failed: {e}")
            return False
    
    def _load_indexed_documents(self):
        """Load indexed documents from metadata."""
        try:
            # In a full implementation, you'd load documents from saved metadata
            # For now, we'll rebuild if needed
            if not self.indexed_documents:
                logger.info("🔄 Document list needs rebuilding from metadata")
                # Could implement proper document persistence here
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load indexed documents: {e}")
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare intelligent RAG search."""
        if not self.index_ready:
            return {
                "error": "RAG index not ready",
                "fallback_queries": ["python data science tutorial"]
            }
        
        # Get search strategy from notebook analysis
        notebook_analysis = shared.get("advanced_notebook_analysis", {})
        search_strategy = notebook_analysis.get("search_strategy", {})
        
        strategic_queries = search_strategy.get("queries", [])
        
        if not strategic_queries:
            # Generate fallback queries
            user_query = shared.get("user_query", "")
            strategic_queries = self._generate_fallback_queries(user_query)
        
        # Ensure we always have at least one query
        if not strategic_queries:
            strategic_queries = [{"query": user_query or "python data science tutorial", "type": "fallback"}]
        
        return {
            "strategic_queries": strategic_queries,
            "notebook_context": notebook_analysis,
            "search_mode": "intelligent_multi_query"
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute intelligent multi-query RAG search."""
        strategic_queries = prep_res["strategic_queries"]
        notebook_context = prep_res.get("notebook_context", {})
        
        search_results = []
        
        logger.info(f"🧠 Executing {len(strategic_queries)} intelligent RAG searches")
        
        for query_info in strategic_queries:
            try:
                result = self._execute_single_search(query_info, notebook_context)
                search_results.append(result)
                
                logger.info(f"✅ {query_info['type']} search: '{query_info['query']}' -> {result.get('total_results', 0)} results")
                
            except Exception as e:
                logger.error(f"❌ Search failed for '{query_info.get('query', 'unknown')}': {e}")
                search_results.append({
                    "query": query_info.get("query", "unknown"),
                    "type": query_info.get("type", "unknown"),
                    "error": str(e),
                    "execution_status": "failed"
                })
        
        logger.info(f"🎯 Intelligent RAG completed: {len(search_results)} searches executed")
        
        return search_results
    
    def _execute_single_search(self, query_info: Dict, context: Dict) -> Dict[str, Any]:
        """Execute a single intelligent search."""
        query_text = query_info["query"]
        query_type = query_info["type"]
        priority = query_info["priority"]
        
        # Generate query embedding
        query_embedding = embedding_manager.get_embedding(query_text)
        
        # Determine search parameters
        k = {"high": 6, "medium": 4, "low": 3}[priority]
        
        # Perform vector search
        indices, similarities = vector_manager.search(query_embedding, k)
        
        # Retrieve and process results
        raw_results = []
        for doc_idx, similarity in zip(indices[0], similarities[0]):
            if doc_idx < len(self.indexed_documents):
                doc = self.indexed_documents[doc_idx]
                raw_results.append({
                    "document": doc,
                    "similarity_score": float(similarity),
                    "doc_index": int(doc_idx)
                })
        
        # Apply advanced ranking if enabled
        if config.enable_advanced_ranking:
            ranked_results = self._apply_advanced_ranking(raw_results, query_type, context)
        else:
            ranked_results = raw_results
        
        # Format results
        formatted_results = []
        for result in ranked_results:
            doc = result["document"]
            formatted_results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity_score": result["similarity_score"],
                "relevance_score": result.get("relevance_score", result["similarity_score"]),
                "source": doc["metadata"]["source"],
                "notebook_name": doc["metadata"]["notebook_name"],
                "cell_type": doc["metadata"]["cell_type"]
            })
        
        return {
            "query": query_text,
            "type": query_type,
            "priority": priority,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "execution_status": "success"
        }
    
    def _apply_advanced_ranking(self, results: List[Dict], query_type: str, context: Dict) -> List[Dict]:
        """Apply advanced ranking with multiple factors."""
        for result in results:
            doc = result["document"]
            metadata = doc["metadata"]
            base_similarity = result["similarity_score"]
            
            ranking_factors = {
                "base_similarity": base_similarity,
                "quality_boost": 0,
                "context_match": 0,
                "type_alignment": 0,
                "chapter_preference": 0
            }
            
            # Quality boost based on content quality score
            quality_score = metadata.get("quality_score", 0.5)
            ranking_factors["quality_boost"] = quality_score * 0.2
            
            # Context matching with workflow stage
            workflow_detection = context.get("workflow_detection", {})
            primary_stage = workflow_detection.get("primary_stage", "")
            
            if self._matches_workflow_stage(doc, primary_stage):
                ranking_factors["context_match"] = 0.15
            
            # Query type alignment boost
            ranking_factors["type_alignment"] = self._calculate_type_alignment_boost(doc, query_type)
            
            # Chapter preference (some chapters are more valuable)
            chapter_num = metadata.get("notebook_metadata", {}).get("chapter", {}).get("number", 0)
            if chapter_num in [3, 5]:  # Pandas and ML chapters are highly valuable
                ranking_factors["chapter_preference"] = 0.1
            
            # Calculate final relevance score
            relevance_score = sum(ranking_factors.values())
            result["relevance_score"] = min(relevance_score, 1.0)
            result["ranking_factors"] = ranking_factors
        
        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results
    
    def _matches_workflow_stage(self, doc: Dict, stage: str) -> bool:
        """Check if document content matches the detected workflow stage."""
        if not stage or stage == "general_analysis":
            return False
        
        content_lower = doc["content"].lower()
        
        stage_keywords = {
            "data_acquisition": ["read_csv", "read_excel", "load", "import", "data", "file"],
            "data_exploration": ["describe", "info", "head", "tail", "explore", "summary", "shape"],
            "data_cleaning": ["fillna", "dropna", "clean", "preprocess", "missing", "duplicates"],
            "feature_engineering": ["feature", "encode", "scale", "transform", "engineer", "select"],
            "modeling": ["fit", "predict", "model", "train", "algorithm", "classifier", "regressor"],
            "visualization": ["plot", "chart", "graph", "visual", "matplotlib", "seaborn", "plotly"],
            "evaluation": ["score", "accuracy", "precision", "recall", "evaluate", "metrics", "performance"]
        }
        
        keywords = stage_keywords.get(stage, [])
        matches = sum(1 for kw in keywords if kw in content_lower)
        
        # Return True if at least 2 keywords match (stronger signal)
        return matches >= 2
    
    def _calculate_type_alignment_boost(self, doc: Dict, query_type: str) -> float:
        """Calculate relevance boost based on query type alignment."""
        metadata = doc["metadata"]
        content = doc["content"]
        
        boost = 0.0
        
        if query_type == "library_specific":
            # Boost code examples for library-specific queries
            if metadata["cell_type"] == "code" and metadata.get("has_code_examples"):
                boost += 0.15
            # Additional boost for import statements
            if "import " in content:
                boost += 0.05
                
        elif query_type == "enhanced_user_query":
            # Boost tutorial and example content for user queries
            semantic_tags = metadata.get("semantic_tags", [])
            if "tutorial" in semantic_tags:
                boost += 0.1
            if "example" in semantic_tags:
                boost += 0.08
                
        elif query_type == "stage_specific":
            # Boost explanatory content for stage-specific queries
            if metadata.get("has_explanations"):
                boost += 0.1
            if metadata["cell_type"] == "markdown":
                boost += 0.05
                
        elif query_type == "theme_based":
            # Boost content with rich semantic information
            semantic_tags = metadata.get("semantic_tags", [])
            boost += min(len(semantic_tags) * 0.02, 0.08)
        
        return boost
    
    def _generate_fallback_queries(self, user_query: str) -> List[Dict]:
        """Generate fallback queries when notebook analysis is not available."""
        # Clean the user query
        clean_query = user_query.replace(".ipynb", "").replace("notebook:", "").strip()
        
        # Generate basic strategic queries
        fallback_queries = []
        
        # Primary query enhancement
        if clean_query and len(clean_query) > 3:
            fallback_queries.append({
                "query": f"{clean_query} python tutorial examples",
                "type": "enhanced_user_query",
                "priority": "high"
            })
        
        # Detect common data science terms and create targeted queries
        query_lower = clean_query.lower()
        
        if any(lib in query_lower for lib in ["pandas", "dataframe"]):
            fallback_queries.append({
                "query": "pandas data manipulation examples advanced techniques",
                "type": "library_specific",
                "priority": "high"
            })
        
        if any(term in query_lower for term in ["visualization", "plot", "chart"]):
            fallback_queries.append({
                "query": "matplotlib seaborn visualization examples tutorial",
                "type": "library_specific",
                "priority": "medium"
            })
        
        if any(term in query_lower for term in ["machine learning", "model", "ml"]):
            fallback_queries.append({
                "query": "scikit learn machine learning workflow examples",
                "type": "library_specific",
                "priority": "high"
            })
        
        # Add generic fallback queries if we don't have enough
        if len(fallback_queries) < 3:
            fallback_queries.extend([
                {
                    "query": "data science workflow best practices python",
                    "type": "fallback",
                    "priority": "medium"
                },
                {
                    "query": "pandas numpy data analysis tutorial",
                    "type": "fallback",
                    "priority": "low"
                }
            ])
        
        return fallback_queries[:config.max_search_queries]  # Respect config limit
    
    def _assess_search_quality(self, search_results: List[Dict]) -> Dict[str, Any]:
        """Assess the overall quality of search results."""
        if not search_results:
            return {"quality_score": 0.0, "assessment": "no_results"}
        
        total_relevance = sum(result.get("relevance_score", 0) for result in search_results)
        avg_relevance = total_relevance / len(search_results)
        
        high_quality_count = len([r for r in search_results if r.get("relevance_score", 0) > 0.7])
        
        quality_assessment = {
            "quality_score": avg_relevance,
            "high_quality_results": high_quality_count,
            "total_results": len(search_results),
            "quality_ratio": high_quality_count / len(search_results),
            "assessment": "excellent" if avg_relevance > 0.8 else "good" if avg_relevance > 0.6 else "fair" if avg_relevance > 0.4 else "poor"
        }
        
        return quality_assessment
    
    def _log_search_performance(self, search_results: List[Dict]):
        """Log detailed search performance metrics."""
        successful_searches = len([r for r in search_results if r.get("execution_status") == "success"])
        total_results = sum(len(r.get("results", [])) for r in search_results)
        
        # Calculate average relevance scores
        all_relevance_scores = []
        for search in search_results:
            for result in search.get("results", []):
                all_relevance_scores.append(result.get("relevance_score", 0))
        
        avg_relevance = sum(all_relevance_scores) / len(all_relevance_scores) if all_relevance_scores else 0
        
        logger.info(f"📈 Search Performance Summary:")
        logger.info(f"   Success Rate: {successful_searches}/{len(search_results)} searches")
        logger.info(f"   Total Results: {total_results} documents retrieved")
        logger.info(f"   Average Relevance: {avg_relevance:.3f}")
        logger.info(f"   High Quality Results: {len([s for s in all_relevance_scores if s > 0.7])}")
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: List[Dict]) -> str:
        """Store intelligent RAG results and performance metrics."""
        shared["intelligent_rag_results"] = exec_res
        shared["rag_method"] = "pocketflow_intelligent"
        shared["total_successful_searches"] = len([r for r in exec_res if r.get("execution_status") == "success"])
        
        # Add performance metrics
        if exec_res:
            all_results = []
            for search in exec_res:
                all_results.extend(search.get("results", []))
            
            shared["rag_performance"] = self._assess_search_quality(all_results)
        
        # Log performance details
        self._log_search_performance(exec_res)
        
        logger.info("🧠 Intelligent PocketFlow RAG completed successfully")
        logger.info(f"   Success Rate: {shared['total_successful_searches']}/{len(exec_res)} searches")
        
        return "default"
