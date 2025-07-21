import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from pocketflow import Node
from ..utils.notebook_utils import extract_notebook_content
from ..utils.content_utils import calculate_content_quality_score
from ..config import config

# Import the proven NotebookReaderTool
try:
    from ...context_retrieval_persona.file_reader_tool import NotebookReaderTool
    NOTEBOOK_READER_AVAILABLE = True
except ImportError:
    NOTEBOOK_READER_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedNotebookAnalysisNode(Node):
    """Advanced notebook analysis node with comprehensive intelligence using NotebookReaderTool."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_strategies = [
            "content_extraction",
            "semantic_analysis", 
            "pattern_recognition",
            "complexity_assessment",
            "recommendation_generation"
        ]
        
        # Initialize the proven NotebookReaderTool
        self.notebook_reader = NotebookReaderTool() if NOTEBOOK_READER_AVAILABLE else None
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare advanced notebook analysis."""
        user_query = shared.get("user_query", "")
        notebook_path = shared.get("notebook_path") or self._extract_notebook_path(user_query)
        
        return {
            "user_query": user_query,
            "notebook_path": notebook_path,
            "analysis_strategies": self.analysis_strategies,
            "enable_deep_analysis": config.enable_deep_analysis
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive notebook analysis."""
        notebook_path = prep_res["notebook_path"]
        
        if not notebook_path or not Path(notebook_path).exists():
            return self._create_fallback_analysis(prep_res["user_query"])
        
        try:
            # Use NotebookReaderTool for comprehensive analysis if available
            if self.notebook_reader:
                logger.info("📖 Using proven NotebookReaderTool for comprehensive analysis")
                notebook_content = self.notebook_reader.extract_rag_context(notebook_path)
                
                # Parse the comprehensive content and enhance with our analysis
                analysis = self._analyze_notebook_content_with_reader(notebook_content, prep_res["user_query"])
                analysis["notebook_reader_used"] = True
            else:
                # Fallback to original extraction method
                logger.info("📖 Using fallback notebook extraction")
                documents = extract_notebook_content(notebook_path)
                
                if not documents:
                    return self._create_fallback_analysis(prep_res["user_query"])
                
                # Perform multi-dimensional analysis
                analysis = {
                    "notebook_path": notebook_path,
                    "extraction_successful": True,
                    "content_analysis": self._analyze_content_structure(documents),
                    "semantic_analysis": self._perform_semantic_analysis(documents),
                    "workflow_detection": self._detect_workflow_patterns(documents),
                    "code_intelligence": self._analyze_code_patterns(documents),
                    "quality_assessment": self._assess_content_quality(documents),
                    "search_strategy": self._generate_search_strategy(documents, prep_res["user_query"]),
                    "recommendations": self._generate_recommendations(documents),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "notebook_reader_used": False
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Advanced notebook analysis failed: {e}")
            return self._create_fallback_analysis(prep_res["user_query"], error=str(e))
    
    def _analyze_content_structure(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze the structure and composition of notebook content."""
        total_content = sum(len(doc["content"]) for doc in documents)
        
        return {
            "total_documents": len(documents),
            "code_cells": len([d for d in documents if d["metadata"]["cell_type"] == "code"]),
            "markdown_cells": len([d for d in documents if d["metadata"]["cell_type"] == "markdown"]),
            "total_content_length": total_content,
            "average_cell_length": total_content / len(documents) if documents else 0,
            "complexity_distribution": self._analyze_complexity_distribution(documents)
        }
    
    def _perform_semantic_analysis(self, documents: List[Dict]) -> Dict[str, Any]:
        """Perform semantic analysis on notebook content."""
        all_content = " ".join([doc["content"] for doc in documents])
        
        return {
            "detected_libraries": self._extract_libraries_advanced(all_content),
            "analysis_themes": self._extract_content_themes(all_content),
            "technical_concepts": self._identify_technical_concepts(all_content),
            "domain_indicators": self._detect_domain_focus(all_content)
        }
    
    def _detect_workflow_patterns(self, documents: List[Dict]) -> Dict[str, Any]:
        """Detect data science workflow patterns in the notebook."""
        all_content = " ".join([doc["content"] for doc in documents]).lower()
        
        workflow_stages = {
            "data_acquisition": {
                "patterns": ["read_csv", "read_excel", "load_data", "import.*data"],
                "weight": 3.0
            },
            "data_exploration": {
                "patterns": ["describe()", "info()", "head()", "shape", "value_counts"],
                "weight": 2.5
            },
            "data_cleaning": {
                "patterns": ["fillna", "dropna", "drop_duplicates", "clean"],
                "weight": 2.0
            },
            "feature_engineering": {
                "patterns": ["feature", "encode", "scale", "transform"],
                "weight": 2.0
            },
            "modeling": {
                "patterns": ["fit(", "predict(", "model", "train"],
                "weight": 3.0
            },
            "visualization": {
                "patterns": ["plot(", "plt.", "sns.", "chart"],
                "weight": 1.5
            },
            "evaluation": {
                "patterns": ["score(", "accuracy", "precision", "evaluate"],
                "weight": 2.5
            }
        }
        
        stage_scores = {}
        for stage, stage_config in workflow_stages.items():
            import re
            score = 0
            for pattern in stage_config["patterns"]:
                matches = len(re.findall(pattern, all_content))
                score += matches * stage_config["weight"]
            stage_scores[stage] = score
        
        # Determine primary stage
        primary_stage = max(stage_scores.keys(), key=lambda k: stage_scores[k]) if any(stage_scores.values()) else "general_analysis"
        
        # Get progression
        significant_stages = [(stage, score) for stage, score in stage_scores.items() if score > 0]
        significant_stages.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "primary_stage": primary_stage,
            "stage_scores": stage_scores,
            "workflow_progression": [stage for stage, _ in significant_stages[:3]],
            "confidence": min(stage_scores.get(primary_stage, 0) / 10, 1.0)
        }
    
    def _analyze_code_patterns(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze code patterns and programming practices."""
        code_docs = [d for d in documents if d["metadata"]["cell_type"] == "code"]
        all_code = " ".join([doc["content"] for doc in code_docs])
        
        if not all_code:
            return {"no_code_detected": True}
        
        import re
        
        patterns = {
            "function_definitions": len(re.findall(r'def\s+\w+', all_code)),
            "class_definitions": len(re.findall(r'class\s+\w+', all_code)),
            "import_statements": len(re.findall(r'import\s+\w+|from\s+\w+\s+import', all_code)),
            "method_calls": len(re.findall(r'\.\w+\(', all_code)),
            "list_comprehensions": len(re.findall(r'\[.*for.*in.*\]', all_code)),
            "error_handling": len(re.findall(r'try:|except:|finally:', all_code)),
            "documentation": len(re.findall(r'""".*?"""|#.*', all_code, re.DOTALL))
        }
        
        # Calculate code quality indicators
        total_lines = len(all_code.split('\n'))
        complexity_score = (
            patterns["function_definitions"] * 2 +
            patterns["class_definitions"] * 3 +
            patterns["error_handling"] * 2
        ) / max(total_lines, 1) * 100
        
        return {
            "code_patterns": patterns,
            "complexity_score": min(complexity_score, 10.0),
            "code_quality_level": "high" if complexity_score > 5 else "medium" if complexity_score > 2 else "basic",
            "total_code_lines": total_lines
        }
    
    def _assess_content_quality(self, documents: List[Dict]) -> Dict[str, Any]:
        """Assess overall quality of notebook content."""
        quality_scores = []
        
        for doc in documents:
            score = calculate_content_quality_score(doc["content"], doc["metadata"])
            quality_scores.append(score)
        
        if not quality_scores:
            return {"quality_assessment_failed": True}
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        high_quality_count = len([s for s in quality_scores if s > 0.7])
        
        return {
            "average_quality_score": avg_quality,
            "quality_distribution": {
                "high_quality": high_quality_count,
                "medium_quality": len([s for s in quality_scores if 0.4 <= s <= 0.7]),
                "low_quality": len([s for s in quality_scores if s < 0.4])
            },
            "overall_quality_level": "high" if avg_quality > 0.7 else "medium" if avg_quality > 0.4 else "low"
        }
    
    def _generate_search_strategy(self, documents: List[Dict], user_query: str) -> Dict[str, Any]:
        """Generate intelligent search strategy based on analysis."""
        # Extract key information from analysis
        semantic_analysis = self._perform_semantic_analysis(documents)
        workflow_detection = self._detect_workflow_patterns(documents)
        
        libraries = [lib["name"] for lib in semantic_analysis.get("detected_libraries", [])]
        primary_stage = workflow_detection.get("primary_stage", "general")
        themes = semantic_analysis.get("analysis_themes", [])
        
        # Generate strategic search queries
        search_queries = []
        
        # 1. User query enhanced with context
        if user_query and len(user_query.strip()) > 5:
            clean_query = self._clean_user_query(user_query)
            if clean_query:
                search_queries.append({
                    "query": f"{clean_query} {libraries[0] if libraries else 'python'} examples",
                    "type": "enhanced_user_query",
                    "priority": "high"
                })
        
        # 2. Stage-specific queries
        if primary_stage != "general":
            search_queries.append({
                "query": f"{primary_stage.replace('_', ' ')} best practices tutorial",
                "type": "stage_specific",
                "priority": "high",
                "stage": primary_stage
            })
        
        # 3. Library-specific queries
        for lib in libraries[:2]:  # Top 2 libraries
            search_queries.append({
                "query": f"{lib} advanced techniques {primary_stage.replace('_', ' ')}",
                "type": "library_specific",
                "priority": "medium",
                "library": lib
            })
        
        # 4. Theme-based queries
        for theme in themes[:2]:  # Top 2 themes
            search_queries.append({
                "query": f"{theme} {libraries[0] if libraries else 'python'} workflow",
                "type": "theme_based",
                "priority": "low",
                "theme": theme
            })
        
        return {
            "strategy_type": "intelligent_multi_query",
            "total_queries": len(search_queries),
            "queries": search_queries[:config.max_search_queries],
            "primary_focus": primary_stage,
            "context_libraries": libraries[:3]
        }
    
    def _generate_recommendations(self, documents: List[Dict]) -> List[str]:
        """Generate specific recommendations based on analysis."""
        recommendations = []
        
        # Analyze code patterns for recommendations
        code_analysis = self._analyze_code_patterns(documents)
        if not code_analysis.get("no_code_detected"):
            patterns = code_analysis.get("code_patterns", {})
            
            if patterns.get("function_definitions", 0) == 0:
                recommendations.append("Consider breaking code into reusable functions for better organization")
            
            if patterns.get("error_handling", 0) == 0:
                recommendations.append("Add error handling (try/except blocks) for more robust code")
            
            if patterns.get("documentation", 0) < 5:
                recommendations.append("Add more comments and docstrings to improve code documentation")
        
        # Quality-based recommendations
        quality_assessment = self._assess_content_quality(documents)
        if quality_assessment.get("average_quality_score", 0) < 0.5:
            recommendations.append("Consider adding more explanatory text to improve content quality")
        
        # Workflow-based recommendations
        workflow_detection = self._detect_workflow_patterns(documents)
        primary_stage = workflow_detection.get("primary_stage")
        
        if primary_stage == "data_exploration":
            recommendations.append("Add comprehensive data profiling and statistical analysis")
        elif primary_stage == "modeling":
            recommendations.append("Implement proper model evaluation and cross-validation techniques")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _extract_libraries_advanced(self, content: str) -> List[Dict[str, Any]]:
        """Advanced library extraction with usage patterns."""
        import re
        
        library_patterns = {
            'pandas': [r'import pandas', r'pd\.', r'DataFrame', r'Series'],
            'numpy': [r'import numpy', r'np\.', r'array\(', r'ndarray'],
            'matplotlib': [r'import matplotlib', r'plt\.', r'pyplot'],
            'seaborn': [r'import seaborn', r'sns\.'],
            'sklearn': [r'from sklearn', r'import sklearn'],
            'scipy': [r'import scipy', r'from scipy']
        }
        
        detected_libraries = []
        content_lower = content.lower()
        
        for lib_name, patterns in library_patterns.items():
            usage_count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                usage_count += len(matches)
            
            if usage_count > 0:
                detected_libraries.append({
                    "name": lib_name,
                    "usage_count": usage_count,
                    "confidence": min(usage_count / 5, 1.0)
                })
        
        return sorted(detected_libraries, key=lambda x: x["confidence"], reverse=True)
    
    def _extract_content_themes(self, content: str) -> List[str]:
        """Extract high-level content themes."""
        content_lower = content.lower()
        themes = []
        
        theme_indicators = {
            "machine_learning": ["model", "train", "predict", "algorithm", "classification", "regression"],
            "data_visualization": ["plot", "chart", "graph", "visualization", "matplotlib", "seaborn"],
            "statistical_analysis": ["statistics", "correlation", "hypothesis", "distribution", "probability"],
            "data_processing": ["clean", "transform", "process", "prepare", "preprocess"],
            "exploratory_analysis": ["explore", "eda", "analyze", "investigate", "discover"]
        }
        
        for theme, indicators in theme_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                themes.append(theme)
        
        return themes
    
    def _identify_technical_concepts(self, content: str) -> List[str]:
        """Identify specific technical concepts mentioned."""
        content_lower = content.lower()
        concepts = []
        
        concept_patterns = {
            "time_series": ["datetime", "timeseries", "time series", "temporal"],
            "natural_language_processing": ["nlp", "text processing", "tokenization"],
            "computer_vision": ["image", "cv2", "opencv", "vision"],
            "deep_learning": ["neural network", "deep learning", "tensorflow", "pytorch"],
            "statistical_modeling": ["statistical model", "hypothesis testing", "p-value"]
        }
        
        for concept, patterns in concept_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                concepts.append(concept)
        
        return concepts
    
    def _detect_domain_focus(self, content: str) -> List[str]:
        """Detect domain-specific focus areas."""
        content_lower = content.lower()
        domains = []
        
        domain_indicators = {
            "finance": ["stock", "financial", "trading", "investment"],
            "healthcare": ["medical", "patient", "clinical", "health"],
            "marketing": ["customer", "marketing", "sales", "advertising"],
            "science": ["research", "experiment", "scientific", "analysis"]
        }
        
        for domain, indicators in domain_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                domains.append(domain)
        
        return domains
    
    def _analyze_complexity_distribution(self, documents: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of complexity across documents."""
        complexity_levels = {"low": 0, "medium": 0, "high": 0}
        
        for doc in documents:
            technical_depth = doc["metadata"].get("technical_depth", "beginner")
            
            if technical_depth == "beginner":
                complexity_levels["low"] += 1
            elif technical_depth == "intermediate":
                complexity_levels["medium"] += 1
            else:
                complexity_levels["high"] += 1
        
        return complexity_levels
    
    def _clean_user_query(self, query: str) -> str:
        """Clean user query for search purposes."""
        import re
        # Remove file paths and special characters
        cleaned = re.sub(r'/[^\s]*\.ipynb', '', query)
        cleaned = re.sub(r'@\w+', '', cleaned)
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    def _extract_notebook_path(self, query: str) -> Optional[str]:
        """Extract notebook path from user query."""
        import re
        
        # Pattern 1: notebook: path
        notebook_match = re.search(r'notebook:\s*([^\s]+\.ipynb)', query, re.IGNORECASE)
        if notebook_match:
            return notebook_match.group(1)
        
        # Pattern 2: Any .ipynb path
        ipynb_match = re.search(r'([^\s]+\.ipynb)', query)
        if ipynb_match:
            return ipynb_match.group(1)
        
        # Pattern 3: Default fallback
        fallback_path = "/Users/jujonahj/jupyter-ai-personas/jupyter_ai_personas/data_science_persona/test_context_retrieval.ipynb"
        if Path(fallback_path).exists():
            return fallback_path
        
        return None
    
    def _create_fallback_analysis(self, user_query: str, error: str = None) -> Dict[str, Any]:
        """Create fallback analysis when notebook processing fails."""
        return {
            "fallback_mode": True,
            "user_query": user_query,
            "error": error,
            "basic_analysis": self._analyze_user_query_for_context(user_query),
            "search_strategy": self._generate_fallback_search_strategy(user_query),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_user_query_for_context(self, query: str) -> Dict[str, Any]:
        """Analyze user query for context clues when notebook is unavailable."""
        query_lower = query.lower()
        
        # Detect mentioned libraries
        detected_libraries = []
        for lib in ["pandas", "numpy", "matplotlib", "seaborn", "sklearn", "scipy"]:
            if lib in query_lower:
                detected_libraries.append({"name": lib, "confidence": 0.8})
        
        # Detect task types
        tasks = []
        if any(word in query_lower for word in ["plot", "chart", "visualize"]):
            tasks.append("visualization")
        if any(word in query_lower for word in ["model", "predict", "train"]):
            tasks.append("modeling")
        if any(word in query_lower for word in ["clean", "preprocess"]):
            tasks.append("data_cleaning")
        
        return {
            "detected_libraries": detected_libraries,
            "detected_tasks": tasks,
            "query_complexity": "advanced" if len(query.split()) > 10 else "basic"
        }
    
    def _generate_fallback_search_strategy(self, user_query: str) -> Dict[str, Any]:
        """Generate basic search strategy from user query alone."""
        clean_query = self._clean_user_query(user_query)
        
        queries = [
            {
                "query": f"{clean_query} python tutorial",
                "type": "enhanced_user_query",
                "priority": "high"
            },
            {
                "query": "data science workflow best practices",
                "type": "fallback",
                "priority": "medium"
            },
            {
                "query": "pandas data analysis examples",
                "type": "fallback",
                "priority": "low"
            }
        ]
        
        return {
            "strategy_type": "fallback_search",
            "queries": queries,
            "total_queries": len(queries)
        }
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store advanced analysis results in shared state."""
        shared["advanced_notebook_analysis"] = exec_res
        shared["analysis_method"] = "pocketflow_advanced"
        shared["analysis_success"] = not exec_res.get("fallback_mode", False)
        
        if exec_res.get("fallback_mode"):
            logger.warning("📊 Notebook analysis completed in fallback mode")
        else:
            logger.info("📊 Advanced notebook analysis completed successfully")
            logger.info(f"   Primary stage: {exec_res.get('workflow_detection', {}).get('primary_stage', 'unknown')}")
            logger.info(f"   Libraries detected: {len(exec_res.get('semantic_analysis', {}).get('detected_libraries', []))}")
            logger.info(f"   Search queries generated: {len(exec_res.get('search_strategy', {}).get('queries', []))}")
        
        return "default"
    
    def _analyze_notebook_content_with_reader(self, notebook_content: str, user_query: str) -> Dict[str, Any]:
        """
        Analyze notebook content extracted by NotebookReaderTool.
        
        This method parses the comprehensive content from NotebookReaderTool
        and performs enhanced analysis using the proven extraction patterns.
        """
        try:
            # Parse the structured content from NotebookReaderTool
            lines = notebook_content.split('\n')
            
            # Extract basic info
            file_path = ""
            kernel_info = ""
            language = ""
            cell_count = 0
            
            # Parse header information
            for line in lines:
                if line.startswith("File: "):
                    file_path = line.replace("File: ", "").strip()
                elif line.startswith("Kernel: "):
                    kernel_info = line.replace("Kernel: ", "").strip()
                elif line.startswith("Language: "):
                    language = line.replace("Language: ", "").strip()
                elif "cells)" in line and "NOTEBOOK CONTENT" in line:
                    # Extract cell count from "=== NOTEBOOK CONTENT (X cells) ==="
                    import re
                    match = re.search(r'\((\d+) cells\)', line)
                    if match:
                        cell_count = int(match.group(1))
            
            # Extract detected libraries section
            libraries = []
            in_libraries_section = False
            for line in lines:
                if "=== DETECTED LIBRARIES ===" in line:
                    in_libraries_section = True
                    continue
                elif line.startswith("===") and in_libraries_section:
                    in_libraries_section = False
                elif in_libraries_section and line.startswith("- "):
                    libraries.append(line.replace("- ", "").strip())
            
            # Extract data science context
            ds_context = ""
            in_ds_section = False
            for line in lines:
                if "=== DATA SCIENCE CONTEXT ===" in line:
                    in_ds_section = True
                    continue
                elif line.startswith("===") and in_ds_section:
                    break
                elif in_ds_section:
                    ds_context += line + "\n"
            
            # Analyze workflow patterns from the comprehensive content
            workflow_stage = self._detect_workflow_from_content(notebook_content)
            
            # Enhanced analysis combining NotebookReaderTool data with our intelligence
            analysis = {
                "notebook_path": file_path,
                "extraction_successful": True,
                "notebook_reader_analysis": {
                    "kernel": kernel_info,
                    "language": language,
                    "cell_count": cell_count,
                    "detected_libraries": libraries,
                    "data_science_context": ds_context.strip()
                },
                "content_analysis": {
                    "total_cells": cell_count,
                    "has_comprehensive_extraction": True,
                    "library_count": len(libraries),
                    "content_richness": "high" if len(notebook_content) > 5000 else "medium"
                },
                "semantic_analysis": {
                    "detected_libraries": [{"name": lib, "usage": "detected"} for lib in libraries],
                    "analysis_themes": self._extract_themes_from_content(ds_context),
                    "complexity_level": self._assess_complexity_from_content(notebook_content)
                },
                "workflow_detection": {
                    "primary_stage": workflow_stage,
                    "confidence": 0.85,  # High confidence with comprehensive extraction
                    "detected_patterns": self._detect_patterns_from_content(notebook_content)
                },
                "code_intelligence": {
                    "code_quality_level": self._assess_code_quality_from_content(notebook_content),
                    "complexity_score": self._calculate_complexity_from_content(notebook_content),
                    "optimization_opportunities": self._detect_optimization_opportunities(notebook_content)
                },
                "search_strategy": self._generate_enhanced_search_strategy(notebook_content, user_query),
                "recommendations": self._generate_enhanced_recommendations(notebook_content, ds_context),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Enhanced analysis with NotebookReaderTool: {cell_count} cells, {len(libraries)} libraries")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ NotebookReaderTool analysis failed: {e}")
            # Fallback to basic analysis
            return self._create_fallback_analysis(user_query, error=str(e))
    
    def _detect_workflow_from_content(self, content: str) -> str:
        """Detect workflow stage from comprehensive notebook content."""
        content_lower = content.lower()
        
        # Enhanced pattern matching using the rich content from NotebookReaderTool
        if any(pattern in content_lower for pattern in ["pd.read", "load_data", "read_csv", "read_json"]):
            return "data_loading"
        elif any(pattern in content_lower for pattern in [".describe()", ".info()", ".head()", "exploratory"]):
            return "data_exploration"  
        elif any(pattern in content_lower for pattern in ["dropna", "fillna", "preprocessing", "clean"]):
            return "data_preprocessing"
        elif any(pattern in content_lower for pattern in ["plt.", "seaborn", "plot", "visualization"]):
            return "visualization"
        elif any(pattern in content_lower for pattern in ["fit(", "model", "sklearn", "machine learning"]):
            return "modeling"
        else:
            return "general_analysis"
    
    def _extract_themes_from_content(self, ds_context: str) -> List[str]:
        """Extract analysis themes from data science context."""
        themes = []
        context_lower = ds_context.lower()
        
        theme_patterns = {
            "data_manipulation": ["dataframe", "pandas", "merge", "join"],
            "statistical_analysis": ["statistics", "correlation", "distribution"],
            "machine_learning": ["model", "fit", "predict", "sklearn"],
            "data_visualization": ["plot", "chart", "graph", "visualization"],
            "time_series": ["datetime", "time", "temporal"]
        }
        
        for theme, patterns in theme_patterns.items():
            if any(pattern in context_lower for pattern in patterns):
                themes.append(theme)
        
        return themes or ["general_analysis"]
    
    def _assess_complexity_from_content(self, content: str) -> str:
        """Assess complexity level from notebook content."""
        content_lines = len(content.split('\n'))
        library_count = content.lower().count('import')
        
        if content_lines > 1000 and library_count > 10:
            return "advanced"
        elif content_lines > 500 and library_count > 5:
            return "intermediate"  
        else:
            return "beginner"
    
    def _detect_patterns_from_content(self, content: str) -> List[str]:
        """Detect workflow patterns from notebook content."""
        patterns = []
        content_lower = content.lower()
        
        if "import" in content_lower:
            patterns.append("library_usage")
        if any(pattern in content_lower for pattern in ["function", "def ", "class "]):
            patterns.append("code_organization")
        if any(pattern in content_lower for pattern in ["for ", "while ", "if "]):
            patterns.append("control_structures")
        if "error:" in content_lower:
            patterns.append("error_handling_needed")
            
        return patterns
    
    def _assess_code_quality_from_content(self, content: str) -> str:
        """Assess code quality from comprehensive content."""
        # Look for quality indicators in the extracted content
        has_comments = "##" in content or "#" in content
        has_functions = "def " in content
        has_error_handling = "try:" in content or "except:" in content
        
        quality_score = 0
        if has_comments:
            quality_score += 1
        if has_functions:
            quality_score += 1  
        if has_error_handling:
            quality_score += 1
            
        if quality_score >= 2:
            return "good"
        elif quality_score == 1:
            return "moderate"
        else:
            return "needs_improvement"
    
    def _calculate_complexity_from_content(self, content: str) -> float:
        """Calculate complexity score from content."""
        # Simple complexity calculation based on content richness
        lines = len(content.split('\n'))
        imports = content.lower().count('import')
        functions = content.lower().count('def ')
        
        # Normalize to 0-10 scale
        complexity = min(10.0, (lines / 100) + (imports * 0.5) + (functions * 0.3))
        return round(complexity, 1)
    
    def _detect_optimization_opportunities(self, content: str) -> List[str]:
        """Detect optimization opportunities from notebook content."""
        opportunities = []
        content_lower = content.lower()
        
        if "for " in content_lower and "pandas" in content_lower:
            opportunities.append("Consider vectorization instead of loops with pandas")
        if ".iterrows()" in content_lower:
            opportunities.append("Replace .iterrows() with vectorized operations")
        if "plt.show()" in content_lower:
            opportunities.append("Consider batch visualization for better performance")
        if content_lower.count("import") > 15:
            opportunities.append("Review import statements for optimization")
            
        return opportunities
    
    def _generate_enhanced_search_strategy(self, content: str, user_query: str) -> Dict[str, Any]:
        """Generate enhanced search strategy using NotebookReaderTool content."""
        # Extract libraries and themes for targeted searches
        libraries = []
        for line in content.split('\n'):
            if line.startswith("- ") and any(lib in line.lower() for lib in ["pandas", "numpy", "matplotlib", "sklearn"]):
                lib_name = line.replace("- ", "").split()[0].replace("import", "").strip()
                libraries.append(lib_name)
        
        # Generate intelligent queries
        queries = [
            {"query": user_query, "type": "user_intent", "priority": "high"}
        ]
        
        # Add library-specific queries  
        for lib in libraries[:3]:  # Top 3 libraries
            queries.append({
                "query": f"{lib} best practices optimization",
                "type": "library_specific", 
                "priority": "medium"
            })
        
        # Add workflow-specific queries
        workflow = self._detect_workflow_from_content(content)
        if workflow != "general_analysis":
            queries.append({
                "query": f"{workflow.replace('_', ' ')} techniques handbook",
                "type": "workflow_specific",
                "priority": "medium"  
            })
        
        return {
            "queries": queries,
            "total_queries": len(queries),
            "strategy": "enhanced_notebook_reader",
            "confidence": 0.9
        }
    
    def _generate_enhanced_recommendations(self, content: str, ds_context: str) -> List[str]:
        """Generate enhanced recommendations using comprehensive analysis."""
        recommendations = []
        
        # Based on detected libraries and patterns
        if "pandas" in content.lower():
            recommendations.append("Optimize pandas operations using vectorization")
        if "matplotlib" in content.lower():
            recommendations.append("Enhance visualizations with professional styling")
        if "sklearn" in content.lower():
            recommendations.append("Implement proper model evaluation and validation")
        
        # Based on data science context
        if "data loading" in ds_context.lower():
            recommendations.append("Consider data validation and error handling")
        if "visualization" in ds_context.lower():
            recommendations.append("Add interactive elements to visualizations")
        
        # Quality improvements
        if "error:" in content.lower():
            recommendations.append("Address errors and implement proper error handling")
        
        return recommendations or ["Apply general data science best practices"]

