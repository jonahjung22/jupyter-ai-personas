"""
utils/notebook_utils.py - Notebook content extraction and analysis
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_notebook_content(notebook_path: str) -> List[Dict[str, Any]]:
    """Extract content from Jupyter notebook with rich metadata."""
    try:
        import nbformat
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        documents = []
        notebook_name = Path(notebook_path).stem
        
        # Extract notebook-level metadata
        nb_metadata = analyze_notebook_structure(nb, notebook_name)
        
        for cell_idx, cell in enumerate(nb.cells):
            content = cell.get('source', '').strip()
            if not content or len(content) < 20:
                continue
            
            doc = {
                "content": content,
                "metadata": {
                    "source": str(notebook_path),
                    "notebook_name": notebook_name,
                    "cell_index": cell_idx,
                    "cell_type": cell.cell_type,
                    "content_length": len(content),
                    "line_count": len(content.split('\n')),
                    "has_code_examples": detect_code_examples(content),
                    "has_explanations": detect_explanations(content),
                    "technical_depth": assess_technical_depth(content),
                    "semantic_tags": extract_semantic_tags(content),
                    "notebook_metadata": nb_metadata,
                    "extraction_timestamp": datetime.now().isoformat()
                }
            }
            
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to extract notebook content from {notebook_path}: {e}")
        return []

def analyze_notebook_structure(nb, notebook_name: str) -> Dict[str, Any]:
    """Analyze notebook structure and extract metadata."""
    return {
        "total_cells": len(nb.cells),
        "code_cells": len([c for c in nb.cells if c.cell_type == "code"]),
        "markdown_cells": len([c for c in nb.cells if c.cell_type == "markdown"]),
        "chapter_info": extract_chapter_info(notebook_name),
        "primary_libraries": extract_notebook_libraries(nb),
        "complexity_level": assess_notebook_complexity(nb)
    }

def extract_chapter_info(notebook_name: str) -> Dict[str, Any]:
    """Extract chapter information from notebook name."""
    chapter_mapping = {
        "01": {"number": 1, "title": "IPython: Beyond Normal Python", "focus": "interactive_python"},
        "02": {"number": 2, "title": "NumPy", "focus": "numerical_computing"},
        "03": {"number": 3, "title": "Pandas", "focus": "data_manipulation"},
        "04": {"number": 4, "title": "Matplotlib", "focus": "visualization"},
        "05": {"number": 5, "title": "Machine Learning", "focus": "scikit_learn"}
    }
    
    for prefix, info in chapter_mapping.items():
        if notebook_name.startswith(prefix):
            return info
    
    return {"number": 0, "title": "General", "focus": "general"}

def extract_notebook_libraries(nb) -> List[str]:
    """Extract libraries used in notebook."""
    libraries = set()
    common_libs = ["numpy", "pandas", "matplotlib", "seaborn", "sklearn", "scipy"]
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            content = cell.get('source', '').lower()
            for lib in common_libs:
                if lib in content:
                    libraries.add(lib)
    
    return list(libraries)

def assess_notebook_complexity(nb) -> str:
    """Assess overall notebook complexity."""
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    if not code_cells:
        return "basic"
    
    complexity_indicators = 0
    for cell in code_cells:
        content = cell.get('source', '')
        complexity_indicators += len([line for line in content.split('\n') 
                                    if any(keyword in line for keyword in ['def ', 'class ', 'for ', 'if '])])
    
    avg_complexity = complexity_indicators / len(code_cells)
    
    if avg_complexity > 3:
        return "advanced"
    elif avg_complexity > 1:
        return "intermediate"
    else:
        return "basic"

def detect_code_examples(content: str) -> bool:
    """Detect if content contains code examples."""
    import re
    code_patterns = [
        r'```python', r'>>> ', r'import \w+', r'def \w+\(',
        r'\w+\.\w+\(', r'= \w+\('
    ]
    return any(re.search(pattern, content) for pattern in code_patterns)

def detect_explanations(content: str) -> bool:
    """Detect if content contains explanatory text."""
    explanation_indicators = [
        "this shows", "we can see", "notice that", "for example",
        "let's", "we'll", "here we", "this demonstrates"
    ]
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in explanation_indicators)

def assess_technical_depth(content: str) -> str:
    """Assess technical depth of content."""
    content_lower = content.lower()
    
    advanced_indicators = [
        "optimization", "performance", "algorithm", "complexity",
        "advanced", "sophisticated", "efficient", "scalable"
    ]
    
    intermediate_indicators = [
        "function", "method", "parameter", "attribute", "module",
        "import", "class", "object"
    ]
    
    if any(indicator in content_lower for indicator in advanced_indicators):
        return "advanced"
    elif any(indicator in content_lower for indicator in intermediate_indicators):
        return "intermediate"
    else:
        return "beginner"

def extract_semantic_tags(content: str) -> List[str]:
    """Extract semantic tags from content."""
    content_lower = content.lower()
    tags = []
    
    tag_patterns = {
        "tutorial": ["tutorial", "guide", "walkthrough", "step-by-step"],
        "example": ["example", "demo", "illustration", "sample"],
        "reference": ["reference", "documentation", "api", "specification"],
        "best_practices": ["best practice", "recommendation", "tip", "advice"],
        "troubleshooting": ["error", "problem", "issue", "debug", "fix"]
    }
    
    for tag, patterns in tag_patterns.items():
        if any(pattern in content_lower for pattern in patterns):
            tags.append(tag)
    
    return tags