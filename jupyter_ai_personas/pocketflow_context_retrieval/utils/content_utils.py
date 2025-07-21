import logging
from typing import List, Dict, Any
from ..config import config
from .notebook_utils import detect_code_examples, detect_explanations, assess_technical_depth, extract_semantic_tags

logger = logging.getLogger(__name__)

def chunk_text_intelligently(content: str, cell_type: str = "markdown") -> List[str]:
    """Intelligently chunk text based on content type."""
    if cell_type == "code":
        return chunk_code_content(content)
    else:
        return chunk_text_content(content)

def chunk_code_content(content: str) -> List[str]:
    """Chunk code content preserving logical structure."""
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line)
        
        # Check for natural breakpoints
        is_breakpoint = (
            line.strip() == "" or
            line.strip().startswith('#') or
            line.startswith('def ') or
            line.startswith('class ') or
            'import ' in line
        )
        
        # Decide whether to start new chunk
        if ((current_size + line_size > config.chunk_size and is_breakpoint and current_chunk) or 
            current_size > config.chunk_size * 1.2):
            
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return [chunk for chunk in chunks if len(chunk.strip()) >= config.min_chunk_size]

def chunk_text_content(content: str) -> List[str]:
    """Chunk text content preserving paragraph structure."""
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        if current_size + para_size > config.chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return [chunk for chunk in chunks if len(chunk.strip()) >= config.min_chunk_size]

def calculate_content_quality_score(content: str, metadata: Dict[str, Any] = None) -> float:
    """Calculate quality score for content."""
    if not content:
        return 0.0
    
    score = 0.0
    
    # Length factor (sweet spot around 100-1000 chars)
    length = len(content)
    if 100 <= length <= 1000:
        score += 0.3
    elif 50 <= length < 100 or 1000 < length <= 2000:
        score += 0.2
    
    # Code and explanation balance
    has_code = detect_code_examples(content)
    has_explanation = detect_explanations(content)
    
    if has_code and has_explanation:
        score += 0.4
    elif has_code or has_explanation:
        score += 0.2
    
    # Technical depth
    depth = assess_technical_depth(content)
    if depth == "intermediate":
        score += 0.2
    elif depth == "advanced":
        score += 0.1
    
    # Semantic richness
    tags = extract_semantic_tags(content)
    score += min(len(tags) * 0.1, 0.2)
    
    return min(score, 1.0)

def filter_low_quality_content(documents: List[Dict]) -> List[Dict]:
    """Filter out low-quality documents."""
    filtered = []
    
    for doc in documents:
        content = doc["content"]
        
        # Skip very short content
        if len(content.strip()) < config.min_chunk_size:
            continue
        
        # Skip pure headers
        if content.strip().startswith('#') and '\n' not in content.strip():
            continue
        
        # Skip just imports
        lines = content.strip().split('\n')
        non_import_lines = [line for line in lines if not line.strip().startswith(('import ', 'from '))]
        if len(non_import_lines) <= 1:
            continue
        
        # Calculate quality score
        quality_score = calculate_content_quality_score(content, doc.get("metadata"))
        
        if quality_score >= config.quality_threshold:
            doc["metadata"]["quality_score"] = quality_score
            filtered.append(doc)
    
    return filtered