"""
PocketFlow Nodes for Data Science Analysis

Three specialized nodes following PocketFlow prep/exec/post pattern:
1. MarkdownReaderNode - Reads repo_context.md 
2. NotebookReaderNode - Reads notebook content based on user query
3. ResponseGeneratorNode - Generates analysis and code recommendations
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .pocketflow import Node
from .file_reader_tool import NotebookReaderTool

logger = logging.getLogger(__name__)


class MarkdownReaderNode(Node):
    """
    Node 1: Read repo_context.md from current directory
    
    Looks for "repo_context.md" in the current working directory
    and loads its content to provide project context.
    """
    
    def prep(self, shared):
        """Prepare to read markdown file"""
        return {
            "working_dir": Path.cwd(),
            "markdown_file": "repo_context.md"
        }
    
    def exec(self, prep_res):
        """Execute markdown file reading"""
        try:
            markdown_path = prep_res["working_dir"] / prep_res["markdown_file"]
            
            if not markdown_path.exists():
                return {
                    "success": False,
                    "error": f"repo_context.md not found in {prep_res['working_dir']}",
                    "markdown_content": ""
                }
            
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            logger.info(f"✅ Read repo_context.md ({len(markdown_content)} characters)")
            
            return {
                "success": True,
                "markdown_content": markdown_content,
                "markdown_path": str(markdown_path)
            }
            
        except Exception as e:
            logger.error(f"❌ Error reading repo_context.md: {e}")
            return {
                "success": False,
                "error": str(e),
                "markdown_content": ""
            }
    
    def post(self, shared, prep_res, exec_res):
        """Store markdown content in shared state"""
        shared["markdown_content"] = exec_res["markdown_content"]
        shared["markdown_success"] = exec_res["success"]
        shared["markdown_path"] = exec_res.get("markdown_path", "")
        return "default"


class NotebookReaderNode(Node):
    """
    Node 2: Read notebook content based on user query
    
    Analyzes user prompt to extract notebook path and reads the notebook content
    using the NotebookReaderTool. If no path specified, looks for .ipynb files.
    """
    
    def __init__(self):
        super().__init__()
        self.notebook_tool = NotebookReaderTool()
    
    def prep(self, shared):
        """Prepare to read notebook"""
        return {
            "user_query": shared.get("user_query", ""),
            "working_dir": Path.cwd()
        }
    
    def exec(self, prep_res):
        """Execute notebook reading"""
        try:
            user_query = prep_res["user_query"]
            working_dir = prep_res["working_dir"]
            
            # Extract notebook path from user query
            notebook_path = self._extract_notebook_path(user_query, working_dir)
            
            if not notebook_path:
                return {
                    "success": False,
                    "error": "No notebook path found in query or directory",
                    "notebook_content": "",
                    "notebook_path": "",
                    "clean_query": user_query
                }
            
            # Read notebook content using NotebookReaderTool
            notebook_content = self.notebook_tool.extract_rag_context(str(notebook_path))
            
            # Check if reading was successful
            if notebook_content.startswith("Error:"):
                return {
                    "success": False,
                    "error": notebook_content,
                    "notebook_content": "",
                    "notebook_path": str(notebook_path),
                    "clean_query": self._remove_notebook_path_from_query(user_query)
                }
            
            logger.info(f"✅ Read notebook: {notebook_path} ({len(notebook_content)} characters)")
            
            return {
                "success": True,
                "notebook_content": notebook_content,
                "notebook_path": str(notebook_path),
                "clean_query": self._remove_notebook_path_from_query(user_query)
            }
            
        except Exception as e:
            logger.error(f"❌ Error reading notebook: {e}")
            return {
                "success": False,
                "error": str(e),
                "notebook_content": "",
                "notebook_path": "",
                "clean_query": prep_res["user_query"]
            }
    
    def _extract_notebook_path(self, query: str, working_dir: Path) -> Optional[Path]:
        """Extract notebook path from user query or find in directory"""
        # Look for explicit notebook path in query
        if "notebook:" in query.lower():
            parts = query.split("notebook:")
            if len(parts) > 1:
                path_part = parts[1].strip().split()[0]
                notebook_path = Path(path_part)
                
                # Convert to absolute path if relative
                if not notebook_path.is_absolute():
                    notebook_path = working_dir / notebook_path
                
                if notebook_path.exists():
                    return notebook_path
        
        # Look for .ipynb files in current directory
        ipynb_files = list(working_dir.glob("*.ipynb"))
        if ipynb_files:
            return ipynb_files[0]  # Return first notebook found
        
        return None
    
    def _remove_notebook_path_from_query(self, query: str) -> str:
        """Remove notebook path from user query"""
        if "notebook:" in query.lower():
            parts = query.split("notebook:")
            if len(parts) > 1:
                # Remove the path part and return the rest
                remaining = parts[1].strip()
                if " " in remaining:
                    return remaining.split(" ", 1)[1]
                return parts[0].strip()
        return query
    
    def post(self, shared, prep_res, exec_res):
        """Store notebook content in shared state"""
        shared["notebook_content"] = exec_res["notebook_content"]
        shared["notebook_success"] = exec_res["success"]
        shared["notebook_path"] = exec_res.get("notebook_path", "")
        shared["clean_query"] = exec_res.get("clean_query", shared.get("user_query", ""))
        return "default"


class ResponseGeneratorNode(Node):
    """
    Node 3: Generate analysis and code recommendations
    
    Uses AWS Bedrock to generate comprehensive analysis and actionable code recommendations
    based on the repo context, notebook content, and user query.
    """
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
    
    def prep(self, shared):
        """Prepare context for response generation"""
        return {
            "user_query": shared.get("clean_query", shared.get("user_query", "")),
            "markdown_content": shared.get("markdown_content", ""),
            "notebook_content": shared.get("notebook_content", ""),
            "notebook_path": shared.get("notebook_path", ""),
            "markdown_path": shared.get("markdown_path", "")
        }
    
    def exec(self, prep_res):
        """Generate comprehensive response"""
        try:
            if not self.model_client:
                return self._generate_fallback_response(prep_res)
            
            # Create structured prompt
            prompt = self._create_analysis_prompt(prep_res)
            
            # Generate response using AWS Bedrock
            response = self.model_client.invoke(prompt)
            
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
            
            logger.info("✅ Generated analysis and recommendations")
            
            return {
                "success": True,
                "response": response_content,
                "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            fallback = self._generate_fallback_response(prep_res)
            return {
                "success": False,
                "error": str(e),
                "response": fallback["response"]
            }
    
    def _create_analysis_prompt(self, prep_res):
        """Create structured prompt for analysis"""
        prompt = f"""You are a data science expert providing analysis and code recommendations.

USER QUERY: {prep_res['user_query']}

REPOSITORY CONTEXT:
{prep_res['markdown_content'][:2000] if prep_res['markdown_content'] else 'No repo context available'}

NOTEBOOK CONTENT:
{prep_res['notebook_content'][:3000] if prep_res['notebook_content'] else 'No notebook content available'}

NOTEBOOK PATH: {prep_res['notebook_path']}

Please provide a comprehensive analysis with:

## 📊 Current State Analysis
- Analyze the current notebook content and identify what's been done
- Point out any issues, inefficiencies, or missing components
- Assess the overall approach and methodology

## 🎯 Recommendations
- Provide specific, actionable recommendations
- Suggest improvements to existing code
- Identify next logical steps in the analysis

## 💻 Implementation Code
- Provide ready-to-use code snippets that can be directly implemented
- Include proper imports and variable handling
- Add comments explaining the approach
- Consider the existing notebook structure and variables

## 🔄 Next Steps
- Outline a clear roadmap for continued development
- Prioritize the most important tasks
- Suggest testing and validation approaches

Focus on practical, implementable solutions that directly address the user's query while building upon the existing work shown in the notebook."""
        
        return prompt
    
    def _generate_fallback_response(self, prep_res):
        """Generate fallback response when LLM is unavailable"""
        return {
            "success": False,
            "response": f"""# Data Science Analysis

## Query: {prep_res['user_query']}

## Current Status
- Repository Context: {'✅ Available' if prep_res['markdown_content'] else '❌ Not found'}
- Notebook Content: {'✅ Available' if prep_res['notebook_content'] else '❌ Not found'}
- Notebook Path: {prep_res['notebook_path'] or 'Not specified'}

## Analysis Summary
Based on the available information, I can see your current work context. However, the AI model is not available to provide detailed analysis.

## Recommendations
1. **Review Current Work**: Examine your notebook content for any obvious issues
2. **Check Data Quality**: Ensure your data is clean and properly formatted
3. **Validate Approach**: Consider if your current methodology aligns with best practices
4. **Plan Next Steps**: Define clear objectives for continued development

## Code Suggestions
```python
# Basic analysis template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and examine your data
df = pd.read_csv('your_data.csv')
print(df.info())
print(df.describe())

# Basic visualization
plt.figure(figsize=(10, 6))
# Add your visualization code here
plt.show()
```

## Next Steps
1. Ensure AWS Bedrock is properly configured
2. Review the notebook content for specific areas needing improvement
3. Apply the basic analysis template above
4. Consider reaching out for more detailed guidance once the AI model is available

*Note: This is a fallback response. Configure AWS Bedrock for detailed AI-powered analysis.*"""
        }
    
    def post(self, shared, prep_res, exec_res):
        """Store final response in shared state"""
        shared["final_response"] = exec_res["response"]
        shared["generation_success"] = exec_res["success"]
        return "default"