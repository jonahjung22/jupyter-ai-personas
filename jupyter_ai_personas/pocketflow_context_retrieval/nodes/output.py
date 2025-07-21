import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from pocketflow import Node
from ..config import config

logger = logging.getLogger(__name__)

class AdvancedOutputNode(Node):
    """Advanced output node with multiple format support."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_formats = ["markdown", "summary"]
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare output generation."""
        return {
            "final_synthesis": shared.get("final_synthesis", ""),
            "synthesis_completed": shared.get("synthesis_completed", False),
            "synthesis_method": shared.get("synthesis_method", "unknown"),
            "output_formats": self.output_formats
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced output generation."""
        final_synthesis = prep_res["final_synthesis"]
        
        if not final_synthesis:
            return {
                "output_successful": False,
                "error": "No synthesis content available for output"
            }
        
        try:
            # Create output directory if needed
            output_dir = Path(".")  # Current directory
            
            # Generate primary markdown report
            primary_file = output_dir / "repo_context.md"
            self._write_file(primary_file, final_synthesis)
            
            files_created = [str(primary_file)]
            
            # Generate executive summary
            if len(final_synthesis) > 1000:  # Only if substantial content
                summary = self._generate_executive_summary(final_synthesis)
                summary_file = output_dir / "context_summary.md"
                self._write_file(summary_file, summary)
                files_created.append(str(summary_file))
            
            # Generate metadata file
            metadata = self._generate_metadata(prep_res)
            metadata_file = output_dir / "analysis_metadata.json"
            self._write_file(metadata_file, metadata)
            files_created.append(str(metadata_file))
            
            return {
                "output_successful": True,
                "files_created": files_created,
                "primary_report": str(primary_file),
                "total_files": len(files_created),
                "content_length": len(final_synthesis),
                "output_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Output generation failed: {e}")
            return {
                "output_successful": False,
                "error": str(e)
            }
    
    def _write_file(self, file_path: Path, content: str):
        """Write content to file with error handling."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"📄 Created: {file_path.name}")
        except Exception as e:
            logger.error(f"❌ Failed to write {file_path}: {e}")
            raise
    
    def _generate_executive_summary(self, full_report: str) -> str:
        """Generate executive summary from full report."""
        lines = full_report.split('\n')
        
        # Extract key sections for summary
        summary_sections = []
        current_section = []
        in_executive = False
        in_recommendations = False
        
        for line in lines:
            # Detect section headers
            if line.startswith('#'):
                if in_executive or in_recommendations:
                    # End current section
                    if current_section:
                        summary_sections.extend(current_section)
                    current_section = []
                    in_executive = False
                    in_recommendations = False
                
                # Check if this is a section we want
                line_lower = line.lower()
                if 'executive' in line_lower or 'summary' in line_lower:
                    in_executive = True
                    summary_sections.append(line)
                elif 'recommendation' in line_lower:
                    in_recommendations = True
                    summary_sections.append(line)
            else:
                # Add content if in relevant section
                if in_executive or in_recommendations:
                    current_section.append(line)
        
        # Add final section if exists
        if current_section:
            summary_sections.extend(current_section)
        
        # Create summary
        if summary_sections:
            summary_content = '\n'.join(summary_sections)
        else:
            # Fallback: first 800 characters
            summary_content = f"# Executive Summary\n\n{full_report[:800]}..."
        
        # Add summary metadata
        summary_header = f"""# 📋 Context Analysis Executive Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Source:** Full PocketFlow analysis report  
**Type:** Key insights and recommendations  

---

"""
        
        return summary_header + summary_content
    
    def _generate_metadata(self, prep_res: Dict[str, Any]) -> str:
        """Generate metadata file with analysis details."""
        import json
        
        metadata = {
            "analysis_metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "synthesis_method": prep_res.get("synthesis_method", "unknown"),
                "synthesis_successful": prep_res.get("synthesis_completed", False),
                "content_length": len(prep_res.get("final_synthesis", "")),
                "output_formats_generated": prep_res.get("output_formats", []),
                "pocketflow_version": "1.0.0",
                "architecture": "advanced_multi_node"
            },
            "system_capabilities": {
                "advanced_notebook_analysis": True,
                "intelligent_multi_query_search": True,
                "quality_filtering": config.enable_quality_filtering,
                "advanced_ranking": config.enable_advanced_ranking,
                "llm_synthesis": config.enable_llm_synthesis,
                "metadata_indexing": config.enable_metadata_indexing
            },
            "configuration": {
                "embedding_model": config.embedding_model,
                "chunk_size": config.chunk_size,
                "max_search_queries": config.max_search_queries,
                "quality_threshold": config.quality_threshold,
                "index_type": config.index_type
            }
        }
        
        return json.dumps(metadata, indent=2)
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store output results."""
        shared["output_results"] = exec_res
        shared["report_saved"] = exec_res.get("output_successful", False)
        shared["output_files"] = exec_res.get("files_created", [])
        
        if exec_res.get("output_successful"):
            logger.info(f"✅ Output generation completed: {exec_res.get('total_files', 0)} files created")
            logger.info(f"   Primary report: {exec_res.get('primary_report', 'repo_context.md')}")
        else:
            logger.error(f"❌ Output generation failed: {exec_res.get('error', 'unknown error')}")
        
        return "default"
    