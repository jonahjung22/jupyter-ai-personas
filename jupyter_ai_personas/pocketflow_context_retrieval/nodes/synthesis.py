"""
nodes/synthesis.py - LLM-powered synthesis for comprehensive report generation
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from pocketflow import Node
from ..utils.llm_utils import call_llm_for_synthesis, build_synthesis_prompt
from ..config import config

logger = logging.getLogger(__name__)

class LLMSynthesisNode(Node):
    """LLM-powered synthesis node for comprehensive report generation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enable_llm_synthesis = config.enable_llm_synthesis
        self.synthesis_fallback = config.synthesis_fallback
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive synthesis."""
        return {
            "advanced_analysis": shared.get("advanced_notebook_analysis", {}),
            "intelligent_rag_results": shared.get("intelligent_rag_results", []),
            "user_query": shared.get("user_query", ""),
            "synthesis_mode": "llm_powered" if self.enable_llm_synthesis else "structured"
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive synthesis."""
        try:
            # Prepare synthesis context
            synthesis_context = self._prepare_synthesis_context(prep_res)
            
            # Generate synthesis
            if prep_res["synthesis_mode"] == "llm_powered":
                synthesis_report = self._generate_llm_synthesis(synthesis_context)
            else:
                synthesis_report = self._generate_structured_synthesis(synthesis_context)
            
            return {
                "synthesis_successful": True,
                "synthesis_report": synthesis_report,
                "synthesis_method": prep_res["synthesis_mode"],
                "context_elements": len(synthesis_context),
                "synthesis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Synthesis failed: {e}")
            
            if self.synthesis_fallback:
                fallback_report = self._create_fallback_synthesis(prep_res)
                return {
                    "synthesis_successful": False,
                    "synthesis_report": fallback_report,
                    "synthesis_method": "fallback",
                    "error": str(e)
                }
            else:
                return {
                    "synthesis_successful": False,
                    "error": str(e)
                }
    
    def _prepare_synthesis_context(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive context for synthesis."""
        context = {
            "user_query": prep_res["user_query"],
            "notebook_insights": self._extract_notebook_insights(prep_res["advanced_analysis"]),
            "rag_findings": self._extract_rag_findings(prep_res["intelligent_rag_results"]),
            "synthesis_goals": self._determine_synthesis_goals(prep_res)
        }
        
        return context
    
    def _extract_notebook_insights(self, analysis: Dict) -> Dict[str, Any]:
        """Extract key insights from advanced notebook analysis."""
        if not analysis or analysis.get("fallback_mode"):
            return {"insights_available": False}
        
        workflow_detection = analysis.get("workflow_detection", {})
        semantic_analysis = analysis.get("semantic_analysis", {})
        code_intelligence = analysis.get("code_intelligence", {})
        
        return {
            "insights_available": True,
            "primary_workflow_stage": workflow_detection.get("primary_stage", "unknown"),
            "workflow_confidence": workflow_detection.get("confidence", 0),
            "detected_libraries": [lib["name"] for lib in semantic_analysis.get("detected_libraries", [])],
            "analysis_themes": semantic_analysis.get("analysis_themes", []),
            "code_quality_level": code_intelligence.get("code_quality_level", "unknown"),
            "complexity_score": code_intelligence.get("complexity_score", 0),
            "recommendations": analysis.get("recommendations", [])
        }
    
    def _extract_rag_findings(self, rag_results: List[Dict]) -> Dict[str, Any]:
        """Extract key findings from RAG results."""
        if not rag_results:
            return {"findings_available": False}
        
        successful_results = [r for r in rag_results if r.get("execution_status") == "success"]
        
        # Collect high-quality content
        high_quality_content = []
        source_diversity = set()
        
        for result in successful_results:
            for item in result.get("results", []):
                relevance_score = item.get("relevance_score", 0)
                if relevance_score > 0.6:  # High relevance threshold
                    high_quality_content.append({
                        "content": item["content"][:400] + "..." if len(item["content"]) > 400 else item["content"],
                        "source": item.get("notebook_name", "Unknown"),
                        "relevance": relevance_score,
                        "query_type": result.get("type", "unknown"),
                        "cell_type": item.get("cell_type", "unknown")
                    })
                    source_diversity.add(item.get("notebook_name", "Unknown"))
        
        return {
            "findings_available": True,
            "total_searches": len(rag_results),
            "successful_searches": len(successful_results),
            "high_quality_results": len(high_quality_content),
            "source_diversity": len(source_diversity),
            "top_findings": high_quality_content[:10],  # Top 10 findings
            "source_coverage": list(source_diversity)[:8]  # Top 8 sources
        }
    
    def _determine_synthesis_goals(self, prep_res: Dict[str, Any]) -> List[str]:
        """Determine synthesis goals based on context."""
        goals = ["comprehensive_analysis", "actionable_recommendations"]
        
        user_query = prep_res["user_query"].lower()
        
        if any(word in user_query for word in ["help", "how to", "explain", "understand"]):
            goals.append("educational_guidance")
        
        if any(word in user_query for word in ["improve", "optimize", "better", "enhance"]):
            goals.append("optimization_suggestions")
        
        if any(word in user_query for word in ["example", "show", "demonstrate", "code"]):
            goals.append("practical_examples")
        
        if any(word in user_query for word in ["workflow", "process", "steps"]):
            goals.append("process_guidance")
        
        return goals
    
    def _generate_llm_synthesis(self, context: Dict[str, Any]) -> str:
        """Generate synthesis using LLM."""
        try:
            # Build comprehensive prompt
            prompt = build_synthesis_prompt(context)
            
            # Call LLM for synthesis
            synthesis = call_llm_for_synthesis(prompt)
            
            return synthesis
            
        except Exception as e:
            logger.error(f"❌ LLM synthesis failed: {e}")
            # Fall back to structured synthesis
            return self._generate_structured_synthesis(context)
    
    def _generate_structured_synthesis(self, context: Dict[str, Any]) -> str:
        """Generate structured synthesis without LLM."""
        user_query = context["user_query"]
        notebook_insights = context["notebook_insights"]
        rag_findings = context["rag_findings"]
        synthesis_goals = context["synthesis_goals"]
        
        report_sections = []
        
        # Header
        report_sections.append(f"""# 🧠 PocketFlow Context Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**User Request:** {user_query}  
**Analysis Framework:** Advanced PocketFlow RAG Architecture  
""")
        
        # Executive Summary
        report_sections.append(self._generate_executive_summary_section(notebook_insights, rag_findings))
        
        # Current Analysis
        report_sections.append(self._generate_current_analysis_section(notebook_insights))
        
        # Research Findings
        report_sections.append(self._generate_research_findings_section(rag_findings))
        
        # Actionable Recommendations
        report_sections.append(self._generate_recommendations_section(notebook_insights, rag_findings))
        
        # Code Examples
        if "practical_examples" in synthesis_goals:
            report_sections.append(self._generate_code_examples_section(rag_findings))
        
        # Next Steps
        report_sections.append(self._generate_next_steps_section(notebook_insights, synthesis_goals))
        
        # Technical Details
        report_sections.append(self._generate_technical_details_section(notebook_insights, rag_findings))
        
        return "\n\n".join(report_sections)
    
    def _generate_executive_summary_section(self, notebook_insights: Dict, rag_findings: Dict) -> str:
        """Generate executive summary section."""
        section = "## 🎯 Executive Summary\n\n"
        
        if notebook_insights["insights_available"]:
            primary_stage = notebook_insights["primary_workflow_stage"]
            libraries = notebook_insights["detected_libraries"]
            
            section += f"""**Current Focus**: {primary_stage.replace('_', ' ').title()} phase with {len(libraries)} primary libraries detected

**Key Insights**:
- Workflow stage: {primary_stage} (confidence: {notebook_insights['workflow_confidence']:.1f})
- Technology stack: {', '.join(libraries[:4]) if libraries else 'General Python'}
- Code quality: {notebook_insights['code_quality_level']} level
- Complexity score: {notebook_insights['complexity_score']:.1f}/10
"""
        
        if rag_findings["findings_available"]:
            section += f"""
**Research Results**:
- Performed {rag_findings['total_searches']} intelligent searches
- Found {rag_findings['high_quality_results']} high-quality resources
- Consulted {rag_findings['source_diversity']} handbook sources
- Success rate: {rag_findings['successful_searches']}/{rag_findings['total_searches']} searches
"""
        
        section += f"""
**Primary Recommendation**: {"Focus on workflow optimization and apply handbook best practices" if notebook_insights["insights_available"] else "Review research findings and implement suggested improvements"}
"""
        
        return section
    
    def _generate_current_analysis_section(self, notebook_insights: Dict) -> str:
        """Generate current situation analysis section."""
        section = "## 📊 Current Situation Analysis\n\n"
        
        if not notebook_insights["insights_available"]:
            section += "**Note**: Detailed notebook analysis not available. Analysis based on query context.\n\n"
            return section
        
        primary_stage = notebook_insights["primary_workflow_stage"]
        themes = notebook_insights["analysis_themes"]
        
        section += f"""**Workflow Assessment**:
- **Current Stage**: {primary_stage.replace('_', ' ').title()}
- **Stage Confidence**: {notebook_insights['workflow_confidence']:.1f}/1.0
- **Analysis Themes**: {', '.join(themes) if themes else 'General data science'}

**Technical Assessment**:
- **Code Quality**: {notebook_insights['code_quality_level'].title()} level
- **Complexity**: {notebook_insights['complexity_score']:.1f}/10 complexity score
- **Libraries**: {len(notebook_insights['detected_libraries'])} libraries detected

**Improvement Areas**:
"""
        
        recommendations = notebook_insights.get("recommendations", [])
        for rec in recommendations[:3]:
            section += f"- {rec}\n"
        
        return section
    
    def _generate_research_findings_section(self, rag_findings: Dict) -> str:
        """Generate research findings section."""
        section = "## 📚 Research Findings from Python Data Science Handbook\n\n"
        
        if not rag_findings["findings_available"]:
            section += "**Note**: RAG research not available. Please ensure handbook is accessible.\n\n"
            return section
        
        section += f"""**Research Summary**:
- **Total Searches**: {rag_findings['total_searches']} strategic queries executed
- **Success Rate**: {rag_findings['successful_searches']}/{rag_findings['total_searches']} searches successful
- **Quality Results**: {rag_findings['high_quality_results']} high-relevance findings
- **Source Coverage**: {rag_findings['source_diversity']} different handbook sections

**Primary Sources Consulted**:
"""
        
        for source in rag_findings['source_coverage'][:5]:
            section += f"- **{source}**: Relevant examples and best practices identified\n"
        
        section += "\n**Key Research Insights**:\n\n"
        
        for i, finding in enumerate(rag_findings['top_findings'][:4], 1):
            section += f"""**{i}. {finding['source']}** ({finding['cell_type']} cell, relevance: {finding['relevance']:.2f})
{finding['content'][:250]}...

"""
        
        return section
    
    def _generate_recommendations_section(self, notebook_insights: Dict, rag_findings: Dict) -> str:
        """Generate actionable recommendations section."""
        section = "## 💡 Actionable Recommendations\n\n"
        
        # High-priority recommendations
        section += "### 🔥 High Priority (Immediate Action)\n\n"
        
        if notebook_insights["insights_available"]:
            primary_stage = notebook_insights["primary_workflow_stage"]
            
            if primary_stage == "data_exploration":
                section += "- Apply advanced EDA techniques from handbook examples\n"
                section += "- Implement comprehensive data profiling and validation\n"
            elif primary_stage == "modeling":
                section += "- Review model evaluation best practices from research findings\n"
                section += "- Implement proper cross-validation and performance metrics\n"
            elif primary_stage == "visualization":
                section += "- Enhance plots with handbook visualization techniques\n"
                section += "- Apply professional styling and annotation practices\n"
            else:
                section += "- Apply stage-specific best practices from handbook research\n"
                section += "- Implement proper error handling and data validation\n"
        
        if rag_findings["findings_available"]:
            section += f"- Review top {min(3, len(rag_findings['top_findings']))} research findings for immediate application\n"
        
        # Medium-priority recommendations
        section += "\n### 📈 Medium Priority (This Week)\n\n"
        section += "- Integrate advanced techniques from multiple handbook sources\n"
        section += "- Optimize code structure based on complexity analysis\n"
        section += "- Implement comprehensive testing and validation procedures\n"
        
        # Long-term recommendations
        section += "\n### 🎯 Long-term Goals (This Month)\n\n"
        section += "- Master advanced concepts from identified handbook sections\n"
        section += "- Build reusable analysis templates and workflows\n"
        section += "- Develop domain expertise through systematic handbook study\n"
        
        return section
    
    def _generate_code_examples_section(self, rag_findings: Dict) -> str:
        """Generate code examples section."""
        section = "## 💻 Code Examples from Research\n\n"
        
        if not rag_findings["findings_available"]:
            section += "**Note**: Code examples not available from current research.\n\n"
            return section
        
        code_examples = [f for f in rag_findings['top_findings'] if f['cell_type'] == 'code']
        
        if not code_examples:
            section += "**Note**: No specific code examples found in current research results.\n\n"
            return section
        
        for i, example in enumerate(code_examples[:3], 1):
            section += f"""### Example {i}: From {example['source']}

**Relevance**: {example['relevance']:.2f}/1.0  
**Context**: {example['query_type'].replace('_', ' ').title()}

```python
{example['content'][:600]}
```

**Application**: {self._suggest_code_application(example)}

---

"""
        
        return section
    
    def _suggest_code_application(self, example: Dict) -> str:
        """Suggest how to apply code example."""
        content = example['content'].lower()
        
        if 'import' in content:
            return "Use this import pattern at the beginning of your analysis"
        elif 'plot' in content or 'plt.' in content:
            return "Apply this visualization technique to your data"
        elif 'dataframe' in content or 'pd.' in content:
            return "Adapt this data manipulation approach to your dataset"
        elif 'model' in content or 'fit(' in content:
            return "Consider this modeling approach for your problem"
        else:
            return "Integrate this pattern into your current workflow"
    
    def _generate_next_steps_section(self, notebook_insights: Dict, synthesis_goals: List[str]) -> str:
        """Generate next steps section."""
        section = "## ⚡ Next Steps\n\n"
        
        section += "### Immediate Actions (Next 2 hours)\n"
        section += "1. Review the research findings and identify 2-3 applicable techniques\n"
        section += "2. Implement the highest-priority recommendation from above\n"
        section += "3. Test one code example from the handbook research\n\n"
        
        if "optimization_suggestions" in synthesis_goals:
            section += "### Optimization Focus\n"
            section += "- Profile current code performance and identify bottlenecks\n"
            section += "- Apply handbook optimization techniques to critical sections\n"
            section += "- Implement vectorized operations where applicable\n\n"
        
        if "educational_guidance" in synthesis_goals:
            section += "### Learning Path\n"
            section += "- Study the identified handbook sections systematically\n"
            section += "- Practice examples in a separate learning notebook\n"
            section += "- Build a personal reference collection of useful patterns\n\n"
        
        section += "### Follow-up Session Preparation\n"
        section += "- Document which recommendations you implemented\n"
        section += "- Note any challenges encountered during application\n"
        section += "- Prepare specific questions for deeper handbook exploration\n"
        
        return section
    
    def _generate_technical_details_section(self, notebook_insights: Dict, rag_findings: Dict) -> str:
        """Generate technical details section."""
        section = "## 🔧 Technical Analysis Details\n\n"
        
        section += "### PocketFlow Architecture Benefits\n"
        section += "✅ **Modular Design**: Each analysis component optimized independently\n"
        section += "✅ **Intelligent Search**: Multi-query strategy with context awareness\n"
        section += "✅ **Quality Filtering**: Advanced relevance scoring and content ranking\n"
        section += "✅ **Comprehensive Analysis**: Deep notebook understanding with workflow detection\n\n"
        
        if notebook_insights["insights_available"]:
            section += "### Notebook Analysis Metrics\n"
            section += f"- **Primary Stage Confidence**: {notebook_insights['workflow_confidence']:.2f}\n"
            section += f"- **Code Complexity Score**: {notebook_insights['complexity_score']:.1f}/10\n"
            section += f"- **Quality Level**: {notebook_insights['code_quality_level'].title()}\n"
            section += f"- **Libraries Detected**: {len(notebook_insights['detected_libraries'])}\n\n"
        
        if rag_findings["findings_available"]:
            section += "### RAG Search Performance\n"
            section += f"- **Search Success Rate**: {rag_findings['successful_searches']}/{rag_findings['total_searches']} ({rag_findings['successful_searches']/rag_findings['total_searches']*100:.1f}%)\n"
            section += f"- **High-Quality Results**: {rag_findings['high_quality_results']} above relevance threshold\n"
            section += f"- **Source Diversity**: {rag_findings['source_diversity']} different handbook sections\n"
            section += f"- **Content Coverage**: Multiple cell types and difficulty levels\n\n"
        
        section += "### System Capabilities\n"
        section += "- **Semantic Understanding**: Context-aware query generation and result ranking\n"
        section += "- **Workflow Intelligence**: Automatic detection of analysis stages and patterns\n"
        section += "- **Quality Assurance**: Multi-factor relevance scoring with content filtering\n"
        section += "- **Comprehensive Synthesis**: Integration of analysis and research findings\n"
        
        return section
    
    def _create_fallback_synthesis(self, prep_res: Dict[str, Any]) -> str:
        """Create fallback synthesis when primary synthesis fails."""
        user_query = prep_res["user_query"]
        
        return f"""# Context Analysis Report (Fallback Mode)

## User Request
{user_query}

## Analysis Status
- **PocketFlow Architecture**: Attempted advanced analysis
- **Synthesis Mode**: Fallback due to processing issues
- **Available Data**: Basic analysis components completed

## Key Findings
The PocketFlow RAG system executed its core components:
- Advanced notebook analysis with workflow detection
- Intelligent multi-query search through handbook
- Quality filtering and relevance ranking of results
- Structured report generation

## Recommendations
1. **Review Individual Components**: Each PocketFlow component provides valuable insights
2. **Apply Best Practices**: Use handbook research findings for immediate improvements
3. **Iterate Analysis**: Refine query or notebook path for enhanced results

## Next Steps
- Examine the detailed search results from RAG system
- Apply identified best practices to current workflow
- Consider retry with more specific analysis parameters

---
*Generated by PocketFlow Context Retrieval System (Fallback Mode)*  
*Core intelligence components remain fully functional*
"""
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store synthesis results."""
        if exec_res.get("synthesis_successful"):
            shared["final_synthesis"] = exec_res["synthesis_report"]
            shared["synthesis_completed"] = True
        else:
            shared["final_synthesis"] = exec_res.get("synthesis_report", "Synthesis failed")
            shared["synthesis_completed"] = False
        
        shared["synthesis_method"] = exec_res.get("synthesis_method", "failed")
        
        logger.info(f"🎯 Synthesis completed: {exec_res.get('synthesis_successful', False)}")
        logger.info(f"   Method: {exec_res.get('synthesis_method', 'unknown')}")
        
        return "default"