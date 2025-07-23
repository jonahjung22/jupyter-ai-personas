"""
PocketFlow Agent for Data Science Analysis

A sophisticated agent that can reason about data science tasks and decide
on appropriate actions based on user queries, repo context, and notebook content.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from .pocketflow import Flow, Node
    from .file_reader_tool import NotebookReaderTool
    from .autogluon_tool import AutoGluonTool
    from agno.models.message import Message as AgnoMessage
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise ImportError(f"Missing dependencies for DataScienceAgent: {e}") from e

logger = logging.getLogger(__name__)


class DecideAction(Node):
    """
    Decision-making node that analyzes the user query and context
    to determine the appropriate action for data science analysis.
    """
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
    
    def prep(self, shared):
        """Prepare context for decision making"""
        return {
            "user_query": shared.get("user_query", ""),
            "repo_context": shared.get("repo_context", ""),
            "notebook_content": shared.get("notebook_content", ""),
            "notebook_path": shared.get("notebook_path", ""),
            "history": shared.get("history", ""),
            "previous_actions": shared.get("action_history", [])
        }
    
    def exec(self, prep_res):
        """Use LLM to decide on the next action"""
        try:
            if not self.model_client:
                return self._default_action(prep_res)
            
            # Create decision prompt
            prompt = self._create_decision_prompt(prep_res)
            
            # Get decision from LLM
            messages = [AgnoMessage(role="user", content=prompt)]
            response = self.model_client.invoke(messages)
            
            # Extract content from Bedrock response format
            if hasattr(response, 'content'):
                decision_text = response.content
            elif isinstance(response, dict):
                # Handle Bedrock response format
                if 'output' in response and 'message' in response['output']:
                    message_content = response['output']['message']['content']
                    if isinstance(message_content, list) and len(message_content) > 0:
                        decision_text = message_content[0].get('text', str(response))
                    else:
                        decision_text = str(message_content)
                else:
                    decision_text = str(response)
            else:
                decision_text = str(response)
            
            logger.debug(f"Raw LLM response: {decision_text[:200]}...")
            
            # Parse the decision
            decision = self._parse_decision(decision_text)
            
            logger.info(f"🤖 Agent decided: {decision.get('action', 'unknown')}")
            return decision
            
        except Exception as e:
            logger.error(f"❌ Decision error: {e}")
            return self._default_action(prep_res)
    
    def _create_decision_prompt(self, prep_res):
        """Create prompt for decision making"""
        return f"""You are a data science agent analyzing a user request. Based on the context provided, decide what action to take next.

                USER QUERY: {prep_res['user_query']}

                REPOSITORY CONTEXT:
                {prep_res['repo_context'][:1000] if prep_res['repo_context'] else 'No repo context available'}

                NOTEBOOK CONTENT:
                {prep_res['notebook_content'][:1500] if prep_res['notebook_content'] else 'No notebook content available'}

                NOTEBOOK PATH: {prep_res['notebook_path']}

                PREVIOUS ACTIONS: {prep_res['previous_actions']}

                Based on this context, decide what action to take. You MUST respond in valid YAML format.

                Choose ONE action from: analyze_data, generate_code, explain_concept, find_issues, create_visualization, optimize_model, debug_code, train_ml_model, hyperparameter_tuning, model_optimization, model_evaluation, feature_engineering, complete_analysis, greeting

                IMPORTANT: Respond with ONLY valid YAML. Do not include any other text.

                ```yaml
                action: [choose one from the list above]
                reasoning: [brief explanation in quotes]
                priority: [high, medium, or low]
                context_summary: [key points in quotes]
                next_steps: [what should happen after this action in quotes]
                ```

                Your YAML response:"""
                    
    def _parse_decision(self, decision_text):
        """Parse the LLM decision response with robust error handling"""
        try:
            # Extract YAML content
            yaml_content = decision_text.strip()
            
            # Try to find YAML block first
            if "```yaml" in decision_text:
                yaml_start = decision_text.find("```yaml") + 7
                yaml_end = decision_text.find("```", yaml_start)
                if yaml_end > yaml_start:
                    yaml_content = decision_text[yaml_start:yaml_end].strip()
            elif "```" in decision_text:
                # Try generic code block
                yaml_start = decision_text.find("```") + 3
                yaml_end = decision_text.find("```", yaml_start)
                if yaml_end > yaml_start:
                    yaml_content = decision_text[yaml_start:yaml_end].strip()
            
            # Clean up common YAML issues
            yaml_content = self._clean_yaml_content(yaml_content)
            
            # Parse YAML
            decision = yaml.safe_load(yaml_content)
            
            # Validate required fields
            if not isinstance(decision, dict):
                logger.warning(f"Decision is not a dict: {type(decision)}")
                return self._extract_decision_from_text(decision_text)
            
            # Ensure required fields exist
            decision.setdefault("action", "complete_analysis")
            decision.setdefault("reasoning", "Fallback to complete analysis")
            decision.setdefault("priority", "medium")
            
            logger.debug(f"Parsed decision: {decision}")
            return decision
            
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error: {e}")
            logger.debug(f"Raw YAML content: {yaml_content}")
            return self._extract_decision_from_text(decision_text)
        except Exception as e:
            logger.error(f"❌ Decision parsing error: {e}")
            return self._default_decision()
    
    def _clean_yaml_content(self, yaml_content):
        """Clean common YAML formatting issues"""
        # Remove extra whitespace
        yaml_content = yaml_content.strip()
        
        # Fix common colon issues
        lines = yaml_content.split('\n')
        cleaned_lines = []
        for line in lines:
            if ':' in line and not line.strip().startswith('#'):
                # Ensure there's a space after colon
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    cleaned_lines.append(f"{key}: {value}")
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_decision_from_text(self, text):
        """Extract decision from text when YAML parsing fails"""
        # Try to extract key information using simple text parsing
        decision = self._default_decision()
        
        text_lower = text.lower()
        
        # Extract action
        actions = ["analyze_data", "generate_code", "explain_concept", "find_issues", 
                  "create_visualization", "optimize_model", "debug_code", "train_ml_model", "complete_analysis", "greeting"]
        
        for action in actions:
            if action in text_lower:
                decision["action"] = action
                break
        
        # Extract reasoning (look for common patterns)
        if "reasoning" in text_lower or "because" in text_lower:
            # Try to extract reasoning text
            for line in text.split('\n'):
                if any(word in line.lower() for word in ["reasoning", "because", "since"]):
                    decision["reasoning"] = line.strip()
                    break
        
        logger.warning(f"Used text extraction fallback: {decision}")
        return decision
    
    def _default_decision(self):
        """Default decision when LLM fails"""
        return {
            "action": "complete_analysis",
            "reasoning": "Fallback to complete analysis",
            "priority": "medium",
            "context_summary": "Limited context available",
            "next_steps": "Provide comprehensive analysis"
        }
    
    def _default_action(self, prep_res):
        """Default action when no model available"""
        return self._default_decision()
    
    def post(self, shared, prep_res, exec_res):
        """Update shared state with decision"""
        shared["current_action"] = exec_res.get("action", "complete_analysis")
        shared["action_reasoning"] = exec_res.get("reasoning", "")
        shared["action_priority"] = exec_res.get("priority", "medium")
        shared["context_summary"] = exec_res.get("context_summary", "")
        
        # Track action history
        action_history = shared.get("action_history", [])
        action_history.append(exec_res.get("action", "complete_analysis"))
        shared["action_history"] = action_history
        
        # Return next node based on action
        action = exec_res.get("action", "complete_analysis")
        
        if action in ["analyze_data", "find_issues", "debug_code"]:
            return "analyze"
        elif action in ["train_ml_model", "hyperparameter_tuning", "model_optimization", "model_evaluation", "feature_engineering"]:
            return "ml_training"
        elif action in ["generate_code", "create_visualization", "optimize_model"]:
            return "complete"  # Route generate actions to complete node for now
        elif action == "explain_concept":
            return "complete"  # Route explain to complete node for now
        elif action == "greeting":
            return "greeting"  # Route to greeting handler
        else:
            return "complete"


class GreetingNode(Node):
    """Node for handling greetings and introductions"""
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
    
    def prep(self, shared):
        """Prepare for greeting"""
        return {
            "user_query": shared.get("user_query", ""),
            "history": shared.get("history", ""),
            "previous_actions": shared.get("action_history", [])
        }
    
    def exec(self, prep_res):
        """Execute greeting response"""
        try:
            # Check if this is a simple greeting
            query_lower = prep_res.get("user_query", "").lower()
            greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
            
            is_greeting = any(word in query_lower for word in greeting_words)
            
            if is_greeting and len(prep_res.get("user_query", "").split()) <= 5:
                # Simple greeting response
                greeting_response = """# Hello! 👋 Welcome to the Data Science Assistant

I'm your advanced data science agent, powered by sophisticated reasoning capabilities and ready to help you with:

## 🔬 **What I Can Do:**
- **Smart Data Analysis**: Analyze your datasets with targeted insights
- **ML Model Training**: Automated machine learning with AutoGluon
- **Code Generation**: Ready-to-use Python code for your projects
- **Problem Solving**: Debug issues and optimize your analysis
- **Context-Aware Help**: I read your notebooks and project context automatically

## 🚀 **Getting Started:**
Just tell me what you'd like to work on! For example:
- "Analyze my sales data for trends"
- "Help me train a classification model" 
- "Debug my notebook: notebook_name.ipynb"
- "Optimize my data preprocessing pipeline"

I'll automatically read your repository context and notebook content to provide targeted, actionable recommendations.

What would you like to explore today? 🎯"""
                
                return {"greeting": greeting_response, "success": True}
            else:
                # More complex query that happens to contain greeting words
                return {"greeting": "", "success": False, "route_to_analysis": True}
                
        except Exception as e:
            logger.error(f"❌ Greeting error: {e}")
            return {"greeting": "Hello! I'm ready to help with your data science tasks.", "success": True}
    
    def post(self, shared, prep_res, exec_res):
        """Handle greeting completion"""
        if exec_res.get("route_to_analysis"):
            # Route complex queries to complete analysis
            return "complete"
        else:
            # Simple greeting completed
            shared["final_response"] = exec_res.get("greeting", "Hello!")
            shared["analysis_complete"] = True
            return "end"


class DataAnalysisNode(Node):
    """Node for focused data analysis tasks"""
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
    
    def prep(self, shared):
        """Prepare for data analysis"""
        return {
            "user_query": shared.get("user_query", ""),
            "notebook_content": shared.get("notebook_content", ""),
            "context_summary": shared.get("context_summary", ""),
            "action_reasoning": shared.get("action_reasoning", "")
        }
    
    def exec(self, prep_res):
        """Execute focused data analysis"""
        try:
            if not self.model_client:
                return self._fallback_analysis(prep_res)
            
            prompt = f"""You are a data science expert performing focused data analysis.

                        USER REQUEST: {prep_res['user_query']}
                        CONTEXT: {prep_res['context_summary']}
                        REASONING: {prep_res['action_reasoning']}

                        NOTEBOOK CONTENT:
                        {prep_res['notebook_content'][:2000] if prep_res['notebook_content'] else 'No notebook content'}

                        Provide a focused analysis with:

                        ## 📊 Data Analysis
                        - Current data state and quality assessment
                        - Key patterns and insights from the data
                        - Statistical summary and observations

                        ## 🔍 Specific Findings
                        - Answer the user's specific question
                        - Highlight important data characteristics
                        - Identify potential issues or opportunities

                        ## 💡 Recommendations
                        - Specific next steps for this analysis
                        - Suggested improvements or additional analysis
                        - Priority actions based on findings

                        Focus on being specific and actionable rather than general."""
            
            messages = [AgnoMessage(role="user", content=prompt)]
            response = self.model_client.invoke(messages)
            
            # Extract content from Bedrock response format
            if hasattr(response, 'content'):
                analysis = response.content
            elif isinstance(response, dict):
                # Handle Bedrock response format
                if 'output' in response and 'message' in response['output']:
                    message_content = response['output']['message']['content']
                    if isinstance(message_content, list) and len(message_content) > 0:
                        analysis = message_content[0].get('text', str(response))
                    else:
                        analysis = str(message_content)
                else:
                    analysis = str(response)
            else:
                analysis = str(response)
            
            return {"analysis": analysis, "success": True}
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return self._fallback_analysis(prep_res)
    
    def _fallback_analysis(self, prep_res):
        """Fallback analysis when model unavailable"""
        return {
            "analysis": f"""## 📊 Data Analysis (Fallback Mode)

            **Query:** {prep_res['user_query']}

            **Context:** {prep_res['context_summary']}

            **Status:** AI model unavailable - providing basic analysis framework.

            ## 🔍 Analysis Framework
            1. **Data Quality Check**: Examine your data for missing values, outliers, and inconsistencies
            2. **Exploratory Analysis**: Use descriptive statistics and visualizations
            3. **Pattern Detection**: Look for trends, correlations, and anomalies
            4. **Validation**: Cross-check findings and validate assumptions

            ## 💡 Next Steps
            - Configure AI model for detailed analysis
            - Apply the framework above to your specific data
            - Consider domain-specific analysis methods
            """,
            "success": False
        }
    
    def post(self, shared, prep_res, exec_res):
        """Store analysis results"""
        shared["analysis_result"] = exec_res.get("analysis", "")
        shared["analysis_success"] = exec_res.get("success", False)
        return "decide"  # Go back to decision node


class MLTrainingNode(Node):
    """Node for automated machine learning training using AutoGluon"""
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
        self.autogluon_tool = AutoGluonTool()
    
    def prep(self, shared):
        """Prepare for ML training"""
        return {
            "user_query": shared.get("user_query", ""),
            "notebook_content": shared.get("notebook_content", ""),
            "notebook_path": shared.get("notebook_path", ""),
            "context_summary": shared.get("context_summary", ""),
            "action_reasoning": shared.get("action_reasoning", "")
        }
    
    def exec(self, prep_res):
        """Execute automated ML training"""
        try:
            # Check AutoGluon availability
            status = self.autogluon_tool.get_status()
            if not status.get("any_available", False):
                return self._fallback_ml_training(prep_res)
            
            # Extract training configuration from context
            training_config = self._extract_training_config(prep_res)
            
            if not training_config.get("success", False):
                return training_config
            
            # Determine training type and execute
            training_type = training_config.get("training_type", "tabular")
            
            if training_type == "tabular":
                result = self._train_tabular_model(training_config)
            elif training_type == "multimodal":
                result = self._train_multimodal_model(training_config)
            elif training_type == "timeseries":
                result = self._train_timeseries_model(training_config)
            else:
                result = {"success": False, "error": f"Unknown training type: {training_type}"}
            
            return result
            
        except Exception as e:
            logger.error(f"❌ ML training error: {e}")
            return self._fallback_ml_training(prep_res)
    
    def _extract_training_config(self, prep_res):
        """Extract training configuration from notebook content"""
        try:
            # First, try to extract data directly from notebook content
            notebook_data = self._extract_data_from_notebook(prep_res['notebook_content'])
            
            if notebook_data["success"]:
                # Use extracted data directly
                return {
                    "success": True,
                    "training_type": "tabular",
                    "data_source": notebook_data["dataframe"],  # Pass DataFrame directly
                    "target_column": notebook_data["target_column"],
                    "problem_type": notebook_data["problem_type"],
                    "time_limit": 600,
                    "data_extraction_method": "notebook_content"
                }
            
            # Fallback: Use LLM to extract training configuration
            if self.model_client:
                prompt = f"""You are analyzing a Jupyter notebook to extract ML training configuration. 

USER QUERY: {prep_res['user_query']}

NOTEBOOK CONTENT:
{prep_res['notebook_content'][:3000] if prep_res['notebook_content'] else 'No notebook content available'}

CRITICAL INSTRUCTIONS:
1. Look for ACTUAL DataFrame variable names (df, data, dataset, etc.) mentioned in the notebook
2. Look for ACTUAL column names shown in data previews, df.head(), df.info(), etc.
3. Look for target/label columns used in machine learning code
4. If you cannot find specific information, respond with "data_not_found" instead of making up placeholders

WHAT TO LOOK FOR:
- DataFrame variables: df = pd.read_csv(...), data = ..., etc.
- Column names from df.head(), df.columns, df.info() outputs
- Target variables: y = df['column_name'], model.fit(X, y), etc.
- Problem type indicators: classification, regression, predict, forecast

Respond in VALID YAML format:

```yaml
success: [true if you found actual data info, false if not]
data_variable: [actual variable name like 'df', 'data' or 'data_not_found']
target_column: [actual column name or 'data_not_found']
problem_type: [classification/regression/forecasting or 'unknown']
confidence: [high/medium/low]
```

Do NOT use generic placeholders like 'target', 'your_data', 'placeholder'. Use actual names from the notebook or 'data_not_found'.

YAML response:"""
                
                messages = [AgnoMessage(role="user", content=prompt)]
                response = self.model_client.invoke(messages)
                
                # Extract content from response
                if hasattr(response, 'content'):
                    config_text = response.content
                elif isinstance(response, dict) and 'output' in response:
                    message_content = response['output']['message']['content']
                    if isinstance(message_content, list) and len(message_content) > 0:
                        config_text = message_content[0].get('text', str(response))
                    else:
                        config_text = str(message_content)
                else:
                    config_text = str(response)
                
                # Parse configuration
                config = self._parse_training_config(config_text)
                if config and config.get("success"):
                    # Convert data_variable to data_source for compatibility
                    data_variable = config.get("data_variable", "")
                    if data_variable != "data_not_found":
                        # For now, we'll create a message indicating we found the variable name
                        # but need the actual data - this is where we'd integrate with notebook execution
                        return {
                            "success": False,  # Set to False until we can access actual data
                            "training_type": "tabular",
                            "data_source": f"notebook_variable_{data_variable}",
                            "target_column": config.get("target_column", ""),
                            "problem_type": config.get("problem_type", "classification"),
                            "time_limit": 600,
                            "error": f"Found data variable '{data_variable}' in notebook but cannot access actual DataFrame yet. Feature under development.",
                            "found_info": config
                        }
                    else:
                        return {
                            "success": False,
                            "error": "Could not find actual data variables or column names in notebook content",
                            "suggestion": "Please ensure your notebook contains data loading code with clear variable names"
                        }
            
            # Fallback to heuristic extraction
            return self._heuristic_training_config(prep_res)
            
        except Exception as e:
            logger.error(f"Config extraction error: {e}")
            return self._heuristic_training_config(prep_res)
    
    def _parse_training_config(self, config_text):
        """Parse training configuration from LLM response"""
        try:
            # Extract YAML content
            yaml_content = config_text.strip()
            if "```yaml" in config_text:
                yaml_start = config_text.find("```yaml") + 7
                yaml_end = config_text.find("```", yaml_start)
                if yaml_end > yaml_start:
                    yaml_content = config_text[yaml_start:yaml_end].strip()
            
            config = yaml.safe_load(yaml_content)
            
            # Validate that we got a proper config
            if isinstance(config, dict):
                # Check if LLM found actual data or responded with "data_not_found"
                success = config.get("success", False)
                data_variable = config.get("data_variable", "")
                target_column = config.get("target_column", "")
                
                # Consider it successful if we found actual data info (not placeholders)
                if success and data_variable not in ["data_not_found", "", None] and target_column not in ["data_not_found", "", None]:
                    return config
                elif success:
                    # Partial success - found some info but not complete
                    return config
                else:
                    logger.warning("LLM could not find data information in notebook")
                    return {"success": False, "error": "No data information found"}
            
        except Exception as e:
            logger.warning(f"Config parsing error: {e}")
            logger.debug(f"YAML content: {yaml_content}")
        
        return None
    
    def _extract_data_from_notebook(self, notebook_content):
        """Extract actual DataFrames from notebook content string"""
        try:
            import pandas as pd
            import re
            import io
            from io import StringIO
            
            if not notebook_content:
                return {"success": False, "error": "No notebook content available"}
            
            # Parse notebook content to find DataFrame outputs
            dataframes = {}
            target_columns = []
            
            # Look for DataFrame outputs (df.head(), df.info(), df.shape, etc.)
            # Pattern to find cell outputs with tabular data
            cell_pattern = r"--- Cell \d+ \(CODE\) ---.*?SOURCE:\n(.*?)(?=OUTPUTS:|--- Cell|\Z)"
            output_pattern = r"OUTPUTS:\s*Output \d+ \([^)]+\):\s*(.*?)(?=\n\s*Output|\n--- Cell|\Z)"
            
            cells = re.findall(cell_pattern, notebook_content, re.DOTALL)
            
            for i, cell_source in enumerate(cells):
                # Look for DataFrame variable assignments
                df_assignments = re.findall(r"(\w+)\s*=.*?pd\.read_\w+\(", cell_source)
                
                # Look for df.head() or similar display commands
                display_commands = re.findall(r"(\w+)\.(?:head|tail|info|describe|shape|columns)", cell_source)
                
                # Combine variable names
                variable_names = list(set(df_assignments + display_commands))
                
                # Extract target column references
                target_refs = re.findall(r"(?:y|target|label)\s*=\s*\w+\[['\"](.*?)['\"]\]", cell_source)
                target_columns.extend(target_refs)
            
            # Look for actual DataFrame output data in the outputs
            outputs = re.findall(output_pattern, notebook_content, re.DOTALL)
            
            for output in outputs:
                # Try to parse tabular data from output
                dataframe = self._parse_tabular_output(output.strip())
                if dataframe is not None:
                    # Assign to first found variable name or default to 'df'
                    var_name = variable_names[0] if variable_names else 'df'
                    dataframes[var_name] = dataframe
                    break  # Use first successfully parsed DataFrame
            
            if dataframes:
                # Get the first DataFrame
                df_name, df = next(iter(dataframes.items()))
                
                # Determine target column
                target_col = None
                if target_columns:
                    # Use first target column that exists in the DataFrame
                    for col in target_columns:
                        if col in df.columns:
                            target_col = col
                            break
                
                # If no explicit target found, try to infer
                if not target_col:
                    target_col = self._infer_target_column_from_df(df)
                
                # Determine problem type
                problem_type = "classification"
                if target_col and target_col in df.columns:
                    if df[target_col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        # Check if it looks like regression (many unique values)
                        unique_ratio = len(df[target_col].unique()) / len(df)
                        if unique_ratio > 0.1:  # More than 10% unique values suggests regression
                            problem_type = "regression"
                
                return {
                    "success": True,
                    "dataframe": df,
                    "target_column": target_col,
                    "problem_type": problem_type,
                    "variable_name": df_name,
                    "dataframe_info": {
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "dtypes": df.dtypes.to_dict()
                    }
                }
            
            return {"success": False, "error": "No DataFrame data found in notebook outputs"}
            
        except Exception as e:
            logger.error(f"Data extraction error: {e}")
            return {"success": False, "error": str(e)}
    
    def _parse_tabular_output(self, output_text):
        """Parse tabular output text to reconstruct DataFrame"""
        try:
            import pandas as pd
            from io import StringIO
            
            lines = output_text.strip().split('\n')
            
            # Look for DataFrame-like output patterns
            # Pattern 1: Standard df.head() output with index and columns
            if any('  ' in line and not line.strip().startswith('[') for line in lines):
                # Try to parse as whitespace-separated tabular data
                # Remove common DataFrame artifacts
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    # Skip empty lines and non-data lines
                    if line and not line.startswith('[') and not line.startswith('...'):
                        clean_lines.append(line)
                
                if len(clean_lines) >= 2:  # At least header + one data row
                    try:
                        # Try parsing with pandas
                        data_text = '\n'.join(clean_lines)
                        df = pd.read_csv(StringIO(data_text), sep=r'\s+', engine='python')
                        
                        # Basic validation
                        if len(df) > 0 and len(df.columns) > 1:
                            return df
                    except Exception:
                        pass
            
            # Pattern 2: CSV-like output
            if ',' in output_text and '\n' in output_text:
                try:
                    df = pd.read_csv(StringIO(output_text))
                    if len(df) > 0 and len(df.columns) > 1:
                        return df
                except Exception:
                    pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Tabular output parsing error: {e}")
            return None
    
    def _infer_target_column_from_df(self, df):
        """Infer likely target column from DataFrame structure"""
        # Common target column names
        target_names = ['target', 'label', 'y', 'class', 'category', 'outcome', 'result', 'price', 'value']
        
        # Check for exact matches
        for col in df.columns:
            if col.lower() in target_names:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for target_name in target_names:
                if target_name in col.lower():
                    return col
        
        # Default: use last column (common ML convention)
        return df.columns[-1] if len(df.columns) > 0 else None
    
    def _heuristic_training_config(self, prep_res):
        """Heuristic training configuration extraction"""
        # Look for actual data files in common locations
        data_source = self._find_data_files()
        
        return {
            "success": bool(data_source),
            "training_type": "tabular",
            "target_column": "target",
            "problem_type": "classification", 
            "time_limit": 600,
            "data_source": data_source,
            "note": "Using heuristic configuration - please provide specific training parameters" if data_source else "No data files found - please specify data source"
        }
    
    def _find_data_files(self):
        """Find data files in common locations"""
        from pathlib import Path
        
        working_dir = Path.cwd()
        data_extensions = ['.csv', '.json', '.xlsx', '.parquet', '.tsv']
        search_paths = [
            working_dir,
            working_dir / 'data',
            working_dir / 'datasets',
            working_dir / 'files'
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for ext in data_extensions:
                    data_files = list(search_path.glob(f'*{ext}'))
                    if data_files:
                        return str(data_files[0])  # Return first found data file
        
        return ""  # No data files found
    
    def _train_tabular_model(self, config):
        """Train tabular model using AutoGluon with enhanced validation"""
        try:
            # Extract data source
            data_source = config.get("data_source", "")
            target_column = config.get("target_column", "target")
            
            # Enhanced data source validation
            if not data_source or data_source.strip() == "":
                return {"success": False, "error": "No data source specified"}
            
            # Check for placeholder text
            placeholders = ["[path to", "placeholder", "your_data", "data_file", "not specified", "no data source"]
            if any(placeholder in data_source.lower() for placeholder in placeholders):
                return {"success": False, "error": f"Invalid data source (placeholder detected): {data_source}"}
            
            logger.info(f"🚀 Training tabular model with data: {data_source}, target: {target_column}")
            
            # Train model using enhanced AutoGluon tool
            result = self.autogluon_tool.train_tabular_model(
                data=data_source,
                target_column=target_column,
                problem_type=config.get("problem_type"),
                time_limit=config.get("time_limit", 600),
                presets=config.get("presets", "best_quality")
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Tabular training error: {e}")
            return {"success": False, "error": str(e)}
    
    def _train_multimodal_model(self, config):
        """Train multimodal model using AutoGluon"""
        try:
            result = self.autogluon_tool.train_multimodal_model(
                data=config.get("data_source", ""),
                target_column=config.get("target_column", "target"),
                problem_type=config.get("problem_type"),
                time_limit=config.get("time_limit", 600),
                presets=config.get("presets", "best_quality")
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Multimodal training error: {e}")
            return {"success": False, "error": str(e)}
    
    def _train_timeseries_model(self, config):
        """Train time series model using AutoGluon"""
        try:
            result = self.autogluon_tool.train_timeseries_model(
                data=config.get("data_source", ""),
                target_column=config.get("target_column", "target"),
                prediction_length=config.get("prediction_length", 24),
                time_limit=config.get("time_limit", 600),
                presets=config.get("presets", "best_quality")
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Time series training error: {e}")
            return {"success": False, "error": str(e)}
    
    def _fallback_ml_training(self, prep_res):
        """Fallback ML training when AutoGluon unavailable"""
        return {
            "success": False,
            "training_result": f"""## 🤖 ML Training (Fallback Mode)

            **Query:** {prep_res['user_query']}

            **Status:** AutoGluon not available - providing ML training guidance.

            ## 🚀 AutoGluon Installation
            ```bash
            # Install AutoGluon (choose based on your needs)
            pip install autogluon              # Full installation
            pip install autogluon.tabular      # Tabular only
            pip install autogluon.multimodal   # Multimodal only
            pip install autogluon.timeseries   # Time series only
            ```

            ## 📊 Training Framework

            ### 1. Tabular Data (Classification/Regression)
            ```python
            from autogluon.tabular import TabularDataset, TabularPredictor

            # Load data
            train_data = TabularDataset('your_data.csv')

            # Train model
            predictor = TabularPredictor(label='target').fit(
                train_data,
                time_limit=600,
                presets='best_quality'
            )

            # Evaluate
            leaderboard = predictor.leaderboard()
            ```

            ### 2. Multimodal Data (Text/Image/Mixed)
            ```python
            from autogluon.multimodal import MultiModalPredictor

            # Train model
            predictor = MultiModalPredictor(label='target').fit(
                train_data,
                time_limit=600
            )
            ```

            ### 3. Time Series Forecasting
            ```python
            from autogluon.timeseries import TimeSeriesPredictor

            # Train forecasting model
            predictor = TimeSeriesPredictor(
                target='value',
                prediction_length=24
            ).fit(data, time_limit=600)
            ```

            ## 💡 Next Steps
            1. Install AutoGluon using the commands above
            2. Prepare your data in the appropriate format
            3. Run the training code for your specific use case
            4. Use the trained model for predictions

            * Note: This is a fallback response. Install AutoGluon for automated ML training.*""",
            "installation_required": True
        }
    
    def post(self, shared, prep_res, exec_res):
        """Store ML training results"""
        shared["ml_training_result"] = exec_res.get("training_result", "")
        shared["ml_training_success"] = exec_res.get("success", False)
        shared["ml_model_path"] = exec_res.get("model_path", "")
        
        if exec_res.get("success"):
            return "complete"  # Go to complete analysis with ML results
        else:
            return "decide"  # Go back to decision node for alternative action


class CompleteAnalysisNode(Node):
    """Node for comprehensive data science analysis"""
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
    
    def prep(self, shared):
        """Prepare for complete analysis"""
        return {
            "user_query": shared.get("user_query", ""),
            "repo_context": shared.get("repo_context", ""),
            "notebook_content": shared.get("notebook_content", ""),
            "notebook_path": shared.get("notebook_path", ""),
            "context_summary": shared.get("context_summary", ""),
            "action_history": shared.get("action_history", [])
        }
    
    def exec(self, prep_res):
        """Execute comprehensive analysis"""
        try:
            if not self.model_client:
                return self._fallback_complete_analysis(prep_res)
            
            prompt = f"""You are a senior data science expert providing comprehensive analysis and recommendations.

                    USER QUERY: {prep_res['user_query']}

                    REPOSITORY CONTEXT:
                    {prep_res['repo_context'][:1500] if prep_res['repo_context'] else 'No repo context available'}

                    NOTEBOOK CONTENT:
                    {prep_res['notebook_content'][:2500] if prep_res['notebook_content'] else 'No notebook content available'}

                    NOTEBOOK PATH: {prep_res['notebook_path']}

                    CONTEXT SUMMARY: {prep_res['context_summary']}

                    PREVIOUS ACTIONS: {prep_res['action_history']}

                    Provide a comprehensive data science analysis with:

                    ## 📊 Current State Analysis
                    - Thorough assessment of the current notebook content
                    - Data quality, structure, and completeness evaluation
                    - Current methodology and approach analysis
                    - Identification of strengths and weaknesses

                    ## 🎯 Targeted Recommendations
                    - Specific, actionable recommendations based on the user's query
                    - Priority-ordered suggestions for improvement
                    - Alternative approaches and methodologies to consider
                    - Best practices and optimization opportunities

                    ## 💻 Implementation Code
                    - Ready-to-use code snippets that can be directly implemented
                    - Proper imports and variable handling
                    - Comments explaining the approach and rationale
                    - Error handling and edge case considerations

                    ## 🔄 Next Steps Roadmap
                    - Clear, prioritized action items
                    - Timeline and dependency considerations
                    - Success metrics and validation approaches
                    - Long-term development suggestions

                    ## 🧪 Testing & Validation
                    - Suggested testing approaches for the analysis
                    - Validation methods for results
                    - Quality assurance recommendations
                    - Performance optimization suggestions

                    Focus on providing actionable, specific guidance that directly addresses the user's needs while building upon existing work."""
                                
            messages = [AgnoMessage(role="user", content=prompt)]
            response = self.model_client.invoke(messages)
            
            # Extract content from Bedrock response format
            if hasattr(response, 'content'):
                complete_analysis = response.content
            elif isinstance(response, dict):
                # Handle Bedrock response format
                if 'output' in response and 'message' in response['output']:
                    message_content = response['output']['message']['content']
                    if isinstance(message_content, list) and len(message_content) > 0:
                        complete_analysis = message_content[0].get('text', str(response))
                    else:
                        complete_analysis = str(message_content)
                else:
                    complete_analysis = str(response)
            else:
                complete_analysis = str(response)
            
            return {"complete_analysis": complete_analysis, "success": True}
            
        except Exception as e:
            logger.error(f"❌ Complete analysis error: {e}")
            return self._fallback_complete_analysis(prep_res)
    
    def _fallback_complete_analysis(self, prep_res):
        """Fallback complete analysis when model unavailable"""
        return {
            "complete_analysis": f"""# Data Science Analysis (Fallback Mode)

            ## Query: {prep_res['user_query']}

            ## Current Status
            - Repository Context: {'✅ Available' if prep_res['repo_context'] else '❌ Not found'}
            - Notebook Content: {'✅ Available' if prep_res['notebook_content'] else '❌ Not found'}
            - Notebook Path: {prep_res['notebook_path'] or 'Not specified'}

            ## Analysis Framework

            ### 📊 Data Assessment
            1. **Data Quality**: Check for missing values, duplicates, and inconsistencies
            2. **Data Structure**: Understand data types, distributions, and relationships
            3. **Data Completeness**: Assess coverage and identify gaps

            ### 🎯 Methodology Review
            1. **Current Approach**: Evaluate existing analysis methods
            2. **Best Practices**: Compare against industry standards
            3. **Optimization**: Identify areas for improvement

            ### 💻 Implementation Template
            ```python
            # Basic analysis template
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Data loading and initial exploration
            df = pd.read_csv('your_data.csv')
            print(df.info())
            print(df.describe())

            # Basic visualizations
            plt.figure(figsize=(12, 8))
            # Add your visualization code here
            plt.show()

            # Statistical analysis
            # Add your statistical analysis code here
            ```

            ### 🔄 Next Steps
            1. **Configure AI Model**: Set up AWS Bedrock for detailed analysis
            2. **Data Exploration**: Apply the template above to your data
            3. **Iterative Improvement**: Refine analysis based on findings
            4. **Validation**: Test and validate your results

            ## Recommendations
            - Set up proper AI model configuration for detailed assistance
            - Follow the analysis framework systematically
            - Document your findings and methodology
            - Consider peer review and validation of results

            *Note: This is a fallback response. Configure the AI model for detailed, context-aware analysis.*""",
                        "success": False
            }
    
    def post(self, shared, prep_res, exec_res):
        """Store complete analysis results"""
        shared["final_response"] = exec_res.get("complete_analysis", "")
        shared["analysis_complete"] = True
        return "end"  # End the analysis


class DataScienceAgent(Flow):
    """
    PocketFlow Agent for Data Science Analysis
    
    A sophisticated agent that can reason about data science tasks,
    make decisions, and provide targeted analysis and recommendations.
    """
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
        
        # Initialize nodes
        self.decide_node = DecideAction(model_client=model_client)
        self.greeting_node = GreetingNode(model_client=model_client)
        self.analyze_node = DataAnalysisNode(model_client=model_client)
        self.ml_training_node = MLTrainingNode(model_client=model_client)
        self.complete_node = CompleteAnalysisNode(model_client=model_client)
        
        # Set up the agent flow
        self.start(self.decide_node)
        
        # Connect decision node to action nodes
        self.decide_node - "greeting" >> self.greeting_node
        self.decide_node - "analyze" >> self.analyze_node
        self.decide_node - "ml_training" >> self.ml_training_node
        self.decide_node - "complete" >> self.complete_node
        
        # Connect analysis node back to decision (for iterative analysis)
        self.analyze_node - "decide" >> self.decide_node
        
        # Connect greeting node to complete for complex queries
        self.greeting_node - "complete" >> self.complete_node
        
        # Connect ML training node back to decision or complete
        self.ml_training_node - "decide" >> self.decide_node
        self.ml_training_node - "complete" >> self.complete_node
        
        logger.info("✅ DataScienceAgent initialized with decision-making, greeting handling, and ML training capabilities")
        logger.debug(f"Agent nodes: {[node.__class__.__name__ for node in [self.decide_node, self.greeting_node, self.analyze_node, self.ml_training_node, self.complete_node]]}")
    
    def prep(self, shared):
        """Agent preparation - load context"""
        logger.info("🚀 Starting agent preparation...")
        
        # Load repo context
        logger.debug("Loading repo context...")
        repo_context = self._load_repo_context()
        shared["repo_context"] = repo_context
        logger.info(f"📋 Repo context: {'✅ Loaded' if repo_context else '❌ Not found'}")
        
        # Load notebook content
        logger.debug("Loading notebook content...")
        user_query = shared.get("user_query", "")
        logger.debug(f"User query for notebook extraction: {user_query}")
        
        notebook_content, notebook_path = self._load_notebook_content(user_query)
        shared["notebook_content"] = notebook_content
        shared["notebook_path"] = notebook_path
        
        logger.info(f"📓 Notebook: {'✅ Loaded' if notebook_content else '❌ Not found'}")
        if notebook_path:
            logger.info(f"📁 Notebook path: {notebook_path}")
        
        # Initialize tracking
        shared["action_history"] = []
        shared["analysis_complete"] = False
        
        prep_result = {
            "agent_initialized": True,
            "context_loaded": bool(repo_context),
            "notebook_loaded": bool(notebook_content)
        }
        
        logger.info(f"✅ Agent preparation complete: {prep_result}")
        return prep_result
    
    def _load_repo_context(self):
        """Load repository context from repo_context.md"""
        try:
            repo_path = Path.cwd() / "repo_context.md"
            if repo_path.exists():
                with open(repo_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"❌ Error loading repo context: {e}")
        return ""
    
    def _load_notebook_content(self, user_query):
        """Load notebook content based on user query"""
        try:
            logger.debug(f"Loading notebook content for query: {user_query[:50]}...")
            notebook_tool = NotebookReaderTool()
            
            # Extract notebook path from query or find default
            notebook_path = self._extract_notebook_path(user_query)
            logger.debug(f"Extracted notebook path: {notebook_path}")
            
            if notebook_path:
                logger.info(f"📖 Reading notebook: {notebook_path}")
                content = notebook_tool.extract_rag_context(str(notebook_path))
                logger.debug(f"Notebook content length: {len(content)} characters")
                
                if content.startswith("Error:"):
                    logger.error(f"❌ Notebook reading failed: {content}")
                    return "", str(notebook_path)
                else:
                    logger.info(f"✅ Successfully read notebook: {notebook_path}")
                    return content, str(notebook_path)
            else:
                logger.warning("❌ No notebook path found")
                return "", ""
            
        except Exception as e:
            logger.error(f"❌ Error loading notebook: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return "", ""
    
    def _extract_notebook_path(self, query):
        """Extract notebook path from query or find default"""
        working_dir = Path.cwd()
        logger.debug(f"Working directory: {working_dir}")
        
        # Look for explicit notebook path with "notebook:" syntax
        if "notebook:" in query.lower():
            logger.debug("Found 'notebook:' in query - extracting explicit path")
            parts = query.split("notebook:")
            if len(parts) > 1:
                path_part = parts[1].strip().split()[0]
                logger.debug(f"Extracted path part: {path_part}")
                notebook_path = Path(path_part)
                
                if not notebook_path.is_absolute():
                    notebook_path = working_dir / notebook_path
                    logger.debug(f"Converted to absolute path: {notebook_path}")
                
                if notebook_path.exists():
                    logger.debug(f"✅ Explicit notebook path exists: {notebook_path}")
                    return notebook_path
                else:
                    logger.warning(f"❌ Explicit notebook path does not exist: {notebook_path}")
        
        # Look for .ipynb file paths directly in the query (without "notebook:" prefix)
        if ".ipynb" in query:
            logger.debug("Found '.ipynb' in query - looking for direct path")
            # Split by whitespace and look for .ipynb files
            words = query.split()
            for word in words:
                if word.endswith('.ipynb'):
                    logger.debug(f"Found potential notebook path: {word}")
                    notebook_path = Path(word)
                    
                    # Try as absolute path first
                    if notebook_path.is_absolute() and notebook_path.exists():
                        logger.info(f"✅ Found absolute notebook path: {notebook_path}")
                        return notebook_path
                    
                    # Try as relative path from working directory
                    relative_path = working_dir / notebook_path
                    if relative_path.exists():
                        logger.info(f"✅ Found relative notebook path: {relative_path}")
                        return relative_path
                    
                    logger.debug(f"Path doesn't exist: {notebook_path}")
        
        # Look for .ipynb files in current directory
        logger.debug("Looking for .ipynb files in current directory")
        ipynb_files = list(working_dir.glob("*.ipynb"))
        logger.debug(f"Found {len(ipynb_files)} .ipynb files: {ipynb_files}")
        
        if ipynb_files:
            selected_notebook = ipynb_files[0]
            logger.debug(f"✅ Selected default notebook: {selected_notebook}")
            return selected_notebook
        
        logger.warning("❌ No notebook files found")
        return None
    
    def run_analysis(self, user_query, **kwargs):
        """Run the data science agent analysis"""
        try:
            # Initialize shared state
            shared = {
                "user_query": user_query,
                "timestamp": kwargs.get("timestamp", ""),
                "history": kwargs.get("history", ""),
                **kwargs
            }
            
            logger.info(f"🤖 Starting agent analysis for: {user_query[:50]}...")
            logger.debug(f"Agent context: history={bool(kwargs.get('history'))}, timestamp={kwargs.get('timestamp')}")
            
            # Run the agent
            result = self.run(shared)
            
            logger.info(f"🤖 Agent analysis completed - Success: {shared.get('analysis_complete', False)}")
            logger.debug(f"Actions taken: {shared.get('action_history', [])}")
            
            # Return results
            return {
                "success": shared.get("analysis_complete", False),
                "response": shared.get("final_response", "No response generated"),
                "context_loaded": bool(shared.get("repo_context", "")),
                "notebook_loaded": bool(shared.get("notebook_content", "")),
                "notebook_path": shared.get("notebook_path", ""),
                "action_history": shared.get("action_history", []),
                "processing_summary": {
                    "repo_context_loaded": bool(shared.get("repo_context", "")),
                    "notebook_loaded": bool(shared.get("notebook_content", "")),
                    "analysis_complete": shared.get("analysis_complete", False),
                    "actions_taken": len(shared.get("action_history", []))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Agent analysis error: {e}")
            return {
                "success": False,
                "response": f"Agent analysis error: {str(e)}",
                "error": str(e),
                "processing_summary": {
                    "repo_context_loaded": False,
                    "notebook_loaded": False,
                    "analysis_complete": False,
                    "actions_taken": 0
                }
            }
    
    def post(self, shared, prep_res, exec_res):
        """Agent completion"""
        shared["agent_completed"] = True
        logger.info(f"🤖 Agent completed - Actions taken: {len(shared.get('action_history', []))}")
        return exec_res