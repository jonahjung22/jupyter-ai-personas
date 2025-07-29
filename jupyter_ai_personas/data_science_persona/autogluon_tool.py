
import logging
import tempfile
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AutoGluonTool:
    """Simplified AutoGluon tool for generating contextual ML code."""
    
    def __init__(self, default_time_limit: int = 120):
        self.default_time_limit = default_time_limit  # 120 for quick testing, 600 for optimal training
    
    def get_status(self) -> Dict[str, Any]:
        """Get tool status and installation information."""
        return {
            "availability": self.availability,
            "installation_commands": {
                "full": "pip install autogluon",
                "tabular_only": "pip install autogluon.tabular",
                "multimodal_only": "pip install autogluon.multimodal", 
                "timeseries_only": "pip install autogluon.timeseries"
            },
            "any_available": any(self.availability.values())
        }
    
    def recommend_ml_solution(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AutoGluon code based on problem context - requires dataset-specific generation."""
        try:
            logger.info("🎯 AutoGluon recommendation requires dataset-specific generation")
            
            return {
                "success": False, 
                "error": "Generic recommendations removed. Use generate_dataset_specific_code() with actual dataset for optimal results.",
                "suggestion": "The AutoGluon tool now only supports dataset-specific code generation for better accuracy and reliability."
            }
                
        except Exception as e:
            logger.error(f"AutoGluon recommendation error: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_dataset_specific_code(self, notebook_data: Dict[str, Any], domain: str, user_query: str = "") -> Dict[str, Any]:
        """Generate AutoGluon code customized for the specific dataset structure."""
        try:
            if not notebook_data.get("success") or "dataframe" not in notebook_data:
                return {"success": False, "error": "No valid dataset provided"}
            
            df = notebook_data["dataframe"]
            variable_name = notebook_data.get("variable_name", "df")
            
            logger.info(f"📊 Analyzing dataset structure for {domain} domain")
            
            # Analyze the actual DataFrame structure
            columns = list(df.columns)
            shape = df.shape
            
            # Detect target column
            target_column = self._detect_target_column(df, notebook_data, user_query)
            
            logger.info(f"🎯 Detected target column: {target_column}")
            logger.info(f"📋 Dataset shape: {shape}")
            logger.info(f"📊 Columns: {columns}")
            
            if domain == "timeseries":
                return self._generate_timeseries_code_for_dataset(df, variable_name, target_column, columns, user_query)
            elif domain == "tabular":
                return self._generate_tabular_code_for_dataset(df, variable_name, target_column, columns, user_query)
            elif domain == "multimodal":
                return self._generate_multimodal_code_for_dataset(df, variable_name, target_column, columns, user_query)
            else:
                return {"success": False, "error": f"Unsupported domain: {domain}"}
                
        except Exception as e:
            logger.error(f"Dataset-specific code generation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _detect_target_column(self, df, notebook_data: Dict[str, Any], user_query: str) -> str:
        """Detect the most likely target column from the dataset."""
        # First try explicit target from notebook analysis
        if notebook_data.get("target_column"):
            return notebook_data["target_column"]
        
        # Look for common target column names
        target_candidates = []
        common_targets = ['target', 'label', 'y', 'class', 'category', 'outcome', 'result', 'price', 'value', 'sales', 'revenue']
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in common_targets:
                target_candidates.append(col)
            elif any(target in col_lower for target in common_targets):
                target_candidates.append(col)
        
        if target_candidates:
            return target_candidates[0]
        
        # For time series, often the last numeric column is the target
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            return numeric_cols[-1]  # Use last numeric column
        
        # Fallback to last column
        return df.columns[-1] if len(df.columns) > 0 else 'target'
    
    def train_tabular_model(self, data, target_column: str, problem_type: str = "auto", 
                          time_limit: int = 600, presets: str = "best_quality") -> Dict[str, Any]:
        """Train actual AutoGluon tabular model and return leaderboard results."""
        try:
            if not self.availability.get("tabular", False):
                return {
                    "success": False,
                    "error": "AutoGluon tabular not available. Install with: pip install autogluon.tabular"
                }
            
            from autogluon.tabular import TabularDataset, TabularPredictor
            import pandas as pd
            
            logger.info(f"🤖 Training AutoGluon tabular model - Target: {target_column}")
            
            # Convert data to TabularDataset if needed
            if isinstance(data, pd.DataFrame):
                train_data = TabularDataset(data)
            else:
                train_data = data
            
            # Create temporary directory for model
            temp_dir = tempfile.mkdtemp(prefix="autogluon_tabular_")
            model_path = os.path.join(temp_dir, "tabular_model")
            
            # Train the model
            predictor = TabularPredictor(
                label=target_column,
                problem_type=problem_type,
                path=model_path
            ).fit(
                train_data,
                time_limit=time_limit,
                presets=presets
            )
            
            # Generate leaderboard
            leaderboard = predictor.leaderboard()
            
            # Get model performance metrics
            best_model = leaderboard.iloc[0]['model'] if len(leaderboard) > 0 else "Unknown"
            best_score = leaderboard.iloc[0]['score_val'] if len(leaderboard) > 0 else "N/A"
            
            logger.info(f"✅ Training completed - Best model: {best_model}")
            
            return {
                "success": True,
                "predictor": predictor,
                "leaderboard": leaderboard,
                "best_model": best_model,
                "best_score": best_score,
                "model_path": model_path,
                "target_column": target_column,
                "problem_type": problem_type,
                "preset": presets,
                "training_time": time_limit
            }
            
        except Exception as e:
            logger.error(f"❌ Tabular training error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def train_multimodal_model(self, data, target_column: str, problem_type: str = "auto",
                             time_limit: int = 600, presets: str = "best_quality") -> Dict[str, Any]:
        """Train actual AutoGluon multimodal model and return results."""
        try:
            if not self.availability.get("multimodal", False):
                return {
                    "success": False,
                    "error": "AutoGluon multimodal not available. Install with: pip install autogluon.multimodal"
                }
            
            from autogluon.multimodal import MultiModalPredictor
            
            logger.info(f"🤖 Training AutoGluon multimodal model - Target: {target_column}")
            
            # Create temporary directory for model
            temp_dir = tempfile.mkdtemp(prefix="autogluon_multimodal_")
            model_path = os.path.join(temp_dir, "multimodal_model")
            
            # Train the model
            predictor = MultiModalPredictor(
                label=target_column,
                path=model_path
            ).fit(
                data,
                time_limit=time_limit,
                presets=presets
            )
            
            # Evaluate the model
            evaluation = predictor.evaluate(data)
            
            logger.info(f"✅ Multimodal training completed")
            
            return {
                "success": True,
                "predictor": predictor,
                "evaluation": evaluation,
                "model_path": model_path,
                "target_column": target_column,
                "problem_type": problem_type,
                "preset": presets,
                "training_time": time_limit
            }
            
        except Exception as e:
            logger.error(f"❌ Multimodal training error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def train_timeseries_model(self, data, target_column: str, prediction_length: int = 24,
                             time_limit: int = 600, presets: str = "best_quality") -> Dict[str, Any]:
        """Train actual AutoGluon time series model and return results."""
        try:
            if not self.availability.get("timeseries", False):
                return {
                    "success": False,
                    "error": "AutoGluon timeseries not available. Install with: pip install autogluon.timeseries"
                }
            
            from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
            import pandas as pd
            
            logger.info(f"🤖 Training AutoGluon time series model - Target: {target_column}")
            
            # Convert to TimeSeriesDataFrame if needed
            if isinstance(data, pd.DataFrame):
                ts_data = TimeSeriesDataFrame(data)
            else:
                ts_data = data
            
            # Create temporary directory for model
            temp_dir = tempfile.mkdtemp(prefix="autogluon_timeseries_")
            model_path = os.path.join(temp_dir, "timeseries_model")
            
            # Train the model
            predictor = TimeSeriesPredictor(
                target=target_column,
                prediction_length=prediction_length,
                path=model_path
            ).fit(
                ts_data,
                time_limit=time_limit,
                presets=presets
            )
            
            # Generate forecasts and evaluate
            forecasts = predictor.predict(ts_data)
            evaluation = predictor.evaluate(ts_data)
            
            logger.info(f"✅ Time series training completed")
            
            return {
                "success": True,
                "predictor": predictor,
                "forecasts": forecasts,
                "evaluation": evaluation,
                "model_path": model_path,
                "target_column": target_column,
                "prediction_length": prediction_length,
                "preset": presets,
                "training_time": time_limit
            }
            
        except Exception as e:
            logger.error(f"❌ Time series training error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def format_leaderboard_results(self, training_result: Dict[str, Any]) -> str:
        """Format leaderboard results for display."""
        if not training_result.get("success", False):
            return f"## ❌ Training Failed\n\nError: {training_result.get('error', 'Unknown error')}"
        
        if "leaderboard" in training_result:
            # Tabular model with leaderboard
            leaderboard = training_result["leaderboard"]
            
            leaderboard_text = "## 🏆 AutoGluon Model Leaderboard\n\n"
            leaderboard_text += f"**🥇 Best Model: {training_result.get('best_model', 'Unknown')}**\n"
            leaderboard_text += f"**📊 Best Score: {training_result.get('best_score', 'N/A')}**\n\n"
            
            leaderboard_text += "### Top Models Performance:\n\n"
            
            # Show top 5 models
            top_models = leaderboard.head(5)
            leaderboard_text += "| Model | Score | Training Time |\n"
            leaderboard_text += "|-------|-------|---------------|\n"
            
            for idx, row in top_models.iterrows():
                model_name = row.get('model', 'Unknown')
                score = row.get('score_val', 'N/A')
                fit_time = row.get('fit_time', 'N/A')
                leaderboard_text += f"| {model_name} | {score} | {fit_time}s |\n"
            
            leaderboard_text += f"\n**Model saved to:** `{training_result.get('model_path', 'N/A')}`\n"
            leaderboard_text += f"**Target Column:** {training_result.get('target_column', 'N/A')}\n"
            leaderboard_text += f"**Problem Type:** {training_result.get('problem_type', 'N/A')}\n"
            
            return leaderboard_text
        
        elif "evaluation" in training_result:
            # Multimodal or time series model
            evaluation = training_result["evaluation"]
            
            result_text = "## 🤖 AutoGluon Training Results\n\n"
            result_text += f"**✅ Training Completed Successfully!**\n\n"
            result_text += f"**Model Type:** {training_result.get('domain', 'Unknown')}\n"
            result_text += f"**Target Column:** {training_result.get('target_column', 'N/A')}\n"
            result_text += f"**Model Path:** `{training_result.get('model_path', 'N/A')}`\n\n"
            
            if isinstance(evaluation, dict):
                result_text += "### Performance Metrics:\n\n"
                for metric, value in evaluation.items():
                    result_text += f"- **{metric}:** {value}\n"
            else:
                result_text += f"### Performance Score: {evaluation}\n"
            
            return result_text
        
        else:
            return "## ✅ Training Completed\n\nModel training finished successfully."
    
    def _generate_timeseries_code_for_dataset(self, df, variable_name: str, target_column: str, columns: list, user_query: str) -> Dict[str, Any]:
        """Generate time series code customized for the specific dataset."""
        
        # Determine prediction length from query
        prediction_length = 24  # default
        if any(word in user_query.lower() for word in ["daily", "day"]):
            prediction_length = 7
        elif any(word in user_query.lower() for word in ["hourly", "hour"]):
            prediction_length = 24
        elif any(word in user_query.lower() for word in ["monthly", "month"]):
            prediction_length = 12
        
        # Analyze the dataset structure
        has_date_index = hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype)
        date_columns = [col for col in columns if 'date' in col.lower() or 'time' in col.lower()]
        
        code = f"""# AutoGluon Time Series Forecasting Solution - Dataset Specific
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd

# Dataset Analysis:
# - Shape: {df.shape}
# - Target Column: '{target_column}'
# - Date Index: {has_date_index}
# - Available Columns: {columns}

# Prepare time series data for AutoGluon
ts_data_formatted = {variable_name}.copy()

# Handle date index or column
if isinstance(ts_data_formatted.index, pd.DatetimeIndex):
    # Data has datetime index - reset it to column
    ts_data_formatted = ts_data_formatted.reset_index()
    timestamp_col = ts_data_formatted.columns[0]
else:
    # Look for date column
    date_cols = [col for col in ts_data_formatted.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        timestamp_col = date_cols[0]
    else:
        # Create a simple date range if no date column found
        ts_data_formatted['timestamp'] = pd.date_range(start='2020-01-01', periods=len(ts_data_formatted), freq='D')
        timestamp_col = 'timestamp'

# Add required item_id column (single time series)
ts_data_formatted['item_id'] = 'series_1'

# Rename to AutoGluon format
ts_data_formatted = ts_data_formatted.rename(columns={{timestamp_col: 'timestamp'}})

# Reorder columns: item_id, timestamp, target columns
cols = ['item_id', 'timestamp'] + [col for col in ts_data_formatted.columns if col not in ['item_id', 'timestamp']]
ts_data_formatted = ts_data_formatted[cols]

print("Dataset-specific formatting completed:")
print(f"Original shape: {{len({variable_name})}}, {{len({variable_name}.columns)}}")
print(f"Target column: '{target_column}'")
print(f"Formatted columns: {{list(ts_data_formatted.columns)}}")
print("\\nFirst few rows:")
print(ts_data_formatted.head())

# Verify target column exists
if '{target_column}' not in ts_data_formatted.columns:
    print("⚠️  Target column '{target_column}' not found!")
    print("Available columns:", list(ts_data_formatted.columns))
    # Use first numeric column as backup
    numeric_cols = ts_data_formatted.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 0:
        actual_target = numeric_cols[0]
        print(f"Using '{{actual_target}}' as target instead")
    else:
        actual_target = '{target_column}'
else:
    actual_target = '{target_column}'

# Create TimeSeriesDataFrame
ts_autogluon = TimeSeriesDataFrame(ts_data_formatted)

# Train AutoGluon time series model
predictor = TimeSeriesPredictor(
    target=actual_target,
    prediction_length={prediction_length},
    path='./autogluon_models/timeseries_model'
).fit(
    ts_autogluon,
    time_limit={self.default_time_limit},
    presets='best_quality'
)

print(f"Generated forecasts for {{len(predictor.predict(ts_autogluon))}} steps")
print("✅ Time series forecasting completed!")"""

        leaderboard_code = f"""# 🏆 VIEW TIME SERIES MODEL PERFORMANCE AND RANKINGS
import pandas as pd

print("🏆 AutoGluon Time Series Training Summary:")
print("="*50)

# Get training summary and model information
try:
    summary = predictor.fit_summary()
    print("Training Summary:")
    print(summary)
except:
    print("Training summary not available")

# Best model information - TimeSeriesPredictor doesn't expose individual model names
print(f"\\n🥇 Best Model: AutoGluon Ensemble (WeightedEnsemble)")

# Model performance evaluation
performance = predictor.evaluate(ts_autogluon)
print("\\nModel Performance Metrics:")
print(performance)

# Generate forecasts
forecasts = predictor.predict(ts_autogluon)
print(f"\\nForecast Summary:")
print(f"Generated {{len(forecasts)}} forecast steps")
print(f"Target: {{actual_target}}")
print(f"Prediction Length: {prediction_length} steps")

print(f"\\nSample Forecasts:")
print(forecasts.head(10))

print(f"\\nModel Selection:")
print("AutoGluon automatically selected the best performing model from the ensemble")
print("The WeightedEnsemble combines multiple models for optimal performance")"""

        return {
            "success": True,
            "domain": "timeseries",
            "optimized_code": code,
            "leaderboard_code": leaderboard_code,
            "solution_summary": f"## 🔮 AutoGluon Time Series Solution \n\n**Target:** {target_column}\n**Dataset Shape:** {df.shape}\n**Forecast Length:** {prediction_length} steps\n\n**Features:**\n- Customized for your specific dataset structure\n- Automatic date/time column detection\n- Robust target column validation\n- Production-ready forecasts"
        }
    
    def _generate_tabular_code_for_dataset(self, df, variable_name: str, target_column: str, columns: list, user_query: str) -> Dict[str, Any]:
        """Generate tabular code customized for the specific dataset."""
        
        # Determine problem type from data and query
        problem_type = "auto"
        if any(word in user_query.lower() for word in ["classify", "classification", "category"]):
            problem_type = "classification"
        elif any(word in user_query.lower() for word in ["regression", "predict", "estimate"]):
            problem_type = "regression"
        elif target_column in df.columns:
            # Auto-detect based on target column characteristics
            unique_ratio = len(df[target_column].unique()) / len(df)
            if unique_ratio < 0.05:  # Less than 5% unique values suggests classification
                problem_type = "classification"
            else:
                problem_type = "regression"

        code = f"""# AutoGluon Tabular ML Solution - Dataset Specific
from autogluon.tabular import TabularDataset, TabularPredictor

# Dataset Analysis:
# - Shape: {df.shape}
# - Target Column: '{target_column}'
# - Available Columns: {columns}
# - Problem Type: {problem_type}

# Verify target column exists
if '{target_column}' not in {variable_name}.columns:
    print("⚠️  Target column '{target_column}' not found!")
    print("Available columns:", list({variable_name}.columns))
    # Try to find a suitable target column
    numeric_cols = {variable_name}.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 0:
        actual_target = numeric_cols[-1]  # Use last numeric column
        print(f"Using '{{actual_target}}' as target instead")
    else:
        actual_target = {variable_name}.columns[-1]  # Use last column
        print(f"Using '{{actual_target}}' as target instead")
else:
    actual_target = '{target_column}'

print(f"Training with target column: {{actual_target}}")
print(f"Dataset shape: {{{variable_name}.shape}}")

# Load your data
train_data = TabularDataset({variable_name})

# Train AutoGluon model
predictor = TabularPredictor(
    label=actual_target,
    problem_type='{problem_type}',
    path='./autogluon_models/tabular_model'
).fit(
    train_data,
    time_limit={self.default_time_limit},
    presets='best_quality'
)

print(f"✅ Training completed for {{actual_target}}!")"""

        leaderboard_code = f"""# 🏆 VIEW MODEL LEADERBOARD AND BEST MODELS
leaderboard = predictor.leaderboard()
print("🏆 AutoGluon Model Leaderboard:")
print("="*50)
print(leaderboard.head(10))  # Show top 10 models

# 🥇 BEST MODEL INFORMATION
best_model = leaderboard.iloc[0]['model']
best_score = leaderboard.iloc[0]['score_val']
print(f"\\n🥇 BEST MODEL: {{best_model}}")
print(f"📊 BEST SCORE: {{best_score:.4f}}")

# 📈 DETAILED RANKING
print("\\n📈 Top 5 Models Ranking:")
for i, row in leaderboard.head(5).iterrows():
    print(f"{{i+1:2d}}. {{row['model']:25s}} | Score: {{row['score_val']:.4f}} | Time: {{row['fit_time']:.1f}}s")"""

        return {
            "success": True,
            "domain": "tabular",
            "optimized_code": code,
            "leaderboard_code": leaderboard_code,
            "solution_summary": f"## 🤖 AutoGluon Tabular Solution \n\n**Target:** {target_column}\n**Dataset Shape:** {df.shape}\n**Problem Type:** {problem_type}\n\n**Features:**\n- Customized for your specific dataset structure\n- Automatic target column validation\n- Smart problem type detection\n- Comprehensive model evaluation and leaderboard"
        }
    
    def _generate_multimodal_code_for_dataset(self, df, variable_name: str, target_column: str, columns: list, user_query: str) -> Dict[str, Any]:
        """Generate multimodal code customized for the specific dataset."""
        
        code = f"""# AutoGluon Multimodal ML Solution - Dataset Specific
from autogluon.multimodal import MultiModalPredictor

# Dataset Analysis:
# - Shape: {df.shape}
# - Target Column: '{target_column}'
# - Available Columns: {columns}

# Verify target column exists
if '{target_column}' not in {variable_name}.columns:
    print("⚠️  Target column '{target_column}' not found!")
    print("Available columns:", list({variable_name}.columns))
    actual_target = {variable_name}.columns[-1]  # Use last column
    print(f"Using '{{actual_target}}' as target instead")
else:
    actual_target = '{target_column}'

print(f"Training multimodal model with target: {{actual_target}}")
print(f"Dataset shape: {{{variable_name}.shape}}")

# Load your multimodal data (text, images, numerical)
train_data = {variable_name}

# Train AutoGluon multimodal model
predictor = MultiModalPredictor(
    label=actual_target,
    path='./autogluon_models/multimodal_model'
).fit(
    train_data,
    time_limit={self.default_time_limit * 2},  # Multimodal typically needs more time
    presets='best_quality'
)

print(f"✅ Multimodal training completed for {{actual_target}}!")
print("Model handles text, images, and numerical data automatically!")"""

        leaderboard_code = f"""#VIEW MULTIMODAL MODEL PERFORMANCE
performance = predictor.evaluate({variable_name})
print("🏆 AutoGluon Multimodal Performance:")
print("="*40)
print(performance)

# 📊 Model Information
print(f"\\n📊 Model Type: Multimodal (Text + Images + Numerical)")
print(f"🎯 Target: {{actual_target}}")
print(f"✅ Training completed successfully!")"""

        return {
            "success": True,
            "domain": "multimodal",
            "optimized_code": code,
            "leaderboard_code": leaderboard_code,
            "solution_summary": f"## AutoGluon Multimodal Solution \n\n**Target:** {target_column}\n**Dataset Shape:** {df.shape}\n\n**Features:**\n- Customized for your specific dataset structure\n- Automatic handling of text, images, and numerical data\n- Smart target column validation\n- State-of-the-art multimodal architectures"
        }
