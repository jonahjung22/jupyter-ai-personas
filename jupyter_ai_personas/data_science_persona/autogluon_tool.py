"""
Comprehensive AutoGluon Tool for Data Science Agent

Provides automated machine learning capabilities across all AutoGluon domains:
- Tabular: Structured data prediction (classification/regression)
- Multimodal: Text, image, and mixed data tasks
- Time Series: Forecasting and temporal pattern analysis
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional, Union, List
import tempfile
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

class AutoGluonTool:
    """
    Comprehensive AutoGluon tool supporting tabular, multimodal, and time series tasks.
    
    Features:
    - Tabular prediction (classification/regression)
    - Multimodal tasks (text, image, mixed data)
    - Time series forecasting
    - Automated model selection and hyperparameter tuning
    - Model evaluation and interpretation
    - Deployment-ready outputs
    """
    
    def __init__(self, work_dir: Optional[str] = None):
        """Initialize AutoGluon tool with availability checks."""
        self.work_dir = Path(work_dir) if work_dir else Path.cwd() / "autogluon_models"
        self.work_dir.mkdir(exist_ok=True)
        
        # Check availability of each AutoGluon domain
        self.availability = self._check_availability()
    
    def _check_availability(self) -> Dict[str, bool]:
        """Check which AutoGluon domains are available."""
        availability = {
            "tabular": False,
            "multimodal": False,
            "timeseries": False
        }
        
        try:
            from autogluon.tabular import TabularPredictor
            availability["tabular"] = True
        except ImportError:
            logger.warning("AutoGluon tabular not available")
        
        try:
            from autogluon.multimodal import MultiModalPredictor
            availability["multimodal"] = True
        except ImportError:
            logger.warning("AutoGluon multimodal not available")
        
        try:
            from autogluon.timeseries import TimeSeriesPredictor
            availability["timeseries"] = True
        except ImportError:
            logger.warning("AutoGluon timeseries not available")
        
        return availability
    
    def get_status(self) -> Dict[str, Any]:
        """Get tool status and installation information."""
        return {
            "availability": self.availability,
            "work_directory": str(self.work_dir),
            "installation_commands": {
                "full": "pip install autogluon",
                "tabular_only": "pip install autogluon.tabular",
                "multimodal_only": "pip install autogluon.multimodal", 
                "timeseries_only": "pip install autogluon.timeseries"
            },
            "any_available": any(self.availability.values())
        }
    
    # TABULAR PREDICTION METHODS
    def train_tabular_model(
        self,
        data: Union[pd.DataFrame, str, Path],
        target_column: str,
        problem_type: Optional[str] = None,
        time_limit: int = 600,
        presets: str = "best_quality",
        eval_metric: Optional[str] = None,
        model_name: Optional[str] = None,
        test_data: Optional[Union[pd.DataFrame, str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train tabular prediction model."""
        if not self.availability["tabular"]:
            return {
                "success": False,
                "error": "AutoGluon tabular not available. Install with: pip install autogluon.tabular"
            }
        
        try:
            from autogluon.tabular import TabularDataset, TabularPredictor
            
            # Load and validate data
            train_data = TabularDataset(str(data) if isinstance(data, (str, Path)) else data)
            
            if target_column not in train_data.columns:
                return {
                    "success": False,
                    "error": f"Target column '{target_column}' not found. Available: {list(train_data.columns)}"
                }
            
            # Setup model
            model_name = model_name or f"tabular_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.work_dir / model_name
            
            logger.info(f"🚀 Training tabular model: {model_name}")
            logger.info(f"📊 Data shape: {train_data.shape}, Target: {target_column}")
            
            # Initialize and train predictor
            predictor = TabularPredictor(
                label=target_column,
                path=str(model_path),
                problem_type=problem_type,
                eval_metric=eval_metric,
                **kwargs
            )
            
            predictor.fit(
                train_data,
                time_limit=time_limit,
                presets=presets,
                verbosity=2
            )
            
            # Evaluate model
            results = self._evaluate_tabular_model(predictor, train_data, test_data)
            results.update({
                "success": True,
                "model_name": model_name,
                "model_path": str(model_path),
                "model_type": "tabular",
                "target_column": target_column,
                "problem_type": predictor.problem_type,
                "training_time": time_limit,
                "preset": presets
            })
            
            logger.info(f"✅ Tabular model training completed: {model_name}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Tabular training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _evaluate_tabular_model(self, predictor, train_data, test_data=None) -> Dict[str, Any]:
        """Evaluate tabular model performance."""
        results = {}
        
        try:
            # Leaderboard
            leaderboard = predictor.leaderboard(silent=True)
            results["leaderboard"] = leaderboard.to_dict('records')
            results["best_model"] = leaderboard.iloc[0]['model']
            results["best_score"] = leaderboard.iloc[0]['score_val']
            
            # Feature importance
            feature_importance = predictor.feature_importance(train_data)
            results["feature_importance"] = feature_importance.to_dict()
            
            # Test evaluation
            if test_data is not None:
                from autogluon.tabular import TabularDataset
                test_data = TabularDataset(str(test_data) if isinstance(test_data, (str, Path)) else test_data)
                test_performance = predictor.evaluate(test_data, silent=True)
                results["test_performance"] = test_performance
                
        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            results["evaluation_error"] = str(e)
        
        return results
    
    # MULTIMODAL PREDICTION METHODS
    def train_multimodal_model(
        self,
        data: Union[pd.DataFrame, str, Path],
        target_column: str,
        problem_type: Optional[str] = None,
        time_limit: int = 600,
        presets: str = "best_quality",
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train multimodal prediction model for text, image, and mixed data."""
        if not self.availability["multimodal"]:
            return {
                "success": False,
                "error": "AutoGluon multimodal not available. Install with: pip install autogluon.multimodal"
            }
        
        try:
            from autogluon.multimodal import MultiModalPredictor
            
            # Load data
            if isinstance(data, (str, Path)):
                train_data = pd.read_csv(str(data))
            else:
                train_data = data
            
            if target_column not in train_data.columns:
                return {
                    "success": False,
                    "error": f"Target column '{target_column}' not found. Available: {list(train_data.columns)}"
                }
            
            # Setup model
            model_name = model_name or f"multimodal_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.work_dir / model_name
            
            logger.info(f"🚀 Training multimodal model: {model_name}")
            logger.info(f"📊 Data shape: {train_data.shape}, Target: {target_column}")
            
            # Initialize and train predictor
            predictor = MultiModalPredictor(
                label=target_column,
                path=str(model_path),
                problem_type=problem_type,
                **kwargs
            )
            
            predictor.fit(
                train_data,
                time_limit=time_limit,
                presets=presets
            )
            
            # Basic evaluation
            try:
                train_score = predictor.evaluate(train_data)
                evaluation = {"train_score": train_score}
            except Exception as e:
                evaluation = {"evaluation_error": str(e)}
            
            logger.info(f"✅ Multimodal model training completed: {model_name}")
            
            return {
                "success": True,
                "model_name": model_name,
                "model_path": str(model_path),
                "model_type": "multimodal",
                "target_column": target_column,
                "problem_type": predictor.problem_type,
                "training_time": time_limit,
                "preset": presets,
                "evaluation": evaluation
            }
            
        except Exception as e:
            logger.error(f"❌ Multimodal training failed: {e}")
            return {"success": False, "error": str(e)}
    
    # TIME SERIES FORECASTING METHODS
    def train_timeseries_model(
        self,
        data: Union[pd.DataFrame, str, Path],
        target_column: str,
        timestamp_column: Optional[str] = None,
        prediction_length: int = 24,
        freq: Optional[str] = None,
        time_limit: int = 600,
        presets: str = "best_quality",
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train time series forecasting model."""
        if not self.availability["timeseries"]:
            return {
                "success": False,
                "error": "AutoGluon timeseries not available. Install with: pip install autogluon.timeseries"
            }
        
        try:
            from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
            
            # Load data
            if isinstance(data, (str, Path)):
                df = pd.read_csv(str(data))
            else:
                df = data
            
            if target_column not in df.columns:
                return {
                    "success": False,
                    "error": f"Target column '{target_column}' not found. Available: {list(df.columns)}"
                }
            
            # Convert to TimeSeriesDataFrame
            if timestamp_column:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
                ts_data = TimeSeriesDataFrame.from_data_frame(
                    df,
                    id_column=None,  # Single time series
                    timestamp_column=timestamp_column
                )
            else:
                # Assume data is already in time series format
                ts_data = TimeSeriesDataFrame(df)
            
            # Setup model
            model_name = model_name or f"timeseries_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.work_dir / model_name
            
            logger.info(f"🚀 Training time series model: {model_name}")
            logger.info(f"📊 Data shape: {ts_data.shape}, Target: {target_column}")
            logger.info(f"🔮 Prediction length: {prediction_length}")
            
            # Initialize and train predictor
            predictor = TimeSeriesPredictor(
                target=target_column,
                prediction_length=prediction_length,
                path=str(model_path),
                freq=freq,
                **kwargs
            )
            
            predictor.fit(
                ts_data,
                time_limit=time_limit,
                presets=presets
            )
            
            # Basic evaluation
            try:
                train_score = predictor.evaluate(ts_data)
                evaluation = {"train_score": train_score}
            except Exception as e:
                evaluation = {"evaluation_error": str(e)}
            
            logger.info(f"✅ Time series model training completed: {model_name}")
            
            return {
                "success": True,
                "model_name": model_name,
                "model_path": str(model_path),
                "model_type": "timeseries",
                "target_column": target_column,
                "prediction_length": prediction_length,
                "frequency": freq,
                "training_time": time_limit,
                "preset": presets,
                "evaluation": evaluation
            }
            
        except Exception as e:
            logger.error(f"❌ Time series training failed: {e}")
            return {"success": False, "error": str(e)}
    
    # UNIVERSAL PREDICTION METHOD
    def predict(
        self,
        model_path: Union[str, Path],
        data: Union[pd.DataFrame, str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Make predictions using any trained AutoGluon model."""
        try:
            model_path = Path(model_path)
            
            # Determine model type from path structure
            model_type = self._detect_model_type(model_path)
            
            if model_type == "tabular":
                return self._predict_tabular(model_path, data, output_path)
            elif model_type == "multimodal":
                return self._predict_multimodal(model_path, data, output_path)
            elif model_type == "timeseries":
                return self._predict_timeseries(model_path, data, output_path)
            else:
                return {"success": False, "error": f"Unknown model type: {model_type}"}
                
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _detect_model_type(self, model_path: Path) -> str:
        """Detect model type from saved model structure."""
        # Check for model-specific files/directories
        if (model_path / "models").exists():
            return "tabular"
        elif (model_path / "predictor.pkl").exists():
            return "multimodal"
        elif (model_path / "predictor.joblib").exists():
            return "timeseries"
        else:
            return "unknown"
    
    def _predict_tabular(self, model_path, data, output_path=None):
        """Make predictions with tabular model."""
        from autogluon.tabular import TabularPredictor, TabularDataset
        
        predictor = TabularPredictor.load(str(model_path))
        pred_data = TabularDataset(str(data) if isinstance(data, (str, Path)) else data)
        
        predictions = predictor.predict(pred_data)
        
        results = {
            "success": True,
            "predictions": predictions.tolist(),
            "model_type": "tabular",
            "num_predictions": len(predictions)
        }
        
        # Add probabilities for classification
        if predictor.problem_type in ['binary', 'multiclass']:
            try:
                probabilities = predictor.predict_proba(pred_data)
                results["probabilities"] = probabilities.to_dict('records')
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        if output_path:
            pred_df = pd.DataFrame({"prediction": predictions})
            pred_df.to_csv(output_path, index=False)
            results["output_path"] = str(output_path)
        
        return results
    
    def _predict_multimodal(self, model_path, data, output_path=None):
        """Make predictions with multimodal model."""
        from autogluon.multimodal import MultiModalPredictor
        
        predictor = MultiModalPredictor.load(str(model_path))
        pred_data = pd.read_csv(str(data)) if isinstance(data, (str, Path)) else data
        
        predictions = predictor.predict(pred_data)
        
        results = {
            "success": True,
            "predictions": predictions.tolist(),
            "model_type": "multimodal",
            "num_predictions": len(predictions)
        }
        
        if output_path:
            pred_df = pd.DataFrame({"prediction": predictions})
            pred_df.to_csv(output_path, index=False)
            results["output_path"] = str(output_path)
        
        return results
    
    def _predict_timeseries(self, model_path, data, output_path=None):
        """Make predictions with time series model."""
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        
        predictor = TimeSeriesPredictor.load(str(model_path))
        
        # Handle different data formats
        if isinstance(data, (str, Path)):
            df = pd.read_csv(str(data))
            ts_data = TimeSeriesDataFrame(df)
        else:
            ts_data = TimeSeriesDataFrame(data)
        
        predictions = predictor.predict(ts_data)
        
        results = {
            "success": True,
            "predictions": predictions.to_dict('records'),
            "model_type": "timeseries",
            "num_predictions": len(predictions)
        }
        
        if output_path:
            predictions.to_csv(output_path)
            results["output_path"] = str(output_path)
        
        return results
    
    # UTILITY METHODS
    def list_models(self) -> Dict[str, Any]:
        """List all available models."""
        try:
            models = []
            
            for model_dir in self.work_dir.iterdir():
                if model_dir.is_dir():
                    model_type = self._detect_model_type(model_dir)
                    if model_type != "unknown":
                        models.append({
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "type": model_type,
                            "created": datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat()
                        })
            
            return {
                "success": True,
                "models": models,
                "total_models": len(models)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_code_examples(self) -> Dict[str, str]:
        """Generate code examples for all AutoGluon domains."""
        return {
            "tabular": self._generate_tabular_example(),
            "multimodal": self._generate_multimodal_example(),
            "timeseries": self._generate_timeseries_example()
        }
    
    def _generate_tabular_example(self) -> str:
        return '''
# AutoGluon Tabular Example
from autogluon.tabular import TabularDataset, TabularPredictor

# Load data
train_data = TabularDataset('train.csv')
test_data = TabularDataset('test.csv')

# Train model
predictor = TabularPredictor(label='target').fit(
    train_data,
    time_limit=600,
    presets='best_quality'
)

# Make predictions
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)

# Evaluate
performance = predictor.evaluate(test_data)
leaderboard = predictor.leaderboard(test_data)
        '''.strip()
    
    def _generate_multimodal_example(self) -> str:
        return '''
# AutoGluon Multimodal Example
from autogluon.multimodal import MultiModalPredictor

# Load data (can contain text, image paths, numerical features)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Train model
predictor = MultiModalPredictor(label='target').fit(
    train_data,
    time_limit=600,
    presets='best_quality'
)

# Make predictions
predictions = predictor.predict(test_data)

# Evaluate
performance = predictor.evaluate(test_data)
        '''.strip()
    
    def _generate_timeseries_example(self) -> str:
        return '''
# AutoGluon Time Series Example
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load time series data
data = TimeSeriesDataFrame.from_data_frame(
    pd.read_csv('timeseries.csv'),
    id_column='id',
    timestamp_column='timestamp'
)

# Train forecasting model
predictor = TimeSeriesPredictor(
    target='value',
    prediction_length=24,
    freq='H'
).fit(data, time_limit=600)

# Make forecasts
forecasts = predictor.predict(data)

# Evaluate
performance = predictor.evaluate(data)
        '''.strip()