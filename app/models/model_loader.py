# app/models/model_loader.py
import pickle
import joblib
from pathlib import Path
from typing import Any, Optional, Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MultiModelLoader:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.label_encoders: Dict[str, Any] = {}
        self.loaded_models: Dict[str, bool] = {}
    
    def load_model(self, model_name: str, model_path: str, encoder_path: str = None) -> bool:
        """Load ML model dan label encoder untuk model tertentu"""
        try:
            # Load main model
            model_file = Path(model_path)
            if not model_file.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model
            try:
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"✅ {model_name} model loaded with joblib")
            except:
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"✅ {model_name} model loaded with pickle")
            
            # Load label encoder berdasarkan model type
            if encoder_path and Path(encoder_path).exists():
                try:
                    self.label_encoders[model_name] = joblib.load(encoder_path)
                    logger.info(f"✅ {model_name} label encoder loaded from {encoder_path}")
                except Exception as e:
                    logger.error(f"Error loading encoder for {model_name}: {e}")
                    self.label_encoders[model_name] = None
            else:
                logger.warning(f"Label encoder not found for {model_name}: {encoder_path}")
                self.label_encoders[model_name] = None
            
            self.loaded_models[model_name] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading {model_name}: {e}")
            self.loaded_models[model_name] = False
            return False
    
    def get_model(self, model_name: str):
        """Get specific model"""
        if model_name not in self.loaded_models or not self.loaded_models[model_name]:
            raise ValueError(f"Model {model_name} not loaded")
        return self.models[model_name]
    
    def get_label_encoders(self, model_name: str):
        """Get label encoders untuk model tertentu"""
        return self.label_encoders.get(model_name)
    
    def predict(self, model_name: str, features):
        """Make prediction using specific model"""
        model = self.get_model(model_name)
        prediction = model.predict(features)
        
        if isinstance(prediction, np.ndarray):
            return [item.item() if isinstance(item, np.number) else item for item in prediction]
        return prediction
    
    def predict_proba(self, model_name: str, features):
        """Get prediction probabilities"""
        model = self.get_model(model_name)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            if isinstance(probabilities, np.ndarray):
                return probabilities.tolist()
            return probabilities
        return None
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if specific model is loaded"""
        return self.loaded_models.get(model_name, False)
    
    def get_loaded_models_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        return self.loaded_models.copy()
