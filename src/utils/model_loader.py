"""
Model loading and inference utilities for Catalan traffic accident mortality prediction.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class MortalityPredictor:
    """Loads XGBoost models and provides mortality prediction functionality."""
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the predictor by loading available models.
        
        Parameters:
        -----------
        models_dir : str
            Path to directory containing model files. Defaults to '../models/'
        """
        if models_dir is None:
            models_dir = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 'models'
            )
        
        self.models_dir = Path(models_dir)
        self.models = {}
        self.model_metadata = {}
        self._load_models()
    
    def _load_models(self):
        """Load all available XGBoost models and their metadata."""
        for pkl_file in self.models_dir.glob('*.pkl'):
            model_name = pkl_file.stem
            metadata_file = self.models_dir / f"{model_name}_metadata.json"
            
            if metadata_file.exists():
                with open(pkl_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                
                with open(metadata_file, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
    
    def get_available_models(self) -> list:
        """Return list of available model names."""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get metadata about a specific model."""
        return self.model_metadata.get(model_name, {})
    
    def predict(
        self, 
        features_df: pd.DataFrame, 
        model_name: str = None,
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mortality for given accident features.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame with preprocessed features
        model_name : str
            Model to use. If None, uses the best model (highest ROC-AUC)
        return_probabilities : bool
            If True, returns probability of mortality. If False, returns binary prediction.
        
        Returns:
        --------
        predictions : np.ndarray
            Binary predictions (0/1) or probabilities depending on return_probabilities
        probabilities : np.ndarray
            Probability of mortality (if return_probabilities=True) or None
        """
        if not self.models:
            raise ValueError("No models loaded. Check models directory.")
        
        # Select best model if none specified
        if model_name is None:
            model_name = self._select_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {self.get_available_models()}")
        
        model = self.models[model_name]
        metadata = self.model_metadata[model_name]
        
        # Get optimal threshold
        threshold = metadata.get('optimal_threshold', 0.5)
        
        # Get probabilities
        probabilities = model.predict_proba(features_df)[:, 1]
        
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        if return_probabilities:
            return predictions, probabilities
        else:
            return predictions, None
    
    def _select_best_model(self) -> str:
        """Select model with highest test ROC-AUC."""
        best_model = None
        best_score = -1
        
        for model_name, metadata in self.model_metadata.items():
            test_roc_auc = metadata.get('evaluation_metrics', {}).get('Test', {}).get('ROC-AUC', 0)
            if test_roc_auc > best_score:
                best_score = test_roc_auc
                best_model = model_name
        
        return best_model or list(self.models.keys())[0]
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Get feature importance from a model."""
        if model_name is None:
            model_name = self._select_best_model()
        
        model = self.models[model_name]
        importance = model.get_booster().get_score(importance_type='weight')
        
        # Convert to DataFrame
        feature_importance = pd.DataFrame(
            list(importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        return feature_importance


class FeaturePreprocessor:
    """Handles feature preprocessing for model input."""
    
    def __init__(self, reference_df: pd.DataFrame = None):
        """
        Initialize preprocessor with reference data.
        
        Parameters:
        -----------
        reference_df : pd.DataFrame
            Reference cleaned dataset to extract categorical levels and scaling info
        """
        self.reference_df = reference_df
        self.categorical_features = [
            'zona', 'nomCom', 'nomDem', 'D_BOIRA', 'D_CARACT_ENTORN',
            'D_CARRIL_ESPECIAL', 'D_CIRCULACIO_MESURES_ESP', 'D_CLIMATOLOGIA',
            'D_FUNC_ESP_VIA', 'D_INTER_SECCIO', 'D_LIMIT_VELOCITAT',
            'D_LLUMINOSITAT', 'D_REGULACIO_PRIORITAT', 'D_SENTITS_VIA',
            'D_SUBTIPUS_ACCIDENT', 'D_SUBTIPUS_TRAM', 'D_SUBZONA',
            'D_SUPERFICIE', 'D_TIPUS_VIA', 'D_TITULARITAT_VIA',
            'D_TRACAT_ALTIMETRIC', 'D_VENT', 'grupHor', 'tipAcc', 'tipDia'
        ]
        
        self.numeric_features = [
            'Any', 'C_VELOCITAT_VIA', 'F_UNITATS_IMPLICADES',
            'F_VIANANTS_IMPLICADES', 'F_BICICLETES_IMPLICADES',
            'F_CICLOMOTORS_IMPLICADES', 'F_MOTOCICLETES_IMPLICADES',
            'F_VEH_LLEUGERS_IMPLICADES', 'F_VEH_PESANTS_IMPLICADES',
            'hor', 'Mes'
        ]
    
    def get_categorical_levels(self, column: str) -> list:
        """Get valid categorical levels for a feature."""
        if self.reference_df is not None and column in self.reference_df.columns:
            return sorted(self.reference_df[column].unique().tolist())
        return []
    
    def create_empty_row(self) -> pd.DataFrame:
        """Create an empty DataFrame with correct columns and types."""
        all_features = self.categorical_features + self.numeric_features
        data = {col: [None] for col in all_features}
        return pd.DataFrame(data)


def load_reference_data(output_dir: str = None) -> pd.DataFrame:
    """Load the cleaned reference dataset."""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'output'
        )
    
    csv_path = os.path.join(output_dir, 'df_cleaned.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return None
