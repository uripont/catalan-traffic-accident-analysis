import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class MortalityPredictor:    
    def __init__(self, models_dir: str = None):
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
        for pkl_file in self.models_dir.glob('*.pkl'):
            model_name = pkl_file.stem
            metadata_file = self.models_dir / f"{model_name}_metadata.json"
            
            if metadata_file.exists():
                with open(pkl_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                
                with open(metadata_file, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
    
    def get_available_models(self) -> list:
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        return self.model_metadata.get(model_name, {})
    
    def predict(
        self, 
        features_df: pd.DataFrame, 
        model_name: str = None,
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def predict_from_dict(
        self,
        scenario_dict: Dict,
        reference_df: pd.DataFrame = None,
        model_name: str = None
    ) -> Tuple[int, float]:
        if model_name is None:
            model_name = self._select_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {self.get_available_models()}")
        
        model = self.models[model_name]
        metadata = self.model_metadata[model_name]
        
        # Get expected feature names from model
        feature_names = model.get_booster().feature_names
        
        # Create a dict with all required features, filling defaults
        features = {}
        for feat in feature_names:
            if feat in scenario_dict:
                features[feat] = scenario_dict[feat]
            elif reference_df is not None and feat in reference_df.columns:
                # Use mode for categorical, mean for numeric
                if reference_df[feat].dtype == 'object':
                    features[feat] = reference_df[feat].mode()[0] if len(reference_df[feat].mode()) > 0 else reference_df[feat].iloc[0]
                else:
                    features[feat] = reference_df[feat].mean()
            else:
                # Default fallback values
                if 'Any' in feat or 'hor' in feat or 'Mes' in feat or 'C_VELOCITAT' in feat:
                    features[feat] = 2015  # Default year/time value
                else:
                    features[feat] = 0  # Default numeric
        
        # Create DataFrame with single row in correct column order
        df = pd.DataFrame([features])
        df = df[feature_names]  # Reorder to match model's expected feature order
        
        # Get prediction
        predictions, probabilities = self.predict(df, model_name=model_name)
        
        return int(predictions[0]), float(probabilities[0])
    
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
    def __init__(self, reference_df: pd.DataFrame = None):
        """
        reference_df should be Reference cleaned dataset to extract categorical levels and scaling info
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
        if self.reference_df is not None and column in self.reference_df.columns:
            return sorted(self.reference_df[column].unique().tolist())
        return []
    
    def create_empty_row(self) -> pd.DataFrame:
        all_features = self.categorical_features + self.numeric_features
        data = {col: [None] for col in all_features}
        return pd.DataFrame(data)


def load_reference_data(output_dir: str = None) -> pd.DataFrame:
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
