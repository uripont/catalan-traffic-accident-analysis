import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import sys
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_loader import MortalityPredictor, load_reference_data

# Load models and data
@st.cache_resource
def load_models_and_data():
    predictor = MortalityPredictor()
    reference_df = load_reference_data()
    return predictor, reference_df

@st.cache_resource
def load_label_encoders():
    output_dir = Path(__file__).parent.parent.parent / 'output'
    encoder_path = output_dir / 'label_encoders.pkl'
    if encoder_path.exists():
        with open(encoder_path, 'rb') as f:
            return pickle.load(f)
    return {}

predictor, reference_df = load_models_and_data()
label_encoders = load_label_encoders()

st.set_page_config(page_title="🔮 Hypothetical scenarios", layout="wide")

# Feature definitions and defaults
NUMERIC_FEATURES = {
    'Any': {'label': 'Year', 'min': 2010, 'max': 2023, 'default': 2020, 'step': 1, 'type': 'int'},
    'hor': {'label': 'Hour of day', 'min': 0.0, 'max': 23.99, 'default': 12.0, 'step': 0.5, 'type': 'float'},
    'C_VELOCITAT_VIA': {'label': 'Speed limit (km/h)', 'min': 0.0, 'max': 120.0, 'default': 50.0, 'step': 5.0, 'type': 'float'},
    'F_UNITATS_IMPLICADES': {'label': 'Units involved', 'min': 1, 'max': 20, 'default': 2, 'step': 1, 'type': 'int'},
    'F_VIANANTS_IMPLICADES': {'label': 'Pedestrians involved', 'min': 0, 'max': 10, 'default': 0, 'step': 1, 'type': 'int'},
    'F_BICICLETES_IMPLICADES': {'label': 'Bicycles involved', 'min': 0, 'max': 10, 'default': 0, 'step': 1, 'type': 'int'},
    'F_CICLOMOTORS_IMPLICADES': {'label': 'Mopeds involved', 'min': 0, 'max': 10, 'default': 0, 'step': 1, 'type': 'int'},
    'F_MOTOCICLETES_IMPLICADES': {'label': 'Motorcycles involved', 'min': 0, 'max': 10, 'default': 0, 'step': 1, 'type': 'int'},
    'F_VEH_LLEUGERS_IMPLICADES': {'label': 'Light vehicles involved', 'min': 0, 'max': 15, 'default': 1, 'step': 1, 'type': 'int'},
    'F_VEH_PESANTS_IMPLICADES': {'label': 'Heavy vehicles involved', 'min': 0, 'max': 10, 'default': 0, 'step': 1, 'type': 'int'},
    'Mes': {'label': 'Month', 'min': 1, 'max': 12, 'default': 6, 'step': 1, 'type': 'int'},
}

CATEGORICAL_FEATURES = {
    'zona': {'label': 'Zone'},
    'nomCom': {'label': 'Region'},
    'nomDem': {'label': 'Administrative demarcation'},
    'D_BOIRA': {'label': 'Fog'},
    'D_CARACT_ENTORN': {'label': 'Environment characteristic'},
    'D_CARRIL_ESPECIAL': {'label': 'Special lane'},
    'D_CIRCULACIO_MESURES_ESP': {'label': 'Special traffic measures'},
    'D_CLIMATOLOGIA': {'label': 'Weather condition'},
    'D_FUNC_ESP_VIA': {'label': 'Special road function'},
    'D_INTER_SECCIO': {'label': 'Intersection type'},
    'D_LIMIT_VELOCITAT': {'label': 'Speed limit category'},
    'D_LLUMINOSITAT': {'label': 'Light condition'},
    'D_REGULACIO_PRIORITAT': {'label': 'Priority regulation'},
    'D_SENTITS_VIA': {'label': 'Road directions'},
    'D_SUBTIPUS_ACCIDENT': {'label': 'Accident subtype'},
    'D_SUBTIPUS_TRAM': {'label': 'Road section subtype'},
    'D_SUBZONA': {'label': 'Subzone'},
    'D_SUPERFICIE': {'label': 'Road surface'},
    'D_TIPUS_VIA': {'label': 'Road type'},
    'D_TITULARITAT_VIA': {'label': 'Road ownership'},
    'D_TRACAT_ALTIMETRIC': {'label': 'Road altitude profile'},
    'D_VENT': {'label': 'Wind'},
    'grupHor': {'label': 'Time period'},
    'tipAcc': {'label': 'Accident type'},
    'tipDia': {'label': 'Day type'},
}

def get_categorical_options(feature_name: str) -> Dict[int, str]:
    """Get label options for a categorical feature from encoders."""
    if feature_name in label_encoders:
        le = label_encoders[feature_name]
        return {idx: label for idx, label in enumerate(le.classes_)}
    return {0: "Default"}

def create_feature_inputs(prefix: str = "") -> Dict:
    """Interactive inputs for all 36 accident features."""
    scenario = {}
    
    st.markdown("### Numeric features")
    cols = st.columns(3)
    for idx, (feat_name, feat_config) in enumerate(NUMERIC_FEATURES.items()):
        col = cols[idx % 3]
        with col:
            scenario[feat_name] = st.slider(
                feat_config['label'],
                min_value=feat_config['min'],
                max_value=feat_config['max'],
                value=feat_config['default'],
                step=feat_config['step'],
                key=f"{feat_name}_{prefix}"
            )
    
    st.markdown("### Categorical features")
    
    cols = st.columns(3)
    for idx, (feat_name, feat_config) in enumerate(CATEGORICAL_FEATURES.items()):
        col = cols[idx % 3]
        with col:
            options = get_categorical_options(feat_name)
            sorted_options = sorted(options.keys())
            
            selected_value = st.selectbox(
                feat_config['label'],
                options=sorted_options,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"{feat_name}_{prefix}"
            )
            scenario[feat_name] = selected_value
    
    return scenario


def display_prediction_card(probability: float, prediction: int) -> None:
    st.metric("Mortality probability", f"{probability*100:.1f}%")


def display_scenario_details(scenario: Dict) -> None:
    with st.expander("Scenario details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Numeric features")
            numeric_dict = {k: scenario.get(k, 0) for k in NUMERIC_FEATURES.keys()}
            st.dataframe(pd.DataFrame(
                [(k, numeric_dict[k]) for k in sorted(numeric_dict.keys())],
                columns=["Feature", "Value"]
            ))
        
        with col2:
            st.subheader("Categorical features")
            categorical_dict = {}
            for k in CATEGORICAL_FEATURES.keys():
                val = scenario.get(k, 0)
                options = get_categorical_options(k)
                label = options.get(val, f"Level {val}")
                categorical_dict[k] = f"{val} - {label}"
            
            st.dataframe(pd.DataFrame(
                [(k, categorical_dict[k]) for k in sorted(categorical_dict.keys())],
                columns=["Feature", "Level & Label"]
            ))


def main():    
    st.markdown("""
    ### 🔮 Hypothetical Scenarios
    Create custom accident scenarios using all 36 original features and predict mortality risk. Even
    with those learned to be not highly influential, they may still impact specific situations.
    """)
    
    st.markdown("---")
    st.markdown("#### Scenario configuration")
    
    scenario = create_feature_inputs(prefix="single")
    
    if st.button("🔍 Predict mortality risk", key="predict_single", use_container_width=True):
        st.markdown("---")
        try:
            pred, prob = predictor.predict_from_dict(scenario, reference_df=reference_df)
            
            st.markdown("### Prediction results")
            display_scenario_details(scenario)

            display_prediction_card(prob, pred)
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
    
if __name__ == "__main__":
    main()
