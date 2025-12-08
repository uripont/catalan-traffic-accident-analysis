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

# Mortality threshold for classification
MORTALITY_THRESHOLD = 0.56
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

# Presets for quick scenario setup
SCENARIO_PRESETS = {
    "🌞 Daytime - Good conditions": {
        'hor': 14.0,
        'D_LLUMINOSITAT': 0,  # Day
        'D_CLIMATOLOGIA': 0,  # No rain
        'D_BOIRA': 0,  # No fog
    },
    "🌙 Night - Low visibility": {
        'hor': 22.0,
        'D_LLUMINOSITAT': 2,  # Night
        'D_CLIMATOLOGIA': 1,  # Rain
        'D_BOIRA': 1,  # Fog
    },
    "🚗 High speed - Highway": {
        'C_VELOCITAT_VIA': 100.0,
        'D_TIPUS_VIA': 1,  # Highway
    },
    "🏘️ Urban - Low speed": {
        'C_VELOCITAT_VIA': 30.0,
        'D_TIPUS_VIA': 0,  # Urban
    },
    "🚕 Multi-vehicle pile-up": {
        'F_UNITATS_IMPLICADES': 5,
        'C_VELOCITAT_VIA': 80.0,
        'D_LLUMINOSITAT': 2,  # Night
    },
    "🚲 Pedestrian crossing": {
        'F_VIANANTS_IMPLICADES': 2,
        'C_VELOCITAT_VIA': 40.0,
        'D_INTER_SECCIO': 1,  # Intersection
    },
    "🌧️ Heavy rain": {
        'D_CLIMATOLOGIA': 3,  # Heavy rain
        'D_SUPERFICIE': 1,  # Wet
        'C_VELOCITAT_VIA': 50.0,
    },
    "🚲 Bicycle collision": {
        'F_BICICLETES_IMPLICADES': 1,
        'F_VEH_LLEUGERS_IMPLICADES': 1,
        'C_VELOCITAT_VIA': 30.0,
    },
    "🛣️ Motorcycle highway": {
        'F_MOTOCICLETES_IMPLICADES': 1,
        'C_VELOCITAT_VIA': 100.0,
        'D_TIPUS_VIA': 1,  # Highway
    },
    "🚛 Truck incident": {
        'F_VEH_PESANTS_IMPLICADES': 1,
        'C_VELOCITAT_VIA': 80.0,
        'F_UNITATS_IMPLICADES': 2,
    },
    "🌫️ Fog in mountain": {
        'D_BOIRA': 1,  # Fog
        'D_TRACAT_ALTIMETRIC': 1,  # Mountain
        'hor': 6.0,  # Early morning
    },
    "⚠️ High-risk scenario": {
        'C_VELOCITAT_VIA': 100.0,
        'D_CLIMATOLOGIA': 1,  # Rain
        'D_LLUMINOSITAT': 2,  # Night
        'F_UNITATS_IMPLICADES': 3,
        'hor': 23.0,  # Late night
    },
}

def get_categorical_options(feature_name: str) -> Dict[int, str]:
    if feature_name in label_encoders:
        le = label_encoders[feature_name]
        return {idx: label for idx, label in enumerate(le.classes_)}
    return {0: "Default"}


def generate_random_values(feature_list: list) -> Dict:
    random_values = {}
    
    for feature_name in feature_list:
        if feature_name in NUMERIC_FEATURES:
            feat_config = NUMERIC_FEATURES[feature_name]
            if feat_config['type'] == 'int':
                random_values[feature_name] = np.random.randint(
                    int(feat_config['min']), 
                    int(feat_config['max']) + 1
                )
            else:  # float
                random_values[feature_name] = np.random.uniform(
                    feat_config['min'], 
                    feat_config['max']
                )
        elif feature_name in CATEGORICAL_FEATURES:
            options = get_categorical_options(feature_name)
            if options:
                random_values[feature_name] = np.random.choice(list(options.keys()))
            else:
                random_values[feature_name] = 0
    
    return random_values


def create_feature_inputs(prefix: str = "", preset_values: Dict = None, preset_id: str = "default", advanced_features: list = None) -> Dict:
    """
    Args:
        prefix: Unique prefix for input keys
        preset_values: Dictionary of preset values to apply
        preset_id: Stable identifier for the preset (changes only when preset actually changes)
        advanced_features: List of advanced feature names for randomization
    """
    if preset_values is None:
        preset_values = {}
    if advanced_features is None:
        advanced_features = []
    
    scenario = {}
    
    st.markdown("### 📅 Time")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        default_year = preset_values.get('Any', int(NUMERIC_FEATURES['Any']['default']))
        scenario['Any'] = st.slider(
            NUMERIC_FEATURES['Any']['label'],
            min_value=int(NUMERIC_FEATURES['Any']['min']),
            max_value=int(NUMERIC_FEATURES['Any']['max']),
            value=default_year,
            step=1,
            key=f"Any_{prefix}_{preset_id}"
        )
    
    with col2:
        default_month = preset_values.get('Mes', 6)
        scenario['Mes'] = st.slider(
            NUMERIC_FEATURES['Mes']['label'],
            min_value=1, max_value=12,
            value=default_month, step=1,
            key=f"Mes_{prefix}_{preset_id}"
        )
    
    with col3:
        default_hour = preset_values.get('hor', 12.0)
        scenario['hor'] = st.slider(
            NUMERIC_FEATURES['hor']['label'],
            min_value=0.0, max_value=23.99,
            value=default_hour, step=0.5,
            key=f"hor_{prefix}_{preset_id}"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        if 'tipDia' in CATEGORICAL_FEATURES:
            options = get_categorical_options('tipDia')
            sorted_options = sorted(options.keys())
            default_tipdia = preset_values.get('tipDia', 0)
            scenario['tipDia'] = st.selectbox(
                CATEGORICAL_FEATURES['tipDia']['label'],
                options=sorted_options,
                index=sorted_options.index(default_tipdia) if default_tipdia in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"tipDia_{prefix}_{preset_id}"
            )
    
    with col2:
        if 'grupHor' in CATEGORICAL_FEATURES:
            options = get_categorical_options('grupHor')
            sorted_options = sorted(options.keys())
            default_gruphor = preset_values.get('grupHor', 0)
            scenario['grupHor'] = st.selectbox(
                CATEGORICAL_FEATURES['grupHor']['label'],
                options=sorted_options,
                index=sorted_options.index(default_gruphor) if default_gruphor in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"grupHor_{prefix}_{preset_id}"
            )
    
    st.markdown("### 🛣️ Road and weather")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        default_speed = preset_values.get('C_VELOCITAT_VIA', 50.0)
        scenario['C_VELOCITAT_VIA'] = st.slider(
            NUMERIC_FEATURES['C_VELOCITAT_VIA']['label'],
            min_value=0.0, max_value=120.0,
            value=default_speed, step=5.0,
            key=f"C_VELOCITAT_VIA_{prefix}_{preset_id}"
        )
    
    with col2:
        if 'D_TIPUS_VIA' in CATEGORICAL_FEATURES:
            options = get_categorical_options('D_TIPUS_VIA')
            sorted_options = sorted(options.keys())
            default_via = preset_values.get('D_TIPUS_VIA', 0)
            scenario['D_TIPUS_VIA'] = st.selectbox(
                CATEGORICAL_FEATURES['D_TIPUS_VIA']['label'],
                options=sorted_options,
                index=sorted_options.index(default_via) if default_via in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"D_TIPUS_VIA_{prefix}_{preset_id}"
            )
    
    with col3:
        if 'D_LLUMINOSITAT' in CATEGORICAL_FEATURES:
            options = get_categorical_options('D_LLUMINOSITAT')
            sorted_options = sorted(options.keys())
            default_light = preset_values.get('D_LLUMINOSITAT', 0)
            scenario['D_LLUMINOSITAT'] = st.selectbox(
                CATEGORICAL_FEATURES['D_LLUMINOSITAT']['label'],
                options=sorted_options,
                index=sorted_options.index(default_light) if default_light in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"D_LLUMINOSITAT_{prefix}_{preset_id}"
            )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'D_CLIMATOLOGIA' in CATEGORICAL_FEATURES:
            options = get_categorical_options('D_CLIMATOLOGIA')
            sorted_options = sorted(options.keys())
            default_weather = preset_values.get('D_CLIMATOLOGIA', 0)
            scenario['D_CLIMATOLOGIA'] = st.selectbox(
                CATEGORICAL_FEATURES['D_CLIMATOLOGIA']['label'],
                options=sorted_options,
                index=sorted_options.index(default_weather) if default_weather in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"D_CLIMATOLOGIA_{prefix}_{preset_id}"
            )
    
    with col2:
        if 'D_BOIRA' in CATEGORICAL_FEATURES:
            options = get_categorical_options('D_BOIRA')
            sorted_options = sorted(options.keys())
            default_fog = preset_values.get('D_BOIRA', 0)
            scenario['D_BOIRA'] = st.selectbox(
                CATEGORICAL_FEATURES['D_BOIRA']['label'],
                options=sorted_options,
                index=sorted_options.index(default_fog) if default_fog in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"D_BOIRA_{prefix}_{preset_id}"
            )
    
    with col3:
        if 'D_VENT' in CATEGORICAL_FEATURES:
            options = get_categorical_options('D_VENT')
            sorted_options = sorted(options.keys())
            default_wind = preset_values.get('D_VENT', 0)
            scenario['D_VENT'] = st.selectbox(
                CATEGORICAL_FEATURES['D_VENT']['label'],
                options=sorted_options,
                index=sorted_options.index(default_wind) if default_wind in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"D_VENT_{prefix}_{preset_id}"
            )
    
    st.markdown("### 🚗 Accident Details")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        default_units = preset_values.get('F_UNITATS_IMPLICADES', 2)
        scenario['F_UNITATS_IMPLICADES'] = st.slider(
            NUMERIC_FEATURES['F_UNITATS_IMPLICADES']['label'],
            min_value=1, max_value=20,
            value=default_units, step=1,
            key=f"F_UNITATS_IMPLICADES_{prefix}_{preset_id}"
        )
    
    with col2:
        default_peds = preset_values.get('F_VIANANTS_IMPLICADES', 0)
        scenario['F_VIANANTS_IMPLICADES'] = st.slider(
            NUMERIC_FEATURES['F_VIANANTS_IMPLICADES']['label'],
            min_value=0, max_value=10,
            value=default_peds, step=1,
            key=f"F_VIANANTS_IMPLICADES_{prefix}_{preset_id}"
        )
    
    with col3:
        default_bikes = preset_values.get('F_BICICLETES_IMPLICADES', 0)
        scenario['F_BICICLETES_IMPLICADES'] = st.slider(
            NUMERIC_FEATURES['F_BICICLETES_IMPLICADES']['label'],
            min_value=0, max_value=10,
            value=default_bikes, step=1,
            key=f"F_BICICLETES_IMPLICADES_{prefix}_{preset_id}"
        )
    
    with col4:
        default_motos = preset_values.get('F_MOTOCICLETES_IMPLICADES', 0)
        scenario['F_MOTOCICLETES_IMPLICADES'] = st.slider(
            NUMERIC_FEATURES['F_MOTOCICLETES_IMPLICADES']['label'],
            min_value=0, max_value=10,
            value=default_motos, step=1,
            key=f"F_MOTOCICLETES_IMPLICADES_{prefix}_{preset_id}"
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        default_mopeds = preset_values.get('F_CICLOMOTORS_IMPLICADES', 0)
        scenario['F_CICLOMOTORS_IMPLICADES'] = st.slider(
            NUMERIC_FEATURES['F_CICLOMOTORS_IMPLICADES']['label'],
            min_value=0, max_value=10,
            value=default_mopeds, step=1,
            key=f"F_CICLOMOTORS_IMPLICADES_{prefix}_{preset_id}"
        )
    
    with col2:
        default_light = preset_values.get('F_VEH_LLEUGERS_IMPLICADES', 1)
        scenario['F_VEH_LLEUGERS_IMPLICADES'] = st.slider(
            NUMERIC_FEATURES['F_VEH_LLEUGERS_IMPLICADES']['label'],
            min_value=0, max_value=15,
            value=default_light, step=1,
            key=f"F_VEH_LLEUGERS_IMPLICADES_{prefix}_{preset_id}"
        )
    
    with col3:
        default_heavy = preset_values.get('F_VEH_PESANTS_IMPLICADES', 0)
        scenario['F_VEH_PESANTS_IMPLICADES'] = st.slider(
            NUMERIC_FEATURES['F_VEH_PESANTS_IMPLICADES']['label'],
            min_value=0, max_value=10,
            value=default_heavy, step=1,
            key=f"F_VEH_PESANTS_IMPLICADES_{prefix}_{preset_id}"
        )
    
    with col4:
        if 'tipAcc' in CATEGORICAL_FEATURES:
            options = get_categorical_options('tipAcc')
            sorted_options = sorted(options.keys())
            default_acc_type = preset_values.get('tipAcc', 0)
            scenario['tipAcc'] = st.selectbox(
                CATEGORICAL_FEATURES['tipAcc']['label'],
                options=sorted_options,
                index=sorted_options.index(default_acc_type) if default_acc_type in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"tipAcc_{prefix}_{preset_id}"
            )
    
    st.markdown("### 📍 Location")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'zona' in CATEGORICAL_FEATURES:
            options = get_categorical_options('zona')
            sorted_options = sorted(options.keys())
            default_zona = preset_values.get('zona', 0)
            scenario['zona'] = st.selectbox(
                CATEGORICAL_FEATURES['zona']['label'],
                options=sorted_options,
                index=sorted_options.index(default_zona) if default_zona in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"zona_{prefix}_{preset_id}"
            )
    
    with col2:
        if 'D_INTER_SECCIO' in CATEGORICAL_FEATURES:
            options = get_categorical_options('D_INTER_SECCIO')
            sorted_options = sorted(options.keys())
            default_inter = preset_values.get('D_INTER_SECCIO', 0)
            scenario['D_INTER_SECCIO'] = st.selectbox(
                CATEGORICAL_FEATURES['D_INTER_SECCIO']['label'],
                options=sorted_options,
                index=sorted_options.index(default_inter) if default_inter in sorted_options else 0,
                format_func=lambda x: options.get(x, f"Level {x}"),
                key=f"D_INTER_SECCIO_{prefix}_{preset_id}"
            )
    
    with st.expander("⚙️ Advanced features", expanded=False):
        st.markdown("Additional characteristics that may affect the prediction")
        
        cols = st.columns(3)
        remaining_features = [f for f in CATEGORICAL_FEATURES.keys() 
                            if f not in scenario]
        
        for idx, feat_name in enumerate(remaining_features):
            col = cols[idx % 3]
            with col:
                options = get_categorical_options(feat_name)
                sorted_options = sorted(options.keys())
                default_val = preset_values.get(feat_name, 0)
                scenario[feat_name] = st.selectbox(
                    CATEGORICAL_FEATURES[feat_name]['label'],
                    options=sorted_options,
                    index=sorted_options.index(default_val) if default_val in sorted_options else 0,
                    format_func=lambda x: options.get(x, f"Level {x}"),
                    key=f"{feat_name}_{prefix}_{preset_id}"
                )
    
    return scenario


def display_scenario_details(scenario: Dict) -> None:
    with st.expander("📋 Full scenario details", expanded=True):
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
    ### 🔮 Hypothetical scenarios
    """)
    
    # Define main vs advanced features
    MAIN_FEATURES = [
        'Any', 'Mes', 'hor', 'tipDia', 'grupHor',
        'C_VELOCITAT_VIA', 'D_TIPUS_VIA', 'D_LLUMINOSITAT', 'D_CLIMATOLOGIA', 'D_BOIRA', 'D_VENT',
        'F_UNITATS_IMPLICADES', 'F_VIANANTS_IMPLICADES', 'F_BICICLETES_IMPLICADES', 'F_MOTOCICLETES_IMPLICADES',
        'F_CICLOMOTORS_IMPLICADES', 'F_VEH_LLEUGERS_IMPLICADES', 'F_VEH_PESANTS_IMPLICADES', 'tipAcc',
        'zona', 'D_INTER_SECCIO'
    ]
    ADVANCED_FEATURES = [f for f in CATEGORICAL_FEATURES.keys() if f not in MAIN_FEATURES]
    
    preset_items = list(SCENARIO_PRESETS.items())
    num_rows = (len(preset_items) + 3) // 4  # Round up division
    
    for row in range(num_rows):
        cols = st.columns(4)
        for col_idx, col in enumerate(cols):
            preset_idx = row * 4 + col_idx
            if preset_idx < len(preset_items):
                preset_name, preset_values = preset_items[preset_idx]
                with col:
                    if st.button(preset_name, use_container_width=True):
                        st.session_state['selected_preset_name'] = preset_name
                        st.session_state['selected_preset'] = preset_values
                        # Clear randomized values and re-randomize the features NOT in this preset
                        st.session_state['randomized_values'] = {}
                        st.session_state['randomized_advanced_values'] = {}
                        st.rerun()

    if st.button("🎲 Randomize", use_container_width=True):
        # Increment counter to force re-randomization
        randomize_count = st.session_state.get('randomize_count', 0) + 1
        st.session_state['randomize_count'] = randomize_count
        
        random_main = generate_random_values(MAIN_FEATURES)
        random_advanced = generate_random_values(ADVANCED_FEATURES)
        st.session_state['randomized_values'] = random_main
        st.session_state['randomized_advanced_values'] = random_advanced
        st.session_state['selected_preset'] = {}
        st.session_state['selected_preset_name'] = 'randomized'
        st.rerun()

    
    st.markdown("---")
    
    # Get preset values if a preset was selected
    preset_values = st.session_state.get('selected_preset', {})
    preset_name = st.session_state.get('selected_preset_name', 'default')
    
    # Apply randomized values if they exist
    if 'randomized_values' in st.session_state:
        preset_values = {**preset_values, **st.session_state['randomized_values']}
    if 'randomized_advanced_values' in st.session_state:
        preset_values = {**preset_values, **st.session_state['randomized_advanced_values']}
    
    # Create feature inputs with preset values
    scenario = create_feature_inputs(prefix="scenario", preset_values=preset_values, preset_id=preset_name, advanced_features=ADVANCED_FEATURES)
    
    st.markdown("---")
    if st.button("🔍 Predict mortality risk", key="predict", use_container_width=True, type="primary"):
        try:
            pred, prob = predictor.predict_from_dict(scenario, reference_df=reference_df)
            
            st.markdown("### 🎯 Prediction Result")
            
            # Determine if it's a mortality case
            is_mortality = prob >= MORTALITY_THRESHOLD
            mortality_label = "💀 FATAL" if is_mortality else "✅ NON-FATAL"
            
            # Main prediction cards
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Mortality risk probability",
                    f"{prob*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Prediction",
                    mortality_label
                )
            
            st.markdown("---")
            
            # Details
            display_scenario_details(scenario)
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
