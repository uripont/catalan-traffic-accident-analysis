import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import pickle


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_loader import MortalityPredictor, load_reference_data

# Define mortality prediction threshold
MORTALITY_THRESHOLD = 0.56

# Load models and data
@st.cache_resource
def load_models_and_data():
    predictor = MortalityPredictor()
    reference_df = load_reference_data()
    
    # Load label encoders to get comarca names    
    encoder_path = Path(__file__).parent.parent.parent / "output" / "label_encoders.pkl"
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)
    
    # Create mapping from encoded values to comarca names
    comarca_encoder = encoders['nomCom']
    comarca_mapping = {i: name for i, name in enumerate(comarca_encoder.classes_)}
    
    # Add comarca names to reference_df
    reference_df['nomCom_name'] = reference_df['nomCom'].map(comarca_mapping).fillna('Unknown')
    
    return predictor, reference_df

predictor, reference_df = load_models_and_data()

st.set_page_config(page_title="📊 Past accident analysis", layout="wide")

# Configure page
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


def load_sample_accidents(df: pd.DataFrame, n_samples: int = 10) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    
    # Filter for accidents with casualties for more interesting samples
    df_casualties = df[df['F_VICTIMES'] > 0]
    
    if len(df_casualties) < n_samples:
        return df_casualties.sample(min(n_samples, len(df_casualties)))
    
    return df_casualties.sample(n_samples)


def extract_scenario_from_accident(row: pd.Series) -> dict:
    scenario = {
        'F_UNITATS_IMPLICADES': row.get('F_UNITATS_IMPLICADES', 1),
        'F_VIANANTS_IMPLICADES': row.get('F_VIANANTS_IMPLICADES', 0),
        'F_BICICLETES_IMPLICADES': row.get('F_BICICLETES_IMPLICADES', 0),
        'F_CICLOMOTORS_IMPLICADES': row.get('F_CICLOMOTORS_IMPLICADES', 0),
        'F_MOTOCICLETES_IMPLICADES': row.get('F_MOTOCICLETES_IMPLICADES', 0),
        'F_VEH_LLEUGERS_IMPLICADES': row.get('F_VEH_LLEUGERS_IMPLICADES', 0),
        'F_VEH_PESANTS_IMPLICADES': row.get('F_VEH_PESANTS_IMPLICADES', 0),
        'F_ALTRES_UNIT_IMPLICADES': row.get('F_ALTRES_UNIT_IMPLICADES', 0),
    }
    return scenario


def display_accident_card(accident: pd.Series, prediction: float, actual_deaths: int) -> None:    
    # Determine if prediction matches outcome
    predicted_fatal = prediction >= MORTALITY_THRESHOLD
    actual_fatal = actual_deaths > 0
    
    # Determine match icon
    if predicted_fatal == actual_fatal:
        # Correct prediction
        match_icon = "✅"
    elif predicted_fatal and not actual_fatal:
        # Predicted death but none occurred
        match_icon = "⚠️"
    else:
        # Actual deaths but not predicted
        match_icon = "❗"
    
    # Build characteristics with emojis only, so that it is more compact
    characteristics = []
    
    # Vehicle/pedestrian types
    if accident.get('F_VIANANTS_IMPLICADES', 0) > 0:
        characteristics.append("🚶")
    if accident.get('F_BICICLETES_IMPLICADES', 0) > 0:
        characteristics.append("🚴")
    if accident.get('F_CICLOMOTORS_IMPLICADES', 0) > 0:
        characteristics.append("🛵")
    if accident.get('F_MOTOCICLETES_IMPLICADES', 0) > 0:
        characteristics.append("🏍️")
    if accident.get('F_VEH_PESANTS_IMPLICADES', 0) > 0:
        characteristics.append("🚚")
    
    # Road and weather conditions
    if accident.get('D_LLUMINOSITAT', 0) in [2, 3]:
        characteristics.append("🌙")
    if accident.get('D_CLIMATOLOGIA', 0) in [2, 3, 4]:
        characteristics.append("🌧️")
    if accident.get('D_BOIRA', 0) == 1:
        characteristics.append("🌫️")
    if accident.get('D_VENT', 0) == 1:
        characteristics.append("💨")
    
    characteristics_str = " ".join(characteristics) if characteristics else "—"
    
    # Injury counts
    deaths = int(accident.get('F_MORTS', 0))
    serious = int(accident.get('F_FERITS_GREUS', 0))
    light = int(accident.get('F_FERITS_LLEUS', 0))
    
    # Location
    comarca = accident.get('nomCom_name', accident.get('nomCom', '?'))
    year = int(accident.get('Any', 0))
    
    # Build concise markdown card
    st.markdown(f"""
<div style="text-align: center; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">
{match_icon}{prediction*100:.1f}%
</div>
 
📅 {year} | 📍 {comarca}  
💀 {deaths} | 🚨 {serious} | 🩹 {light} | 🚗 {int(accident.get('F_UNITATS_IMPLICADES', 0))}  
{characteristics_str}
""", unsafe_allow_html=True)



def display_accident_statistics(predictions: list, df_analyzed: pd.DataFrame) -> None:    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_risk = np.mean(predictions) * 100
        st.metric("Average Risk", f"{avg_risk:.1f}%")
    
    with col2:
        high_risk = sum(1 for p in predictions if p >= MORTALITY_THRESHOLD)
        st.metric("High Risk Cases", high_risk)
    
    with col3:
        actual_deaths = df_analyzed['F_MORTS'].sum()
        st.metric("Actual Deaths", int(actual_deaths))
    
    with col4:
        accuracy = sum(1 for p, deaths in zip(predictions, df_analyzed['F_MORTS']) 
                      if (p >= MORTALITY_THRESHOLD) == (deaths > 0)) / len(predictions) * 100
        st.metric("Prediction Match", f"{accuracy:.1f}%")


def plot_prediction_distribution(predictions: list, actual_outcomes: list) -> None:
    """Plot distribution of predictions vs actual outcomes."""
    
    df_plot = pd.DataFrame({
        'Predicted Probability': predictions,
        'Actual Outcome': ['Fatal' if death > 0 else 'Non-Fatal' for death in actual_outcomes]
    })
    
    fig = px.box(
        df_plot,
        x='Actual Outcome',
        y='Predicted Probability',
        color='Actual Outcome',
        color_discrete_map={'Fatal': '#d32f2f', 'Non-Fatal': '#388e3c'},
        title='Prediction Distribution: Predicted vs Actual Outcomes'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():    
    st.markdown("""
    ### 📊 Past accident analysis
    """)
    st.markdown(f"""
    **Legend:** ✅ Correct prediction | ⚠️ Predicted death but none occurred | ❗ Missed: deaths occurred but not predicted
    
    **Threshold:** Predictions ≥ {MORTALITY_THRESHOLD*100:.0f}% are considered fatal
    """)
    
    n_samples = st.slider("Number of accidents to analyze", 5, 200, 50) # 50 cases as default instead
    
    col1, col2 = st.columns([1, 4])
    with col1:
        load_button = st.button("🔄 Sample past cases")
    
    if load_button:
        df_sample = load_sample_accidents(reference_df, n_samples)
        
        if len(df_sample) > 0:
            predictions = []
            
            st.markdown("---")
            st.markdown(f"#### Analyzing {len(df_sample)} accidents...")
            
            # Display statistics
            display_accident_statistics(
                [predictor.predict_from_dict(extract_scenario_from_accident(row))[1] 
                 for _, row in df_sample.iterrows()],
                df_sample
            )
            
            st.markdown("---")
            
            # Show all cards in grid and track prediction outcomes
            cols = st.columns(5)
            icon_counts = {"✅": 0, "⚠️": 0, "❗": 0}
            
            for idx, (_, accident) in enumerate(df_sample.iterrows()):
                scenario = extract_scenario_from_accident(accident)
                pred, prob = predictor.predict_from_dict(scenario)
                actual_deaths = accident.get('F_MORTS', 0)
                predictions.append(prob)
                
                # Track icon type
                predicted_fatal = prob >= MORTALITY_THRESHOLD
                actual_fatal = actual_deaths > 0
                
                if predicted_fatal == actual_fatal:
                    icon_counts["✅"] += 1
                elif predicted_fatal and not actual_fatal:
                    icon_counts["⚠️"] += 1
                else:
                    icon_counts["❗"] += 1
                
                col_idx = idx % 5
                with cols[col_idx]:
                    st.markdown(f"**Accident #{idx + 1}**")
                    display_accident_card(accident, prob, actual_deaths)
            
            # Show summary of prediction outcomes
            st.markdown("---")
            total = len(df_sample)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pct = (icon_counts["✅"] / total * 100) if total > 0 else 0
                st.metric("✅ Correct", f"{icon_counts['✅']} ({pct:.1f}%)")
            
            with col2:
                pct = (icon_counts["⚠️"] / total * 100) if total > 0 else 0
                st.metric("⚠️ False positive", f"{icon_counts['⚠️']} ({pct:.1f}%)")
            
            with col3:
                pct = (icon_counts["❗"] / total * 100) if total > 0 else 0
                st.metric("❗ Missed (false negative)", f"{icon_counts['❗']} ({pct:.1f}%)")
            
            st.markdown("---")
            plot_prediction_distribution(predictions, df_sample['F_MORTS'].tolist())
            
if __name__ == "__main__":
    main()
