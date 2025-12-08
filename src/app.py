import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_loader import MortalityPredictor, load_reference_data

# Configure page
st.set_page_config(
    page_title="Understanding mortality in Catalan traffic accidents",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def load_models_and_data():
    predictor = MortalityPredictor()
    reference_df = load_reference_data()
    return predictor, reference_df

predictor, reference_df = load_models_and_data()

# Home page content
st.title("🚗 Understanding mortality in Catalan traffic accidents")
st.markdown("### Visual Analytics Final Project, by Felipe López and Oriol Pont")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 📌 Project introduction
    
    This application analyzes traffic accident data from Catalonia (2010-2023) to predict 
    and understand mortality risk. With **24.5k accident records**, we've trained machine 
    learning models to identify patterns that correlate with deadly outcomes.
    """)

with col2:
    st.info("""
    - 📊 24,500+ accident records
    - 🗓️ Time span: 2010-2023 (13 years)
    - 🎯 40+ features analyzed
    - ⚙️ More than 50k model variations explored
    """)

st.markdown("---")
st.markdown("### 🗺️ Use cases")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    #### 🔮 Hypothetical scenarios
    Simulate new accident characteristics and predict mortality risk in "what-if" scenarios.
    """)

with col2:
    st.markdown("""
    #### 📊 Past accident analysis
    Upload or select historical accidents to assess the likelihood of observed outcomes.
    """)

with col3:
    st.markdown("""
    #### ⚠️ Risk prevention
    Identify which factors most influence mortality risk to guide prevention strategies.
    """)

with col4:
    st.markdown("""
    #### 🔄 Comparative analysis
    Compare multiple accident scenarios side-by-side to understand relative risk levels.
    """)

st.markdown("---")
st.markdown("### 📈 Model information")
best_model = predictor._select_best_model()
model_info = predictor.get_model_info(best_model)
st.write(f"**Selected Model:** {best_model}")
if model_info:
    test_metrics = model_info.get('evaluation_metrics', {}).get('Test', {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", f"{test_metrics.get('ROC-AUC', 0):.3f}")
    with col2:
        st.metric("Recall", f"{test_metrics.get('Recall', 0):.3f}")
    with col3:
        st.metric("Precision", f"{test_metrics.get('Precision', 0):.3f}")
    with col4:
        st.metric("F1 Score", f"{test_metrics.get('F1', 0):.3f}")
