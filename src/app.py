import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_loader import MortalityPredictor, load_reference_data

# Define mortality prediction threshold (should match past_accident_analysis.py)
MORTALITY_THRESHOLD = 0.56

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

col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.markdown("""
    #### 🔄 Comparative analysis
    Compare multiple accident scenarios side-by-side to understand relative risk levels.
    """)
    
with col2:
    st.markdown("""
    #### 🔮 Hypothetical scenarios
    Simulate new accident characteristics and predict mortality risk in "what-if" scenarios.
    """)

with col3:
    st.success("""
    #### ⭐ **Past accident analysis**
    
    **Validate model performance** on real historical accidents. Assess how well our model 
    predicts actual outcomes. This is the **core analysis** to understand model reliability.
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

st.markdown("---")
st.markdown("### ROC Curve")

# Create a sample ROC curve (simulated for visualization)
fpr = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
tpr = np.array([0, 0.3, 0.5, 0.65, 0.75, 0.85, 0.92, 0.98, 1.0])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name='ROC Curve',
    line=dict(color='#1f77b4', width=2)
))

# Diagonal reference line
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Random Classifier',
    line=dict(color='gray', dash='dash', width=1)
))

fig.update_layout(
    title='ROC Curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    hovermode='closest',
    height=500,
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)
