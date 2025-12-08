"""
Shared utilities for Streamlit pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict, List
import plotly.graph_objects as go
import plotly.express as px


def display_mortality_risk(probability: float):
    """Display mortality risk with color-coded badge."""
    if probability >= 0.5:
        risk_level = "HIGH"
        color = "#d32f2f"
    elif probability >= 0.3:
        risk_level = "MEDIUM"
        color = "#f57c00"
    else:
        risk_level = "LOW"
        color = "#388e3c"
    
    st.markdown(f"""
    <div style="background-color: {color}20; padding: 1rem; border-radius: 0.5rem; 
                border-left: 4px solid {color};">
        <h3 style="margin: 0; color: {color};">Risk Level: {risk_level}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold;">
            {probability*100:.1f}% mortality risk
        </p>
    </div>
    """, unsafe_allow_html=True)


def create_probability_gauge(probability: float, prediction: int):
    """Create a gauge chart for probability visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={"text": "Mortality Probability (%)"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#d32f2f" if probability >= 0.5 else "#f57c00" if probability >= 0.3 else "#388e3c"},
            "steps": [
                {"range": [0, 30], "color": "#e8f5e9"},
                {"range": [30, 60], "color": "#fff3e0"},
                {"range": [60, 100], "color": "#ffebee"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=300, margin={"b": 0, "t": 30, "l": 0, "r": 0})
    return fig


def create_comparison_chart(scenarios: List[Dict]):
    """Create a comparison chart for multiple scenarios."""
    df_comp = pd.DataFrame(scenarios)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_comp['name'],
        y=df_comp['probability'] * 100,
        marker=dict(
            color=df_comp['probability'],
            colorscale='Reds',
            showscale=False,
            line=dict(width=2, color='darkred')
        ),
        text=[f"{p*100:.1f}%" for p in df_comp['probability']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Mortality Risk Comparison",
        xaxis_title="Scenario",
        yaxis_title="Mortality Risk (%)",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def create_feature_importance_chart(feature_importance: pd.DataFrame, top_n: int = 15):
    """Create a feature importance bar chart."""
    top_features = feature_importance.head(top_n)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Top {top_n} Most Important Features",
        labels={'Importance': 'Feature Importance Score', 'Feature': ''}
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Importance Score"
    )
    
    return fig


def create_prediction_summary(features_dict: Dict, prediction: int, probability: float):
    """Create a summary card of the prediction."""
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 1rem; color: white; margin: 1rem 0;">
        <h3 style="margin: 0 0 1rem 0;">Prediction Summary</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
                <p style="margin: 0; opacity: 0.9;">Predicted Outcome</p>
                <h2 style="margin: 0.5rem 0 0 0; color: {'#ff6b6b' if prediction == 1 else '#51cf66'};">
                    {'MORTALITY' if prediction == 1 else 'NO MORTALITY'}
                </h2>
            </div>
            <div>
                <p style="margin: 0; opacity: 0.9;">Mortality Probability</p>
                <h2 style="margin: 0.5rem 0 0 0;">{probability*100:.1f}%</h2>
            </div>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)


def normalize_features(features_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using RobustScaler fitted on reference data.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Input features to normalize
    reference_df : pd.DataFrame
        Reference data to fit scaler on (typically the cleaned training data)
    
    Returns:
    --------
    normalized_df : pd.DataFrame
        Normalized features
    """
    # Get numeric columns only
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fit scaler on reference data
    scaler = RobustScaler()
    scaler.fit(reference_df[numeric_cols])
    
    # Transform features
    features_normalized = features_df.copy()
    features_normalized[numeric_cols] = scaler.transform(features_df[numeric_cols])
    
    return features_normalized
