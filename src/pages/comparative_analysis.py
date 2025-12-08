import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_loader import MortalityPredictor, load_reference_data

# Load models and data
@st.cache_resource
def load_models_and_data():
    predictor = MortalityPredictor()
    reference_df = load_reference_data()
    return predictor, reference_df

predictor, reference_df = load_models_and_data()

st.set_page_config(page_title="🔄 Comparative Analysis", layout="wide")

def create_risk_comparison_chart(names: list, risks: list, colors_list: list = None) -> None:
    if colors_list is None:
        colors_list = ['#d32f2f' if r >= 0.5 else '#f57c00' if r >= 0.3 else '#388e3c' 
                      for r in risks]
    
    fig = go.Figure(go.Bar(
        x=names,
        y=[r*100 for r in risks],
        marker=dict(color=colors_list),
        text=[f"{r*100:.1f}%" for r in risks],
        textposition='outside',
    ))
    
    fig.update_layout(
        title="Mortality Risk Comparison",
        yaxis_title="Risk Probability (%)",
        xaxis_title="Scenario",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_accident_type_comparison(reference_df: pd.DataFrame) -> None:    
    df_analysis = reference_df.copy()
    
    def categorize_accident(row):
        if row['F_MOTOCICLETES_IMPLICADES'] > 0:
            return 'Motorcycle Involved'
        elif row['F_VIANANTS_IMPLICADES'] > 0:
            return 'Pedestrian Involved'
        elif row['F_VEH_PESANTS_IMPLICADES'] > 0:
            return 'Heavy Vehicle Involved'
        elif row['F_UNITATS_IMPLICADES'] > 2:
            return 'Multi-Vehicle'
        else:
            return 'Standard (2 vehicles)'
    
    df_analysis['accident_type'] = df_analysis.apply(categorize_accident, axis=1)
    
    # Create visualization
    type_data = df_analysis.groupby('accident_type').apply(
        lambda x: (x['F_MORTS'] > 0).sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    fig = px.bar(
        x=type_data.index,
        y=type_data.values,
        labels={'y': 'Fatal Accident Rate (%)', 'x': 'Accident Type'},
        title='Fatal Accident Rates by Type',
        color=type_data.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=400, template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def display_comparison_table(scenarios: dict, predictions: dict) -> None:    
    data = []
    for name, scenario in scenarios.items():
        risk = predictions.get(name, 0)
        risk_level = "🔴 HIGH" if risk >= 0.5 else "🟠 MEDIUM" if risk >= 0.3 else "🟢 LOW"
        
        data.append({
            'Scenario': name,
            'Units': scenario.get('F_UNITATS_IMPLICADES', 0),
            'Pedestrians': scenario.get('F_VIANANTS_IMPLICADES', 0),
            'Motorcycles': scenario.get('F_MOTOCICLETES_IMPLICADES', 0),
            'Heavy Vehicles': scenario.get('F_VEH_PESANTS_IMPLICADES', 0),
            'Risk Probability': f"{risk*100:.1f}%",
            'Risk Level': risk_level,
        })
    
    df_comparison = pd.DataFrame(data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)


def main():    
    st.markdown("""
    ### 🔄 Comparative analysis
    Compare different accident scenarios side-by-side to understand relative risk levels.
    """)
    
    st.markdown("""
    Create multiple accident scenarios and compare their predicted mortality risks, 
    compositions, and historical patterns in similar accidents.
    """)
    
    # Tabs for different comparsion modes
    tab1, tab2, tab3 = st.tabs([
        "Custom scenario comparison",
        "Accident type analysis",
        "Advanced comparison"
    ])
    
    with tab1:
        st.markdown("---")
        st.markdown("#### 📋 Create scenarios to compare")
        
        num_scenarios = st.slider("Number of scenarios", 2, 4, 2, key="num_scenarios")
        
        scenarios = {}
        cols = st.columns(num_scenarios)
        
        for idx, col in enumerate(cols):
            with col:
                scenario_name = st.text_input(f"Scenario {idx+1} name", 
                                             f"Scenario {idx+1}", 
                                             key=f"name_{idx}")
                
                st.markdown(f"##### {scenario_name}")
                
                scenario = {
                    'F_UNITATS_IMPLICADES': st.slider(
                        "Units", 1, 6, 2, key=f"units_{idx}"
                    ),
                    'F_VIANANTS_IMPLICADES': st.slider(
                        "Pedestrians", 0, 3, 0, key=f"peds_{idx}"
                    ),
                    'F_MOTOCICLETES_IMPLICADES': st.slider(
                        "Motorcycles", 0, 3, 0, key=f"moto_{idx}"
                    ),
                    'F_VEH_PESANTS_IMPLICADES': st.slider(
                        "Heavy vehicles", 0, 2, 0, key=f"heavy_{idx}"
                    ),
                    'F_VEH_LLEUGERS_IMPLICADES': st.slider(
                        "Light vehicles", 0, 4, 1, key=f"light_{idx}"
                    ),
                    'F_BICICLETES_IMPLICADES': 0,
                    'F_CICLOMOTORS_IMPLICADES': 0,
                    'F_ALTRES_UNIT_IMPLICADES': 0,
                }
                
                scenarios[scenario_name] = scenario
        
        if st.button("Compare scenarios", key="compare_custom"):
            st.markdown("---")
            
            predictions = {}
            try:
                for name, scenario in scenarios.items():
                    _, prob = predictor.predict_from_dict(scenario, reference_df=reference_df)
                    predictions[name] = prob
                
                st.markdown("#### Detailed comparison")
                display_comparison_table(scenarios, predictions)
                
                st.markdown("---")
                
                create_risk_comparison_chart(
                    list(scenarios.keys()),
                    list(predictions.values())
                )
                
            except Exception as e:
                st.error(f"Comparison error: {str(e)}")
    
    with tab2:
        st.markdown("---")
        st.markdown("#### 🚗 Accident type analysis")
        
        st.write("""
        Analyze how different types of accidents have historically affected mortality rates 
        in the Catalan dataset.
        """)
        
        create_accident_type_comparison(reference_df)
        
        st.markdown("---")
        st.markdown("#### 📈 Type breakdown")
        
        # Calculate statistics by type
        df_analysis = reference_df.copy()
        
        def categorize_accident(row):
            if row['F_MOTOCICLETES_IMPLICADES'] > 0:
                return 'Motorcycle Involved'
            elif row['F_VIANANTS_IMPLICADES'] > 0:
                return 'Pedestrian Involved'
            elif row['F_VEH_PESANTS_IMPLICADES'] > 0:
                return 'Heavy Vehicle Involved'
            elif row['F_UNITATS_IMPLICADES'] > 2:
                return 'Multi-Vehicle'
            else:
                return 'Standard (2 vehicles)'
        
        df_analysis['accident_type'] = df_analysis.apply(categorize_accident, axis=1)
        
        for acc_type in df_analysis['accident_type'].unique():
            df_type = df_analysis[df_analysis['accident_type'] == acc_type]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"{acc_type} - Count", len(df_type))
            
            with col2:
                fatal_pct = (df_type['F_MORTS'] > 0).sum() / len(df_type) * 100
                st.metric("Fatal Rate", f"{fatal_pct:.1f}%")
            
            with col3:
                total_deaths = df_type['F_MORTS'].sum()
                st.metric("Total Deaths", int(total_deaths))
            
            with col4:
                avg_victims = df_type['F_VICTIMES'].mean()
                st.metric("Avg Victims", f"{avg_victims:.1f}")
    
    with tab3:
        st.markdown("---")
        st.markdown("#### 🎯 Advanced comparison")
        
        st.write("""
        Compare scenarios with similar historical accidents to understand 
        how your scenario relates to real-world data.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Your scenario")
            your_scenario = {
                'F_UNITATS_IMPLICADES': st.slider("Units", 1, 6, 2, key="adv_units"),
                'F_VIANANTS_IMPLICADES': st.slider("Pedestrians", 0, 3, 0, key="adv_peds"),
                'F_MOTOCICLETES_IMPLICADES': st.slider("Motorcycles", 0, 3, 1, key="adv_moto"),
                'F_VEH_PESANTS_IMPLICADES': st.slider("Heavy vehicles", 0, 2, 0, key="adv_heavy"),
                'F_VEH_LLEUGERS_IMPLICADES': st.slider("Light vehicles", 0, 4, 1, key="adv_light"),
                'F_BICICLETES_IMPLICADES': 0,
                'F_CICLOMOTORS_IMPLICADES': 0,
                'F_ALTRES_UNIT_IMPLICADES': 0,
            }
        
        with col2:
            st.markdown("##### Statistics")
            _, your_risk = predictor.predict_from_dict(your_scenario, reference_df=reference_df)
            st.metric("Predicted Risk", f"{your_risk*100:.1f}%")
            
            # Find similar accidents
            similar_mask = (
                (reference_df['F_UNITATS_IMPLICADES'] == your_scenario['F_UNITATS_IMPLICADES']) &
                (reference_df['F_VIANANTS_IMPLICADES'] <= your_scenario['F_VIANANTS_IMPLICADES'] + 1) &
                (reference_df['F_MOTOCICLETES_IMPLICADES'] <= your_scenario['F_MOTOCICLETES_IMPLICADES'] + 1)
            )
            
            similar_count = similar_mask.sum()
            if similar_count > 0:
                similar_fatal_rate = (reference_df[similar_mask]['F_MORTS'] > 0).sum() / similar_count * 100
                st.metric("Historical Match", f"{similar_count} accidents")
                st.metric("Historical Fatal Rate", f"{similar_fatal_rate:.1f}%")

if __name__ == "__main__":
    main()
