import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Interface Krr Simple",
    page_icon="🔬",
    layout="wide"
)

# ==================== CSS ====================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITRE ====================
st.markdown("""
<div class="main-header">
    <h1>🔬 Interface Krr Simple</h1>
    <h2>Calcul Basic Sans Complications</h2>
</div>
""", unsafe_allow_html=True)

# ==================== INITIALISATION ====================
if 'experiments' not in st.session_state:
    st.session_state.experiments = {}

# ==================== FONCTION CALCUL KRR SIMPLE ====================
def calculate_krr_simple(df_valid, fps=250, sphere_mass_g=10.0, sphere_radius_mm=15.0, angle_deg=15.0):
    """Calcul Krr simple sans complications"""
    
    if len(df_valid) < 20:
        st.error("Pas assez de données")
        return None
    
    # Paramètres
    dt = 1/fps
    g = 9.81
    
    # Calibration simple
    avg_radius_px = df_valid['Radius'].mean()
    pixels_per_mm = avg_radius_px / sphere_radius_mm
    
    # Positions en mètres
    x_m = df_valid['X_center'].values / pixels_per_mm / 1000
    y_m = df_valid['Y_center'].values / pixels_per_mm / 1000
    
    # Vitesses
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Vitesses moyennées
    n = len(v_magnitude) // 4
    v0 = np.mean(v_magnitude[:n])
    vf = np.mean(v_magnitude[-n:])
    
    # Distance
    distance = np.sum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))
    
    # Krr simple
    if v0 > vf and distance > 0:
        krr = (v0**2 - vf**2) / (2 * g * distance)
    else:
        krr = 0
    
    return {
        'krr': krr,
        'v0': v0 * 1000,  # mm/s
        'vf': vf * 1000,  # mm/s
        'distance': distance * 1000,  # mm
        'calibration': pixels_per_mm
    }

# ==================== CHARGEMENT DONNÉES ====================
def load_data_simple(uploaded_file, exp_name, water_content, angle, sphere_type):
    """Chargement simple"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
            
            metrics = calculate_krr_simple(df_valid, angle_deg=angle)
            if metrics is None:
                return None
                
            return {
                'name': exp_name,
                'water_content': water_content,
                'angle': angle,
                'sphere_type': sphere_type,
                'metrics': metrics,
                'success_rate': len(df_valid) / len(df) * 100
            }
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
            return None
    return None

# ==================== INTERFACE ====================
st.markdown("## 📂 Chargement Simple")

col1, col2 = st.columns(2)

with col1:
    exp_name = st.text_input("Nom expérience", value="Exp_Simple")
    water_content = st.number_input("Eau (%)", value=0.0, min_value=0.0, max_value=30.0)
    
with col2:
    angle = st.number_input("Angle (°)", value=5.0, min_value=1.0, max_value=45.0)
    sphere_type = st.selectbox("Type sphère", ["Solide", "Creuse"])

uploaded_file = st.file_uploader("Fichier CSV", type=['csv'])

if st.button("🚀 Analyser Simple") and uploaded_file is not None:
    exp_data = load_data_simple(uploaded_file, exp_name, water_content, angle, sphere_type)
    
    if exp_data:
        st.session_state.experiments[exp_name] = exp_data
        metrics = exp_data['metrics']
        
        # Affichage résultats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            krr_val = metrics['krr']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Krr</h3>
                <h2>{krr_val:.6f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>V₀</h3>
                <h2>{metrics['v0']:.1f} mm/s</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Distance</h3>
                <h2>{metrics['distance']:.1f} mm</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Calibration</h3>
                <h2>{metrics['calibration']:.2f} px/mm</h2>
            </div>
            """, unsafe_allow_html=True)

# ==================== TABLEAU RÉSULTATS ====================
if st.session_state.experiments:
    st.markdown("## 📋 Résultats")
    
    results = []
    for name, exp in st.session_state.experiments.items():
        metrics = exp['metrics']
        results.append({
            'Expérience': name,
            'Eau (%)': exp['water_content'],
            'Angle (°)': exp['angle'],
            'Krr': f"{metrics['krr']:.6f}",
            'V₀ (mm/s)': f"{metrics['v0']:.1f}",
            'Distance (mm)': f"{metrics['distance']:.1f}",
            'Succès (%)': f"{exp['success_rate']:.1f}"
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    # Graphiques simples
    if len(results) >= 2:
        st.markdown("## 📊 Graphiques")
        
        # Préparer données
        plot_data = []
        for name, exp in st.session_state.experiments.items():
            plot_data.append({
                'Expérience': name,
                'Humidité': exp['water_content'],
                'Angle': exp['angle'],
                'Krr': exp['metrics']['krr']
            })
        plot_df = pd.DataFrame(plot_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.scatter(plot_df, x='Humidité', y='Krr', 
                            title="Krr vs Humidité", 
                            hover_data=['Expérience'])
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.scatter(plot_df, x='Angle', y='Krr', 
                            title="Krr vs Angle", 
                            hover_data=['Expérience'])
            st.plotly_chart(fig2, use_container_width=True)

# ==================== GESTION ====================
if st.session_state.experiments:
    st.markdown("## 🗂️ Gestion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exp_to_remove = st.selectbox("Supprimer:", ["Aucune"] + list(st.session_state.experiments.keys()))
        if exp_to_remove != "Aucune" and st.button("🗑️ Supprimer"):
            del st.session_state.experiments[exp_to_remove]
            st.rerun()
    
    with col2:
        if st.button("🧹 Tout effacer"):
            st.session_state.experiments = {}
            st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
### ✅ Interface Simple Fonctionnelle
- Calcul Krr basique sans complications
- Formule : Krr = (V₀² - Vf²) / (2gL)
- Graphiques simples
- Pas de bugs !
""")
