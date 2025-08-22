import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    <h1>🔬 Interface Krr - Analyse Friction</h1>
    <h2>Calcul et Comparaison des Coefficients</h2>
</div>
""", unsafe_allow_html=True)

# ==================== INITIALISATION ====================
if 'experiments' not in st.session_state:
    st.session_state.experiments = {}

# ==================== FONCTION CALCUL KRR ====================
def calculate_krr_simple(df_valid, fps=250, sphere_mass_g=10.0, sphere_radius_mm=15.0, angle_deg=15.0):
    """Calcul Krr simple et fonctionnel"""
    
    if len(df_valid) < 20:
        return None
    
    # Paramètres
    dt = 1/fps
    g = 9.81
    
    # Calibration
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
    
    # Krr
    if v0 > vf and distance > 0:
        krr = (v0**2 - vf**2) / (2 * g * distance)
    else:
        krr = 0
    
    return {
        'krr': krr,
        'v0': v0 * 1000,
        'vf': vf * 1000,
        'distance': distance * 1000,
        'calibration': pixels_per_mm
    }

# ==================== CHARGEMENT DONNÉES ====================
def load_data_simple(uploaded_file, exp_name, water_content, angle, sphere_type):
    """Chargement simple des données"""
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

# ==================== INTERFACE CHARGEMENT ====================
st.markdown("## 📂 Chargement des Données")

col1, col2 = st.columns(2)

with col1:
    exp_name = st.text_input("Nom expérience", value=f"Exp_{len(st.session_state.experiments)+1}")
    water_content = st.number_input("Teneur en eau (%)", value=0.0, min_value=0.0, max_value=30.0)
    
with col2:
    angle = st.number_input("Angle (°)", value=5.0, min_value=1.0, max_value=45.0)
    sphere_type = st.selectbox("Type sphère", ["Solide", "Creuse"])

uploaded_file = st.file_uploader("Fichier CSV", type=['csv'], key=f"file_uploader_{len(st.session_state.experiments)}")

if st.button("🚀 Analyser et Ajouter à la Comparaison") and uploaded_file is not None:
    # Vérifier si le nom existe déjà
    if exp_name in st.session_state.experiments:
        st.warning(f"⚠️ Expérience '{exp_name}' existe déjà. Changez le nom ou elle sera remplacée.")
    
    exp_data = load_data_simple(uploaded_file, exp_name, water_content, angle, sphere_type)
    
    if exp_data:
        # AJOUTER (pas remplacer) l'expérience
        st.session_state.experiments[exp_name] = exp_data
        metrics = exp_data['metrics']
        
        st.success(f"✅ Expérience '{exp_name}' AJOUTÉE à la comparaison!")
        st.info(f"📊 Total expériences: {len(st.session_state.experiments)}")
        
        # Affichage résultats de la nouvelle expérience
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            krr_val = metrics['krr']
            status = "✅ NORMAL" if 0.03 <= krr_val <= 0.15 else "⚠️ ÉLEVÉ"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Krr</h3>
                <h2>{krr_val:.6f}</h2>
                <p>{status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>V₀</h3>
                <h2>{metrics['v0']:.1f} mm/s</h2>
                <p>Vitesse initiale</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Distance</h3>
                <h2>{metrics['distance']:.1f} mm</h2>
                <p>Distance parcourue</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Calibration</h3>
                <h2>{metrics['calibration']:.2f} px/mm</h2>
                <p>Pixels par mm</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.rerun()

# ==================== AFFICHAGE EXPÉRIENCES ACTUELLES ====================
if st.session_state.experiments:
    st.markdown(f"### 📋 Expériences Chargées ({len(st.session_state.experiments)})")
    
    exp_summary = []
    for name, exp in st.session_state.experiments.items():
        exp_summary.append({
            'Nom': name,
            'Eau (%)': exp['water_content'],
            'Angle (°)': exp['angle'],
            'Type': exp['sphere_type'],
            'Krr': f"{exp['metrics']['krr']:.6f}",
            'Status': '✅' if 0.03 <= exp['metrics']['krr'] <= 0.15 else '⚠️'
        })
    
    summary_df = pd.DataFrame(exp_summary)
    st.dataframe(summary_df, use_container_width=True)

# ==================== TESTS RAPIDES ====================
st.markdown("### 🧪 Tests Rapides")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧪 Test Sec (0% eau, 5°)"):
        np.random.seed(42)
        test_metrics = {
            'krr': 0.055 + np.random.normal(0, 0.005),
            'v0': 180 + np.random.normal(0, 20),
            'distance': 250 + np.random.normal(0, 30),
            'calibration': 1.33
        }
        st.session_state.experiments['Test_Sec_0%'] = {
            'name': 'Test_Sec_0%',
            'water_content': 0.0,
            'angle': 5.0,
            'sphere_type': 'Solide',
            'metrics': test_metrics,
            'success_rate': 85.0
        }
        st.success("✅ Test sec ajouté!")
        st.rerun()

with col2:
    if st.button("🧪 Test Humide (10% eau, 15°)"):
        np.random.seed(123)
        test_metrics = {
            'krr': 0.072 + np.random.normal(0, 0.005),
            'v0': 160 + np.random.normal(0, 15),
            'distance': 220 + np.random.normal(0, 25),
            'calibration': 1.33
        }
        st.session_state.experiments['Test_Humide_10%'] = {
            'name': 'Test_Humide_10%',
            'water_content': 10.0,
            'angle': 15.0,
            'sphere_type': 'Solide',
            'metrics': test_metrics,
            'success_rate': 78.0
        }
        st.success("✅ Test humide ajouté!")
        st.rerun()

with col3:
    if st.button("🧪 Test Angle (5% eau, 30°)"):
        np.random.seed(456)
        test_metrics = {
            'krr': 0.063 + np.random.normal(0, 0.006),
            'v0': 220 + np.random.normal(0, 25),
            'distance': 280 + np.random.normal(0, 35),
            'calibration': 1.33
        }
        st.session_state.experiments['Test_Angle_30°'] = {
            'name': 'Test_Angle_30°',
            'water_content': 5.0,
            'angle': 30.0,
            'sphere_type': 'Solide',
            'metrics': test_metrics,
            'success_rate': 82.0
        }
        st.success("✅ Test angle ajouté!")
        st.rerun()

# ==================== TABLEAU RÉSULTATS ====================
if st.session_state.experiments:
    st.markdown("## 📋 Tableau de Comparaison")
    
    results = []
    for name, exp in st.session_state.experiments.items():
        metrics = exp['metrics']
        results.append({
            'Expérience': name,
            'Eau (%)': exp['water_content'],
            'Angle (°)': exp['angle'],
            'Type': exp['sphere_type'],
            'Krr': f"{metrics['krr']:.6f}",
            'V₀ (mm/s)': f"{metrics['v0']:.1f}",
            'Distance (mm)': f"{metrics['distance']:.1f}",
            'Succès (%)': f"{exp['success_rate']:.1f}"
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

# ==================== GRAPHIQUES ====================
if st.session_state.experiments:
    if len(st.session_state.experiments) >= 1:
        st.markdown("## 📊 Graphiques de Comparaison")
        
        # Préparer données
        plot_data = []
        for name, exp in st.session_state.experiments.items():
            plot_data.append({
                'Expérience': name,
                'Humidité': exp['water_content'],
                'Angle': exp['angle'],
                'Krr': exp['metrics']['krr'],
                'V0': exp['metrics']['v0'],
                'Distance': exp['metrics']['distance'],
                'Sphere_Type': exp['sphere_type'],
                'Success_Rate': exp['success_rate']
            })
        plot_df = pd.DataFrame(plot_data)
        
        # === GRAPHIQUES PRINCIPAUX ===
        col1, col2 = st.columns(2)
        
        with col1:
            # Krr vs Humidité
            fig1 = px.scatter(plot_df, x='Humidité', y='Krr', 
                            color='Sphere_Type', size='Success_Rate',
                            hover_data=['Expérience', 'V0', 'Angle'],
                            title="💧 Krr vs Teneur en Eau",
                            labels={'Krr': 'Coefficient Krr', 'Humidité': 'Teneur en Eau (%)'})
            
            # Références Van Wal
            fig1.add_hline(y=0.052, line_dash="dash", line_color="red", annotation_text="Van Wal min: 0.052")
            fig1.add_hline(y=0.066, line_dash="dash", line_color="red", annotation_text="Van Wal max: 0.066")
            
            # Courbe de tendance
            if len(plot_df) >= 2:
                try:
                    z = np.polyfit(plot_df['Humidité'], plot_df['Krr'], min(2, len(plot_df)-1))
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Humidité'].min(), plot_df['Humidité'].max(), 100)
                    fig1.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                            name='Tendance', 
                                            line=dict(dash='dot', color='purple', width=3)))
                except:
                    pass
            
            fig1.update_layout(height=500)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Krr vs Angle
            fig2 = px.scatter(plot_df, x='Angle', y='Krr', 
                            color='Humidité', size='V0',
                            hover_data=['Expérience', 'Distance'],
                            title="📐 Krr vs Angle d'Inclinaison",
                            labels={'Krr': 'Coefficient Krr', 'Angle': 'Angle (°)'})
            
            # Références Van Wal
            fig2.add_hline(y=0.052, line_dash="dash", line_color="red", annotation_text="Van Wal min")
            fig2.add_hline(y=0.066, line_dash="dash", line_color="red", annotation_text="Van Wal max")
            
            # Courbe de tendance
            if len(plot_df) >= 2:
                try:
                    z = np.polyfit(plot_df['Angle'], plot_df['Krr'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Angle'].min(), plot_df['Angle'].max(), 100)
                    fig2.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                            name='Tendance', 
                                            line=dict(dash='dot', color='orange', width=3)))
                except:
                    pass
            
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
        
        # === GRAPHIQUE COMPARATIF ===
        st.markdown("### 📊 Comparaison Directe")
        
        fig_comparison = go.Figure()
        
        # Barres colorées par type
        colors = ['lightblue' if stype == 'Solide' else 'lightcoral' for stype in plot_df['Sphere_Type']]
        
        fig_comparison.add_trace(go.Bar(
            x=[f"{row['Expérience']}\n({row['Humidité']}%, {row['Angle']}°)" for _, row in plot_df.iterrows()],
            y=plot_df['Krr'],
            name='Krr',
            text=[f"{val:.4f}" for val in plot_df['Krr']],
            textposition='auto',
            marker_color=colors
        ))
        
        # Références Van Wal
        fig_comparison.add_hline(y=0.052, line_dash="dash", line_color="red", annotation_text="Van Wal min: 0.052")
        fig_comparison.add_hline(y=0.066, line_dash="dash", line_color="red", annotation_text="Van Wal max: 0.066")
        
        # Moyenne expériences
        mean_krr = plot_df['Krr'].mean()
        fig_comparison.add_hline(y=mean_krr, line_dash="dot", line_color="green", 
                               annotation_text=f"Moyenne: {mean_krr:.4f}")
        
        fig_comparison.update_layout(
            title="📊 Comparaison Krr - Toutes Expériences",
            xaxis_title="Expériences",
            yaxis_title="Coefficient Krr",
            height=500,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # === ANALYSE TENDANCES ===
        if len(plot_df) >= 2:
            st.markdown("### 🔍 Analyse des Tendances")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if plot_df['Humidité'].nunique() > 1:
                    humid_krr_corr = plot_df[['Humidité', 'Krr']].corr().iloc[0, 1]
                    st.metric("Corrélation Humidité ↔ Krr", f"{humid_krr_corr:.3f}")
                    
                    if humid_krr_corr > 0.5:
                        st.success("📈 Krr augmente avec humidité")
                    elif humid_krr_corr < -0.5:
                        st.info("📉 Krr diminue avec humidité")
                    else:
                        st.warning("➡️ Peu d'effet humidité")
                else:
                    st.info("Une seule humidité")
            
            with col2:
                if plot_df['Angle'].nunique() > 1:
                    angle_krr_corr = plot_df[['Angle', 'Krr']].corr().iloc[0, 1]
                    st.metric("Corrélation Angle ↔ Krr", f"{angle_krr_corr:.3f}")
                    
                    if abs(angle_krr_corr) > 0.7:
                        direction = "augmente" if angle_krr_corr > 0 else "diminue"
                        st.warning(f"⚠️ Krr {direction} avec angle")
                    else:
                        st.success("✅ Krr indépendant angle (Van Wal)")
                else:
                    st.info("Un seul angle")
            
            with col3:
                krr_values = plot_df['Krr'].values
                van_wal_range = [0.052, 0.066]
                in_range = np.sum((krr_values >= van_wal_range[0]) & (krr_values <= van_wal_range[1]))
                total = len(krr_values)
                
                st.metric("Dans plage Van Wal", f"{in_range}/{total}")
                
                if in_range / total > 0.7:
                    st.success("✅ Cohérent littérature")
                else:
                    st.warning("⚠️ Écart Van Wal")
        
        # === EXPORT ===
        st.markdown("### 📥 Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📋 Export Tableau",
                data=csv_data,
                file_name="resultats_krr.csv",
                mime="text/csv"
            )
        
        with col2:
            plot_csv = plot_df.to_csv(index=False)
            st.download_button(
                label="📊 Export Graphiques",
                data=plot_csv,
                file_name="donnees_graphiques.csv",
                mime="text/csv"
            )

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
### ✅ Interface Krr Fonctionnelle
- Calcul Krr : Krr = (V₀² - Vf²) / (2gL)
- Graphiques comparatifs avec courbes de tendance
- Références Van Wal intégrées
- Export des données et résultats
""")
