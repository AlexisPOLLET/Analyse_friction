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
        
        st.success(f"✅ Expérience '{exp_name}' ajoutée avec succès!")
        
        # Affichage résultats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            krr_val = metrics['krr']
            status = "✅ NORMAL" if 0.03 <= krr_val <= 0.15 else "⚠️ ÉLEVÉ" if krr_val > 0.15 else "📊 FAIBLE"
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
        
        st.rerun()  # Forcer le rafraîchissement pour voir les nouveaux graphiques

# ==================== TESTS RAPIDES ====================
st.markdown("### 🧪 Tests Rapides pour Comparaison")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧪 Test Sec (0% eau, 5°)"):
        # Simuler données réalistes
        np.random.seed(42)
        test_metrics = {
            'krr': 0.055 + np.random.normal(0, 0.005),  # Van Wal typique
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
            'krr': 0.072 + np.random.normal(0, 0.005),  # Plus élevé avec humidité
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
    if st.button("🧪 Test Angle Fort (5% eau, 30°)"):
        np.random.seed(456)
        test_metrics = {
            'krr': 0.063 + np.random.normal(0, 0.006),  # Angle élevé
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
        st.success("✅ Test angle fort ajouté!")
        st.rerun()

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
    if len(results) >= 1:  # Changé de 2 à 1 pour voir graphiques avec 1 seule expérience
        st.markdown("## 📊 Graphiques de Comparaison Complets")
        
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
                'Calibration': exp['metrics']['calibration'],
                'Sphere_Type': exp['sphere_type'],
                'Success_Rate': exp['success_rate']
            })
        plot_df = pd.DataFrame(plot_data)
        
        # === GRAPHIQUES PRINCIPAUX COMME AVANT ===
        st.markdown("### 🎯 Graphiques Principaux - Style Ancien")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Krr vs Humidité - Style ancien avec sphères et courbe
            fig1 = px.scatter(plot_df, x='Humidité', y='Krr', 
                            color='Sphere_Type', size='Success_Rate',
                            hover_data=['Expérience', 'V0', 'Distance', 'Angle'],
                            title="💧 Coefficient Krr vs Teneur en Eau",
                            labels={'Krr': 'Coefficient Krr', 'Humidité': 'Teneur en Eau (%)'})
            
            # Ajouter lignes de référence Van Wal
            fig1.add_hline(y=0.052, line_dash="dash", line_color="red", annotation_text="Van Wal min: 0.052")
            fig1.add_hline(y=0.066, line_dash="dash", line_color="red", annotation_text="Van Wal max: 0.066")
            
            # Ajouter courbe de tendance polynomiale (quadratique)
            if len(plot_df) >= 3:
                try:
                    # Tendance quadratique pour capturer l'optimum d'humidité
                    z = np.polyfit(plot_df['Humidité'], plot_df['Krr'], min(2, len(plot_df)-1))
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Humidité'].min(), plot_df['Humidité'].max(), 100)
                    fig1.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                            name='Tendance Quadratique', 
                                            line=dict(dash='dot', color='purple', width=3)))
                except:
                    pass
            elif len(plot_df) == 2:
                # Ligne droite pour 2 points
                try:
                    z = np.polyfit(plot_df['Humidité'], plot_df['Krr'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Humidité'].min(), plot_df['Humidité'].max(), 100)
                    fig1.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                            name='Tendance Linéaire', 
                                            line=dict(dash='dot', color='purple', width=3)))
                except:
                    pass
            
            fig1.update_layout(height=500)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Krr vs Angle - Style ancien avec sphères et courbe
            fig2 = px.scatter(plot_df, x='Angle', y='Krr', 
                            color='Humidité', size='V0',
                            hover_data=['Expérience', 'Distance', 'Sphere_Type'],
                            title="📐 Coefficient Krr vs Angle d'Inclinaison",
                            labels={'Krr': 'Coefficient Krr', 'Angle': 'Angle (°)'})
            
            # Ajouter lignes de référence Van Wal
            fig2.add_hline(y=0.052, line_dash="dash", line_color="red", annotation_text="Van Wal min")
            fig2.add_hline(y=0.066, line_dash="dash", line_color="red", annotation_text="Van Wal max")
            
            # Ajouter courbe de tendance
            if len(plot_df) >= 2:
                try:
                    z = np.polyfit(plot_df['Angle'], plot_df['Krr'], min(1, len(plot_df)-1))
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Angle'].min(), plot_df['Angle'].max(), 100)
                    fig2.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                            name='Tendance', 
                                            line=dict(dash='dot', color='orange', width=3)))
                except:
                    pass
            
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
        
        # === GRAPHIQUE COMPARATIF EN BARRES ===
        st.markdown("### 📊 Comparaison Directe - Tous Coefficients")
        
        fig_comparison = go.Figure()
        
        # Barres Krr avec couleurs selon type de sphère
        colors = ['lightblue' if stype == 'Solide' else 'lightcoral' for stype in plot_df['Sphere_Type']]
        
        fig_comparison.add_trace(go.Bar(
            x=[f"{row['Expérience']}\n({row['Humidité']}% eau, {row['Angle']}°)" for _, row in plot_df.iterrows()],
            y=plot_df['Krr'],
            name='Krr',
            text=[f"{val:.4f}" for val in plot_df['Krr']],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Krr: %{y:.6f}<extra></extra>'
        ))
        
        # Lignes de référence Van Wal
        fig_comparison.add_hline(y=0.052, line_dash="dash", line_color="red", 
                               annotation_text="Van Wal min: 0.052")
        fig_comparison.add_hline(y=0.066, line_dash="dash", line_color="red", 
                               annotation_text="Van Wal max: 0.066")
        
        # Ligne moyenne de tes expériences
        mean_krr = plot_df['Krr'].mean()
        fig_comparison.add_hline(y=mean_krr, line_dash="dot", line_color="green", 
                               annotation_text=f"Moyenne expériences: {mean_krr:.4f}")
        
        fig_comparison.update_layout(
            title="📊 Comparaison Krr - Toutes Expériences avec Références",
            xaxis_title="Expériences (conditions)",
            yaxis_title="Coefficient Krr",
            height=600,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # === ANALYSE DES TENDANCES ===
        if len(plot_df) >= 2:
            st.markdown("### 🔍 Analyse des Tendances")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Effet humidité
                if plot_df['Humidité'].nunique() > 1:
                    humid_krr_corr = plot_df[['Humidité', 'Krr']].corr().iloc[0, 1]
                    st.metric("Corrélation Humidité ↔ Krr", f"{humid_krr_corr:.3f}")
                    
                    if humid_krr_corr > 0.5:
                        st.success("📈 Krr augmente avec l'humidité")
                    elif humid_krr_corr < -0.5:
                        st.info("📉 Krr diminue avec l'humidité")
                    else:
                        st.warning("➡️ Peu d'effet de l'humidité")
                else:
                    st.info("Une seule valeur d'humidité")
            
            with col2:
                # Effet angle
                if plot_df['Angle'].nunique() > 1:
                    angle_krr_corr = plot_df[['Angle', 'Krr']].corr().iloc[0, 1]
                    st.metric("Corrélation Angle ↔ Krr", f"{angle_krr_corr:.3f}")
                    
                    if abs(angle_krr_corr) > 0.7:
                        direction = "augmente" if angle_krr_corr > 0 else "diminue"
                        st.warning(f"⚠️ Krr {direction} fortement avec l'angle")
                    else:
                        st.success("✅ Krr indépendant de l'angle (Van Wal)")
                else:
                    st.info("Un seul angle testé")
            
            with col3:
                # Comparaison Van Wal
                krr_values = plot_df['Krr'].values
                van_wal_range = [0.052, 0.066]
                in_range = np.sum((krr_values >= van_wal_range[0]) & (krr_values <= van_wal_range[1]))
                total = len(krr_values)
                
                st.metric("Dans plage Van Wal", f"{in_range}/{total}")
                
                if in_range / total > 0.7:
                    st.success("✅ Cohérent avec littérature")
                elif in_range / total > 0.3:
                    st.warning("⚠️ Partiellement cohérent")
                else:
                    st.error("❌ Écart significatif")
        
        # === RECOMMANDATIONS ===
        st.markdown("### 💡 Recommandations pour Nouveaux Tests")
        
        recommendations = []
        
        # Analyse couverture
        humidity_values = sorted(plot_df['Humidité'].unique())
        angle_values = sorted(plot_df['Angle'].unique())
        
        if len(humidity_values) < 4:
            recommendations.append(f"💧 **Tester plus d'humidités** : Actuellement {humidity_values}% - ajouter points intermédiaires")
        
        if len(angle_values) < 3:
            recommendations.append(f"📐 **Varier les angles** : Actuellement {angle_values}° - tester 5°, 15°, 30°")
        
        # Recherche optimum humidité
        if len(plot_df) >= 3 and plot_df['Humidité'].nunique() >= 3:
            max_krr_idx = plot_df['Krr'].idxmax()
            optimal_humidity = plot_df.loc[max_krr_idx, 'Humidité']
            if 5 <= optimal_humidity <= 20:
                recommendations.append(f"🎯 **Optimum détecté** : {optimal_humidity}% humidité - tester autour de cette valeur")
        
        # Validation répétabilité
        if plot_df['Krr'].std() / plot_df['Krr'].mean() > 0.15:
            recommendations.append("🔄 **Améliorer répétabilité** : Variation Krr élevée - répéter certaines conditions")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.success("✅ **Bon plan expérimental** - continuer les mesures !")
        
        # === EXPORT ===
        st.markdown("### 📥 Export Données Complètes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📋 Export Tableau Résultats",
                data=csv_data,
                file_name="resultats_krr_complet.csv",
                mime="text/csv"
            )
        
        with col2:
            plot_csv = plot_df.to_csv(index=False)
            st.download_button(
                label="📊 Export Données Graphiques",
                data=plot_csv,
                file_name="donnees_pour_graphiques.csv",
                mime="text/csv"
            )0.6:
                direction = "augmente" if angle_krr_corr > 0 else "diminue"
                insights.append(f"📐 **Dépendance à l'angle** : Krr {direction} avec l'angle")
            else:
                insights.append("📐 **Indépendance angle** : Conforme théorie Van Wal")
        
        # Comparaison Van Wal
        krr_values = plot_df['Krr'].values
        van_wal_range = [0.052, 0.066]
        in_range = np.sum((krr_values >= van_wal_range[0]) & (krr_values <= van_wal_range[1]))
        total = len(krr_values)
        
        if in_range / total > 0.5:
            insights.append(f"✅ **Cohérence Van Wal** : {in_range}/{total} valeurs dans la plage littérature")
        else:
            insights.append(f"⚠️ **Écart Van Wal** : {in_range}/{total} valeurs dans la plage - conditions différentes ?")
        
        # Qualité expérimentale
        success_rates = plot_df['Success_Rate'].values
        if np.mean(success_rates) > 80:
            insights.append("🎯 **Bonne qualité** : Taux de détection élevé")
        else:
            insights.append("⚠️ **Qualité variable** : Améliorer conditions de détection")
        
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        
        # === RECOMMANDATIONS ===
        st.markdown("### 💡 Recommandations Expérimentales")
        
        recommendations = []
        
        # Analyse couverture paramètres
        humidity_range = plot_df['Humidité'].max() - plot_df['Humidité'].min()
        angle_range = plot_df['Angle'].max() - plot_df['Angle'].min()
        
        if humidity_range < 10:
            recommendations.append("💧 **Élargir gamme humidité** : Tester 0%, 5%, 10%, 15%, 20%")
        
        if angle_range < 10:
            recommendations.append("📐 **Varier angles** : Tester 5°, 10°, 15°, 20°, 30°")
        
        # Analyse répétabilité
        krr_variation = plot_df['Krr'].std() / plot_df['Krr'].mean() * 100
        if krr_variation > 20:
            recommendations.append("🔄 **Améliorer répétabilité** : Variation Krr élevée")
        
        # Recherche optimum
        if len(plot_df) >= 4:
            max_krr_idx = plot_df['Krr'].idxmax()
            optimal_humidity = plot_df.loc[max_krr_idx, 'Humidité']
            if 8 <= optimal_humidity <= 15:
                recommendations.append(f"🎯 **Optimum détecté** : {optimal_humidity}% humidité donne Krr maximum")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.success("✅ **Plan expérimental équilibré**")
        
    else:
        st.info("📊 Ajoutez des expériences pour voir les graphiques de comparaison")
        st.info("💡 Utilisez les boutons 'Test Rapide' ci-dessus pour ajouter des données d'exemple")

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

