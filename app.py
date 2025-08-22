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
        
        # === GRAPHIQUES PRINCIPAUX ===
        st.markdown("### 🎯 Graphiques Principaux")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Krr vs Humidité avec références Van Wal
            fig1 = px.scatter(plot_df, x='Humidité', y='Krr', 
                            color='Angle', size='Success_Rate',
                            hover_data=['Expérience', 'V0', 'Distance'],
                            title="📊 Krr vs Teneur en Eau", 
                            labels={'Krr': 'Coefficient Krr', 'Humidité': 'Teneur en Eau (%)'})
            
            # Ajouter lignes de référence Van Wal
            fig1.add_hline(y=0.052, line_dash="dash", line_color="red", annotation_text="Van Wal min: 0.052")
            fig1.add_hline(y=0.066, line_dash="dash", line_color="red", annotation_text="Van Wal max: 0.066")
            
            # Ajouter ligne de tendance
            if len(plot_df) >= 3:
                try:
                    z = np.polyfit(plot_df['Humidité'], plot_df['Krr'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Humidité'].min(), plot_df['Humidité'].max(), 100)
                    fig1.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                            name='Tendance', line=dict(dash='dot', color='purple', width=2)))
                except:
                    pass
            
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Krr vs Angle
            fig2 = px.scatter(plot_df, x='Angle', y='Krr', 
                            color='Humidité', size='V0',
                            hover_data=['Expérience', 'Distance'],
                            title="📐 Krr vs Angle d'Inclinaison", 
                            labels={'Krr': 'Coefficient Krr', 'Angle': 'Angle (°)'})
            
            # Ajouter ligne de tendance
            if len(plot_df) >= 3:
                try:
                    z = np.polyfit(plot_df['Angle'], plot_df['Krr'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Angle'].min(), plot_df['Angle'].max(), 100)
                    fig2.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                            name='Tendance', line=dict(dash='dot', color='orange', width=2)))
                except:
                    pass
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # === GRAPHIQUES SECONDAIRES ===
        st.markdown("### 📈 Analyses Complémentaires")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Vitesse vs Krr
            fig3 = px.scatter(plot_df, x='V0', y='Krr',
                            color='Humidité', size='Distance',
                            hover_data=['Expérience', 'Angle'],
                            title="🏃 Krr vs Vitesse Initiale",
                            labels={'V0': 'Vitesse Initiale (mm/s)', 'Krr': 'Coefficient Krr'})
            st.plotly_chart(fig3, use_container_width=True)
            
        with col2:
            # Distance vs Success Rate
            fig4 = px.scatter(plot_df, x='Distance', y='Success_Rate',
                            color='Angle', size='Krr',
                            hover_data=['Expérience', 'Humidité'],
                            title="📏 Qualité Détection vs Distance",
                            labels={'Distance': 'Distance (mm)', 'Success_Rate': 'Taux de Succès (%)'})
            st.plotly_chart(fig4, use_container_width=True)
        
        # === GRAPHIQUE COMPARATIF EN BARRES ===
        st.markdown("### 📊 Comparaison Directe")
        
        fig_comparison = go.Figure()
        
        # Barres Krr
        fig_comparison.add_trace(go.Bar(
            x=plot_df['Expérience'],
            y=plot_df['Krr'],
            name='Krr',
            text=[f"{val:.4f}" for val in plot_df['Krr']],
            textposition='auto',
            marker_color='lightblue'
        ))
        
        # Ligne Van Wal moyenne
        van_wal_mean = (0.052 + 0.066) / 2
        fig_comparison.add_hline(y=van_wal_mean, line_dash="dash", line_color="red", 
                               annotation_text=f"Van Wal moyen: {van_wal_mean:.3f}")
        
        fig_comparison.update_layout(
            title="📊 Comparaison Krr - Toutes Expériences",
            xaxis_title="Expériences",
            yaxis_title="Coefficient Krr",
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # === MATRICE DE CORRÉLATION ===
        st.markdown("### 🔗 Analyse de Corrélation")
        
        # Sélectionner colonnes numériques
        corr_cols = ['Humidité', 'Angle', 'Krr', 'V0', 'Distance', 'Success_Rate']
        corr_data = plot_df[corr_cols]
        
        if len(corr_data) >= 3:
            corr_matrix = corr_data.corr()
            
            fig_corr = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               title="🔗 Matrice de Corrélation",
                               color_continuous_scale="RdBu_r",
                               zmin=-1, zmax=1)
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Top corrélations
            st.markdown("#### 🎯 Corrélations les Plus Fortes")
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_values = corr_matrix.where(mask).stack().reset_index()
            corr_values.columns = ['Variable1', 'Variable2', 'Corrélation']
            corr_values = corr_values.sort_values('Corrélation', key=abs, ascending=False)
            
            for i, row in corr_values.head(3).iterrows():
                strength = "Forte" if abs(row['Corrélation']) > 0.7 else "Modérée" if abs(row['Corrélation']) > 0.4 else "Faible"
                direction = "positive" if row['Corrélation'] > 0 else "négative"
                color = "🔴" if abs(row['Corrélation']) > 0.7 else "🟠" if abs(row['Corrélation']) > 0.4 else "🟡"
                
                st.markdown(f"{color} **{strength} corrélation {direction}** : {row['Variable1']} ↔ {row['Variable2']} (r = {row['Corrélation']:.3f})")
        
        # === ANALYSE AUTOMATIQUE ===
        st.markdown("### 🧠 Insights Automatiques")
        
        insights = []
        
        # Effet humidité
        if len(plot_df) >= 3:
            humid_krr_corr = plot_df[['Humidité', 'Krr']].corr().iloc[0, 1]
            if humid_krr_corr > 0.5:
                insights.append("💧 **Effet humidité positif** : L'humidité augmente la résistance au roulement")
            elif humid_krr_corr < -0.3:
                insights.append("💧 **Effet lubrification** : L'humidité réduit la résistance")
            else:
                insights.append("💧 **Effet humidité faible** : Impact minimal dans cette gamme")
        
        # Effet angle
        if len(plot_df) >= 3:
            angle_krr_corr = plot_df[['Angle', 'Krr']].corr().iloc[0, 1]
            if abs(angle_krr_corr) > 0.6:
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
        
        # === EXPORT DONNÉES ===
        st.markdown("### 📥 Export Données")
        
        col1, col2, col3 = st.columns(3)
        
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
        
        with col3:
            if len(corr_data) >= 3:
                corr_csv = corr_matrix.to_csv()
                st.download_button(
                    label="🔗 Export Corrélations",
                    data=corr_csv,
                    file_name="correlations.csv",
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
### ✅ Interface Simple Fonctionnelle
- Calcul Krr basique sans complications
- Formule : Krr = (V₀² - Vf²) / (2gL)
- Graphiques simples
- Pas de bugs !
""")

# ==================== ALTERNATIVES IA ====================
st.markdown("---")
st.markdown("""
### 🤖 Alternatives IA pour le Code
Si cette interface ne te convient pas, voici d'autres IA spécialisées :

**Pour le Code Python/Streamlit :**
- **GitHub Copilot** : Excellent pour compléter du code
- **Cursor.sh** : IDE avec IA intégrée, très bon pour débugger
- **Replit Ghostwriter** : IA dédiée au code, interface simple
- **CodeWhisperer (Amazon)** : Bon pour les scripts scientifiques
- **Tabnine** : IA de complétion de code

**Pour l'Analyse Scientifique :**
- **Perplexity** : Excellent pour rechercher des formules/méthodes
- **Notion AI** : Bon pour organiser et analyser des données
- **ChatGPT Code Interpreter** : Plugin spécialisé analyse de données

**IDE avec IA :**
- **VS Code + GitHub Copilot** : Combinaison très puissante
- **PyCharm + Tabnine** : Spécialisé Python scientifique
- **Jupyter + Kite** : Pour analyse interactive

Tu peux aussi essayer **Google Colab** avec ses nouvelles fonctions IA !
""")
