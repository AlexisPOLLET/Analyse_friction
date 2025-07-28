# Analyseur de Résultats de Friction
# Sphères sur Substrat Granulaire Humide - Université d'Osaka

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🔬 Analyseur de Résultats de Friction",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .parameter-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .analysis-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .comparison-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des données de session
if 'friction_experiments' not in st.session_state:
    st.session_state.friction_experiments = {}

# En-tête principal
st.markdown("""
<div class="header-section">
    <h1>🔬 Analyseur de Résultats de Friction</h1>
    <h2>Sphères sur Substrat Granulaire Humide</h2>
    <p><em>Département des Sciences de la Terre Cosmique - Université d'Osaka</em></p>
</div>
""", unsafe_allow_html=True)

# Fonctions utilitaires
def load_results_csv(uploaded_file, experiment_name="Expérience", water_content=0.0, sphere_type="Steel"):
    """Charge et traite un fichier CSV de résultats"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Vérifier la structure attendue
        if 'Parametre' not in df.columns or 'Valeur' not in df.columns:
            st.error("❌ Le fichier doit contenir les colonnes 'Parametre' et 'Valeur'")
            return None, None
        
        # Convertir en dictionnaire pour faciliter l'accès
        results_dict = dict(zip(df['Parametre'], df['Valeur']))
        
        # Métadonnées de l'expérience
        metadata = {
            'experiment_name': experiment_name,
            'water_content': water_content,
            'sphere_type': sphere_type,
            'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters_count': len(results_dict)
        }
        
        return results_dict, metadata
    return None, None

def analyze_friction_parameters(results_dict):
    """Analyse les paramètres de friction et génère des insights"""
    insights = []
    
    # Analyse du Krr
    if 'Krr' in results_dict:
        krr = results_dict['Krr']
        if 0.03 <= krr <= 0.10:
            insights.append(f"✅ **Krr = {krr:.6f}** - Cohérent avec Van Wal (2017)")
        elif krr < 0.03:
            insights.append(f"⬇️ **Krr = {krr:.6f}** - Résistance faible (surface lisse)")
        else:
            insights.append(f"⬆️ **Krr = {krr:.6f}** - Résistance élevée (surface rugueuse)")
    
    # Analyse des coefficients de friction
    friction_coeffs = ['μ_Cinétique', 'μ_Roulement', 'μ_Énergétique']
    for coeff in friction_coeffs:
        if coeff in results_dict:
            value = results_dict[coeff]
            if coeff == 'μ_Énergétique' and value < 0:
                insights.append(f"⚠️ **{coeff} = {value:.4f}** - Valeur négative (perte d'énergie)")
            elif value > 1:
                insights.append(f"🔥 **{coeff} = {value:.4f}** - Friction élevée")
            else:
                insights.append(f"📊 **{coeff} = {value:.4f}** - Friction modérée")
    
    # Analyse de l'efficacité énergétique
    if 'Efficacite_Energie_%' in results_dict:
        eff = results_dict['Efficacite_Energie_%']
        if eff == 0:
            insights.append("⚡ **Efficacité = 0%** - Énergie totalement dissipée")
        elif eff < 20:
            insights.append(f"⚡ **Efficacité = {eff:.1f}%** - Forte dissipation énergétique")
        else:
            insights.append(f"⚡ **Efficacité = {eff:.1f}%** - Conservation énergétique modérée")
    
    return insights

def create_parameter_visualization(results_dict, title="Paramètres de Friction"):
    """Crée une visualisation des paramètres"""
    
    # Séparer les paramètres par catégories
    kinematic_params = ['Vitesse_Max_mm/s', 'Distance_mm', 'Duree_s']
    friction_params = ['Krr', 'μ_Cinétique', 'μ_Roulement', 'μ_Énergétique']
    energy_params = ['Efficacite_Energie_%', 'Force_Normale_mN']
    
    # Créer des sous-graphiques
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🏃 Paramètres Cinématiques', '🔧 Coefficients de Friction',
                       '⚡ Paramètres Énergétiques', '📊 Vue d\'Ensemble'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Graphique 1: Paramètres cinématiques
    kinematic_values = [results_dict.get(p, 0) for p in kinematic_params if p in results_dict]
    kinematic_labels = [p.replace('_', ' ') for p in kinematic_params if p in results_dict]
    
    if kinematic_values:
        fig.add_trace(
            go.Bar(x=kinematic_labels, y=kinematic_values, 
                   name='Cinématique', marker_color='lightblue'),
            row=1, col=1
        )
    
    # Graphique 2: Coefficients de friction
    friction_values = [results_dict.get(p, 0) for p in friction_params if p in results_dict]
    friction_labels = [p.replace('_', ' ') for p in friction_params if p in results_dict]
    
    if friction_values:
        colors = ['red' if v < 0 else 'green' for v in friction_values]
        fig.add_trace(
            go.Bar(x=friction_labels, y=friction_values,
                   name='Friction', marker_color=colors),
            row=1, col=2
        )
    
    # Graphique 3: Paramètres énergétiques
    energy_values = [results_dict.get(p, 0) for p in energy_params if p in results_dict]
    energy_labels = [p.replace('_', ' ').replace('%', '') for p in energy_params if p in results_dict]
    
    if energy_values:
        fig.add_trace(
            go.Bar(x=energy_labels, y=energy_values,
                   name='Énergie', marker_color='orange'),
            row=2, col=1
        )
    
    # Graphique 4: Vue d'ensemble (gauge pour Krr)
    if 'Krr' in results_dict:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=results_dict['Krr'],
                delta={'reference': 0.06, 'position': "top"},
                gauge={'axis': {'range': [None, 0.15]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.03], 'color': "lightgray"},
                                {'range': [0.03, 0.10], 'color': "lightgreen"},
                                {'range': [0.10, 0.15], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.06}},
                title={'text': "Krr Coefficient"}),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text=title)
    return fig

# Interface principale
st.sidebar.markdown("### 📋 Navigation")
mode = st.sidebar.radio("Sélectionnez le mode:", [
    "📊 Analyse Individuelle", 
    "🔍 Comparaison Multi-Expériences",
    "📈 Analyse de Tendances"
])

# Mode 1: Analyse Individuelle
if mode == "📊 Analyse Individuelle":
    
    st.markdown("## 📊 Analyse des Résultats Individuels")
    
    # Section de téléchargement
    st.markdown("### 📁 Chargement des Données")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        experiment_name = st.text_input("Nom de l'expérience", value="Exp_Friction_1")
    with col2:
        water_content = st.number_input("Taux d'humidité (%)", value=0.0, min_value=0.0, max_value=30.0)
    with col3:
        sphere_type = st.selectbox("Type de sphère", ["Steel", "Plastic", "Glass"])
    
    # Upload du fichier
    uploaded_file = st.file_uploader(
        "Téléchargez votre fichier CSV de résultats",
        type=['csv'],
        help="Format attendu: colonnes 'Parametre' et 'Valeur'"
    )
    
    # Traitement des données
    if uploaded_file is not None:
        results_dict, metadata = load_results_csv(uploaded_file, experiment_name, water_content, sphere_type)
        
        if results_dict is not None:
            st.success(f"✅ Fichier chargé avec succès! {len(results_dict)} paramètres détectés")
            
            # Affichage des métriques principales
            st.markdown("### 🎯 Métriques Principales")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'Krr' in results_dict:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{results_dict['Krr']:.6f}</h3>
                        <p>Coefficient Krr</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if 'Vitesse_Max_mm/s' in results_dict:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{results_dict['Vitesse_Max_mm/s']:.1f}</h3>
                        <p>Vitesse Max (mm/s)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if 'Distance_mm' in results_dict:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{results_dict['Distance_mm']:.1f}</h3>
                        <p>Distance (mm)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                if 'μ_Roulement' in results_dict:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{results_dict['μ_Roulement']:.4f}</h3>
                        <p>μ Roulement</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualisation des paramètres
            st.markdown("### 📈 Visualisation des Paramètres")
            fig = create_parameter_visualization(results_dict, f"Analyse de {experiment_name}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé des paramètres
            st.markdown("### 📋 Paramètres Détaillés")
            
            # Créer un DataFrame pour l'affichage
            display_data = []
            for param, value in results_dict.items():
                # Déterminer l'unité et la catégorie
                if 'mm/s' in param:
                    unit = "mm/s"
                    category = "Cinématique"
                elif 'mm' in param:
                    unit = "mm"
                    category = "Géométrie"
                elif 'mN' in param:
                    unit = "mN"
                    category = "Force"
                elif '%' in param:
                    unit = "%"
                    category = "Énergie"
                elif 'μ' in param:
                    unit = "-"
                    category = "Friction"
                elif param == 'Krr':
                    unit = "-"
                    category = "Résistance"
                elif 's' in param:
                    unit = "s"
                    category = "Temps"
                else:
                    unit = "-"
                    category = "Autre"
                
                display_data.append({
                    'Paramètre': param.replace('_', ' '),
                    'Valeur': f"{value:.6f}" if abs(value) < 1 else f"{value:.2f}",
                    'Unité': unit,
                    'Catégorie': category
                })
            
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True)
            
            # Analyse physique
            st.markdown("### 🧠 Analyse Physique")
            insights = analyze_friction_parameters(results_dict)
            
            for insight in insights:
                st.markdown(f"""
                <div class="parameter-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
            
            # Comparaison avec la littérature
            st.markdown("### 📚 Comparaison avec la Littérature")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="comparison-section">
                    <h4>📖 Références Littérature</h4>
                    <ul>
                        <li><strong>Van Wal (2017):</strong> Krr = 0.05-0.07 (sols secs)</li>
                        <li><strong>Darbois Texier (2018):</strong> δ/R ∝ (ρs/ρg)^0.75</li>
                        <li><strong>De Blasio (2009):</strong> Krr indépendant de la vitesse</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if 'Krr' in results_dict:
                    krr_value = results_dict['Krr']
                    if 0.05 <= krr_value <= 0.07:
                        status = "✅ Conforme"
                        color = "green"
                    elif krr_value < 0.05:
                        status = "⬇️ Inférieur"
                        color = "blue"
                    else:
                        status = "⬆️ Supérieur"
                        color = "orange"
                    
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>🎯 Évaluation de Krr</h4>
                        <p><strong>Votre valeur:</strong> {krr_value:.6f}</p>
                        <p><strong>Status:</strong> <span style="color: {color};">{status}</span></p>
                        <p><strong>Effet humidité:</strong> {water_content}% d'eau</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Sauvegarde pour comparaison
            st.markdown("### 💾 Sauvegarde")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Sauvegarder pour comparaison"):
                    st.session_state.friction_experiments[experiment_name] = {
                        'results': results_dict,
                        'metadata': metadata
                    }
                    st.success(f"✅ Expérience '{experiment_name}' sauvegardée!")
            
            with col2:
                # Export enrichi
                export_data = pd.DataFrame([{
                    'Experiment': experiment_name,
                    'Water_Content_%': water_content,
                    'Sphere_Type': sphere_type,
                    **results_dict
                }])
                
                csv_export = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger résultats enrichis",
                    data=csv_export,
                    file_name=f"resultats_enrichis_{experiment_name}.csv",
                    mime="text/csv"
                )

# Mode 2: Comparaison Multi-Expériences
elif mode == "🔍 Comparaison Multi-Expériences":
    
    st.markdown("## 🔍 Comparaison Multi-Expériences")
    
    if not st.session_state.friction_experiments:
        st.warning("⚠️ Aucune expérience sauvegardée. Utilisez d'abord le mode 'Analyse Individuelle'.")
        
        # Possibilité de charger des données de démonstration
        if st.button("🚀 Charger des données de démonstration"):
            # Simuler quelques expériences
            demo_experiments = {
                'Exp_W0%': {
                    'results': {'Krr': 0.054, 'Vitesse_Max_mm/s': 2669, 'μ_Roulement': 1.23, 'Efficacite_Energie_%': 0},
                    'metadata': {'water_content': 0, 'sphere_type': 'Steel'}
                },
                'Exp_W5%': {
                    'results': {'Krr': 0.061, 'Vitesse_Max_mm/s': 2450, 'μ_Roulement': 1.38, 'Efficacite_Energie_%': 2},
                    'metadata': {'water_content': 5, 'sphere_type': 'Steel'}
                },
                'Exp_W10%': {
                    'results': {'Krr': 0.068, 'Vitesse_Max_mm/s': 2280, 'μ_Roulement': 1.52, 'Efficacite_Energie_%': 5},
                    'metadata': {'water_content': 10, 'sphere_type': 'Steel'}
                }
            }
            st.session_state.friction_experiments.update(demo_experiments)
            st.success("✅ Données de démonstration chargées!")
            st.rerun()
    
    else:
        # Sélection des expériences à comparer
        st.markdown("### 🎯 Sélection des Expériences")
        
        selected_experiments = st.multiselect(
            "Choisissez les expériences à comparer:",
            options=list(st.session_state.friction_experiments.keys()),
            default=list(st.session_state.friction_experiments.keys())
        )
        
        if len(selected_experiments) >= 2:
            # Créer le tableau de comparaison
            comparison_data = []
            
            for exp_name in selected_experiments:
                exp = st.session_state.friction_experiments[exp_name]
                row = {
                    'Expérience': exp_name,
                    'Humidité (%)': exp['metadata']['water_content'],
                    'Type Sphère': exp['metadata']['sphere_type']
                }
                row.update(exp['results'])
                comparison_data.append(row)
            
            comp_df = pd.DataFrame(comparison_data)
            
            # Graphiques de comparaison
            st.markdown("### 📊 Analyses Comparatives")
            
            # Graphique 1: Krr vs Humidité
            if 'Krr' in comp_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_krr = px.scatter(
                        comp_df, 
                        x='Humidité (%)', 
                        y='Krr',
                        color='Type Sphère',
                        size='Vitesse_Max_mm/s' if 'Vitesse_Max_mm/s' in comp_df.columns else None,
                        title="🎯 Coefficient Krr vs Taux d'Humidité",
                        hover_data=['Expérience']
                    )
                    fig_krr.add_hline(y=0.06, line_dash="dash", line_color="red", 
                                     annotation_text="Référence Van Wal")
                    st.plotly_chart(fig_krr, use_container_width=True)
                
                with col2:
                    # Graphique des coefficients de friction
                    friction_cols = [col for col in comp_df.columns if 'μ_' in col]
                    if friction_cols:
                        fig_friction = go.Figure()
                        
                        for col in friction_cols:
                            fig_friction.add_trace(go.Scatter(
                                x=comp_df['Humidité (%)'],
                                y=comp_df[col],
                                mode='lines+markers',
                                name=col.replace('μ_', 'μ ')
                            ))
                        
                        fig_friction.update_layout(
                            title="🔧 Évolution des Coefficients de Friction",
                            xaxis_title="Humidité (%)",
                            yaxis_title="Coefficient de Friction"
                        )
                        st.plotly_chart(fig_friction, use_container_width=True)
            
            # Graphique 2: Vitesse vs Humidité
            if 'Vitesse_Max_mm/s' in comp_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_vitesse = px.line(
                        comp_df,
                        x='Humidité (%)',
                        y='Vitesse_Max_mm/s',
                        markers=True,
                        title="🏃 Vitesse Maximale vs Humidité"
                    )
                    st.plotly_chart(fig_vitesse, use_container_width=True)
                
                with col2:
                    # Efficacité énergétique
                    if 'Efficacite_Energie_%' in comp_df.columns:
                        fig_efficiency = px.bar(
                            comp_df,
                            x='Expérience',
                            y='Efficacite_Energie_%',
                            color='Humidité (%)',
                            title="⚡ Efficacité Énergétique par Expérience"
                        )
                        st.plotly_chart(fig_efficiency, use_container_width=True)
            
            # Tableau de comparaison détaillé
            st.markdown("### 📋 Tableau de Comparaison Détaillé")
            st.dataframe(comp_df, use_container_width=True)
            
            # Analyse de corrélation
            st.markdown("### 🔗 Analyse de Corrélation")
            
            numeric_cols = comp_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = comp_df[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="📊 Matrice de Corrélation des Paramètres",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Insights automatiques
            st.markdown("### 🧠 Insights Automatiques")
            
            if 'Krr' in comp_df.columns and 'Humidité (%)' in comp_df.columns:
                krr_humidity_corr = comp_df['Krr'].corr(comp_df['Humidité (%)'])
                
                if krr_humidity_corr > 0.7:
                    st.success(f"✅ **Forte corrélation positive** entre Krr et humidité (r = {krr_humidity_corr:.3f})")
                elif krr_humidity_corr < -0.7:
                    st.info(f"📉 **Forte corrélation négative** entre Krr et humidité (r = {krr_humidity_corr:.3f})")
                else:
                    st.warning(f"⚠️ **Corrélation faible** entre Krr et humidité (r = {krr_humidity_corr:.3f})")

# Mode 3: Analyse de Tendances
elif mode == "📈 Analyse de Tendances":
    
    st.markdown("## 📈 Analyse de Tendances et Modélisation")
    
    if not st.session_state.friction_experiments:
        st.warning("⚠️ Aucune expérience disponible pour l'analyse de tendances.")
    else:
        # Compilation des données
        all_data = []
        for exp_name, exp in st.session_state.friction_experiments.items():
            row = {
                'Expérience': exp_name,
                'Humidité': exp['metadata']['water_content'],
                'Type': exp['metadata']['sphere_type']
            }
            row.update(exp['results'])
            all_data.append(row)
        
        trend_df = pd.DataFrame(all_data)
        
        st.markdown("### 🎯 Modélisation des Relations")
        
        # Sélection des paramètres pour la modélisation
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox("Paramètre X (indépendant):", 
                                 ['Humidité'] + [col for col in trend_df.columns if col not in ['Expérience', 'Type']])
        
        with col2:
            y_param = st.selectbox("Paramètre Y (dépendant):", 
                                 [col for col in trend_df.columns if col not in ['Expérience', 'Type', x_param]])
        
        if len(trend_df) >= 3:
            # Ajustement polynomial
            st.markdown("#### 📊 Aj
