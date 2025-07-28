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
            st.markdown("#### 📊 Ajustement de Courbe et Prédiction")
            
            # Nettoyage des données pour la régression
            clean_data = trend_df[[x_param, y_param]].dropna()
            
            if len(clean_data) >= 3:
                x_data = clean_data[x_param].values
                y_data = clean_data[y_param].values
                
                # Choix du type d'ajustement
                fit_type = st.selectbox("Type d'ajustement:", [
                    "Linéaire", 
                    "Polynomial (degré 2)", 
                    "Polynomial (degré 3)",
                    "Exponentiel",
                    "Logarithmique"
                ])
                
                # Calcul de l'ajustement
                x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                
                if fit_type == "Linéaire":
                    coeffs = np.polyfit(x_data, y_data, 1)
                    y_fit = np.polyval(coeffs, x_fit)
                    equation = f"y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}"
                    
                elif "Polynomial" in fit_type:
                    degree = int(fit_type.split("degré ")[1].split(")")[0])
                    coeffs = np.polyfit(x_data, y_data, degree)
                    y_fit = np.polyval(coeffs, x_fit)
                    
                    if degree == 2:
                        equation = f"y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}"
                    else:
                        equation = f"y = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.4f}"
                
                elif fit_type == "Exponentiel":
                    # Ajustement exponentiel: y = a * exp(b * x)
                    log_y = np.log(np.maximum(y_data, 1e-10))
                    coeffs = np.polyfit(x_data, log_y, 1)
                    a = np.exp(coeffs[1])
                    b = coeffs[0]
                    y_fit = a * np.exp(b * x_fit)
                    equation = f"y = {a:.4f} * exp({b:.4f} * x)"
                
                elif fit_type == "Logarithmique":
                    # Ajustement logarithmique: y = a * ln(x) + b
                    log_x = np.log(np.maximum(x_data, 1e-10))
                    coeffs = np.polyfit(log_x, y_data, 1)
                    log_x_fit = np.log(np.maximum(x_fit, 1e-10))
                    y_fit = coeffs[0] * log_x_fit + coeffs[1]
                    equation = f"y = {coeffs[0]:.4f} * ln(x) + {coeffs[1]:.4f}"
                
                # Calcul du R²
                y_pred_data = np.interp(x_data, x_fit, y_fit)
                ss_res = np.sum((y_data - y_pred_data) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Visualisation
                fig_trend = go.Figure()
                
                # Points expérimentaux
                fig_trend.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    name='Données expérimentales',
                    marker=dict(size=10, color='red')
                ))
                
                # Courbe d'ajustement
                fig_trend.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name=f'Ajustement {fit_type}',
                    line=dict(color='blue', width=3)
                ))
                
                fig_trend.update_layout(
                    title=f"📈 Relation {y_param} vs {x_param}",
                    xaxis_title=x_param,
                    yaxis_title=y_param,
                    annotations=[
                        dict(
                            x=0.02, y=0.98,
                            xref="paper", yref="paper",
                            text=f"<b>Équation:</b> {equation}<br><b>R² = {r_squared:.4f}</b>",
                            showarrow=False,
                            align="left",
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1
                        )
                    ]
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Évaluation de la qualité de l'ajustement
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{r_squared:.4f}</h3>
                        <p>Coefficient R²</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    rmse = np.sqrt(np.mean((y_data - y_pred_data) ** 2))
                    st.markdown(f"""
                    <div class="metric-card">  
                        <h3>{rmse:.4f}</h3>
                        <p>RMSE</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    mae = np.mean(np.abs(y_data - y_pred_data))
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{mae:.4f}</h3>
                        <p>MAE</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prédictions
                st.markdown("#### 🔮 Prédictions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    prediction_value = st.number_input(
                        f"Valeur de {x_param} pour prédiction:",
                        value=float(x_data.mean()),
                        min_value=float(x_data.min() * 0.5),
                        max_value=float(x_data.max() * 1.5)
                    )
                
                with col2:
                    # Calcul de la prédiction
                    if fit_type == "Linéaire":
                        pred_y = np.polyval(coeffs, prediction_value)
                    elif "Polynomial" in fit_type:
                        pred_y = np.polyval(coeffs, prediction_value)
                    elif fit_type == "Exponentiel":
                        pred_y = a * np.exp(b * prediction_value)
                    elif fit_type == "Logarithmique":
                        pred_y = coeffs[0] * np.log(max(prediction_value, 1e-10)) + coeffs[1]
                    
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>🎯 Prédiction</h4>
                        <p><strong>{x_param} = {prediction_value}</strong></p>
                        <p><strong>{y_param} = {pred_y:.4f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Analyse des relations théoriques
        st.markdown("### 🧠 Relations Théoriques Attendues")
        
        theoretical_relations = {
            "Krr vs Humidité": {
                "forme": "Krr = K_sec(1 + αw + βw²)",
                "physique": "Effet linéaire et quadratique de l'humidité",
                "littérature": "Van Wal (2017) - sols secs uniquement"
            },
            "μ_Roulement vs Humidité": {
                "forme": "μ = μ_sec + γw·exp(-δw)",
                "physique": "Augmentation initiale puis saturation",
                "littérature": "Ponts capillaires - cohésion maximale ~10-15%"
            },
            "δ/R vs ρs/ρg": {
                "forme": "δ/R = A(ρs/ρg)^n",
                "physique": "Pénétration proportionnelle au ratio de densités",
                "littérature": "Darbois Texier (2018) - n ≈ 0.75"
            }
        }
        
        for relation, info in theoretical_relations.items():
            st.markdown(f"""
            <div class="comparison-section">
                <h4>📐 {relation}</h4>
                <p><strong>Forme théorique:</strong> {info['forme']}</p>
                <p><strong>Physique:</strong> {info['physique']}</p>
                <p><strong>Référence:</strong> {info['littérature']}</p>
            </div>
            """, unsafe_allow_html=True)

# Section d'export global
st.markdown("---")
st.markdown("### 💾 Export et Sauvegarde Globale")

if st.session_state.friction_experiments:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export de toutes les expériences
        all_experiments = []
        for exp_name, exp in st.session_state.friction_experiments.items():
            row = {
                'Expérience': exp_name,
                'Date': exp['metadata']['date'],
                'Humidité (%)': exp['metadata']['water_content'],
                'Type Sphère': exp['metadata']['sphere_type']
            }
            row.update(exp['results'])
            all_experiments.append(row)
        
        if all_experiments:
            export_df = pd.DataFrame(all_experiments)
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Export complet (CSV)",
                data=csv_data,
                file_name="analyse_friction_complete.csv",
                mime="text/csv"
            )
    
    with col2:
        # Nettoyage des données
        if st.button("🧹 Nettoyer toutes les données"):
            st.session_state.friction_experiments = {}
            st.success("✅ Toutes les données ont été supprimées!")
            st.rerun()
    
    with col3:
        # Statistiques de la session
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(st.session_state.friction_experiments)}</h3>
            <p>Expériences sauvegardées</p>
        </div>
        """, unsafe_allow_html=True)

# Guide d'utilisation
st.markdown("---")
st.markdown("## 📚 Guide d'Utilisation Rapide")

with st.expander("🎯 Format des fichiers CSV"):
    st.markdown("""
    **Structure attendue:**
    ```csv
    Parametre,Valeur
    Krr,0.05375906313075665
    Vitesse_Max_mm/s,2669.2493426183214
    Distance_mm,369.02196089885706
    μ_Cinétique,1.3163109285264338
    μ_Roulement,1.2288222650005098
    Efficacite_Energie_%,0.0
    Force_Normale_mN,48.863349941400124
    ```
    
    **Paramètres reconnus automatiquement:**
    - `Krr`: Coefficient de résistance au roulement
    - `Vitesse_Max_mm/s`: Vitesse maximale en mm/s
    - `Distance_mm`: Distance parcourue en mm
    - `Duree_s`: Durée de l'expérience en secondes
    - `μ_*`: Coefficients de friction (Cinétique, Roulement, Énergétique)
    - `Efficacite_Energie_%`: Efficacité énergétique en %
    - `Force_Normale_mN`: Force normale en mN
    """)

with st.expander("🔬 Interprétation des résultats"):
    st.markdown("""
    **Coefficient Krr:**
    - 0.03-0.05: Résistance faible (surface lisse)
    - 0.05-0.07: Valeurs typiques littérature (Van Wal 2017)
    - 0.07-0.10: Résistance élevée (effet humidité)
    - >0.10: Résistance très élevée (substrat déformable)
    
    **Coefficients de friction:**
    - μ_Cinétique: Friction pendant le mouvement
    - μ_Roulement: Résistance spécifique au roulement
    - μ_Énergétique: Bilan énergétique (peut être négatif)
    
    **Efficacité énergétique:**
    - 0%: Énergie totalement dissipée
    - 1-20%: Dissipation importante (substrat mou)
    - >20%: Conservation partielle d'énergie
    """)

with st.expander("📊 Analyse comparative"):
    st.markdown("""
    **Tendances attendues avec l'humidité:**
    1. **Krr augmente** avec l'humidité (cohésion capillaire)
    2. **Optimum vers 10-15%** (ponts capillaires maximaux)
    3. **Saturation à haute humidité** (25%+)
    4. **Vitesse diminue** avec l'augmentation de Krr
    5. **Efficacité énergétique diminue** (plus de dissipation)
    
    **Relations théoriques à vérifier:**
    - Krr = K_sec(1 + αw + βw²)
    - δ/R = A(ρs/ρg)^0.75
    - φ_eff = φ_sec + γw·exp(-δw)
    """)

# Footer avec informations du projet
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
    <h4>🔬 Projet de Recherche</h4>
    <p><strong>"Rolling Resistance of Spheres on Wet Granular Material"</strong></p>
    <p>Département des Sciences de la Terre Cosmique - Université d'Osaka</p>
    <p><em>Innovation: Première étude de l'effet de l'humidité sur la résistance au roulement</em></p>
    <hr>
    <p>📊 <strong>Objectif:</strong> Quantifier l'effet de l'humidité sur la friction de roulement</p>
    <p>🎯 <strong>Innovation:</strong> Extension des modèles existants (sols secs) aux conditions humides</p>
    <p>📈 <strong>Applications:</strong> Géotechnique, mécanique des sols, ingénierie</p>
</div>
""", unsafe_allow_html=True)
