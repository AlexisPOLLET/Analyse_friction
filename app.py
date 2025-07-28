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

# Fonctions utilitaires avec gestion d'erreurs améliorée
def safe_convert_to_float(value):
    """Conversion sécurisée en float"""
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def load_results_csv(uploaded_file, experiment_name="Expérience", water_content=0.0, sphere_type="Steel"):
    """Charge et traite un fichier CSV de résultats avec gestion d'erreurs"""
    try:
        if uploaded_file is not None:
            # Lecture du fichier avec gestion d'encodage
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            
            # Vérifier la structure attendue
            if 'Parametre' not in df.columns or 'Valeur' not in df.columns:
                st.error("❌ Le fichier doit contenir les colonnes 'Parametre' et 'Valeur'")
                return None, None
            
            # Nettoyer les données
            df = df.dropna()  # Supprimer les lignes vides
            df['Valeur'] = df['Valeur'].apply(safe_convert_to_float)
            
            # Convertir en dictionnaire pour faciliter l'accès
            results_dict = dict(zip(df['Parametre'], df['Valeur']))
            
            # Métadonnées de l'expérience
            metadata = {
                'experiment_name': experiment_name,
                'water_content': safe_convert_to_float(water_content),
                'sphere_type': sphere_type,
                'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'parameters_count': len(results_dict)
            }
            
            return results_dict, metadata
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
        return None, None
    
    return None, None

def analyze_friction_parameters(results_dict):
    """Analyse les paramètres de friction et génère des insights"""
    insights = []
    
    try:
        # Analyse du Krr
        if 'Krr' in results_dict:
            krr = safe_convert_to_float(results_dict['Krr'])
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
                value = safe_convert_to_float(results_dict[coeff])
                if coeff == 'μ_Énergétique' and value < 0:
                    insights.append(f"⚠️ **{coeff} = {value:.4f}** - Valeur négative (perte d'énergie)")
                elif value > 1:
                    insights.append(f"🔥 **{coeff} = {value:.4f}** - Friction élevée")
                else:
                    insights.append(f"📊 **{coeff} = {value:.4f}** - Friction modérée")
        
        # Analyse de l'efficacité énergétique
        if 'Efficacite_Energie_%' in results_dict:
            eff = safe_convert_to_float(results_dict['Efficacite_Energie_%'])
            if eff == 0:
                insights.append("⚡ **Efficacité = 0%** - Énergie totalement dissipée")
            elif eff < 20:
                insights.append(f"⚡ **Efficacité = {eff:.1f}%** - Forte dissipation énergétique")
            else:
                insights.append(f"⚡ **Efficacité = {eff:.1f}%** - Conservation énergétique modérée")
        
    except Exception as e:
        st.error(f"Erreur dans l'analyse des paramètres: {str(e)}")
    
    return insights

def create_parameter_visualization(results_dict, title="Paramètres de Friction"):
    """Crée une visualisation des paramètres avec gestion d'erreurs"""
    
    try:
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
        kinematic_values = [safe_convert_to_float(results_dict.get(p, 0)) for p in kinematic_params if p in results_dict]
        kinematic_labels = [p.replace('_', ' ') for p in kinematic_params if p in results_dict]
        
        if kinematic_values and kinematic_labels:
            fig.add_trace(
                go.Bar(x=kinematic_labels, y=kinematic_values, 
                       name='Cinématique', marker_color='lightblue'),
                row=1, col=1
            )
        
        # Graphique 2: Coefficients de friction
        friction_values = [safe_convert_to_float(results_dict.get(p, 0)) for p in friction_params if p in results_dict]
        friction_labels = [p.replace('_', ' ') for p in friction_params if p in results_dict]
        
        if friction_values and friction_labels:
            colors = ['red' if v < 0 else 'green' for v in friction_values]
            fig.add_trace(
                go.Bar(x=friction_labels, y=friction_values,
                       name='Friction', marker_color=colors),
                row=1, col=2
            )
        
        # Graphique 3: Paramètres énergétiques
        energy_values = [safe_convert_to_float(results_dict.get(p, 0)) for p in energy_params if p in results_dict]
        energy_labels = [p.replace('_', ' ').replace('%', '') for p in energy_params if p in results_dict]
        
        if energy_values and energy_labels:
            fig.add_trace(
                go.Bar(x=energy_labels, y=energy_values,
                       name='Énergie', marker_color='orange'),
                row=2, col=1
            )
        
        # Graphique 4: Vue d'ensemble (gauge pour Krr)
        if 'Krr' in results_dict:
            krr_value = safe_convert_to_float(results_dict['Krr'])
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=krr_value,
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
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la visualisation: {str(e)}")
        # Retourner un graphique vide en cas d'erreur
        fig = go.Figure()
        fig.update_layout(title="Erreur dans la visualisation")
        return fig

def safe_create_dataframe(data_list):
    """Création sécurisée d'un DataFrame"""
    try:
        if not data_list:
            return pd.DataFrame()
        
        # Nettoyer les données avant création du DataFrame
        cleaned_data = []
        for row in data_list:
            cleaned_row = {}
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    cleaned_row[key] = safe_convert_to_float(value)
                else:
                    cleaned_row[key] = str(value) if value is not None else ""
            cleaned_data.append(cleaned_row)
        
        return pd.DataFrame(cleaned_data)
    except Exception as e:
        st.error(f"Erreur lors de la création du DataFrame: {str(e)}")
        return pd.DataFrame()

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
        
        if results_dict is not None and metadata is not None:
            st.success(f"✅ Fichier chargé avec succès! {len(results_dict)} paramètres détectés")
            
            # Affichage des métriques principales
            st.markdown("### 🎯 Métriques Principales")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'Krr' in results_dict:
                    krr_val = safe_convert_to_float(results_dict['Krr'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{krr_val:.6f}</h3>
                        <p>Coefficient Krr</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if 'Vitesse_Max_mm/s' in results_dict:
                    vitesse_val = safe_convert_to_float(results_dict['Vitesse_Max_mm/s'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{vitesse_val:.1f}</h3>
                        <p>Vitesse Max (mm/s)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if 'Distance_mm' in results_dict:
                    distance_val = safe_convert_to_float(results_dict['Distance_mm'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{distance_val:.1f}</h3>
                        <p>Distance (mm)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                if 'μ_Roulement' in results_dict:
                    mu_val = safe_convert_to_float(results_dict['μ_Roulement'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{mu_val:.4f}</h3>
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
                try:
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
                    
                    value_float = safe_convert_to_float(value)
                    formatted_value = f"{value_float:.6f}" if abs(value_float) < 1 else f"{value_float:.2f}"
                    
                    display_data.append({
                        'Paramètre': param.replace('_', ' '),
                        'Valeur': formatted_value,
                        'Unité': unit,
                        'Catégorie': category
                    })
                except Exception as e:
                    st.warning(f"Erreur avec le paramètre {param}: {str(e)}")
            
            if display_data:
                df_display = safe_create_dataframe(display_data)
                if not df_display.empty:
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
            
            # Sauvegarde pour comparaison
            st.markdown("### 💾 Sauvegarde")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Sauvegarder pour comparaison"):
                    try:
                        st.session_state.friction_experiments[experiment_name] = {
                            'results': results_dict,
                            'metadata': metadata
                        }
                        st.success(f"✅ Expérience '{experiment_name}' sauvegardée!")
                    except Exception as e:
                        st.error(f"Erreur lors de la sauvegarde: {str(e)}")
            
            with col2:
                # Export enrichi
                try:
                    export_data = [{
                        'Experiment': experiment_name,
                        'Water_Content_%': water_content,
                        'Sphere_Type': sphere_type,
                        **results_dict
                    }]
                    
                    export_df = safe_create_dataframe(export_data)
                    if not export_df.empty:
                        csv_export = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger résultats enrichis",
                            data=csv_export,
                            file_name=f"resultats_enrichis_{experiment_name}.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Erreur lors de l'export: {str(e)}")

# Mode 2: Comparaison Multi-Expériences
elif mode == "🔍 Comparaison Multi-Expériences":
    
    st.markdown("## 🔍 Comparaison Multi-Expériences")
    
    if not st.session_state.friction_experiments:
        st.warning("⚠️ Aucune expérience sauvegardée. Utilisez d'abord le mode 'Analyse Individuelle'.")
        
        # Possibilité de charger des données de démonstration
        if st.button("🚀 Charger des données de démonstration"):
            try:
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
            except Exception as e:
                st.error(f"Erreur lors du chargement des données de démonstration: {str(e)}")
    
    else:
        try:
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
                    try:
                        exp = st.session_state.friction_experiments[exp_name]
                        row = {
                            'Expérience': exp_name,
                            'Humidité (%)': safe_convert_to_float(exp['metadata']['water_content']),
                            'Type Sphère': exp['metadata']['sphere_type']
                        }
                        
                        # Ajouter les résultats avec conversion sécurisée
                        for key, value in exp['results'].items():
                            row[key] = safe_convert_to_float(value)
                        
                        comparison_data.append(row)
                    except Exception as e:
                        st.warning(f"Erreur avec l'expérience {exp_name}: {str(e)}")
                
                if comparison_data:
                    comp_df = safe_create_dataframe(comparison_data)
                    
                    if not comp_df.empty:
                        # Graphiques de comparaison
                        st.markdown("### 📊 Analyses Comparatives")
                        
                        # Graphique 1: Krr vs Humidité
                        if 'Krr' in comp_df.columns and 'Humidité (%)' in comp_df.columns:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                try:
                                    fig_krr = px.scatter(
                                        comp_df, 
                                        x='Humidité (%)', 
                                        y='Krr',
                                        color='Type Sphère',
                                        title="🎯 Coefficient Krr vs Taux d'Humidité",
                                        hover_data=['Expérience']
                                    )
                                    fig_krr.add_hline(y=0.06, line_dash="dash", line_color="red", 
                                                     annotation_text="Référence Van Wal")
                                    st.plotly_chart(fig_krr, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Erreur dans le graphique Krr: {str(e)}")
                            
                            with col2:
                                # Graphique des coefficients de friction
                                friction_cols = [col for col in comp_df.columns if 'μ_' in col]
                                if friction_cols:
                                    try:
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
                                    except Exception as e:
                                        st.error(f"Erreur dans le graphique friction: {str(e)}")
                        
                        # Tableau de comparaison détaillé
                        st.markdown("### 📋 Tableau de Comparaison Détaillé")
                        st.dataframe(comp_df, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Erreur dans la comparaison multi-expériences: {str(e)}")

# Mode 3: Analyse de Tendances (simplifié pour éviter les erreurs)
elif mode == "📈 Analyse de Tendances":
    
    st.markdown("## 📈 Analyse de Tendances et Modélisation")
    
    if not st.session_state.friction_experiments:
        st.warning("⚠️ Aucune expérience disponible pour l'analyse de tendances.")
    else:
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
        try:
            all_experiments = []
            for exp_name, exp in st.session_state.friction_experiments.items():
                row = {
                    'Expérience': exp_name,
                    'Date': exp['metadata']['date'],
                    'Humidité (%)': safe_convert_to_float(exp['metadata']['water_content']),
                    'Type Sphère': exp['metadata']['sphere_type']
                }
                
                # Ajouter les résultats avec conversion sécurisée
                for key, value in exp['results'].items():
                    row[key] = safe_convert_to_float(value)
                
                all_experiments.append(row)
            
            if all_experiments:
                export_df = safe_create_dataframe(all_experiments)
                if not export_df.empty:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export complet (CSV)",
                        data=csv_data,
                        file_name="analyse_friction_complete.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Erreur lors de l'export: {str(e)}")
    
    with col2:
        # Nettoyage des données
        if st.button("🧹 Nettoyer toutes les données"):
            try:
                st.session_state.friction_experiments = {}
                st.success("✅ Toutes les données ont été supprimées!")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors du nettoyage: {str(e)}")
    
    with col3:
        # Statistiques de la session
        try:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(st.session_state.friction_experiments)}</h3>
                <p>Expériences sauvegardées</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur affichage statistiques: {str(e)}")

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

with st.expander("⚠️ Résolution des problèmes"):
    st.markdown("""
    **Erreurs communes et solutions:**
    
    **Erreur de chargement CSV:**
    - Vérifiez que votre fichier contient les colonnes 'Parametre' et 'Valeur'
    - Assurez-vous que les valeurs numériques sont bien formatées
    - Évitez les caractères spéciaux dans les noms de paramètres
    
    **Erreur de visualisation:**
    - Vérifiez que vos données contiennent des valeurs numériques valides
    - Les valeurs manquantes ou non-numériques seront converties en 0
    
    **Erreur de comparaison:**
    - Assurez-vous d'avoir au moins 2 expériences sauvegardées
    - Vérifiez que les expériences ont des paramètres en commun
    
    **Performance lente:**
    - Limitez le nombre d'expériences comparées simultanément
    - Nettoyez régulièrement les données sauvegardées
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
    <br>
    <p><small>Version robuste avec gestion d'erreurs améliorée</small></p>
</div>
""", unsafe_allow_html=True)
