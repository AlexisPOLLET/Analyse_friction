# Workflow d'Intégration CSV - De l'Analyse à la Comparaison
# Compatible avec tous vos outputs d'analyse

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🔗 CSV Integration Workflow",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
# 🔗 Workflow d'Intégration CSV
## De votre Code d'Analyse → Plateforme de Comparaison
""")

# Sidebar pour le type de fichier
st.sidebar.markdown("### 📁 Type de Fichier CSV")
file_type = st.sidebar.selectbox("Sélectionnez le type de votre fichier:", [
    "🎯 Données de Trajectoire (Frame, X_center, Y_center, Radius)",
    "🧮 Résultats d'Analyse Complète (time_s, velocities, forces, etc.)",
    "📊 Données de Comparaison (Experiment, Water_Content, Krr, etc.)",
    "🔍 Détection Automatique"
])

def detect_csv_type(df):
    """Détection automatique du type de fichier CSV"""
    columns = df.columns.tolist()
    
    # Fichier de trajectoire standard
    if all(col in columns for col in ['Frame', 'X_center', 'Y_center', 'Radius']):
        return "trajectory"
    
    # Fichier d'analyse complète
    elif any('time_s' in col.lower() for col in columns) and any('krr' in col.lower() for col in columns):
        return "complete_analysis"
    
    # Fichier de comparaison
    elif 'Experiment' in columns and 'Water_Content' in columns and 'Krr' in columns:
        return "comparison"
    
    # Fichier avec métriques avancées
    elif any(col in columns for col in ['v_magnitude_ms', 'acceleration_ms2', 'F_resistance_N']):
        return "advanced_metrics"
    
    # Fichier de détection simple
    elif all(col in columns for col in ['X_center', 'Y_center']) and len(columns) <= 6:
        return "simple_detection"
    
    else:
        return "unknown"

def extract_experiment_metadata(filename):
    """Extraction automatique des métadonnées depuis le nom de fichier"""
    metadata = {
        'experiment_name': filename,
        'water_content': 0.0,
        'angle_degrees': 15.0,
        'sphere_type': 'Unknown'
    }
    
    # Patterns courants dans vos noms de fichiers
    import re
    
    # Recherche du pourcentage d'eau: W10%, _10%, etc.
    water_match = re.search(r'[Ww](\d+(?:\.\d+)?)%?', filename)
    if water_match:
        metadata['water_content'] = float(water_match.group(1))
    
    # Recherche de l'angle: A15°, _15deg, etc.
    angle_match = re.search(r'[Aa](\d+(?:\.\d+)?)°?', filename)
    if angle_match:
        metadata['angle_degrees'] = float(angle_match.group(1))
    
    # Recherche du type de sphère
    if 'steel' in filename.lower():
        metadata['sphere_type'] = 'Steel'
    elif 'plastic' in filename.lower():
        metadata['sphere_type'] = 'Plastic'
    elif 'glass' in filename.lower():
        metadata['sphere_type'] = 'Glass'
    
    # Recherche de la taille: 2cm, 3cm, etc.
    size_match = re.search(r'(\d+)cm', filename.lower())
    if size_match:
        metadata['sphere_type'] += f"_{size_match.group(1)}cm"
    
    return metadata

def process_trajectory_csv(df, metadata):
    """Traitement des fichiers de trajectoire standard"""
    st.markdown("### 🎯 Fichier de Trajectoire Détecté")
    
    # Vérification de la structure
    required_cols = ['Frame', 'X_center', 'Y_center', 'Radius']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Colonnes manquantes: {missing_cols}")
        return None
    
    # Statistiques de base
    df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
    success_rate = len(df_valid) / len(df) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Frames Totaux", len(df))
    with col2:
        st.metric("Détections Valides", len(df_valid))
    with col3:
        st.metric("Taux de Succès", f"{success_rate:.1f}%")
    
    # Calcul automatique du Krr
    if len(df_valid) > 10:
        st.markdown("#### 🧮 Calcul Automatique du Krr")
        
        # Paramètres configurables
        col1, col2, col3 = st.columns(3)
        with col1:
            fps = st.number_input("FPS Caméra", value=250.0, min_value=1.0)
        with col2:
            pixels_per_mm = st.number_input("Calibration (px/mm)", value=5.0, min_value=0.1)
        with col3:
            sphere_mass_g = st.number_input("Masse Sphère (g)", value=10.0, min_value=0.1)
        
        if st.button("🚀 Calculer le Krr"):
            # Conversion en unités physiques
            dt = 1 / fps
            x_m = df_valid['X_center'].values / pixels_per_mm / 1000
            y_m = df_valid['Y_center'].values / pixels_per_mm / 1000
            
            # Calcul des vitesses
            vx = np.gradient(x_m, dt)
            vy = np.gradient(y_m, dt)
            v_magnitude = np.sqrt(vx**2 + vy**2)
            
            # Vitesses initiale et finale
            n_avg = min(3, len(v_magnitude)//4)
            v0 = np.mean(v_magnitude[:n_avg]) if n_avg > 0 else v_magnitude[0]
            vf = np.mean(v_magnitude[-n_avg:]) if n_avg > 0 else v_magnitude[-1]
            
            # Distance totale
            distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
            total_distance = np.sum(distances)
            
            # Calcul Krr
            if total_distance > 0 and v0 > vf:
                krr = (v0**2 - vf**2) / (2 * 9.81 * total_distance)
                
                # Affichage des résultats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("V₀ (m/s)", f"{v0:.4f}")
                with col2:
                    st.metric("Vf (m/s)", f"{vf:.4f}")
                with col3:
                    st.metric("Distance (m)", f"{total_distance:.4f}")
                with col4:
                    st.metric("**Krr**", f"{krr:.6f}")
                
                # Mise ร jour des mรฉtadonnรฉes
                metadata['krr_calculated'] = krr
                metadata['v0'] = v0
                metadata['vf'] = vf
                metadata['distance'] = total_distance
                
                # Visualisation rapide
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_valid['X_center'], 
                    y=df_valid['Y_center'],
                    mode='markers+lines',
                    marker=dict(color=df_valid['Frame'], colorscale='viridis'),
                    name='Trajectoire'
                ))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(title="Trajectoire de la Sphère", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    return metadata

def process_complete_analysis_csv(df, metadata):
    """Traitement des fichiers d'analyse complète"""
    st.markdown("### 🧮 Fichier d'Analyse Complète Détecté")
    
    # Identification des colonnes importantes
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    velocity_cols = [col for col in df.columns if 'velocity' in col.lower() or 'v_' in col.lower()]
    force_cols = [col for col in df.columns if 'force' in col.lower() or 'f_' in col.lower()]
    energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'e_' in col.lower()]
    krr_cols = [col for col in df.columns if 'krr' in col.lower()]
    
    st.markdown("#### 📋 Colonnes Identifiées")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Temps:**")
        for col in time_cols:
            st.markdown(f"- {col}")
    
    with col2:
        st.markdown("**Cinématique:**")
        for col in velocity_cols[:5]:  # Limite ร  5 pour l'affichage
            st.markdown(f"- {col}")
    
    with col3:
        st.markdown("**Dynamique:**")
        for col in force_cols + energy_cols:
            st.markdown(f"- {col}")
    
    # Extraction des métriques clés
    if krr_cols:
        krr_col = krr_cols[0]
        avg_krr = df[krr_col].mean()
        std_krr = df[krr_col].std()
        
        st.markdown("#### 🎯 Métriques Clés Extraites")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Krr Moyen", f"{avg_krr:.6f}")
        with col2:
            st.metric("Écart-Type Krr", f"{std_krr:.6f}")
        with col3:
            st.metric("Coefficient de Variation", f"{std_krr/avg_krr*100:.1f}%")
        
        metadata['krr_calculated'] = avg_krr
        metadata['krr_std'] = std_krr
    
    # Visualisations automatiques
    if time_cols and velocity_cols:
        st.markdown("#### 📈 Visualisations Automatiques")
        
        time_col = time_cols[0]
        
        # Graphique de vitesse
        if velocity_cols:
            fig_vel = go.Figure()
            for vel_col in velocity_cols[:3]:  # Max 3 courbes
                fig_vel.add_trace(go.Scatter(
                    x=df[time_col],
                    y=df[vel_col],
                    mode='lines',
                    name=vel_col
                ))
            fig_vel.update_layout(
                title="Évolution des Vitesses",
                xaxis_title="Temps (s)",
                yaxis_title="Vitesse",
                height=400
            )
            st.plotly_chart(fig_vel, use_container_width=True)
    
    return metadata

def process_comparison_csv(df, metadata):
    """Traitement des fichiers de comparaison"""
    st.markdown("### 📊 Fichier de Comparaison Détecté")
    
    # Vérification des colonnes essentielles
    required_cols = ['Experiment', 'Water_Content', 'Krr']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Colonnes manquantes: {missing_cols}")
        return None
    
    # Statistiques du dataset
    n_experiments = len(df)
    n_humidity_levels = df['Water_Content'].nunique()
    krr_range = df['Krr'].max() - df['Krr'].min()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expériences", n_experiments)
    with col2:
        st.metric("Niveaux d'Humidité", n_humidity_levels)
    with col3:
        st.metric("Plage Krr", f"{krr_range:.6f}")
    
    # Aperçu des données
    st.markdown("#### 📋 Aperçu des Données")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Analyse rapide de l'effet de l'humidité
    if 'Water_Content' in df.columns and 'Krr' in df.columns:
        st.markdown("#### 💧 Analyse Rapide - Effet de l'Humidité")
        
        fig_humidity = px.scatter(
            df, 
            x='Water_Content', 
            y='Krr',
            color='Sphere_Type' if 'Sphere_Type' in df.columns else None,
            title="Krr vs Humidité - Vue d'Ensemble"
        )
        st.plotly_chart(fig_humidity, use_container_width=True)
        
        # Statistiques par niveau d'humidité
        humidity_stats = df.groupby('Water_Content')['Krr'].agg(['mean', 'std', 'count'])
        st.markdown("#### 📊 Statistiques par Niveau d'Humidité")
        st.dataframe(humidity_stats, use_container_width=True)
    
    return metadata

# Interface principale
st.markdown("## 📁 Upload et Analyse de votre Fichier CSV")

uploaded_file = st.file_uploader(
    "Choisissez votre fichier CSV d'analyse",
    type=['csv'],
    help="Uploadez n'importe quel fichier CSV généré par vos codes d'analyse"
)

if uploaded_file is not None:
    # Lecture du fichier
    try:
        df = pd.read_csv(uploaded_file)
        filename = uploaded_file.name
        
        st.success(f"✅ Fichier '{filename}' chargé avec succès!")
        st.markdown(f"**Dimensions:** {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        # Extraction automatique des métadonnées
        metadata = extract_experiment_metadata(filename)
        
        st.markdown("### 🏷️ Métadonnées Extraites Automatiquement")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metadata['experiment_name'] = st.text_input("Nom Expérience", metadata['experiment_name'])
        with col2:
            metadata['water_content'] = st.number_input("Humidité (%)", value=metadata['water_content'])
        with col3:
            metadata['angle_degrees'] = st.number_input("Angle (°)", value=metadata['angle_degrees'])
        with col4:
            metadata['sphere_type'] = st.text_input("Type Sphère", metadata['sphere_type'])
        
        # Détection automatique du type
        if "🔍 Détection Automatique" in file_type:
            detected_type = detect_csv_type(df)
            type_names = {
                "trajectory": "🎯 Données de Trajectoire",
                "complete_analysis": "🧮 Analyse Complète", 
                "comparison": "📊 Comparaison",
                "advanced_metrics": "📈 Métriques Avancées",
                "simple_detection": "🔍 Détection Simple",
                "unknown": "❓ Type Inconnu"
            }
            st.info(f"🤖 Type détecté automatiquement: **{type_names.get(detected_type, 'Inconnu')}**")
        else:
            detected_type = {
                "🎯 Données de Trajectoire (Frame, X_center, Y_center, Radius)": "trajectory",
                "🧮 Résultats d'Analyse Complète (time_s, velocities, forces, etc.)": "complete_analysis",
                "📊 Données de Comparaison (Experiment, Water_Content, Krr, etc.)": "comparison"
            }.get(file_type, "unknown")
        
        # Traitement selon le type
        if detected_type == "trajectory":
            processed_metadata = process_trajectory_csv(df, metadata)
        elif detected_type == "complete_analysis":
            processed_metadata = process_complete_analysis_csv(df, metadata)
        elif detected_type == "comparison":
            processed_metadata = process_comparison_csv(df, metadata)
        else:
            st.warning("⚠️ Type de fichier non reconnu. Affichage des données brutes.")
            st.dataframe(df, use_container_width=True)
            processed_metadata = metadata
        
        # Options d'export et de sauvegarde
        if processed_metadata:
            st.markdown("---")
            st.markdown("### 💾 Sauvegarde pour Analyse Comparative")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Sauvegarder dans la Base de Comparaison"):
                    # Initialiser la session state si nécessaire
                    if 'research_experiments' not in st.session_state:
                        st.session_state.research_experiments = {}
                    
                    # Sauvegarder l'expérience
                    exp_name = processed_metadata['experiment_name']
                    st.session_state.research_experiments[exp_name] = {
                        'data': df,
                        'metadata': processed_metadata
                    }
                    
                    st.success(f"✅ Expérience '{exp_name}' sauvegardée!")
                    st.info("👉 Allez maintenant dans la section 'Analyse Comparative' pour comparer vos résultats")
            
            with col2:
                # Export des métadonnées enrichies
                enriched_csv = df.copy()
                for key, value in processed_metadata.items():
                    if key not in enriched_csv.columns:
                        enriched_csv[f'meta_{key}'] = value
                
                csv_data = enriched_csv.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger CSV Enrichi",
                    data=csv_data,
                    file_name=f"enriched_{filename}",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
        st.info("💡 Vérifiez que votre fichier CSV est bien formaté et accessible")

# Guide d'utilisation
st.markdown("---")
st.markdown("## 📚 Guide d'Utilisation")

with st.expander("🎯 Pour les fichiers de Trajectoire"):
    st.markdown("""
    **Format attendu:** `Frame, X_center, Y_center, Radius`
    
    **Ce que fait la plateforme:**
    - ✅ Calcul automatique du Krr
    - ✅ Visualisation de la trajectoire
    - ✅ Extraction des vitesses initiale/finale
    - ✅ Métadonnées automatiques depuis le nom de fichier
    
    **Exemple de nom de fichier:** `A15°_W10%_Steel_2cm_trajectory.csv`
    """)

with st.expander("🧮 Pour les fichiers d'Analyse Complète"):
    st.markdown("""
    **Colonnes reconnues automatiquement:**
    - ⏱️ Temps: `time_s`, `Time`, etc.
    - 🏃 Vitesses: `velocity`, `v_magnitude_ms`, `vx_ms`, etc.
    - 🔧 Forces: `F_resistance_N`, `force`, etc.
    - ⚡ Énergies: `E_total_J`, `energy`, etc.
    - 📊 Krr: `Krr_instantaneous`, `krr`, etc.
    
    **Avantages:**
    - ✅ Métriques déjà calculées
    - ✅ Visualisations automatiques
    - ✅ Statistiques avancées
    """)

with st.expander("📊 Pour les fichiers de Comparaison"):
    st.markdown("""
    **Format attendu:** `Experiment, Water_Content, Krr, [autres métriques]`
    
    **Fonctionnalités:**
    - ✅ Import direct dans l'analyse comparative  
    - ✅ Visualisations humidité vs Krr
    - ✅ Statistiques par niveau d'humidité
    - ✅ Intégration immédiate dans la recherche
    """)

# Status de la base de données
if 'research_experiments' in st.session_state and st.session_state.research_experiments:
    st.markdown("---")
    st.markdown("### 🗄️ Base de Données Actuelle")
    
    exp_summary = []
    for name, exp in st.session_state.research_experiments.items():
        meta = exp['metadata']
        exp_summary.append({
            'Expérience': name,
            'Humidité (%)': meta.get('water_content', 'N/A'),
            'Angle (°)': meta.get('angle_degrees', 'N/A'),
            'Type Sphère': meta.get('sphere_type', 'N/A'),
            'Krr': meta.get('krr_calculated', 'N/A')
        })
    
    summary_df = pd.DataFrame(exp_summary)
    st.dataframe(summary_df, use_container_width=True)
    
    st.success(f"🎯 {len(exp_summary)} expériences prêtes pour l'analyse comparative!")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>🔗 Intégration Parfaite: Vos Analyses → Comparaisons Avancées</strong></p>
    <p>Upload → Détection Automatique → Calculs → Comparaisons → Rapport</p>
</div>
""", unsafe_allow_html=True)
