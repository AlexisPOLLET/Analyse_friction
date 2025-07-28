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
    page_title="Analyseur de Friction - Substrat Humide",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .comparison-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .conclusion-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
<div class="main-header">
    <h1>🔬 Analyseur Comparatif de Friction</h1>
    <h2>Sphères sur Substrat Granulaire Humide</h2>
    <p><em>Analyse de l'effet de la teneur en eau et de l'angle de pente sur la résistance au roulement</em></p>
</div>
""", unsafe_allow_html=True)

# Initialisation des données de session
if 'experiments_data' not in st.session_state:
    st.session_state.experiments_data = {}

# Fonctions d'analyse
def load_detection_data(uploaded_file, experiment_name, water_content, angle, sphere_type):
    """Charge les données de détection et calcule les métriques"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Vérification des colonnes requises
        required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ Le fichier doit contenir les colonnes : {required_columns}")
            return None
        
        # Filtrer les détections valides
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        
        if len(df_valid) < 10:
            st.warning("⚠️ Pas assez de données valides pour l'analyse")
            return None
        
        # Calculer les métriques avancées
        metrics = calculate_friction_metrics(df_valid, water_content, angle, sphere_type)
        
        return {
            'name': experiment_name,
            'data': df,
            'valid_data': df_valid,
            'water_content': water_content,
            'angle': angle,
            'sphere_type': sphere_type,
            'metrics': metrics,
            'success_rate': len(df_valid) / len(df) * 100
        }
    return None

def load_results_data(uploaded_file, experiment_name, water_content, angle, sphere_type):
    """Charge les données de résultats pré-calculés"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Convertir en dictionnaire de métriques
        metrics = {}
        for _, row in df.iterrows():
            param = row['Parametre']
            value = row['Valeur']
            metrics[param] = value
        
        return {
            'name': experiment_name,
            'water_content': water_content,
            'angle': angle,
            'sphere_type': sphere_type,
            'metrics': metrics,
            'data_type': 'results'
        }
    return None

def calculate_friction_metrics(df_valid, water_content, angle, sphere_type):
    """Calcule les métriques de friction et cinématiques"""
    
    # Paramètres par défaut
    fps = 250.0  # Images par seconde
    pixels_per_mm = 5.0  # Calibration camera
    sphere_mass_g = 10.0  # Masse de la sphère en grammes
    sphere_radius_mm = 15.0  # Rayon en mm
    
    # Constantes physiques
    g = 9.81  # m/s²
    dt = 1 / fps
    mass_kg = sphere_mass_g / 1000
    radius_m = sphere_radius_mm / 1000
    angle_rad = np.radians(angle)
    
    # Facteur d'inertie selon le type de sphère
    j_factor = 2/5 if sphere_type == "Solide" else 2/3
    
    # Conversion en unités physiques
    x_m = df_valid['X_center'].values / pixels_per_mm / 1000
    y_m = df_valid['Y_center'].values / pixels_per_mm / 1000
    
    # Calculs cinématiques
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Accélérations
    acceleration = np.gradient(v_magnitude, dt)
    
    # Vitesses initiale et finale
    n_avg = min(3, len(v_magnitude)//4)
    v0 = np.mean(v_magnitude[:n_avg]) if len(v_magnitude) >= n_avg else v_magnitude[0]
    vf = np.mean(v_magnitude[-n_avg:]) if len(v_magnitude) >= n_avg else v_magnitude[-1]
    
    # Distance totale
    distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
    total_distance = np.sum(distances)
    
    # Coefficient de résistance au roulement Krr
    if total_distance > 0 and v0 > vf:
        krr = (v0**2 - vf**2) / (2 * g * total_distance)
    else:
        krr = None
    
    # Forces et énergies
    F_resistance = mass_kg * np.abs(acceleration)
    F_gravity = mass_kg * g * np.sin(angle_rad)
    
    # Énergies cinétiques
    E_trans = 0.5 * mass_kg * v_magnitude**2
    I = j_factor * mass_kg * radius_m**2
    omega = v_magnitude / radius_m
    E_rot = 0.5 * I * omega**2
    E_total = E_trans + E_rot
    
    # Métriques de qualité de trajectoire
    y_variation = np.std(y_m) * 1000  # mm
    path_length = np.sum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))
    straight_distance = np.sqrt((x_m[-1] - x_m[0])**2 + (y_m[-1] - y_m[0])**2)
    trajectory_efficiency = (straight_distance / path_length * 100) if path_length > 0 else 0
    
    return {
        'Krr': krr,
        'v0_ms': v0,
        'vf_ms': vf,
        'v0_mms': v0 * 1000,
        'vf_mms': vf * 1000,
        'max_velocity_mms': np.max(v_magnitude) * 1000,
        'avg_velocity_mms': np.mean(v_magnitude) * 1000,
        'max_acceleration_mms2': np.max(np.abs(acceleration)) * 1000,
        'avg_acceleration_mms2': np.mean(np.abs(acceleration)) * 1000,
        'total_distance_mm': total_distance * 1000,
        'max_resistance_force_mN': np.max(F_resistance) * 1000,
        'avg_resistance_force_mN': np.mean(F_resistance) * 1000,
        'energy_initial_mJ': E_total[0] * 1000 if len(E_total) > 0 else 0,
        'energy_final_mJ': E_total[-1] * 1000 if len(E_total) > 0 else 0,
        'energy_dissipated_mJ': (E_total[0] - E_total[-1]) * 1000 if len(E_total) > 0 else 0,
        'energy_efficiency_percent': (E_total[-1] / E_total[0] * 100) if len(E_total) > 0 and E_total[0] > 0 else 0,
        'trajectory_efficiency_percent': trajectory_efficiency,
        'vertical_variation_mm': y_variation,
        'duration_s': (len(df_valid) - 1) * dt,
        'j_factor': j_factor,
        'friction_coefficient_eff': krr + np.tan(angle_rad) if krr is not None else None,
        
        # Séries temporelles pour les graphiques
        'time_series': {
            'time': np.arange(len(df_valid)) * dt,
            'velocity_mms': v_magnitude * 1000,
            'acceleration_mms2': acceleration * 1000,
            'resistance_force_mN': F_resistance * 1000,
            'energy_total_mJ': E_total * 1000
        }
    }

# Interface utilisateur
st.sidebar.title("🎛️ Configuration des Expériences")

# Section de chargement des données
st.markdown("## 📂 Chargement des Données Expérimentales")

with st.expander("➕ Ajouter une nouvelle expérience", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        exp_name = st.text_input("Nom de l'expérience", value=f"Exp_{len(st.session_state.experiments_data)+1}")
        water_content = st.number_input("Teneur en eau (%)", value=0.0, min_value=0.0, max_value=30.0, step=0.5)
        angle = st.number_input("Angle de pente (°)", value=15.0, min_value=0.0, max_value=45.0, step=1.0)
    
    with col2:
        sphere_type = st.selectbox("Type de sphère", ["Solide", "Creuse"])
        data_type = st.radio("Type de données", ["Données de détection (CSV)", "Résultats pré-calculés (CSV)"])
    
    uploaded_file = st.file_uploader(
        "Charger le fichier de données",
        type=['csv'],
        help="Fichier CSV avec données de détection ou résultats pré-calculés"
    )
    
    if st.button("📊 Ajouter l'expérience") and uploaded_file is not None:
        if data_type == "Données de détection (CSV)":
            exp_data = load_detection_data(uploaded_file, exp_name, water_content, angle, sphere_type)
        else:
            exp_data = load_results_data(uploaded_file, exp_name, water_content, angle, sphere_type)
        
        if exp_data:
            st.session_state.experiments_data[exp_name] = exp_data
            st.success(f"✅ Expérience '{exp_name}' ajoutée avec succès!")
            st.rerun()

# Affichage des expériences chargées
if st.session_state.experiments_data:
    st.markdown("### 📋 Expériences Chargées")
    
    exp_summary = []
    for name, data in st.session_state.experiments_data.items():
        exp_summary.append({
            'Expérience': name,
            'Teneur en eau (%)': data['water_content'],
            'Angle (°)': data['angle'],
            'Type de sphère': data['sphere_type'],
            'Krr': f"{data['metrics'].get('Krr', 'N/A'):.6f}" if data['metrics'].get('Krr') is not None else "N/A",
            'Taux de succès (%)': f"{data.get('success_rate', 'N/A'):.1f}" if 'success_rate' in data else "N/A"
        })
    
    st.dataframe(pd.DataFrame(exp_summary), use_container_width=True)
    
    # Sélection des expériences à comparer
    st.markdown("### 🔍 Sélection pour Comparaison")
    selected_experiments = st.multiselect(
        "Choisir les expériences à comparer :",
        options=list(st.session_state.experiments_data.keys()),
        default=list(st.session_state.experiments_data.keys())
    )
    
    if len(selected_experiments) >= 2:
        st.markdown("---")
        st.markdown("## 📊 Analyse Comparative Détaillée")
        
        # Préparer les données pour la comparaison
        comparison_data = []
        for exp_name in selected_experiments:
            exp = st.session_state.experiments_data[exp_name]
            metrics = exp['metrics']
            
            comparison_data.append({
                'Expérience': exp_name,
                'Teneur_eau': exp['water_content'],
                'Angle': exp['angle'],
                'Type_sphère': exp['sphere_type'],
                **metrics
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Analyse de l'effet de la teneur en eau
        st.markdown("### 💧 Effet de la Teneur en Eau sur la Friction")
        
        if comp_df['Krr'].notna().any():
            col1, col2 = st.columns(2)
            
            with col1:
                fig_water = px.scatter(
                    comp_df, 
                    x='Teneur_eau', 
                    y='Krr',
                    color='Type_sphère',
                    hover_data=['Expérience'],
                    title="🔍 Coefficient Krr vs Teneur en Eau",
                    labels={'Teneur_eau': 'Teneur en eau (%)', 'Krr': 'Coefficient Krr'}
                )
                
                # Ajouter une ligne de tendance seulement s'il y a assez de données
                valid_trend_data = comp_df.dropna(subset=['Teneur_eau', 'Krr'])
                if len(valid_trend_data) >= 2:
                    try:
                        z = np.polyfit(valid_trend_data['Teneur_eau'], valid_trend_data['Krr'], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(valid_trend_data['Teneur_eau'].min(), valid_trend_data['Teneur_eau'].max(), 100)
                        fig_water.add_trace(go.Scatter(
                            x=x_trend, 
                            y=p(x_trend),
                            mode='lines',
                            name='Tendance',
                            line=dict(dash='dash', color='red', width=2)
                        ))
                    except:
                        pass  # Ignorer si la régression échoue
                
                st.plotly_chart(fig_water, use_container_width=True)
            
            with col2:
                # Graphique de l'accélération vs teneur en eau
                if 'max_acceleration_mms2' in comp_df.columns and comp_df['max_acceleration_mms2'].notna().any():
                    valid_accel = comp_df.dropna(subset=['max_acceleration_mms2'])
                    if len(valid_accel) > 0:
                        fig_accel = px.bar(
                            valid_accel,
                            x='Expérience',
                            y='max_acceleration_mms2',
                            color='Teneur_eau',
                            title="🚀 Accélération Maximale par Expérience",
                            labels={'max_acceleration_mms2': 'Accélération max (mm/s²)'}
                        )
                        fig_accel.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_accel, use_container_width=True)
                    else:
                        st.info("Données d'accélération non disponibles")
                else:
                    st.info("Données d'accélération non disponibles")
        
        # Analyse de l'effet de l'angle
        st.markdown("### 📐 Effet de l'Angle de Pente sur la Friction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if comp_df['Krr'].notna().any():
                # Filtrer les données valides pour éviter les erreurs
                valid_data = comp_df.dropna(subset=['Angle', 'Krr'])
                if len(valid_data) > 0:
                    # Utiliser une taille fixe si max_velocity_mms n'est pas disponible
                    size_col = 'max_velocity_mms' if 'max_velocity_mms' in valid_data.columns and valid_data['max_velocity_mms'].notna().any() else None
                    
                    fig_angle = px.scatter(
                        valid_data,
                        x='Angle',
                        y='Krr',
                        color='Teneur_eau',
                        size=size_col,
                        hover_data=['Expérience'],
                        title="📈 Coefficient Krr vs Angle de Pente",
                        labels={'Angle': 'Angle (°)', 'Krr': 'Coefficient Krr'}
                    )
                    st.plotly_chart(fig_angle, use_container_width=True)
                else:
                    st.warning("Pas assez de données valides pour l'analyse angle-Krr")
        
        with col2:
            # Vitesse finale vs angle
            if 'vf_mms' in comp_df.columns and comp_df['vf_mms'].notna().any():
                valid_data_vel = comp_df.dropna(subset=['Angle', 'vf_mms'])
                if len(valid_data_vel) > 0:
                    # Utiliser une taille fixe si Krr n'est pas disponible
                    size_col_vel = None
                    if 'Krr' in valid_data_vel.columns and valid_data_vel['Krr'].notna().any():
                        # Normaliser les valeurs de Krr pour la taille
                        krr_values = valid_data_vel['Krr'].fillna(valid_data_vel['Krr'].mean())
                        # Créer des tailles relatives
                        size_values = ((krr_values - krr_values.min()) / (krr_values.max() - krr_values.min()) * 20 + 10) if krr_values.max() != krr_values.min() else [15] * len(krr_values)
                        valid_data_vel = valid_data_vel.copy()
                        valid_data_vel['size_normalized'] = size_values
                        size_col_vel = 'size_normalized'
                    
                    fig_vel = px.scatter(
                        valid_data_vel,
                        x='Angle',
                        y='vf_mms',
                        color='Teneur_eau',
                        size=size_col_vel,
                        hover_data=['Expérience'],
                        title="🏃 Vitesse Finale vs Angle",
                        labels={'Angle': 'Angle (°)', 'vf_mms': 'Vitesse finale (mm/s)'}
                    )
                    st.plotly_chart(fig_vel, use_container_width=True)
                else:
                    st.warning("Pas assez de données valides pour l'analyse vitesse-angle")
            else:
                st.info("Données de vitesse finale non disponibles")
        
        # Métriques énergétiques
        st.markdown("### ⚡ Analyse Énergétique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'energy_dissipated_mJ' in comp_df.columns and comp_df['energy_dissipated_mJ'].notna().any():
                valid_energy = comp_df.dropna(subset=['energy_dissipated_mJ'])
                if len(valid_energy) > 0:
                    fig_energy = px.bar(
                        valid_energy,
                        x='Expérience',
                        y='energy_dissipated_mJ',
                        color='Teneur_eau',
                        title="🔋 Énergie Dissipée par Expérience",
                        labels={'energy_dissipated_mJ': 'Énergie dissipée (mJ)'}
                    )
                    fig_energy.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_energy, use_container_width=True)
                else:
                    st.info("Données d'énergie dissipée non disponibles")
            else:
                st.info("Données d'énergie dissipée non disponibles")
        
        with col2:
            if 'energy_efficiency_percent' in comp_df.columns and comp_df['energy_efficiency_percent'].notna().any():
                valid_efficiency = comp_df.dropna(subset=['energy_efficiency_percent', 'Teneur_eau'])
                if len(valid_efficiency) > 0:
                    # Gérer la taille du graphique
                    size_col_eff = None
                    if 'Krr' in valid_efficiency.columns and valid_efficiency['Krr'].notna().any():
                        krr_vals = valid_efficiency['Krr'].fillna(valid_efficiency['Krr'].mean())
                        if krr_vals.max() != krr_vals.min():
                            size_vals_eff = (krr_vals - krr_vals.min()) / (krr_vals.max() - krr_vals.min()) * 20 + 10
                        else:
                            size_vals_eff = [15] * len(krr_vals)
                        valid_efficiency = valid_efficiency.copy()
                        valid_efficiency['size_krr'] = size_vals_eff
                        size_col_eff = 'size_krr'
                    
                    fig_eff = px.scatter(
                        valid_efficiency,
                        x='Teneur_eau',
                        y='energy_efficiency_percent',
                        color='Angle',
                        size=size_col_eff,
                        hover_data=['Expérience'],
                        title="🎯 Efficacité Énergétique vs Teneur en Eau",
                        labels={'Teneur_eau': 'Teneur en eau (%)', 'energy_efficiency_percent': 'Efficacité énergétique (%)'}
                    )
                    st.plotly_chart(fig_eff, use_container_width=True)
                else:
                    st.info("Données d'efficacité énergétique non disponibles")
            else:
                st.info("Données d'efficacité énergétique non disponibles")
        
        # Tableau de comparaison détaillé
        st.markdown("### 📋 Tableau de Comparaison Détaillé")
        
        # Sélectionner les colonnes importantes pour l'affichage
        display_cols = [
            'Expérience', 'Teneur_eau', 'Angle', 'Type_sphère', 'Krr',
            'max_acceleration_mms2', 'max_velocity_mms', 'total_distance_mm',
            'energy_dissipated_mJ', 'trajectory_efficiency_percent'
        ]
        
        available_cols = [col for col in display_cols if col in comp_df.columns]
        display_df = comp_df[available_cols].copy()
        
        # Formatage des colonnes numériques
        if 'Krr' in display_df.columns:
            display_df['Krr'] = display_df['Krr'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
        
        for col in ['max_acceleration_mms2', 'max_velocity_mms', 'total_distance_mm', 'energy_dissipated_mJ']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        if 'trajectory_efficiency_percent' in display_df.columns:
            display_df['trajectory_efficiency_percent'] = display_df['trajectory_efficiency_percent'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Conclusions et analyse
        st.markdown("---")
        st.markdown("## 🎯 Conclusions et Analyse")
        
        # Analyse automatique des tendances
        conclusions = []
        
        # Effet de la teneur en eau
        if len(comp_df) >= 2 and comp_df['Krr'].notna().sum() >= 2:
            # Filtrer les données valides pour la corrélation
            valid_corr_data = comp_df.dropna(subset=['Teneur_eau', 'Krr'])
            if len(valid_corr_data) >= 2:
                try:
                    water_krr_corr = valid_corr_data[['Teneur_eau', 'Krr']].corr().iloc[0, 1]
                    
                    if not pd.isna(water_krr_corr):
                        if water_krr_corr > 0.3:
                            conclusions.append({
                                'type': 'Effet de l\'humidité',
                                'finding': 'AUGMENTATION de la friction avec l\'humidité',
                                'correlation': f'{water_krr_corr:.3f}',
                                'explanation': 'L\'augmentation de la teneur en eau crée des ponts capillaires qui augmentent la cohésion du substrat, augmentant ainsi la résistance au roulement.',
                                'color': 'warning'
                            })
                        elif water_krr_corr < -0.3:
                            conclusions.append({
                                'type': 'Effet de l\'humidité',
                                'finding': 'DIMINUTION de la friction avec l\'humidité',
                                'correlation': f'{water_krr_corr:.3f}',
                                'explanation': 'L\'eau agit comme un lubrifiant, réduisant la friction entre la sphère et le substrat granulaire.',
                                'color': 'success'
                            })
                        else:
                            conclusions.append({
                                'type': 'Effet de l\'humidité',
                                'finding': 'EFFET MINIMAL de l\'humidité sur la friction',
                                'correlation': f'{water_krr_corr:.3f}',
                                'explanation': 'Dans la gamme testée, la teneur en eau n\'a pas d\'effet significatif sur le coefficient de résistance au roulement.',
                                'color': 'info'
                            })
                except Exception as e:
                    st.warning(f"Erreur dans le calcul de corrélation humidité-Krr: {str(e)}")
        
        # Effet de l'angle
        if len(comp_df) >= 2 and comp_df['Krr'].notna().sum() >= 2:
            valid_angle_data = comp_df.dropna(subset=['Angle', 'Krr'])
            if len(valid_angle_data) >= 2:
                try:
                    angle_krr_corr = valid_angle_data[['Angle', 'Krr']].corr().iloc[0, 1]
                    
                    if not pd.isna(angle_krr_corr):
                        if angle_krr_corr > 0.3:
                            conclusions.append({
                                'type': 'Effet de l\'angle',
                                'finding': 'La friction AUGMENTE avec l\'angle de pente',
                                'correlation': f'{angle_krr_corr:.3f}',
                                'explanation': 'Les angles plus élevés peuvent causer une pénétration plus profonde de la sphère dans le substrat, augmentant la résistance.',
                                'color': 'warning'
                            })
                        elif angle_krr_corr < -0.3:
                            conclusions.append({
                                'type': 'Effet de l\'angle',
                                'finding': 'La friction DIMINUE avec l\'angle de pente',
                                'correlation': f'{angle_krr_corr:.3f}',
                                'explanation': 'Les angles plus élevés facilitent le roulement en réduisant le contact avec le substrat.',
                                'color': 'success'
                            })
                except Exception as e:
                    st.warning(f"Erreur dans le calcul de corrélation angle-Krr: {str(e)}")
        
        # Analyse des performances par expérience
        if 'max_acceleration_mms2' in comp_df.columns and comp_df['max_acceleration_mms2'].notna().any():
            try:
                best_accel_idx = comp_df['max_acceleration_mms2'].idxmax()
                best_accel_exp = comp_df.loc[best_accel_idx]
                conclusions.append({
                    'type': 'Performance maximale',
                    'finding': f'Accélération maximale : {best_accel_exp["Expérience"]}',
                    'correlation': f'{best_accel_exp["max_acceleration_mms2"]:.2f} mm/s²',
                    'explanation': f'Conditions : {best_accel_exp["Teneur_eau"]}% d\'humidité, {best_accel_exp["Angle"]}° d\'angle',
                    'color': 'info'
                })
            except Exception as e:
                st.warning(f"Erreur dans l'analyse des performances: {str(e)}")
        
        # Affichage des conclusions
        for conclusion in conclusions:
            color_class = {
                'success': 'conclusion-card',
                'warning': 'warning-card',
                'info': 'comparison-section'
            }.get(conclusion['color'], 'comparison-section')
            
            st.markdown(f"""
            <div class="{color_class}">
                <h4>🔍 {conclusion['type']}</h4>
                <h5><strong>{conclusion['finding']}</strong></h5>
                <p><strong>Corrélation :</strong> {conclusion['correlation']}</p>
                <p><strong>Explication :</strong> {conclusion['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Métriques de synthèse
        st.markdown("### 📊 Métriques de Synthèse")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if comp_df['Krr'].notna().any():
                try:
                    avg_krr = comp_df['Krr'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{avg_krr:.6f}</h3>
                        <p>Krr Moyen</p>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>N/A</h3>
                        <p>Krr Moyen</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            if 'max_acceleration_mms2' in comp_df.columns and comp_df['max_acceleration_mms2'].notna().any():
                try:
                    max_accel = comp_df['max_acceleration_mms2'].max()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{max_accel:.1f}</h3>
                        <p>Accél. Max (mm/s²)</p>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>N/A</h3>
                        <p>Accél. Max (mm/s²)</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col3:
            try:
                water_range = comp_df['Teneur_eau'].max() - comp_df['Teneur_eau'].min()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{water_range:.1f}%</h3>
                    <p>Gamme d'Humidité</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>N/A</h3>
                    <p>Gamme d'Humidité</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            try:
                angle_range = comp_df['Angle'].max() - comp_df['Angle'].min()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{angle_range:.1f}°</h3>
                    <p>Gamme d'Angles</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>N/A</h3>
                    <p>Gamme d'Angles</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Export des résultats
        st.markdown("### 💾 Export des Résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_comparison = comp_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les données de comparaison (CSV)",
                data=csv_comparison,
                file_name="comparaison_friction_complete.csv",
                mime="text/csv"
            )
        
        with col2:
            # Créer un rapport de conclusions
            conclusions_text = "# Rapport d'Analyse - Friction sur Substrat Granulaire Humide\n\n"
            for i, conclusion in enumerate(conclusions, 1):
                conclusions_text += f"## {i}. {conclusion['type']}\n"
                conclusions_text += f"**Résultat :** {conclusion['finding']}\n"
                conclusions_text += f"**Corrélation :** {conclusion['correlation']}\n"
                conclusions_text += f"**Explication :** {conclusion['explanation']}\n\n"
            
            st.download_button(
                label="📄 Télécharger le rapport de conclusions (TXT)",
                data=conclusions_text,
                file_name="rapport_conclusions_friction.txt",
                mime="text/plain"
            )
    
    elif len(selected_experiments) == 1:
        st.info("📊 Sélectionnez au moins 2 expériences pour effectuer une comparaison")
        
        # Affichage détaillé d'une seule expérience
        exp_name = selected_experiments[0]
        exp_data = st.session_state.experiments_data[exp_name]
        
        st.markdown(f"### 🔍 Analyse Détaillée - {exp_name}")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = exp_data['metrics']
        
        with col1:
            krr_val = metrics.get('Krr', 'N/A')
            krr_display = f"{krr_val:.6f}" if krr_val is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{krr_display}</h3>
                <p>Coefficient Krr</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            max_accel = metrics.get('max_acceleration_mms2', 'N/A')
            accel_display = f"{max_accel:.1f}" if max_accel != 'N/A' else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{accel_display}</h3>
                <p>Accél. Max (mm/s²)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            max_vel = metrics.get('max_velocity_mms', 'N/A')
            vel_display = f"{max_vel:.1f}" if max_vel != 'N/A' else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{vel_display}</h3>
                <p>Vitesse Max (mm/s)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            distance = metrics.get('total_distance_mm', 'N/A')
            dist_display = f"{distance:.1f}" if distance != 'N/A' else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{dist_display}</h3>
                <p>Distance (mm)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Graphiques de séries temporelles si disponibles
        if 'time_series' in metrics and metrics['time_series']:
            st.markdown("#### 📈 Évolution Temporelle")
            
            ts = metrics['time_series']
            
            # Créer un graphique avec sous-graphiques
            fig_ts = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Vitesse vs Temps', 'Accélération vs Temps', 
                               'Force de Résistance vs Temps', 'Énergie Totale vs Temps')
            )
            
            # Vitesse
            fig_ts.add_trace(
                go.Scatter(x=ts['time'], y=ts['velocity_mms'], 
                          mode='lines', name='Vitesse', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Accélération
            fig_ts.add_trace(
                go.Scatter(x=ts['time'], y=ts['acceleration_mms2'], 
                          mode='lines', name='Accélération', line=dict(color='red')),
                row=1, col=2
            )
            
            # Force de résistance
            fig_ts.add_trace(
                go.Scatter(x=ts['time'], y=ts['resistance_force_mN'], 
                          mode='lines', name='Force', line=dict(color='green')),
                row=2, col=1
            )
            
            # Énergie
            fig_ts.add_trace(
                go.Scatter(x=ts['time'], y=ts['energy_total_mJ'], 
                          mode='lines', name='Énergie', line=dict(color='purple')),
                row=2, col=2
            )
            
            fig_ts.update_xaxes(title_text="Temps (s)")
            fig_ts.update_yaxes(title_text="Vitesse (mm/s)", row=1, col=1)
            fig_ts.update_yaxes(title_text="Accélération (mm/s²)", row=1, col=2)
            fig_ts.update_yaxes(title_text="Force (mN)", row=2, col=1)
            fig_ts.update_yaxes(title_text="Énergie (mJ)", row=2, col=2)
            
            fig_ts.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_ts, use_container_width=True)
    
    else:
        st.info("🎯 Sélectionnez des expériences à comparer pour commencer l'analyse")

# Gestion des expériences
if st.session_state.experiments_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗂️ Gestion des Expériences")
    
    # Supprimer une expérience
    exp_to_remove = st.sidebar.selectbox(
        "Supprimer une expérience :",
        options=["Aucune"] + list(st.session_state.experiments_data.keys())
    )
    
    if exp_to_remove != "Aucune" and st.sidebar.button("🗑️ Supprimer"):
        del st.session_state.experiments_data[exp_to_remove]
        st.success(f"Expérience '{exp_to_remove}' supprimée!")
        st.rerun()
    
    # Effacer toutes les expériences
    if st.sidebar.button("🧹 Effacer Tout"):
        st.session_state.experiments_data = {}
        st.success("Toutes les expériences ont été supprimées!")
        st.rerun()

else:
    st.markdown("""
    ## 🚀 Pour Commencer
    
    ### 📋 Instructions d'utilisation :
    
    1. **📂 Chargez vos données expérimentales** :
       - Données de détection : CSV avec colonnes `Frame`, `X_center`, `Y_center`, `Radius`
       - Résultats pré-calculés : CSV avec colonnes `Parametre` et `Valeur`
    
    2. **⚙️ Configurez les paramètres** :
       - Nom de l'expérience
       - Teneur en eau (%)
       - Angle de pente (°)
       - Type de sphère (Solide/Creuse)
    
    3. **🔍 Comparez les résultats** :
       - Sélectionnez 2+ expériences
       - Analysez l'effet de l'humidité et de l'angle
       - Consultez les conclusions automatiques
    
    ### 📊 Types d'analyses disponibles :
    
    - **💧 Effet de la teneur en eau** sur le coefficient Krr
    - **📐 Effet de l'angle de pente** sur la friction
    - **🚀 Analyse cinématique** (vitesse, accélération)
    - **⚡ Analyse énergétique** (dissipation, efficacité)
    - **🛤️ Qualité de trajectoire**
    
    ### 🎯 Conclusions automatiques :
    
    L'application détermine automatiquement :
    - Si l'humidité **augmente** ou **diminue** la friction
    - L'effet de l'angle sur la résistance au roulement
    - Les conditions optimales pour chaque métrique
    - Les corrélations statistiques entre paramètres
    """)

# Section d'aide et informations
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Informations sur le Projet")

with st.sidebar.expander("🔬 Contexte Scientifique"):
    st.markdown("""
    **Projet :** Rolling Resistance of Spheres on Wet Granular Material
    
    **Objectif :** Quantifier l'effet de l'humidité sur la friction de roulement
    
    **Innovation :** Première étude sur l'humidité (littérature = sols secs uniquement)
    
    **Paramètres étudiés :**
    - Teneur en eau : 0-25%
    - Angles : 5-45°
    - Types de sphères : Solides/Creuses
    """)

with st.sidebar.expander("📐 Formules Utilisées"):
    st.markdown("""
    **Coefficient Krr :**
    ```
    Krr = (V₀² - Vf²) / (2gL)
    ```
    
    **Facteur d'inertie :**
    - Sphère solide : j = 2/5
    - Sphère creuse : j = 2/3
    
    **Friction effective :**
    ```
    μeff = Krr + tan(θ)
    ```
    
    **Énergie cinétique totale :**
    ```
    E = Etrans + Erot
    E = ½mv² + ½Iω²
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🎓 <strong>Analyseur de Friction - Substrat Granulaire Humide</strong><br>
    <em>Développé pour l'analyse de la résistance au roulement de sphères sur matériau granulaire humide</em><br>
    📧 Université d'Osaka - Département des Sciences de la Terre Cosmique
</div>
""", unsafe_allow_html=True)
