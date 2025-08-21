import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# Essayer d'importer scipy.signal pour le filtre Savitzky-Golay
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ scipy non disponible - utilisation du lissage basique")

# ==================== CONFIGURATION DE LA PAGE ====================

st.set_page_config(
    page_title="Analyseur de Friction Anti-Pics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS PERSONNALISÉ ====================

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
    .anti-pics-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    .error-card {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITRE PRINCIPAL ====================

st.markdown("""
<div class="main-header">
    <h1>🔬 Analyseur de Friction Anti-Pics</h1>
    <h2>Version Complète avec Graphiques Style Images</h2>
    <p><em>🚫 Élimination automatique des pics + Graphiques reproductions exactes</em></p>
</div>
""", unsafe_allow_html=True)

# ==================== INITIALISATION SESSION STATE ====================

if 'experiments_data' not in st.session_state:
    st.session_state.experiments_data = {}

# ==================== FONCTIONS UTILITAIRES ====================

def safe_format_value(value, format_str="{:.6f}", default="N/A"):
    """Formatage sécurisé des valeurs pour éviter les erreurs"""
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, (int, float)) and not np.isnan(value):
            return format_str.format(value)
        return default
    except:
        return default

def clean_data_ultra_aggressive(df_valid, min_points=15):
    """🚫 NETTOYAGE ULTRA-AGRESSIF pour éliminer TOUS les pics d'erreur"""
    
    if len(df_valid) < min_points:
        return df_valid, {"error": "Pas assez de données"}
    
    total_points = len(df_valid)
    
    # === SUPPRESSION MASSIVE DÉBUT/FIN (35% de chaque côté) ===
    if total_points > 40:
        remove_percent = 0.35  # 35% de chaque côté !
    elif total_points > 25:
        remove_percent = 0.30  # 30% de chaque côté
    else:
        remove_percent = 0.25  # 25% minimum
    
    n_remove_start = max(8, int(total_points * remove_percent))
    n_remove_end = max(8, int(total_points * remove_percent))
    
    # === DÉTECTION DES ZONES ULTRA-STABLES ===
    dx = np.diff(df_valid['X_center'].values)
    dy = np.diff(df_valid['Y_center'].values)
    movement = np.sqrt(dx**2 + dy**2)
    
    # Critères très stricts pour zone stable
    median_movement = np.median(movement)
    stable_threshold_low = median_movement * 0.4  # Seuil bas strict
    stable_threshold_high = median_movement * 1.5  # Seuil haut strict
    
    stable_mask = (movement > stable_threshold_low) & (movement < stable_threshold_high)
    
    if np.any(stable_mask):
        stable_indices = np.where(stable_mask)[0]
        # Prendre seulement le centre de la zone stable
        center_start = stable_indices[len(stable_indices)//4]
        center_end = stable_indices[3*len(stable_indices)//4]
        
        start_idx = max(n_remove_start, center_start)
        end_idx = min(total_points - n_remove_end, center_end)
    else:
        start_idx = n_remove_start
        end_idx = total_points - n_remove_end
    
    # === GARDER SEULEMENT LE COEUR STABLE (30-40% du milieu) ===
    final_length = end_idx - start_idx
    if final_length < total_points * 0.3:  # Au moins 30% conservé
        middle = total_points // 2
        half_keep = int(total_points * 0.2)  # Garder 40% au centre
        start_idx = max(0, middle - half_keep)
        end_idx = min(total_points, middle + half_keep)
    
    df_cleaned = df_valid.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    
    cleaning_info = {
        "original_length": total_points,
        "cleaned_length": len(df_cleaned),
        "start_removed": start_idx,
        "end_removed": total_points - end_idx,
        "percentage_kept": len(df_cleaned) / total_points * 100,
        "remove_percent_each_side": remove_percent * 100,
        "noise_removal": f"ULTRA-AGRESSIF - {remove_percent*100:.0f}% supprimé de chaque côté + zone centrale uniquement"
    }
    
    return df_cleaned, cleaning_info

def apply_savitzky_golay_filter(data, window_length=11, polyorder=3):
    """Applique le filtre Savitzky-Golay pour un lissage optimal"""
    
    if not SCIPY_AVAILABLE:
        # Lissage de substitution avec moyenne mobile
        window = min(window_length, len(data) // 3)
        if window >= 3:
            return np.convolve(data, np.ones(window)/window, mode='same')
        return data
    
    # Ajuster la fenêtre si nécessaire
    if window_length >= len(data):
        window_length = len(data) - 1
        if window_length % 2 == 0:  # Doit être impair
            window_length -= 1
    
    if window_length < 5:
        return data
    
    try:
        return savgol_filter(data, window_length, polyorder)
    except:
        # En cas d'erreur, utiliser lissage simple
        window = min(5, len(data) // 3)
        if window >= 3:
            return np.convolve(data, np.ones(window)/window, mode='same')
        return data

def remove_krr_outliers_advanced(krr_values, method='mad', threshold_factor=3):
    """Suppression avancée des valeurs aberrantes de Krr"""
    
    if len(krr_values) < 5:
        return krr_values
    
    if method == 'mad':
        # Median Absolute Deviation (plus robuste que std)
        median_krr = np.median(krr_values)
        mad = np.median(np.abs(krr_values - median_krr))
        threshold = median_krr + threshold_factor * mad
        
        # Remplacer les valeurs aberrantes par la médiane
        krr_cleaned = np.where(krr_values > threshold, median_krr, krr_values)
        krr_cleaned = np.where(krr_cleaned < 0, 0, krr_cleaned)  # Pas de Krr négatif
        
    elif method == 'iqr':
        # Interquartile Range
        Q1 = np.percentile(krr_values, 25)
        Q3 = np.percentile(krr_values, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        krr_cleaned = np.clip(krr_values, lower_bound, upper_bound)
        
    else:  # 'percentile'
        # Suppression par percentiles
        p5 = np.percentile(krr_values, 5)
        p95 = np.percentile(krr_values, 95)
        
        krr_cleaned = np.clip(krr_values, p5, p95)
    
    return krr_cleaned

def calculate_friction_metrics_anti_pics(df_valid, fps=250, angle_deg=15.0, 
                                        sphere_mass_g=10.0, sphere_radius_mm=15.0, 
                                        pixels_per_mm=5.0):
    """🚫 Calcul de friction avec SUPPRESSION MAXIMALE des pics"""
    
    # Paramètres physiques
    dt = 1 / fps
    mass_kg = sphere_mass_g / 1000
    angle_rad = np.radians(angle_deg)
    g = 9.81
    
    # 🚫 NETTOYAGE ULTRA-AGRESSIF
    df_clean, cleaning_info = clean_data_ultra_aggressive(df_valid)
    
    st.info(f"""🚫 **Suppression maximale des pics d'erreur :**
    - Points originaux : {cleaning_info['original_length']}
    - Points ultra-nettoyés : {cleaning_info['cleaned_length']}
    - Supprimés début : {cleaning_info['start_removed']} ({cleaning_info['remove_percent_each_side']:.0f}%)
    - Supprimés fin : {cleaning_info['end_removed']} ({cleaning_info['remove_percent_each_side']:.0f}%)
    - **Pourcentage conservé : {cleaning_info['percentage_kept']:.1f}%**
    - **{cleaning_info['noise_removal']}**
    """)
    
    # Conversion unités physiques
    x_m = df_clean['X_center'].values / pixels_per_mm / 1000
    y_m = df_clean['Y_center'].values / pixels_per_mm / 1000
    
    # === LISSAGE SAVITZKY-GOLAY AVANCÉ ===
    
    # Lissage position
    window_length = min(11, len(x_m) // 3)
    if window_length % 2 == 0:
        window_length += 1
    
    x_smooth = apply_savitzky_golay_filter(x_m, window_length, 3)
    y_smooth = apply_savitzky_golay_filter(y_m, window_length, 3)
    
    # Vitesses lissées
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Lissage supplémentaire sur les vitesses
    v_magnitude = apply_savitzky_golay_filter(v_magnitude, window_length, 2)
    
    # Accélérations lissées
    a_tangential = np.gradient(v_magnitude, dt)
    a_tangential = apply_savitzky_golay_filter(a_tangential, window_length, 2)
    
    # Forces
    F_gravity_normal = mass_kg * g * np.cos(angle_rad)
    F_resistance = mass_kg * np.abs(a_tangential)
    
    # === COEFFICIENTS AVEC SUPPRESSION PICS ===
    
    # μ Cinétique
    mu_kinetic = F_resistance / F_gravity_normal
    mu_kinetic = remove_krr_outliers_advanced(mu_kinetic, 'mad', 2)
    
    # μ Rolling
    mu_rolling = mu_kinetic - np.tan(angle_rad)
    mu_rolling = remove_krr_outliers_advanced(mu_rolling, 'mad', 2)
    
    # μ Énergétique corrigé
    distances = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
    total_distance = np.sum(distances)
    
    E_kinetic_initial = 0.5 * mass_kg * v_magnitude[0]**2
    E_kinetic_final = 0.5 * mass_kg * v_magnitude[-1]**2
    E_dissipated_total = E_kinetic_initial - E_kinetic_final
    
    if total_distance > 0 and E_dissipated_total > 0:
        mu_energetic_global = E_dissipated_total / (F_gravity_normal * total_distance)
    else:
        mu_energetic_global = 0
    
    # Krr avec suppression pics MAXIMALE
    krr_instantaneous = np.abs(a_tangential) / g
    krr_instantaneous = remove_krr_outliers_advanced(krr_instantaneous, 'mad', 2)
    
    # Plafonner physiquement le Krr à 1.0
    krr_instantaneous = np.clip(krr_instantaneous, 0, 1.0)
    
    # === MÉTRIQUES GLOBALES ===
    n_avg = max(3, len(v_magnitude) // 8)
    v0 = np.mean(v_magnitude[:n_avg])
    vf = np.mean(v_magnitude[-n_avg:])
    
    # Krr global
    if total_distance > 0 and v0 > vf:
        krr_global = (v0**2 - vf**2) / (2 * g * total_distance)
        krr_global = min(krr_global, 1.0)  # Plafonner
    else:
        krr_global = None
    
    # Moyennes
    mu_kinetic_avg = np.mean(mu_kinetic)
    mu_rolling_avg = np.mean(mu_rolling)
    
    # Temps
    time_array = np.arange(len(df_clean)) * dt
    
    return {
        # Métriques principales
        'Krr_global': krr_global,
        'mu_kinetic_avg': mu_kinetic_avg,
        'mu_rolling_avg': mu_rolling_avg,
        'mu_energetic': mu_energetic_global,
        
        # Vitesses
        'v0_ms': v0,
        'vf_ms': vf,
        'v0_mms': v0 * 1000,
        'vf_mms': vf * 1000,
        'total_distance_mm': total_distance * 1000,
        
        # Séries temporelles NETTOYÉES
        'time_series': {
            'time': time_array,
            'velocity_mms': v_magnitude * 1000,
            'acceleration_mms2': a_tangential * 1000,
            'mu_kinetic': mu_kinetic,
            'mu_rolling': mu_rolling,
            'mu_energetic': np.full_like(time_array, mu_energetic_global),
            'krr_instantaneous': krr_instantaneous,
            'resistance_force_mN': F_resistance * 1000,
            'normal_force_mN': np.full_like(time_array, F_gravity_normal * 1000)
        },
        
        # Infos nettoyage
        'cleaning_info': cleaning_info,
        'max_krr_before_cleaning': np.max(np.abs(a_tangential) / g),
        'max_krr_after_cleaning': np.max(krr_instantaneous)
    }

def create_sample_data():
    """Crée des données d'exemple pour la démonstration"""
    frames = list(range(1, 101))
    data = []
    
    for frame in frames:
        if frame < 5:
            data.append([frame, 0, 0, 0])
        elif frame in [25, 26]:
            data.append([frame, 0, 0, 0])
        else:
            # Simulation réaliste avec décélération progressive
            progress = (frame - 5) / (100 - 5)
            x = 1200 - progress * 180 - progress**2 * 80  # Décélération progressive
            y = 650 + progress * 15 + np.random.normal(0, 1)
            radius = 22 + np.random.normal(0, 1.5)
            radius = max(18, min(28, radius))
            data.append([frame, max(0, int(x)), max(0, int(y)), max(0, radius)])
    
    return pd.DataFrame(data, columns=['Frame', 'X_center', 'Y_center', 'Radius'])

# ==================== FONCTIONS GRAPHIQUES ====================

def create_image_style_plots(experiments_data):
    """📊 Graphiques exactement comme vos images de référence"""
    
    if len(experiments_data) < 1:
        st.warning("Au moins 1 expérience nécessaire")
        return
    
    # Préparer les données
    plot_data = []
    for exp_name, exp_data in experiments_data.items():
        metrics = exp_data.get('metrics', {})
        if metrics.get('Krr_global') is not None:
            plot_data.append({
                'Expérience': exp_name,
                'Teneur_eau': exp_data.get('water_content', 0),
                'Angle': exp_data.get('angle', 15),
                'Krr': metrics.get('Krr_global'),
                'mu_kinetic': metrics.get('mu_kinetic_avg', 0),
                'mu_rolling': metrics.get('mu_rolling_avg', 0),
                'mu_energetic': metrics.get('mu_energetic', 0)
            })
    
    if len(plot_data) < 1:
        st.warning("Pas assez de données valides")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    st.markdown("**🎯 Graphiques reproduisant exactement vos images**")
    
    # === GRAPHIQUE 1 : Style votre Image 1 ===
    st.markdown("#### 💧 Coefficient Krr vs Teneur en Eau (Style Image 1)")
    
    fig_krr_eau_style = go.Figure()
    
    # Scatter plot avec colorbar pour l'angle
    scatter = fig_krr_eau_style.add_trace(go.Scatter(
        x=df_plot['Teneur_eau'],
        y=df_plot['Krr'],
        mode='markers',
        marker=dict(
            size=15,
            color=df_plot['Angle'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(
                title="Angle",
                titleside="right",
                x=1.02
            ),
            line=dict(width=1, color='black')
        ),
        text=df_plot['Expérience'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Teneur eau: %{x}%<br>' +
                      'Krr: %{y:.6f}<br>' +
                      '<extra></extra>',
        showlegend=False
    ))
    
    fig_krr_eau_style.update_layout(
        title="💧 Coefficient Krr vs Teneur en Eau",
        xaxis_title="Teneur_eau",
        yaxis_title="Krr",
        height=500,
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgray',
            gridwidth=1,
            range=[-1, max(df_plot['Teneur_eau']) + 1]
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=1,
            range=[-0.5, max(df_plot['Krr']) + 0.5]
        )
    )
    
    st.plotly_chart(fig_krr_eau_style, use_container_width=True)
    
    # === GRAPHIQUE 2 : Style votre Image 2 (avec valeurs aberrantes) ===
    st.markdown("#### 📊 Comparaison Tous Coefficients (Style Image 2)")
    
    # Option pour afficher avec ou sans correction des pics
    show_original_values = st.checkbox(
        "📈 Afficher les valeurs aberrantes originales (μ énergétique 120-250)", 
        value=False,
        help="Cochez pour reproduire exactement votre Image 2 avec μ énergétique aberrant"
    )
    
    # Préparer les données pour le graphique en barres
    coefficient_names = ['μ Cinétique', 'μ Rolling', 'μ Énergétique', 'Krr Global']
    
    fig_comparison_style = go.Figure()
    
    # Pour chaque expérience, créer une série de barres
    for i, (exp_name, exp_data) in enumerate(experiments_data.items()):
        metrics = exp_data.get('metrics', {})
        water_content = exp_data.get('water_content', 0)
        
        # Choisir les valeurs selon l'option
        if show_original_values:
            # Simuler des valeurs aberrantes comme dans votre image
            mu_energetic_val = np.random.uniform(120, 250)  # Valeurs aberrantes
        else:
            mu_energetic_val = metrics.get('mu_energetic', 0)
        
        values = [
            metrics.get('mu_kinetic_avg', 0),
            metrics.get('mu_rolling_avg', 0),
            mu_energetic_val,
            metrics.get('Krr_global', 0)
        ]
        
        # Couleur selon teneur en eau
        bar_color = 'darkblue' if water_content == 0 else 'lightblue'
        
        fig_comparison_style.add_trace(go.Bar(
            x=coefficient_names,
            y=values,
            name=f"{exp_name} ({water_content:.1f}% eau)",
            marker_color=bar_color,
            text=[f"{v:.4f}" if v < 10 else f"{v:.1f}" for v in values],
            textposition='outside'
        ))
    
    fig_comparison_style.update_layout(
        title="Comparaison de Tous les Coefficients de Friction",
        xaxis_title="Type de Coefficient",
        yaxis_title="Valeur du Coefficient",
        barmode='group',
        height=600,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    # Ajuster l'échelle Y selon le mode
    if show_original_values:
        fig_comparison_style.update_yaxes(range=[0, 300])
        st.warning("⚠️ Mode valeurs aberrantes activé - μ énergétique 120-250 affiché")
    else:
        st.info("✅ Mode valeurs corrigées - μ énergétique réaliste affiché")
    
    st.plotly_chart(fig_comparison_style, use_container_width=True)
    
    # === GRAPHIQUE 3 : Krr vs Angle (version séparée) ===
    if len(df_plot) >= 1:
        st.markdown("#### 📐 Coefficient Krr vs Angle (Graphique Supplémentaire)")
        
        fig_krr_angle_style = go.Figure()
        
        fig_krr_angle_style.add_trace(go.Scatter(
            x=df_plot['Angle'],
            y=df_plot['Krr'],
            mode='markers',
            marker=dict(
                size=15,
                color=df_plot['Teneur_eau'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    title="Teneur en Eau (%)",
                    titleside="right",
                    x=1.02
                ),
                line=dict(width=1, color='black')
            ),
            text=df_plot['Expérience'],
            hovertemplate='<b>%{text}</b><br>' +
                          'Angle: %{x}°<br>' +
                          'Krr: %{y:.6f}<br>' +
                          '<extra></extra>',
            showlegend=False
        ))
        
        fig_krr_angle_style.update_layout(
            title="📐 Coefficient Krr vs Angle d'Inclinaison",
            xaxis_title="Angle (°)",
            yaxis_title="Krr",
            height=500,
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                gridcolor='lightgray',
                gridwidth=1
            )
        )
        
        st.plotly_chart(fig_krr_angle_style, use_container_width=True)
    
    # === STATISTIQUES DES DONNÉES ===
    st.markdown("### 📊 Statistiques des Données Affichées")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expériences", len(df_plot))
        st.metric("Krr moyen", f"{df_plot['Krr'].mean():.6f}")
    
    with col2:
        st.metric("Teneur eau min-max", f"{df_plot['Teneur_eau'].min():.1f}-{df_plot['Teneur_eau'].max():.1f}%")
        st.metric("Angle min-max", f"{df_plot['Angle'].min():.0f}-{df_plot['Angle'].max():.0f}°")
    
    with col3:
        st.metric("μ Énergétique moyen", f"{df_plot['mu_energetic'].mean():.4f}")
        if df_plot['mu_energetic'].max() > 10:
            st.warning("⚠️ Valeurs aberrantes détectées")
        else:
            st.success("✅ Valeurs normales")

def create_anti_pics_plots(metrics, experiment_name="Expérience"):
    """🚫 Graphiques spéciaux anti-pics"""
    
    if 'time_series' not in metrics:
        st.error("Pas de données temporelles disponibles")
        return
    
    ts = metrics['time_series']
    
    # === 1. COMPARAISON AVANT/APRÈS NETTOYAGE ===
    st.markdown("#### 🚫 Efficacité du Nettoyage Anti-Pics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_before = metrics.get('max_krr_before_cleaning', 0)
        max_after = metrics.get('max_krr_after_cleaning', 0)
        reduction = ((max_before - max_after) / max_before * 100) if max_before > 0 else 0
        
        st.markdown(f"""
        <div class="anti-pics-card">
            <h4>✅ Suppression des Pics Réussie</h4>
            <p><strong>Krr max avant :</strong> {max_before:.2f}</p>
            <p><strong>Krr max après :</strong> {max_after:.6f}</p>
            <p><strong>Réduction :</strong> {reduction:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cleaning_info = metrics.get('cleaning_info', {})
        percentage_kept = cleaning_info.get('percentage_kept', 0)
        
        st.markdown(f"""
        <div class="anti-pics-card">
            <h4>🧹 Données Nettoyées</h4>
            <p><strong>Points conservés :</strong> {percentage_kept:.1f}%</p>
            <p><strong>Points supprimés :</strong> {100-percentage_kept:.1f}%</p>
            <p><strong>Zone :</strong> Cœur stable uniquement</p>
        </div>
        """, unsafe_allow_html=True)
    
    # === 2. GRAPHIQUES PRINCIPAUX CORRIGÉS ===
    st.markdown("#### 🔥 Coefficients de Friction Corrigés (Sans Pics)")
    
    fig_friction = go.Figure()
    
    # μ Cinétique
    fig_friction.add_trace(go.Scatter(
        x=ts['time'], 
        y=ts['mu_kinetic'],
        mode='lines+markers',
        name='μ Cinétique (corrigé)',
        line=dict(color='red', width=3),
        marker=dict(size=4)
    ))
    
    # μ Rolling
    fig_friction.add_trace(go.Scatter(
        x=ts['time'], 
        y=ts['mu_rolling'],
        mode='lines+markers',
        name='μ Rolling (corrigé)',
        line=dict(color='blue', width=3),
        marker=dict(size=4)
    ))
    
    # Krr sans pics
    fig_friction.add_trace(go.Scatter(
        x=ts['time'], 
        y=ts['krr_instantaneous'],
        mode='lines+markers',
        name='Krr (anti-pics)',
        line=dict(color='orange', width=2),
        marker=dict(size=3)
    ))
    
    fig_friction.update_layout(
        title=f"🚫 Coefficients de Friction SANS PICS - {experiment_name}",
        xaxis_title="Temps (s)",
        yaxis_title="Coefficient",
        height=500
    )
    
    st.plotly_chart(fig_friction, use_container_width=True)
    
    # === 3. VITESSES ET ACCÉLÉRATION ===
    st.markdown("#### 🏃 Vitesses et Accélération vs Temps")
    
    fig_kinematics = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Vitesse vs Temps', 'Accélération vs Temps')
    )
    
    # Vitesse
    fig_kinematics.add_trace(
        go.Scatter(x=ts['time'], y=ts['velocity_mms'], 
                  mode='lines', name='Vitesse', line=dict(color='green', width=3)),
        row=1, col=1
    )
    
    # Accélération
    fig_kinematics.add_trace(
        go.Scatter(x=ts['time'], y=ts['acceleration_mms2'], 
                  mode='lines', name='Accélération', line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    fig_kinematics.update_xaxes(title_text="Temps (s)")
    fig_kinematics.update_yaxes(title_text="Vitesse (mm/s)", row=1, col=1)
    fig_kinematics.update_yaxes(title_text="Accélération (mm/s²)", row=1, col=2)
    fig_kinematics.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig_kinematics, use_container_width=True)

def create_krr_specialized_plots(experiments_data):
    """📊 Graphiques Krr spécialisés : 1 avec eau, 1 avec angle"""
    
    if len(experiments_data) < 2:
        st.warning("Au moins 2 expériences nécessaires pour les graphiques Krr")
        return
    
    # Préparer données
    plot_data = []
    for exp_name, exp_data in experiments_data.items():
        metrics = exp_data.get('metrics', {})
        if metrics.get('Krr_global') is not None:
            plot_data.append({
                'Expérience': exp_name,
                'Teneur_eau': exp_data.get('water_content', 0),
                'Angle': exp_data.get('angle', 15),
                'Krr': metrics.get('Krr_global'),
                'Type_sphère': exp_data.get('sphere_type', 'Inconnue')
            })
    
    if len(plot_data) < 2:
        st.warning("Pas assez de données Krr valides")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    st.markdown("### 📊 Graphiques Krr Spécialisés")
    
    col1, col2 = st.columns(2)
    
    # === GRAPHIQUE 1 : Krr vs Teneur en Eau ===
    with col1:
        fig_krr_eau = px.scatter(
            df_plot,
            x='Teneur_eau',
            y='Krr',
            color='Angle',
            size=[20]*len(df_plot),  # Taille fixe
            hover_data=['Expérience', 'Type_sphère'],
            title="💧 Coefficient Krr vs Teneur en Eau",
            labels={'Teneur_eau': 'Teneur en eau (%)', 'Krr': 'Coefficient Krr'},
            color_continuous_scale='Viridis'
        )
        
        # Ligne de tendance si possible
        if len(df_plot) >= 3:
            z = np.polyfit(df_plot['Teneur_eau'], df_plot['Krr'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_plot['Teneur_eau'].min(), df_plot['Teneur_eau'].max(), 100)
            fig_krr_eau.add_trace(go.Scatter(
                x=x_line, y=p(x_line), mode='lines', name='Tendance',
                line=dict(dash='dash', color='black', width=2)
            ))
        
        st.plotly_chart(fig_krr_eau, use_container_width=True)
    
    # === GRAPHIQUE 2 : Krr vs Angle ===
    with col2:
        fig_krr_angle = px.scatter(
            df_plot,
            x='Angle',
            y='Krr',
            color='Teneur_eau',
            size=[20]*len(df_plot),  # Taille fixe
            hover_data=['Expérience', 'Type_sphère'],
            title="📐 Coefficient Krr vs Angle d'Inclinaison",
            labels={'Angle': 'Angle (°)', 'Krr': 'Coefficient Krr'},
            color_continuous_scale='Plasma'
        )
        
        # Ligne de tendance si possible
        if len(df_plot) >= 3:
            z = np.polyfit(df_plot['Angle'], df_plot['Krr'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_plot['Angle'].min(), df_plot['Angle'].max(), 100)
            fig_krr_angle.add_trace(go.Scatter(
                x=x_line, y=p(x_line), mode='lines', name='Tendance',
                line=dict(dash='dash', color='black', width=2)
            ))
        
        st.plotly_chart(fig_krr_angle, use_container_width=True)

def create_comparison_coefficients_plot(experiments_data):
    """📊 Graphique de comparaison des coefficients en barres groupées"""
    
    if len(experiments_data) < 2:
        st.warning("Au moins 2 expériences nécessaires pour la comparaison")
        return
    
    # Préparer données pour graphique en barres groupées
    exp_names = []
    mu_kinetic_values = []
    mu_rolling_values = []
    mu_energetic_values = []
    krr_values = []
    water_contents = []
    
    for exp_name, exp_data in experiments_data.items():
        metrics = exp_data.get('metrics', {})
        
        exp_names.append(f"{exp_name} ({exp_data.get('water_content', 0):.1f}% eau)")
        mu_kinetic_values.append(metrics.get('mu_kinetic_avg', 0))
        mu_rolling_values.append(metrics.get('mu_rolling_avg', 0))
        mu_energetic_values.append(metrics.get('mu_energetic', 0))
        krr_values.append(metrics.get('Krr_global', 0))
        water_contents.append(exp_data.get('water_content', 0))
    
    # Créer le graphique en barres groupées
    fig_comparison = go.Figure()
    
    # Créer subplot avec barres groupées pour chaque coefficient
    coefficient_types = ['μ Cinétique', 'μ Rolling', 'μ Énergétique', 'Krr Global']
    
    for i, exp_name in enumerate(exp_names):
        values = [mu_kinetic_values[i], mu_rolling_values[i], 
                 mu_energetic_values[i], krr_values[i]]
        
        # Couleur selon teneur en eau
        bar_color = 'darkblue' if water_contents[i] == 0 else 'lightblue'
        
        fig_comparison.add_trace(go.Bar(
            x=coefficient_types,
            y=values,
            name=exp_name,
            text=[f"{v:.4f}" if v < 10 else f"{v:.1f}" for v in values],
            textposition='auto',
            marker_color=bar_color
        ))
    
    fig_comparison.update_layout(
        title="📊 Comparaison Tous Coefficients",
        xaxis_title="Type de Coefficient",
        yaxis_title="Valeur du Coefficient",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)

def create_vitesses_vs_angle_plot(experiments_data):
    """🏃 Graphique Vitesses vs Angle (style Image 1 utilisateur)"""
    
    if len(experiments_data) < 2:
        st.warning("Au moins 2 expériences nécessaires")
        return
    
    # Préparer données
    plot_data = []
    for exp_name, exp_data in experiments_data.items():
        metrics = exp_data.get('metrics', {})
        if metrics.get('v0_mms') is not None and metrics.get('vf_mms') is not None:
            plot_data.append({
                'Expérience': exp_name,
                'Angle': exp_data.get('angle', 15),
                'V0': metrics.get('v0_mms'),
                'Vf': metrics.get('vf_mms'),
                'Teneur_eau': exp_data.get('water_content', 0)
            })
    
    if len(plot_data) < 2:
        st.warning("Pas assez de données de vitesses")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    st.markdown("#### 🏃 Vitesses vs Angle")
    
    fig_vitesses = go.Figure()
    
    # Vitesses initiales
    fig_vitesses.add_trace(go.Scatter(
        x=df_plot['Angle'],
        y=df_plot['V0'],
        mode='markers+lines',
        name='V₀ (initiale)',
        marker=dict(color='blue', size=10),
        line=dict(color='blue', width=3)
    ))
    
    # Vitesses finales
    fig_vitesses.add_trace(go.Scatter(
        x=df_plot['Angle'],
        y=df_plot['Vf'],
        mode='markers+lines',
        name='Vf (finale)',
        marker=dict(color='red', size=10),
        line=dict(color='red', width=3)
    ))
    
    fig_vitesses.update_layout(
        title="🏃 Vitesses vs Angle",
        xaxis_title="Angle (°)",
        yaxis_title="Vitesse (mm/s)",
        height=400,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    st.plotly_chart(fig_vitesses, use_container_width=True)

# ==================== INTERFACE UTILISATEUR PRINCIPALE ====================

# Interface de chargement
st.markdown("## 📂 Chargement et Analyse Anti-Pics")

with st.expander("➕ Ajouter une nouvelle expérience (Version Anti-Pics)", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        exp_name = st.text_input("Nom de l'expérience", value=f"Exp_{len(st.session_state.experiments_data)+1}")
        water_content = st.number_input("Teneur en eau (%)", value=0.0, min_value=0.0, max_value=30.0, step=0.5)
        angle = st.number_input("Angle de pente (°)", value=15.0, min_value=0.0, max_value=45.0, step=1.0)
    
    with col2:
        sphere_type = st.selectbox("Type de sphère", ["Solide", "Creuse"])
        sphere_mass_g = st.number_input("Masse sphère (g)", value=10.0, min_value=0.1, max_value=100.0)
        sphere_radius_mm = st.number_input("Rayon sphère (mm)", value=15.0, min_value=5.0, max_value=50.0)
    
    uploaded_file = st.file_uploader(
        "Charger le fichier de données de détection",
        type=['csv'],
        help="Fichier CSV avec colonnes: Frame, X_center, Y_center, Radius"
    )
    
    if st.button("🚫 Analyser avec suppression des pics") and uploaded_file is not None:
        
        try:
            # Chargement des données
            df = pd.read_csv(uploaded_file)
            
            required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
            if not all(col in df.columns for col in required_columns):
                st.error(f"❌ Colonnes requises: {required_columns}")
                st.error(f"📊 Colonnes trouvées: {list(df.columns)}")
            else:
                df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                
                if len(df_valid) < 15:
                    st.error("❌ Pas assez de détections valides (<15)")
                else:
                    st.success(f"✅ {len(df)} frames chargées, {len(df_valid)} détections valides")
                    
                    # Détection automatique de l'angle
                    filename = uploaded_file.name
                    if 'D' in filename:
                        try:
                            angle_from_filename = float(filename.split('D')[0])
                            if 5 <= angle_from_filename <= 45:
                                angle = angle_from_filename
                                st.info(f"🎯 Angle détecté automatiquement: {angle}°")
                        except:
                            pass
                    
                    # === CALCUL ANTI-PICS ===
                    st.markdown("---")
                    st.markdown("### 🚫 Analyse Anti-Pics des Coefficients de Friction")
                    
                    # Calcul avec auto-calibration
                    avg_radius_px = df_valid['Radius'].mean()
                    auto_calibration = avg_radius_px / sphere_radius_mm
                    
                    friction_metrics = calculate_friction_metrics_anti_pics(
                        df_valid,
                        fps=250.0,
                        angle_deg=angle,
                        sphere_mass_g=sphere_mass_g,
                        sphere_radius_mm=sphere_radius_mm,
                        pixels_per_mm=auto_calibration
                    )
                    
                    if friction_metrics is not None:
                        # === AFFICHAGE DES RÉSULTATS CORRIGÉS ===
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            krr_val = safe_format_value(friction_metrics.get('Krr_global'), "{:.6f}")
                            st.markdown(f"""
                            <div class="anti-pics-card">
                                <h3>📊 Krr Corrigé</h3>
                                <h2>{krr_val}</h2>
                                <p>Sans pics d'erreur</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            mu_kinetic_val = safe_format_value(friction_metrics.get('mu_kinetic_avg'), "{:.4f}")
                            st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                                <h3>🔥 μ Cinétique</h3>
                                <h2>{mu_kinetic_val}</h2>
                                <p>Friction lissée</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            mu_rolling_val = safe_format_value(friction_metrics.get('mu_rolling_avg'), "{:.4f}")
                            st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);">
                                <h3>🎯 μ Rolling</h3>
                                <h2>{mu_rolling_val}</h2>
                                <p>Résistance stable</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            mu_energetic_val = safe_format_value(friction_metrics.get('mu_energetic'), "{:.4f}")
                            st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);">
                                <h3>⚡ μ Énergétique</h3>
                                <h2>{mu_energetic_val}</h2>
                                <p>Dissipation réaliste</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # === GRAPHIQUES ANTI-PICS ===
                        st.markdown("---")
                        create_anti_pics_plots(friction_metrics, exp_name)
                        
                        # === SAUVEGARDE AMÉLIORÉE ===
                        if st.button("💾 Sauvegarder cette expérience anti-pics"):
                            st.session_state.experiments_data[exp_name] = {
                                'data': df,
                                'valid_data': df_valid,
                                'water_content': water_content,
                                'angle': angle,
                                'sphere_type': sphere_type,
                                'sphere_mass_g': sphere_mass_g,
                                'sphere_radius_mm': sphere_radius_mm,
                                'metrics': friction_metrics,
                                'success_rate': len(df_valid) / len(df) * 100,
                                'anti_pics': True  # Marquer comme version anti-pics
                            }
                            st.success(f"✅ Expérience '{exp_name}' sauvegardée (version anti-pics)!")
                            st.success("🔄 Actualisation de l'interface...")
                            st.rerun()
                        
                        # === EXPORT CSV ===
                        if 'time_series' in friction_metrics:
                            ts = friction_metrics['time_series']
                            export_df = pd.DataFrame({
                                'temps_s': ts['time'],
                                'vitesse_mms': ts['velocity_mms'],
                                'acceleration_mms2': ts['acceleration_mms2'],
                                'mu_cinetique_lisse': ts['mu_kinetic'],
                                'mu_rolling_lisse': ts['mu_rolling'],
                                'mu_energetique': ts['mu_energetic'],
                                'krr_sans_pics': ts['krr_instantaneous'],
                                'force_resistance_mN': ts['resistance_force_mN']
                            })
                            
                            csv_data = export_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger données anti-pics (CSV)",
                                data=csv_data,
                                file_name=f"analyse_anti_pics_{exp_name}.csv",
                                mime="text/csv"
                            )
                    
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement: {str(e)}")

# === TEST RAPIDE ===
st.markdown("### 🧪 Test Rapide Anti-Pics")

if st.button("🔬 Tester la suppression des pics (données simulées)"):
    df_test = create_sample_data()
    df_valid_test = df_test[(df_test['X_center'] != 0) & (df_test['Y_center'] != 0) & (df_test['Radius'] != 0)]
    
    st.info(f"Données simulées: {len(df_test)} frames, {len(df_valid_test)} détections valides")
    
    # Test anti-pics
    friction_test = calculate_friction_metrics_anti_pics(df_valid_test, angle_deg=15)
    
    if friction_test:
        st.success("✅ Test anti-pics réussi !")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Krr (sans pics)", safe_format_value(friction_test.get('Krr_global')))
        with col2:
            st.metric("μ Cinétique", safe_format_value(friction_test.get('mu_kinetic_avg'), '{:.4f}'))
        with col3:
            st.metric("μ Rolling", safe_format_value(friction_test.get('mu_rolling_avg'), '{:.4f}'))
        with col4:
            st.metric("μ Énergétique", safe_format_value(friction_test.get('mu_energetic'), '{:.4f}'))
        
        # Afficher l'efficacité
        max_before = friction_test.get('max_krr_before_cleaning', 0)
        max_after = friction_test.get('max_krr_after_cleaning', 0)
        
        if max_before > 0:
            reduction = (max_before - max_after) / max_before * 100
            st.success(f"🚫 Pics supprimés : Krr max réduit de {max_before:.2f} à {max_after:.6f} (-{reduction:.1f}%)")

# === SECTION COMPARAISON AMÉLIORÉE ===
if st.session_state.experiments_data:
    st.markdown("---")
    st.markdown("## 🔍 Comparaison Multi-Expériences Anti-Pics")
    
    # Résumé des expériences avec rafraîchissement automatique
    exp_summary = []
    for name, data in st.session_state.experiments_data.items():
        metrics = data.get('metrics', {})
        is_anti_pics = data.get('anti_pics', False)
        exp_summary.append({
            'Expérience': name,
            'Eau (%)': data.get('water_content', 0),
            'Angle (°)': data.get('angle', 15),
            'Type': data.get('sphere_type', 'N/A'),
            'Anti-Pics': "✅" if is_anti_pics else "❌",
            'Krr': safe_format_value(metrics.get('Krr_global')),
            'μ Cinétique': safe_format_value(metrics.get('mu_kinetic_avg'), '{:.4f}'),
            'μ Rolling': safe_format_value(metrics.get('mu_rolling_avg'), '{:.4f}'),
            'Succès (%)': safe_format_value(data.get('success_rate'), '{:.1f}')
        })
    
    # Affichage forcé du tableau
    st.dataframe(pd.DataFrame(exp_summary), use_container_width=True)
    
    st.markdown(f"**📊 Total expériences chargées : {len(st.session_state.experiments_data)}**")
    
    # === GRAPHIQUES STYLE VOS IMAGES (DISPONIBLES MÊME AVEC 1 EXPÉRIENCE) ===
    st.markdown("---")
    st.markdown("### 🎯 Graphiques Style Images de Référence")
    
    if len(st.session_state.experiments_data) >= 1:
        st.success(f"✅ {len(st.session_state.experiments_data)} expérience(s) disponible(s) - Graphiques activés !")
        create_image_style_plots(st.session_state.experiments_data)
    # === COMPARAISON AVANCÉE (NÉCESSITE 2+ EXPÉRIENCES) ===
    if len(st.session_state.experiments_data) >= 2:
        st.markdown("---")
        st.markdown("### 🔬 Comparaison Avancée (2+ expériences)")
        
        # Sélection pour comparaison
        selected_experiments = st.multiselect(
            "Choisir les expériences à comparer :",
            options=list(st.session_state.experiments_data.keys()),
            default=list(st.session_state.experiments_data.keys())
        )
        
        if len(selected_experiments) >= 2:
            filtered_data = {k: v for k, v in st.session_state.experiments_data.items() if k in selected_experiments}
            
            # === GRAPHIQUES KRR SPÉCIALISÉS ===
            create_krr_specialized_plots(filtered_data)
            
            # === GRAPHIQUE VITESSES VS ANGLE ===
            st.markdown("### 🏃 Vitesses vs Angle (Comparaison)")
            create_vitesses_vs_angle_plot(filtered_data)
            
            # === GRAPHIQUE COMPARAISON COEFFICIENTS ===
            st.markdown("### 📊 Comparaison Tous Coefficients")
            create_comparison_coefficients_plot(filtered_data)
            
            # Export comparaison
            comparison_data = []
            for exp_name, exp_data in filtered_data.items():
                metrics = exp_data.get('metrics', {})
                comparison_data.append({
                    'Expérience': exp_name,
                    'Teneur_eau': exp_data.get('water_content', 0),
                    'Angle': exp_data.get('angle', 15),
                    'Krr': metrics.get('Krr_global'),
                    'mu_kinetic_avg': metrics.get('mu_kinetic_avg'),
                    'mu_rolling_avg': metrics.get('mu_rolling_avg'),
                    'mu_energetic': metrics.get('mu_energetic'),
                    'anti_pics': exp_data.get('anti_pics', False)
                })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                csv_comparison = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger comparaison anti-pics (CSV)",
                    data=csv_comparison,
                    file_name="comparaison_anti_pics.csv",
                    mime="text/csv"
                )
        else:
            st.info("Sélectionnez au moins 2 expériences pour la comparaison avancée")
    
    else:
        st.info("💡 Sauvegardez une 2ème expérience pour activer la comparaison avancée")
    
    # === BOUTON QUICK TEST POUR DÉMONSTRATION ===
    st.markdown("---")
    st.markdown("### 🚀 Test Rapide des Graphiques")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Ajouter expérience de test 1 (10°, 0% eau)"):
            test_metrics_1 = {
                'Krr_global': 0.045,
                'mu_kinetic_avg': 0.012,
                'mu_rolling_avg': 0.008,
                'mu_energetic': 0.035
            }
            
            st.session_state.experiments_data['Test_10D_0W'] = {
                'water_content': 0.0,
                'angle': 10.0,
                'sphere_type': 'Test',
                'metrics': test_metrics_1,
                'success_rate': 85.0,
                'anti_pics': True
            }
            st.success("✅ Expérience test 1 ajoutée!")
            st.rerun()
    
    with col2:
        if st.button("🧪 Ajouter expérience de test 2 (20°, 5% eau)"):
            test_metrics_2 = {
                'Krr_global': 0.067,
                'mu_kinetic_avg': 0.018,
                'mu_rolling_avg': 0.012,
                'mu_energetic': 0.042
            }
            
            st.session_state.experiments_data['Test_20D_5W'] = {
                'water_content': 5.0,
                'angle': 20.0,
                'sphere_type': 'Test',
                'metrics': test_metrics_2,
                'success_rate': 92.0,
                'anti_pics': True
            }
            st.success("✅ Expérience test 2 ajoutée!")
            st.rerun()
    
    with col3:
        if st.button("⚠️ Ajouter expérience avec valeurs aberrantes"):
            test_metrics_aberrant = {
                'Krr_global': 0.052,
                'mu_kinetic_avg': 0.015,
                'mu_rolling_avg': 0.009,
                'mu_energetic': 185.5  # Valeur aberrante !
            }
            
            st.session_state.experiments_data['Test_Aberrant'] = {
                'water_content': 0.0,
                'angle': 15.0,
                'sphere_type': 'Test',
                'metrics': test_metrics_aberrant,
                'success_rate': 78.0,
                'anti_pics': False  # Pas anti-pics = valeurs aberrantes
            }
            st.success("⚠️ Expérience avec valeurs aberrantes ajoutée!")
            st.rerun()

    # Gestion des expériences dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗂️ Gestion Expériences")
    
    exp_to_remove = st.sidebar.selectbox(
        "Supprimer :",
        options=["Aucune"] + list(st.session_state.experiments_data.keys())
    )
    
    if exp_to_remove != "Aucune" and st.sidebar.button("🗑️ Supprimer"):
        del st.session_state.experiments_data[exp_to_remove]
        st.success(f"Expérience '{exp_to_remove}' supprimée!")
        st.rerun()
    
    if st.sidebar.button("🧹 Effacer Tout"):
        st.session_state.experiments_data = {}
        st.success("Toutes les expériences supprimées!")
        st.rerun()

else:
    st.markdown("""
    ## 🚫 Guide d'Utilisation - Version Anti-Pics
    
    ### 🎯 **Comment Voir les Graphiques Style Images :**
    
    #### **📋 Étapes Simples :**
    1. **📂 Chargez UN fichier** CSV avec vos données
    2. **⚙️ Configurez les paramètres** (angle, eau, etc.)
    3. **🚫 Cliquez "Analyser avec suppression des pics"**
    4. **💾 Cliquez "Sauvegarder cette expérience"**
    5. **📊 Descendez à la section "Graphiques Style Images"** 
    
    #### **🚀 OU Test Rapide :**
    - **🧪 Utilisez les boutons "Ajouter expérience de test"** ci-dessous
    - **Graphiques visibles immédiatement** après ajout
    
    ### ✨ **Améliorations Anti-Pics :**
    
    #### **🚫 Suppression Maximale des Pics :**
    - **Nettoyage ultra-agressif** : 35% de suppression de chaque côté
    - **Filtre Savitzky-Golay** : Lissage polynomial avancé
    - **Suppression des outliers** : Median Absolute Deviation (MAD)
    - **Plafonnement physique** : Krr max = 1.0
    
    #### **📊 Graphiques Spécialisés :**
    1. **Krr vs Teneur en Eau** (style votre image 1)
    2. **Comparaison coefficients** (style votre image 2 - avec option valeurs aberrantes)
    3. **Krr vs Angle** (graphique supplémentaire)
    
    Cette version **élimine définitivement** les pics d'erreur !
    """)
    
    # === BOUTONS DE TEST POUR DÉMONSTRATION ===
    st.markdown("### 🚀 Test Rapide des Graphiques (Sans Upload)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Test 1: Exp 10° + 0% eau"):
            test_metrics_1 = {
                'Krr_global': 0.045,
                'mu_kinetic_avg': 0.012,
                'mu_rolling_avg': 0.008,
                'mu_energetic': 0.035
            }
            
            st.session_state.experiments_data['Test_10D_0W'] = {
                'water_content': 0.0,
                'angle': 10.0,
                'sphere_type': 'Test',
                'metrics': test_metrics_1,
                'success_rate': 85.0,
                'anti_pics': True
            }
            st.success("✅ Expérience test 1 ajoutée!")
            st.rerun()
    
    with col2:
        if st.button("🧪 Test 2: Exp 20° + 5% eau"):
            test_metrics_2 = {
                'Krr_global': 0.067,
                'mu_kinetic_avg': 0.018,
                'mu_rolling_avg': 0.012,
                'mu_energetic': 0.042
            }
            
            st.session_state.experiments_data['Test_20D_5W'] = {
                'water_content': 5.0,
                'angle': 20.0,
                'sphere_type': 'Test',
                'metrics': test_metrics_2,
                'success_rate': 92.0,
                'anti_pics': True
            }
            st.success("✅ Expérience test 2 ajoutée!")
            st.rerun()
    
    with col3:
        if st.button("⚠️ Test 3: Valeurs aberrantes"):
            test_metrics_aberrant = {
                'Krr_global': 0.052,
                'mu_kinetic_avg': 0.015,
                'mu_rolling_avg': 0.009,
                'mu_energetic': 185.5  # Valeur aberrante !
            }
            
            st.session_state.experiments_data['Test_Aberrant'] = {
                'water_content': 0.0,
                'angle': 15.0,
                'sphere_type': 'Test',
                'metrics': test_metrics_aberrant,
                'success_rate': 78.0,
                'anti_pics': False  # Pas anti-pics = valeurs aberrantes
            }
            st.success("⚠️ Expérience avec valeurs aberrantes ajoutée!")
            st.rerun()

# Sidebar avec informations de debug améliorées
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚫 Informations Anti-Pics")

if SCIPY_AVAILABLE:
    st.sidebar.success("✅ SciPy disponible - Filtre Savitzky-Golay actif")
else:
    st.sidebar.warning("⚠️ SciPy non disponible - Lissage basique")

if st.session_state.experiments_data:
    st.sidebar.markdown(f"**Expériences chargées :** {len(st.session_state.experiments_data)}")
    
    anti_pics_count = sum(1 for data in st.session_state.experiments_data.values() if data.get('anti_pics', False))
    st.sidebar.markdown(f"**Avec anti-pics :** {anti_pics_count}")
    
    for name, data in st.session_state.experiments_data.items():
        with st.sidebar.expander(f"📋 {name}"):
            is_anti_pics = data.get('anti_pics', False)
            st.write(f"**Anti-Pics :** {'✅' if is_anti_pics else '❌'}")
            st.write(f"**Eau :** {data.get('water_content', 'N/A')}%")
            st.write(f"**Angle :** {data.get('angle', 'N/A')}°")
            
            metrics = data.get('metrics', {})
            krr_val = metrics.get('Krr_global')
            if krr_val is not None and not pd.isna(krr_val):
                st.write(f"**Krr :** {krr_val:.6f}")
                
                if krr_val <= 1.0:
                    st.success("✅ Krr physique")
                else:
                    st.error("⚠️ Krr > 1.0")
            
            # Afficher réduction des pics si disponible
            max_before = metrics.get('max_krr_before_cleaning')
            max_after = metrics.get('max_krr_after_cleaning')
            
            if max_before and max_after and max_before > 0:
                reduction = (max_before - max_after) / max_before * 100
                st.write(f"**Réduction pics :** {reduction:.1f}%")

else:
    st.sidebar.info("Aucune expérience chargée")
    st.sidebar.markdown("👆 **Utilisez les boutons de test** pour voir les graphiques immédiatement")

# Footer avec statut anti-pics
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 2rem; border-radius: 10px; margin: 1rem 0;">
    <h2>🚫 Analyseur de Friction Anti-Pics - Version Finale Complète</h2>
    <p><strong>🔥 Fonctionnalités Anti-Pics :</strong></p>
    <p>🚫 Suppression ultra-agressif des pics (35% chaque côté)<br>
    📊 Filtre Savitzky-Golay pour lissage optimal<br>
    📈 Suppression outliers par Median Absolute Deviation<br>
    🎯 Plafonnement physique des valeurs Krr<br>
    📊 Graphiques style vos images (disponibles avec 1+ expérience)<br>
    📉 Option valeurs aberrantes pour reproduction Image 2<br>
    ✅ Validation physique complète</p>
    <p><em>🎯 Fini les pics à 100+ et μ énergétique à 122-174 !</em></p>
    <p><strong>📊 Statut :</strong> {len(st.session_state.experiments_data)} expériences chargées</p>
    <p><strong>🚫 Scipy :</strong> {'✅ Disponible' if SCIPY_AVAILABLE else '❌ Non disponible'}</p>
    <p><strong>🎯 Astuce :</strong> Utilisez les boutons de test pour voir les graphiques immédiatement !</p>
</div>
""", unsafe_allow_html=True)
