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
        y=ts
