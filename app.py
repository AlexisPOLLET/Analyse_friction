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
    .diagnostic-card {
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
    .friction-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
<div class="main-header">
    <h1>🔬 Analyseur Avancé de Friction</h1>
    <h2>Sphères sur Substrat Granulaire Humide</h2>
    <p><em>🔥 Analyse complète des coefficients de friction grain-sphère + trace</em></p>
</div>
""", unsafe_allow_html=True)

# Initialisation des données de session
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

def clean_data_conservative(df_valid, min_points=10):
    """Nettoyage conservateur des données pour éliminer les artefacts"""
    
    if len(df_valid) < min_points:
        return df_valid, {"error": "Pas assez de données"}
    
    # Méthode conservative (5% de chaque côté maximum)
    n_remove_start = max(1, min(3, len(df_valid) // 20))
    n_remove_end = max(1, min(3, len(df_valid) // 20))
    
    # Calculer les déplacements inter-frames
    dx = np.diff(df_valid['X_center'].values)
    dy = np.diff(df_valid['Y_center'].values)
    movement = np.sqrt(dx**2 + dy**2)
    
    # Identifier les zones de mouvement stable
    median_movement = np.median(movement)
    stable_threshold = median_movement * 0.3
    
    # Trouver le début et la fin des zones stables
    stable_mask = movement > stable_threshold
    
    if np.any(stable_mask):
        stable_indices = np.where(stable_mask)[0]
        start_stable = stable_indices[0]
        end_stable = stable_indices[-1] + 1
        
        start_idx = min(start_stable + 2, n_remove_start)
        end_idx = max(end_stable - 2, len(df_valid) - n_remove_end)
    else:
        start_idx = n_remove_start
        end_idx = len(df_valid) - n_remove_end
    
    # S'assurer qu'on garde assez de données
    if end_idx - start_idx < min_points:
        start_idx = max(0, len(df_valid)//4)
        end_idx = min(len(df_valid), len(df_valid) - len(df_valid)//4)
    
    df_cleaned = df_valid.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    
    cleaning_info = {
        "original_length": len(df_valid),
        "cleaned_length": len(df_cleaned),
        "start_removed": start_idx,
        "end_removed": len(df_valid) - end_idx,
        "percentage_kept": len(df_cleaned) / len(df_valid) * 100,
        "median_movement": median_movement
    }
    
    return df_cleaned, cleaning_info

# ==================== CALCUL KRR DE BASE ====================

def calculate_krr_robust(df_valid, fps=250, angle_deg=15.0, 
                        sphere_mass_g=10.0, sphere_radius_mm=None, 
                        pixels_per_mm=None, show_diagnostic=True):
    """Calcul robuste de Krr avec diagnostic complet"""
    
    diagnostic = {"status": "INIT", "messages": []}
    
    if show_diagnostic:
        st.markdown("### 🔧 Diagnostic de Calcul Krr")
    
    try:
        # 1. Vérification des données de base
        if len(df_valid) < 10:
            diagnostic["status"] = "ERROR"
            diagnostic["messages"].append("❌ Moins de 10 points valides")
            return None, diagnostic
        
        diagnostic["messages"].append(f"✅ Données de base: {len(df_valid)} points valides")
        
        # 2. Nettoyage des données
        df_clean, cleaning_info = clean_data_conservative(df_valid)
        
        if "error" in cleaning_info:
            diagnostic["status"] = "ERROR"
            diagnostic["messages"].append("❌ Échec du nettoyage des données")
            return None, diagnostic
        
        diagnostic["messages"].append(f"🧹 Nettoyage: {cleaning_info['cleaned_length']}/{cleaning_info['original_length']} points conservés ({cleaning_info['percentage_kept']:.1f}%)")
        
        # 3. Calibration automatique intelligente
        if pixels_per_mm is None or sphere_radius_mm is None:
            avg_radius_px = df_clean['Radius'].mean()
            
            if sphere_radius_mm is None:
                estimated_radius_mm = 15.0
                if avg_radius_px > 30:
                    estimated_radius_mm = 20.0
                elif avg_radius_px < 15:
                    estimated_radius_mm = 10.0
                sphere_radius_mm = estimated_radius_mm
            
            if pixels_per_mm is None:
                auto_calibration = avg_radius_px / sphere_radius_mm
                if 2.0 <= auto_calibration <= 15.0:
                    pixels_per_mm = auto_calibration
                    diagnostic["messages"].append(f"🎯 Calibration automatique: {pixels_per_mm:.2f} px/mm")
                else:
                    pixels_per_mm = 5.0
                    diagnostic["messages"].append(f"⚠️ Calibration automatique douteuse ({auto_calibration:.2f}), utilisation valeur par défaut: {pixels_per_mm} px/mm")
        else:
            diagnostic["messages"].append(f"📏 Calibration manuelle: {pixels_per_mm:.2f} px/mm")
        
        # 4. Conversion en unités physiques
        dt = 1 / fps
        g = 9.81
        
        x_m = df_clean['X_center'].values / pixels_per_mm / 1000
        y_m = df_clean['Y_center'].values / pixels_per_mm / 1000
        
        # Vérification du mouvement
        dx_total = abs(x_m[-1] - x_m[0]) * 1000
        dy_total = abs(y_m[-1] - y_m[0]) * 1000
        
        diagnostic["messages"].append(f"📏 Déplacement total: ΔX={dx_total:.1f}mm, ΔY={dy_total:.1f}mm")
        
        if dx_total < 5:
            diagnostic["status"] = "ERROR"
            diagnostic["messages"].append("❌ Déplacement horizontal insuffisant (<5mm)")
            return None, diagnostic
        
        # 5. Calcul des vitesses avec lissage optionnel
        window_size = min(3, len(x_m) // 5)
        if window_size >= 3:
            x_smooth = np.convolve(x_m, np.ones(window_size)/window_size, mode='same')
            y_smooth = np.convolve(y_m, np.ones(window_size)/window_size, mode='same')
            diagnostic["messages"].append(f"🔄 Lissage appliqué (fenêtre: {window_size})")
        else:
            x_smooth = x_m
            y_smooth = y_m
        
        vx = np.gradient(x_smooth, dt)
        vy = np.gradient(y_smooth, dt)
        v_magnitude = np.sqrt(vx**2 + vy**2)
        
        # 6. Calcul des vitesses initiale et finale
        n_avg = max(2, len(v_magnitude) // 6)
        
        v0 = np.mean(v_magnitude[:n_avg])
        vf = np.mean(v_magnitude[-n_avg:])
        
        diagnostic["messages"].append(f"🏃 Vitesses (moyennage sur {n_avg} points): v0={v0*1000:.2f} mm/s, vf={vf*1000:.2f} mm/s")
        
        # 7. Vérifications physiques
        if v0 <= 0:
            diagnostic["status"] = "ERROR"
            diagnostic["messages"].append("❌ Vitesse initiale nulle ou négative")
            return None, diagnostic
        
        if vf >= v0:
            diagnostic["status"] = "ERROR"
            diagnostic["messages"].append("❌ La sphère accélère au lieu de décélérer")
            diagnostic["messages"].append("   → Vérifiez la calibration ou l'angle")
            return None, diagnostic
        
        deceleration_percent = (v0 - vf) / v0 * 100
        diagnostic["messages"].append(f"📉 Décélération: {deceleration_percent:.1f}%")
        
        # 8. Calcul de la distance
        distances = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
        total_distance = np.sum(distances)
        
        diagnostic["messages"].append(f"📏 Distance parcourue: {total_distance*1000:.2f} mm")
        
        if total_distance <= 0:
            diagnostic["status"] = "ERROR"
            diagnostic["messages"].append("❌ Distance parcourue nulle")
            return None, diagnostic
        
        # 9. Calcul final de Krr
        krr = (v0**2 - vf**2) / (2 * g * total_distance)
        
        diagnostic["messages"].append(f"📊 Krr calculé: {krr:.6f}")
        
        # 10. Validation du résultat
        if krr < 0:
            diagnostic["status"] = "ERROR"
            diagnostic["messages"].append("❌ Krr négatif (impossible physiquement)")
            return None, diagnostic
        elif krr > 1.0:
            diagnostic["status"] = "WARNING"
            diagnostic["messages"].append("⚠️ Krr très élevé (>1.0) - vérifiez les paramètres")
        elif 0.03 <= krr <= 0.15:
            diagnostic["status"] = "SUCCESS"
            diagnostic["messages"].append("✅ Krr dans la gamme littérature (0.03-0.15)")
        else:
            diagnostic["status"] = "WARNING"
            diagnostic["messages"].append("⚠️ Krr hors gamme typique mais possible")
        
        # 11. Résultats complets
        results = {
            'Krr': krr,
            'v0_ms': v0,
            'vf_ms': vf,
            'v0_mms': v0 * 1000,
            'vf_mms': vf * 1000,
            'total_distance_mm': total_distance * 1000,
            'deceleration_percent': deceleration_percent,
            'dx_mm': dx_total,
            'dy_mm': dy_total,
            'calibration_px_per_mm': pixels_per_mm,
            'sphere_radius_mm': sphere_radius_mm,
            'points_original': len(df_valid),
            'points_used': len(df_clean),
            'cleaning_info': cleaning_info
        }
        
        diagnostic["messages"].append("✅ Calcul Krr terminé avec succès")
        return results, diagnostic
        
    except Exception as e:
        diagnostic["status"] = "ERROR"
        diagnostic["messages"].append(f"❌ Erreur inattendue: {str(e)}")
        return None, diagnostic

# ==================== ANALYSE DE FRICTION AVANCÉE ====================

def calculate_advanced_friction_metrics(df_valid, fps=250, angle_deg=15.0, 
                                       sphere_mass_g=10.0, sphere_radius_mm=15.0, 
                                       pixels_per_mm=5.0):
    """Calcul des métriques de friction avancées avec séries temporelles"""
    
    # Paramètres physiques
    dt = 1 / fps
    mass_kg = sphere_mass_g / 1000
    radius_m = sphere_radius_mm / 1000
    angle_rad = np.radians(angle_deg)
    g = 9.81
    
    # Nettoyage des données
    df_clean, cleaning_info = clean_data_conservative(df_valid)
    
    # Conversion en unités physiques
    x_m = df_clean['X_center'].values / pixels_per_mm / 1000
    y_m = df_clean['Y_center'].values / pixels_per_mm / 1000
    
    # Lissage
    window_size = min(3, len(x_m) // 5)
    if window_size >= 3:
        x_smooth = np.convolve(x_m, np.ones(window_size)/window_size, mode='same')
        y_smooth = np.convolve(y_m, np.ones(window_size)/window_size, mode='same')
    else:
        x_smooth = x_m
        y_smooth = y_m
    
    # Cinématique
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Accélérations
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    a_magnitude = np.sqrt(ax**2 + ay**2)
    a_tangential = np.gradient(v_magnitude, dt)
    
    # Forces
    F_gravity_tangential = mass_kg * g * np.sin(angle_rad)
    F_gravity_normal = mass_kg * g * np.cos(angle_rad)
    F_resistance = mass_kg * np.abs(a_tangential)
    F_net = mass_kg * a_tangential
    
    # Coefficients de friction temporels
    mu_kinetic = F_resistance / F_gravity_normal
    mu_rolling = mu_kinetic - np.tan(angle_rad)
    
    # Métriques énergétiques
    E_kinetic = 0.5 * mass_kg * v_magnitude**2
    P_dissipated = F_resistance * v_magnitude
    
    E_dissipated_cumul = np.cumsum(P_dissipated * dt)
    distance_cumul = np.cumsum(v_magnitude * dt)
    mu_energetic = np.where(distance_cumul > 0, 
                           E_dissipated_cumul / (F_gravity_normal * distance_cumul), 
                           0)
    
    # Krr instantané
    krr_instantaneous = np.abs(a_tangential) / (g * np.cos(angle_rad))
    
    # Métriques globales
    n_avg = max(2, len(v_magnitude) // 6)
    v0 = np.mean(v_magnitude[:n_avg])
    vf = np.mean(v_magnitude[-n_avg:])
    
    distances = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
    total_distance = np.sum(distances)
    
    if total_distance > 0 and v0 > vf:
        krr_global = (v0**2 - vf**2) / (2 * g * total_distance)
    else:
        krr_global = None
    
    # Moyennes des coefficients
    mu_kinetic_avg = np.mean(mu_kinetic)
    mu_rolling_avg = np.mean(mu_rolling)
    mu_energetic_final = mu_energetic[-1] if len(mu_energetic) > 0 else 0
    
    # Analyse statistique
    mu_kinetic_std = np.std(mu_kinetic)
    mu_rolling_std = np.std(mu_rolling)
    
    correlation_v_mu = np.corrcoef(v_magnitude, mu_kinetic)[0, 1] if len(v_magnitude) > 3 else 0
    
    time_array = np.arange(len(df_clean)) * dt
    
    if len(time_array) > 3:
        mu_kinetic_trend = np.polyfit(time_array, mu_kinetic, 1)[0]
        mu_rolling_trend = np.polyfit(time_array, mu_rolling, 1)[0]
    else:
        mu_kinetic_trend = 0
        mu_rolling_trend = 0
    
    return {
        'Krr_global': krr_global,
        'mu_kinetic_avg': mu_kinetic_avg,
        'mu_rolling_avg': mu_rolling_avg,
        'mu_energetic': mu_energetic_final,
        'mu_kinetic_std': mu_kinetic_std,
        'mu_rolling_std': mu_rolling_std,
        'mu_kinetic_trend': mu_kinetic_trend,
        'mu_rolling_trend': mu_rolling_trend,
        'correlation_velocity_friction': correlation_v_mu,
        'v0_ms': v0,
        'vf_ms': vf,
        'v0_mms': v0 * 1000,
        'vf_mms': vf * 1000,
        'total_distance_mm': total_distance * 1000,
        'time_series': {
            'time': time_array,
            'velocity_mms': v_magnitude * 1000,
            'acceleration_mms2': a_tangential * 1000,
            'mu_kinetic': mu_kinetic,
            'mu_rolling': mu_rolling,
            'mu_energetic': mu_energetic,
            'krr_instantaneous': krr_instantaneous,
            'resistance_force_mN': F_resistance * 1000,
            'power_dissipated_mW': P_dissipated * 1000,
            'energy_dissipated_cumul_mJ': E_dissipated_cumul * 1000,
            'normal_force_mN': np.full_like(time_array, F_gravity_normal * 1000),
            'tangential_force_mN': np.full_like(time_array, F_gravity_tangential * 1000)
        },
        'cleaning_info': cleaning_info
    }

def create_friction_summary_cards(metrics):
    """Crée les cartes de résumé des métriques de friction"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mu_kinetic_val = safe_format_value(metrics.get('mu_kinetic_avg'), "{:.4f}")
        st.markdown(f"""
        <div class="friction-card">
            <h3>🔥 μ Cinétique</h3>
            <h2>{mu_kinetic_val}</h2>
            <p>Friction directe grain-sphère</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        mu_rolling_val = safe_format_value(metrics.get('mu_rolling_avg'), "{:.4f}")
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4834d4 0%, #686de0 100%);">
            <h3>🎯 μ Rolling</h3>
            <h2>{mu_rolling_val}</h2>
            <p>Résistance pure au roulement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        mu_energetic_val = safe_format_value(metrics.get('mu_energetic'), "{:.4f}")
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);">
            <h3>⚡ μ Énergétique</h3>
            <h2>{mu_energetic_val}</h2>
            <p>Basé sur dissipation énergie</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        krr_val = safe_format_value(metrics.get('Krr_global'), "{:.6f}")
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);">
            <h3>📊 Krr Référence</h3>
            <h2>{krr_val}</h2>
            <p>Coefficient traditionnel</p>
        </div>
        """, unsafe_allow_html=True)

def create_advanced_friction_plots(metrics, experiment_name="Expérience"):
    """Crée les graphiques avancés de friction"""
    
    if 'time_series' not in metrics:
        st.error("Pas de données temporelles disponibles")
        return
    
    ts = metrics['time_series']
    
    # === GRAPHIQUE 1: COEFFICIENTS DE FRICTION VS TEMPS ===
    st.markdown("#### 🔥 Coefficients de Friction vs Temps")
    
    fig_friction_time = go.Figure()
    
    # μ Cinétique
    fig_friction_time.add_trace(go.Scatter(
        x=ts['time'], 
        y=ts['mu_kinetic'],
        mode='lines',
        name='μ Cinétique',
        line=dict(color='red', width=2),
        hovertemplate='Temps: %{x:.3f}s<br>μ Cinétique: %{y:.4f}<extra></extra>'
    ))
    
    # μ Rolling
    fig_friction_time.add_trace(go.Scatter(
        x=ts['time'], 
        y=ts['mu_rolling'],
        mode='lines',
        name='μ Rolling',
        line=dict(color='blue', width=2),
        hovertemplate='Temps: %{x:.3f}s<br>μ Rolling: %{y:.4f}<extra></extra>'
    ))
    
    # μ Énergétique
    fig_friction_time.add_trace(go.Scatter(
        x=ts['time'], 
        y=ts['mu_energetic'],
        mode='lines',
        name='μ Énergétique',
        line=dict(color='purple', width=2),
        hovertemplate='Temps: %{x:.3f}s<br>μ Énergétique: %{y:.4f}<extra></extra>'
    ))
    
    # Krr instantané
    fig_friction_time.add_trace(go.Scatter(
        x=ts['time'], 
        y=ts['krr_instantaneous'],
        mode='lines',
        name='Krr Instantané',
        line=dict(color='orange', width=1, dash='dash'),
        hovertemplate='Temps: %{x:.3f}s<br>Krr: %{y:.4f}<extra></extra>'
    ))
    
    fig_friction_time.update_layout(
        title=f"Évolution des Coefficients de Friction - {experiment_name}",
        xaxis_title="Temps (s)",
        yaxis_title="Coefficient de Friction",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_friction_time, use_container_width=True)
    
    # === GRAPHIQUE 2: ANALYSE FORCES ===
    st.markdown("#### ⚖️ Analyse des Forces")
    
    fig_forces = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Forces vs Temps', 'Puissance Dissipée', 
                       'Énergie Dissipée Cumulée', 'Corrélation Vitesse-Friction'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Forces
    fig_forces.add_trace(
        go.Scatter(x=ts['time'], y=ts['resistance_force_mN'], 
                  mode='lines', name='F Résistance', line=dict(color='red')),
        row=1, col=1
    )
    fig_forces.add_trace(
        go.Scatter(x=ts['time'], y=ts['normal_force_mN'], 
                  mode='lines', name='F Normale', line=dict(color='blue', dash='dash')),
        row=1, col=1
    )
    fig_forces.add_trace(
        go.Scatter(x=ts['time'], y=ts['tangential_force_mN'], 
                  mode='lines', name='F Tangentielle', line=dict(color='green', dash='dash')),
        row=1, col=1
    )
    
    # Puissance dissipée
    fig_forces.add_trace(
        go.Scatter(x=ts['time'], y=ts['power_dissipated_mW'], 
                  mode='lines', name='Puissance', line=dict(color='purple')),
        row=1, col=2
    )
    
    # Énergie dissipée cumulée
    fig_forces.add_trace(
        go.Scatter(x=ts['time'], y=ts['energy_dissipated_cumul_mJ'], 
                  mode='lines', name='Énergie', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Corrélation vitesse-friction
    fig_forces.add_trace(
        go.Scatter(x=ts['velocity_mms'], y=ts['mu_kinetic'], 
                  mode='markers', name='V vs μ', marker=dict(color='red', size=4)),
        row=2, col=2
    )
    
    fig_forces.update_xaxes(title_text="Temps (s)", row=1, col=1)
    fig_forces.update_xaxes(title_text="Temps (s)", row=1, col=2)
    fig_forces.update_xaxes(title_text="Temps (s)", row=2, col=1)
    fig_forces.update_xaxes(title_text="Vitesse (mm/s)", row=2, col=2)
    
    fig_forces.update_yaxes(title_text="Force (mN)", row=1, col=1)
    fig_forces.update_yaxes(title_text="Puissance (mW)", row=1, col=2)
    fig_forces.update_yaxes(title_text="Énergie (mJ)", row=2, col=1)
    fig_forces.update_yaxes(title_text="μ Cinétique", row=2, col=2)
    
    fig_forces.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_forces, use_container_width=True)
    
    # === GRAPHIQUE 3: HISTOGRAMMES DES COEFFICIENTS ===
    st.markdown("#### 📊 Distribution des Coefficients de Friction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_hist_kinetic = px.histogram(
            x=ts['mu_kinetic'], 
            nbins=20,
            title="Distribution μ Cinétique",
            labels={'x': 'μ Cinétique', 'y': 'Fréquence'}
        )
        fig_hist_kinetic.update_layout(height=300)
        st.plotly_chart(fig_hist_kinetic, use_container_width=True)
    
    with col2:
        fig_hist_rolling = px.histogram(
            x=ts['mu_rolling'],
            nbins=20,
            title="Distribution μ Rolling",
            labels={'x': 'μ Rolling', 'y': 'Fréquence'}
        )
        fig_hist_rolling.update_layout(height=300)
        st.plotly_chart(fig_hist_rolling, use_container_width=True)
    
    with col3:
        fig_hist_energetic = px.histogram(
            x=ts['mu_energetic'],
            nbins=20,
            title="Distribution μ Énergétique",
            labels={'x': 'μ Énergétique', 'y': 'Fréquence'}
        )
        fig_hist_energetic.update_layout(height=300)
        st.plotly_chart(fig_hist_energetic, use_container_width=True)

def display_diagnostic_messages(diagnostic):
    """Affiche les messages de diagnostic avec formatage approprié"""
    
    if not diagnostic or "messages" not in diagnostic:
        return
    
    status = diagnostic.get("status", "UNKNOWN")
    messages = diagnostic.get("messages", [])
    
    if status == "SUCCESS":
        card_class = "diagnostic-card"
    elif status == "WARNING":
        card_class = "warning-card"
    elif status == "ERROR":
        card_class = "error-card"
    else:
        card_class = "metric-card"
    
    with st.expander(f"🔍 Messages de diagnostic ({len(messages)} messages)", expanded=(status == "ERROR")):
        for message in messages:
            st.markdown(f"""
            <div class="{card_class}" style="margin: 0.2rem 0; padding: 0.5rem;">
                {message}
            </div>
            """, unsafe_allow_html=True)

def create_comparison_analysis(experiments_data):
    """Analyse comparative de plusieurs expériences"""
    
    if len(experiments_data) < 2:
        st.warning("Au moins 2 expériences sont nécessaires pour la comparaison")
        return
    
    st.markdown("### 🔄 Analyse Comparative Multi-Expériences")
    
    # Préparer les données de comparaison
    comparison_data = []
    
    for exp_name, exp_data in experiments_data.items():
        metrics = exp_data.get('friction_metrics', {})
        basic_metrics = exp_data.get('basic_metrics', {})
        
        comparison_data.append({
            'Expérience': exp_name,
            'Teneur_eau': exp_data.get('water_content', 0),
            'Angle': exp_data.get('angle', 15),
            'Type_sphère': exp_data.get('sphere_type', 'Acier'),
            'Krr': metrics.get('Krr_global', basic_metrics.get('Krr')),
            'μ_cinétique': metrics.get('mu_kinetic_avg'),
            'μ_rolling': metrics.get('mu_rolling_avg'),
            'μ_énergétique': metrics.get('mu_energetic'),
            'v0_mms': metrics.get('v0_mms', basic_metrics.get('v0_mms')),
            'vf_mms': metrics.get('vf_mms', basic_metrics.get('vf_mms')),
            'distance_mm': metrics.get('total_distance_mm', basic_metrics.get('total_distance_mm')),
            'corrélation_v_μ': metrics.get('correlation_velocity_friction'),
            'μ_cinétique_std': metrics.get('mu_kinetic_std'),
            'μ_trend': metrics.get('mu_kinetic_trend')
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Tableau de comparaison
    st.markdown("#### 📋 Tableau de Comparaison Détaillé")
    st.dataframe(comp_df, use_container_width=True)
    
    # Graphiques de comparaison
    col1, col2 = st.columns(2)
    
    with col1:
        if comp_df['Krr'].notna().any():
            fig_krr_water = px.scatter(
                comp_df, 
                x='Teneur_eau', 
                y='Krr',
                color='Type_sphère',
                size='distance_mm',
                hover_data=['Expérience'],
                title="🔥 Krr vs Teneur en Eau",
                labels={'Teneur_eau': 'Teneur en eau (%)', 'Krr': 'Coefficient Krr'}
            )
            st.plotly_chart(fig_krr_water, use_container_width=True)
    
    with col2:
        if comp_df['μ_cinétique'].notna().any():
            fig_mu_comparison = px.bar(
                comp_df,
                x='Expérience',
                y=['μ_cinétique', 'μ_rolling', 'μ_énergétique'],
                title="🎯 Comparaison des Coefficients μ",
                labels={'value': 'Coefficient μ', 'variable': 'Type de μ'}
            )
            fig_mu_comparison.update_xaxes(tickangle=45)
            st.plotly_chart(fig_mu_comparison, use_container_width=True)
    
    # Analyse de corrélation
    st.markdown("#### 🔗 Analyse de Corrélations")
    
    numeric_cols = ['Teneur_eau', 'Angle', 'Krr', 'μ_cinétique', 'μ_rolling', 'μ_énergétique', 'v0_mms', 'vf_mms']
    available_cols = [col for col in numeric_cols if col in comp_df.columns and comp_df[col].notna().any()]
    
    if len(available_cols) >= 3:
        corr_matrix = comp_df[available_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="🔗 Matrice de Corrélation - Paramètres de Friction",
            color_continuous_scale="RdBu_r"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Insights automatiques
        st.markdown("#### 🧠 Insights Automatiques")
        
        # Corrélation humidité-friction
        if 'Teneur_eau' in comp_df.columns and 'μ_cinétique' in comp_df.columns:
            corr_humidity_friction = comp_df[['Teneur_eau', 'μ_cinétique']].corr().iloc[0, 1]
            if not pd.isna(corr_humidity_friction):
                if corr_humidity_friction > 0.5:
                    st.success(f"✅ **Corrélation forte positive** entre humidité et friction cinétique (r = {corr_humidity_friction:.3f})")
                elif corr_humidity_friction < -0.5:
                    st.warning(f"⚠️ **Corrélation forte négative** entre humidité et friction cinétique (r = {corr_humidity_friction:.3f})")
                else:
                    st.info(f"ℹ️ **Corrélation modérée** entre humidité et friction cinétique (r = {corr_humidity_friction:.3f})")
        
        # Stabilité des coefficients
        if 'μ_cinétique_std' in comp_df.columns:
            avg_std = comp_df['μ_cinétique_std'].mean()
            if avg_std < 0.01:
                st.success("✅ **Coefficients de friction très stables** (faible variabilité temporelle)")
            elif avg_std > 0.05:
                st.warning("⚠️ **Forte variabilité temporelle** des coefficients de friction")
            else:
                st.info("ℹ️ **Variabilité modérée** des coefficients de friction")
    
    # Export des résultats
    st.markdown("#### 💾 Export des Résultats")
    
    csv_comparison = comp_df.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger la comparaison (CSV)",
        data=csv_comparison,
        file_name="comparaison_friction_complete.csv",
        mime="text/csv"
    )

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
            x = 1200 - (frame - 5) * 10 + np.random.normal(0, 2)
            y = 650 + (frame - 5) * 0.8 + np.random.normal(0, 2)
            radius = 22 + np.random.normal(0, 2)
            radius = max(18, min(28, radius))
            data.append([frame, max(0, x), max(0, y), max(0, radius)])
    
    return pd.DataFrame(data, columns=['Frame', 'X_center', 'Y_center', 'Radius'])

# ==================== INTERFACE PRINCIPALE ====================

# Barre latérale de navigation
st.sidebar.title("🧭 Navigation")
analysis_mode = st.sidebar.selectbox(
    "Mode d'analyse:",
    ["🔬 Analyse Simple", "🔥 Analyse Friction Avancée", "🔄 Comparaison Multi-Expériences", "📚 Données d'Exemple"]
)

# ==================== MODE ANALYSE SIMPLE ====================
if analysis_mode == "🔬 Analyse Simple":
    
    st.markdown("## 🔬 Analyse Simple - Calcul Krr de Base")
    
    # Paramètres d'expérience
    st.markdown("### ⚙️ Paramètres de l'Expérience")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        experiment_name = st.text_input("Nom de l'expérience", value="Exp_Simple")
        water_content = st.number_input("Teneur en eau (%)", value=0.0, min_value=0.0, max_value=30.0)
        sphere_type = st.selectbox("Type de sphère", ["Acier", "Plastique", "Verre"])
    
    with col2:
        fps = st.number_input("FPS caméra", value=250.0, min_value=30.0, max_value=1000.0)
        angle_deg = st.number_input("Angle d'inclinaison (°)", value=15.0, min_value=1.0, max_value=45.0)
        sphere_mass_g = st.number_input("Masse sphère (g)", value=10.0, min_value=0.1, max_value=100.0)
    
    with col3:
        sphere_radius_mm = st.number_input("Rayon sphère (mm)", value=15.0, min_value=5.0, max_value=50.0)
        pixels_per_mm = st.number_input("Calibration (px/mm)", value=0.0, min_value=0.0, help="0 = auto")
        
        if pixels_per_mm == 0.0:
            pixels_per_mm = None
            st.info("🎯 Calibration automatique activée")
    
    # Upload de fichier
    st.markdown("### 📂 Chargement des Données")
    
    uploaded_file = st.file_uploader(
        "Fichier CSV avec données de détection", 
        type=['csv'],
        help="Format requis: Frame, X_center, Y_center, Radius"
    )
    
    df = None
    df_valid = None
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
            if not all(col in df.columns for col in required_columns):
                st.error(f"❌ Colonnes requises: {required_columns}")
                st.error(f"📊 Colonnes trouvées: {list(df.columns)}")
            else:
                df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                st.success(f"✅ Fichier chargé: {len(df)} frames, {len(df_valid)} détections valides")
                
        except Exception as e:
            st.error(f"❌ Erreur de lecture: {str(e)}")
    
    else:
        if st.button("🔬 Utiliser données d'exemple"):
            df = create_sample_data()
            df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
            st.info("📊 Données d'exemple chargées")
    
    # Analyse si données disponibles
    if df is not None and len(df_valid) > 0:
        
        # Aperçu des données
        st.markdown("### 📊 Aperçu des Données")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Frames total", len(df))
        with col2:
            st.metric("Détections valides", len(df_valid))
        with col3:
            success_rate = len(df_valid) / len(df) * 100
            st.metric("Taux de succès", f"{success_rate:.1f}%")
        with col4:
            avg_radius = df_valid['Radius'].mean()
            st.metric("Rayon moyen", f"{avg_radius:.1f} px")
        
        # Calcul Krr
        if st.button("🚀 Calculer Krr", type="primary"):
            
            results, diagnostic = calculate_krr_robust(
                df_valid, 
                fps=fps, 
                angle_deg=angle_deg,
                sphere_mass_g=sphere_mass_g,
                sphere_radius_mm=sphere_radius_mm,
                pixels_per_mm=pixels_per_mm,
                show_diagnostic=True
            )
            
            display_diagnostic_messages(diagnostic)
            
            if results is not None:
                
                # Résultats principaux
                st.markdown("### 🎯 Résultats Principaux")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    krr_val = safe_format_value(results['Krr'], "{:.6f}")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Coefficient Krr</h3>
                        <h2>{krr_val}</h2>
                        <p>Résistance au roulement</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    v0_val = safe_format_value(results['v0_mms'], "{:.2f}")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🏃 Vitesse Initiale</h3>
                        <h2>{v0_val} mm/s</h2>
                        <p>Début de trajectoire</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    vf_val = safe_format_value(results['vf_mms'], "{:.2f}")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Vitesse Finale</h3>
                        <h2>{vf_val} mm/s</h2>
                        <p>Fin de trajectoire</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    dist_val = safe_format_value(results['total_distance_mm'], "{:.2f}")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📏 Distance</h3>
                        <h2>{dist_val} mm</h2>
                        <p>Distance parcourue</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Sauvegarde pour comparaison
                if st.button("💾 Sauvegarder pour comparaison"):
                    st.session_state.experiments_data[experiment_name] = {
                        'basic_metrics': results,
                        'water_content': water_content,
                        'angle': angle_deg,
                        'sphere_type': sphere_type,
                        'sphere_mass_g': sphere_mass_g,
                        'sphere_radius_mm': sphere_radius_mm,
                        'fps': fps
                    }
                    st.success(f"✅ Expérience '{experiment_name}' sauvegardée!")
                
                # Export des résultats
                results_df = pd.DataFrame([{
                    'Parametre': k,
                    'Valeur': v
                } for k, v in results.items() if isinstance(v, (int, float))])
                
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger résultats (CSV)",
                    data=csv_results,
                    file_name=f"resultats_{experiment_name}.csv",
                    mime="text/csv"
                )

# ==================== MODE ANALYSE FRICTION AVANCÉE ====================
elif analysis_mode == "🔥 Analyse Friction Avancée":
    
    st.markdown("## 🔥 Analyse Avancée des Coefficients de Friction")
    st.markdown("*Analyse complète des mécanismes de friction grain-sphère avec séries temporelles*")
    
    # Paramètres d'expérience
    st.markdown("### ⚙️ Paramètres de l'Expérience")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        experiment_name = st.text_input("Nom de l'expérience", value="Exp_Friction")
        water_content = st.number_input("Teneur en eau (%)", value=5.0, min_value=0.0, max_value=30.0)
        sphere_type = st.selectbox("Type de sphère", ["Acier", "Plastique", "Verre"], key="friction_sphere")
    
    with col2:
        fps = st.number_input("FPS caméra", value=250.0, min_value=30.0, max_value=1000.0, key="friction_fps")
        angle_deg = st.number_input("Angle d'inclinaison (°)", value=15.0, min_value=1.0, max_value=45.0, key="friction_angle")
        sphere_mass_g = st.number_input("Masse sphère (g)", value=10.0, min_value=0.1, max_value=100.0, key="friction_mass")
    
    with col3:
        sphere_radius_mm = st.number_input("Rayon sphère (mm)", value=15.0, min_value=5.0, max_value=50.0, key="friction_radius")
        pixels_per_mm = st.number_input("Calibration (px/mm)", value=5.0, min_value=1.0, max_value=20.0, key="friction_calib")
        advanced_smoothing = st.checkbox("Lissage avancé", value=True)
    
    # Upload de fichier
    st.markdown("### 📂 Chargement des Données")
    
    uploaded_file_friction = st.file_uploader(
        "Fichier CSV avec données de détection", 
        type=['csv'],
        help="Format requis: Frame, X_center, Y_center, Radius",
        key="friction_upload"
    )
    
    df_friction = None
    df_valid_friction = None
    
    if uploaded_file_friction is not None:
        try:
            df_friction = pd.read_csv(uploaded_file_friction)
            
            required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
            if not all(col in df_friction.columns for col in required_columns):
                st.error(f"❌ Colonnes requises: {required_columns}")
            else:
                df_valid_friction = df_friction[(df_friction['X_center'] != 0) & 
                                               (df_friction['Y_center'] != 0) & 
                                               (df_friction['Radius'] != 0)]
                st.success(f"✅ Fichier chargé: {len(df_friction)} frames, {len(df_valid_friction)} détections valides")
                
        except Exception as e:
            st.error(f"❌ Erreur de lecture: {str(e)}")
    
    else:
        if st.button("🔬 Utiliser données d'exemple (friction)", key="friction_sample"):
            df_friction = create_sample_data()
            df_valid_friction = df_friction[(df_friction['X_center'] != 0) & 
                                          (df_friction['Y_center'] != 0) & 
                                          (df_friction['Radius'] != 0)]
            st.info("📊 Données d'exemple chargées")
    
    # Analyse si données disponibles
    if df_friction is not None and len(df_valid_friction) > 0:
        
        if st.button("🔥 Lancer Analyse Friction Complète", type="primary"):
            
            with st.spinner("🔄 Calcul des métriques de friction avancées..."):
                
                friction_metrics = calculate_advanced_friction_metrics(
                    df_valid_friction,
                    fps=fps,
                    angle_deg=angle_deg,
                    sphere_mass_g=sphere_mass_g,
                    sphere_radius_mm=sphere_radius_mm,
                    pixels_per_mm=pixels_per_mm
                )
                
                # Cartes de résumé
                st.markdown("### 🔥 Résumé des Coefficients de Friction")
                create_friction_summary_cards(friction_metrics)
                
                # Graphiques avancés
                st.markdown("### 📈 Analyses Temporelles Avancées")
                create_advanced_friction_plots(friction_metrics, experiment_name)
                
                # Métriques statistiques détaillées
                st.markdown("### 📊 Statistiques Détaillées")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🔥 Friction Cinétique**")
                    st.metric("Moyenne", safe_format_value(friction_metrics.get('mu_kinetic_avg'), "{:.4f}"))
                    st.metric("Écart-type", safe_format_value(friction_metrics.get('mu_kinetic_std'), "{:.4f}"))
                    st.metric("Tendance", safe_format_value(friction_metrics.get('mu_kinetic_trend'), "{:.6f}"))
                
                with col2:
                    st.markdown("**🎯 Friction Rolling**")
                    st.metric("Moyenne", safe_format_value(friction_metrics.get('mu_rolling_avg'), "{:.4f}"))
                    st.metric("Écart-type", safe_format_value(friction_metrics.get('mu_rolling_std'), "{:.4f}"))
                    st.metric("Tendance", safe_format_value(friction_metrics.get('mu_rolling_trend'), "{:.6f}"))
                
                with col3:
                    st.markdown("**⚡ Métriques Énergétiques**")
                    st.metric("μ Énergétique", safe_format_value(friction_metrics.get('mu_energetic'), "{:.4f}"))
                    st.metric("Corrélation V-μ", safe_format_value(friction_metrics.get('correlation_velocity_friction'), "{:.3f}"))
                    st.metric("Krr Global", safe_format_value(friction_metrics.get('Krr_global'), "{:.6f}"))
                
                # Sauvegarde pour comparaison
                if st.button("💾 Sauvegarder analyse friction", key="save_friction"):
                    st.session_state.experiments_data[experiment_name] = {
                        'friction_metrics': friction_metrics,
                        'water_content': water_content,
                        'angle': angle_deg,
                        'sphere_type': sphere_type,
                        'sphere_mass_g': sphere_mass_g,
                        'sphere_radius_mm': sphere_radius_mm,
                        'fps': fps
                    }
                    st.success(f"✅ Analyse friction '{experiment_name}' sauvegardée!")
                
                # Export détaillé
                if 'time_series' in friction_metrics:
                    ts = friction_metrics['time_series']
                    detailed_df = pd.DataFrame({
                        'temps_s': ts['time'],
                        'vitesse_mms': ts['velocity_mms'],
                        'acceleration_mms2': ts['acceleration_mms2'],
                        'mu_cinétique': ts['mu_kinetic'],
                        'mu_rolling': ts['mu_rolling'],
                        'mu_énergétique': ts['mu_energetic'],
                        'krr_instantané': ts['krr_instantaneous'],
                        'force_résistance_mN': ts['resistance_force_mN'],
                        'puissance_dissipée_mW': ts['power_dissipated_mW']
                    })
                    
                    csv_detailed = detailed_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger analyse complète (CSV)",
                        data=csv_detailed,
                        file_name=f"analyse_friction_{experiment_name}.csv",
                        mime="text/csv"
                    )

# ==================== MODE COMPARAISON MULTI-EXPÉRIENCES ====================
elif analysis_mode == "🔄 Comparaison Multi-Expériences":
    
    st.markdown("## 🔄 Comparaison Multi-Expériences")
    
    if not st.session_state.experiments_data:
        st.warning("⚠️ Aucune expérience sauvegardée. Veuillez d'abord analyser des expériences dans les autres modes.")
        
        # Option pour charger des données d'exemple
        if st.button("📚 Charger expériences d'exemple"):
            # Créer des données d'exemple pour la comparaison
            sample_experiments = {
                "5D-5w-1": {
                    'basic_metrics': {'Krr': 0.054, 'v0_mms': 120, 'vf_mms': 85, 'total_distance_mm': 95},
                    'water_content': 5.0,
                    'angle': 5.0,
                    'sphere_type': 'Acier'
                },
                "10D-5w-2": {
                    'basic_metrics': {'Krr': 0.062, 'v0_mms': 115, 'vf_mms': 78, 'total_distance_mm': 88},
                    'water_content': 10.0,
                    'angle': 5.0,
                    'sphere_type': 'Acier'
                },
                "15D-5w-3": {
                    'basic_metrics': {'Krr': 0.071, 'v0_mms': 108, 'vf_mms': 69, 'total_distance_mm': 82},
                    'water_content': 15.0,
                    'angle': 5.0,
                    'sphere_type': 'Acier'
                }
            }
            
            st.session_state.experiments_data.update(sample_experiments)
            st.success("✅ Expériences d'exemple chargées!")
            st.rerun()
    
    else:
        # Afficher les expériences disponibles
        st.markdown("### 📋 Expériences Disponibles")
        
        exp_summary = []
        for name, data in st.session_state.experiments_data.items():
            exp_summary.append({
                'Nom': name,
                'Teneur eau (%)': data.get('water_content', 'N/A'),
                'Angle (°)': data.get('angle', 'N/A'),
                'Type sphère': data.get('sphere_type', 'N/A'),
                'Type analyse': 'Friction' if 'friction_metrics' in data else 'Simple'
            })
        
        summary_df = pd.DataFrame(exp_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        # Sélection des expériences à comparer
        selected_experiments = st.multiselect(
            "Sélectionnez les expériences à comparer:",
            options=list(st.session_state.experiments_data.keys()),
            default=list(st.session_state.experiments_data.keys())
        )
        
        if len(selected_experiments) >= 2:
            filtered_data = {k: v for k, v in st.session_state.experiments_data.items() if k in selected_experiments}
            create_comparison_analysis(filtered_data)
        else:
            st.info("Sélectionnez au moins 2 expériences pour la comparaison")
        
        # Gestion des expériences
        st.markdown("### 🗂️ Gestion des Expériences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Supprimer une expérience:**")
            exp_to_remove = st.selectbox(
                "Sélectionner:",
                options=["Aucune"] + list(st.session_state.experiments_data.keys())
            )
            
            if exp_to_remove != "Aucune" and st.button("🗑️ Supprimer"):
                del st.session_state.experiments_data[exp_to_remove]
                st.success(f"✅ Expérience '{exp_to_remove}' supprimée!")
                st.rerun()
        
        with col2:
            st.markdown("**Tout effacer:**")
            st.write("⚠️ Supprimera toutes les expériences sauvegardées")
            if st.button("🧹 Tout effacer"):
                st.session_state.experiments_data = {}
                st.success("✅ Toutes les expériences supprimées!")
                st.rerun()

# ==================== MODE DONNÉES D'EXEMPLE ====================
elif analysis_mode == "📚 Données d'Exemple":
    
    st.markdown("## 📚 Données d'Exemple et Documentation")
    
    st.markdown("""
    ### 🎯 Objectif du Projet
    
    Cette application analyse la **résistance au roulement de sphères sur substrat granulaire humide**.
    
    **Innovation:** Première étude systématique des effets de l'humidité sur la friction granulaire.
    
    **Applications:**
    - Géotechnique (fondations sur sols humides)
    - Transport sédimentaire
    - Exploration planétaire
    - Agriculture et mining
    """)
    
    # Générer et afficher données d'exemple
    st.markdown("### 📊 Exemple de Données de Détection")
    
    if st.button("🔬 Générer données d'exemple"):
        sample_data = create_sample_data()
        
        st.markdown("**Format des données d'entrée:**")
        st.dataframe(sample_data.head(10))
        
        st.markdown("**Description des colonnes:**")
        col_desc = pd.DataFrame({
            'Colonne': ['Frame', 'X_center', 'Y_center', 'Radius'],
            'Description': [
                'Numéro de frame (image)',
                'Position X du centre de la sphère (pixels)',
                'Position Y du centre de la sphère (pixels)',
                'Rayon détecté de la sphère (pixels)'
            ],
            'Unité': ['#', 'px', 'px', 'px']
        })
        st.dataframe(col_desc)
        
        # Téléchargement des données d'exemple
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger données d'exemple (CSV)",
            data=csv_sample,
            file_name="donnees_exemple_friction.csv",
            mime="text/csv"
        )
    
    st.markdown("### 🔬 Types d'Analyses Disponibles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔬 Analyse Simple:**
        - Calcul du coefficient Krr de base
        - Vitesses initiale et finale
        - Distance parcourue
        - Diagnostic automatique
        
        **Formule Krr:** `(v₀² - vf²) / (2gL)`
        """)
    
    with col2:
        st.markdown("""
        **🔥 Analyse Friction Avancée:**
        - Coefficients μ cinétique, rolling, énergétique
        - Séries temporelles complètes
        - Analyse des forces et puissances
        - Corrélations vitesse-friction
        
        **Innovation:** Séparation des mécanismes de friction
        """)
    
    st.markdown("### 📈 Métriques Calculées")
    
    metrics_table = pd.DataFrame({
        'Métrique': [
            'Krr', 'μ Cinétique', 'μ Rolling', 'μ Énergétique', 
            'Force Résistance', 'Puissance Dissipée', 'Corrélation V-μ'
        ],
        'Description': [
            'Coefficient de résistance au roulement traditionnel',
            'Friction directe grain-sphère (force/normale)',
            'Friction pure de roulement (cinétique - pente)',
            'Basé sur dissipation énergétique cumulée',
            'Force de résistance instantanée',
            'Puissance dissipée par friction',
            'Corrélation entre vitesse et friction'
        ],
        'Unité': ['[-]', '[-]', '[-]', '[-]', '[mN]', '[mW]', '[-]'],
        'Gamme Typique': [
            '0.03-0.15', '0.1-0.8', '0.05-0.5', '0.02-0.3',
            '1-50', '0.1-10', '-1 à +1'
        ]
    })
    
    st.dataframe(metrics_table, use_container_width=True)
    
    st.markdown("### 🔍 Effets de l'Humidité (Hypothèses)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 Humidité Faible (0-8%):**
        - Ponts capillaires isolés
        - Augmentation modérée de la friction
        - Effet proportionnel à w%
        """)
        
        st.markdown("""
        **🎯 Humidité Optimale (8-15%):**
        - Réseau de ponts capillaires
        - Maximum de résistance
        - Percolation des forces cohésives
        """)
    
    with col2:
        st.markdown("""
        **💧 Humidité Élevée (>15%):**
        - Films d'eau continus
        - Effet de lubrification
        - Stabilisation ou diminution
        """)
        
        st.markdown("""
        **🔬 Modèle Empirique:**
        `Krr(w) = Krr_sec × [1 + αw + βw²]`
        
        α > 0 (effet capillaire)
        β < 0 (saturation/lubrification)
        """)
    
    st.markdown("### 📚 Références Scientifiques")
    
    references = pd.DataFrame({
        'Auteur': [
            'Van Wal et al. (2017)',
            'De Blasio & Sæter (2009)',
            'Darbois Texier et al. (2018)',
            'Cette étude (2024)'
        ],
        'Contribution': [
            'Modèle micro-collision, régime no-plowing',
            'Régime plowing, sphères petites',
            'Lois d\'échelle δ/R ∝ (ρs/ρg)^0.75',
            'Premier effet humidité systématique'
        ],
        'Krr Typique': [
            '0.052-0.066',
            '0.45-0.65',
            'Variables',
            '0.054-0.084'
        ],
        'Conditions': [
            'Sphères grandes, gravier sec',
            'Sphères petites, sable sec',
            'Diverses tailles et densités',
            'Sphères moyennes, sable humide'
        ]
    })
    
    st.dataframe(references, use_container_width=True)
    
    st.markdown("### 🛠️ Guide d'Utilisation")
    
    with st.expander("📋 Instructions détaillées"):
        st.markdown("""
        **1. Préparation des données:**
        - Format CSV requis: Frame, X_center, Y_center, Radius
        - Données de détection par computer vision
        - Éliminer les frames sans détection (valeurs = 0)
        
        **2. Paramètres expérimentaux:**
        - **FPS:** Fréquence d'acquisition (typique: 250 fps)
        - **Angle:** Inclinaison du plan (5-45°)
        - **Masse/Rayon:** Propriétés physiques de la sphère
        - **Calibration:** pixels/mm (0 = automatique)
        
        **3. Choix du mode d'analyse:**
        - **Simple:** Krr de base, diagnostic rapide
        - **Avancé:** Analyse friction complète, séries temporelles
        - **Comparaison:** Analyse multi-expériences
        
        **4. Interprétation:**
        - Krr 0.03-0.15: gamme littérature
        - μ > 0.5: friction élevée
        - Corrélation V-μ: dépendance vitesse
        - Tendances temporelles: stabilité
        
        **5. Export et sauvegarde:**
        - CSV détaillé pour analyse externe
        - Sauvegarde pour comparaison
        - Graphiques interactifs
        """)
    
    st.markdown("### 🚀 Perspectives")
    
    st.markdown("""
    **Extensions possibles:**
    - Autres matériaux granulaires
    - Effets de température
    - Validation terrain
    - Applications industrielles
    
    **Impact scientifique:**
    - Premier modèle humidité-friction granulaire
    - Applications géotechniques directes
    - Fondements pour exploration spatiale
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
### 🔬 Analyseur de Friction - Substrat Granulaire Humide
*Développé pour l'analyse quantitative des effets d'humidité sur la résistance au roulement*

**Institution:** Département Cosmic Earth Science, Université d'Osaka  
**Innovation:** Première étude systématique humidité-friction granulaire  
**Applications:** Géotechnique, transport sédimentaire, exploration planétaire
""")

# Sidebar avec informations
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Statistiques Projet")
st.sidebar.markdown(f"""
- **Expériences sauvegardées:** {len(st.session_state.experiments_data)}
- **Types d'analyse:** 2 (Simple + Avancée)
- **Métriques calculées:** 15+
- **Format export:** CSV interactif
""")

if st.session_state.experiments_data:
    st.sidebar.markdown("### 💾 Expériences Récentes")
    for exp_name in list(st.session_state.experiments_data.keys())[-3:]:
        exp_data = st.session_state.experiments_data[exp_name]
        st.sidebar.markdown(f"""
        **{exp_name}**
        - Eau: {exp_data.get('water_content', 'N/A')}%
        - Type: {exp_data.get('sphere_type', 'N/A')}
        """)

st.sidebar.markdown("""
### 🎓 Contexte Recherche
**Domaine:** Mécanique granulaire  
**Innovation:** Effets d'humidité  
**Impact:** Applications ingénierie  
**Méthode:** Computer vision + physique
""")

# Aide rapide
with st.sidebar.expander("❓ Aide Rapide"):
    st.markdown("""
    **Problèmes fréquents:**
    
    **Krr négatif:** Vérifiez calibration
    **Pas de mouvement:** Augmentez angle
    **Données bruitées:** Activez lissage
    **Calibration auto:** Rayon détecté cohérent
    
    **Contact:** Support technique disponible
    """)

# Debug info (masqué par défaut)
if st.sidebar.checkbox("🔧 Mode Debug", value=False):
    st.sidebar.markdown("### 🔧 Informations Debug")
    st.sidebar.json({
        "session_experiments": len(st.session_state.experiments_data),
        "current_mode": analysis_mode,
        "available_modes": ["🔬 Analyse Simple", "🔥 Analyse Friction Avancée", "🔄 Comparaison Multi-Expériences", "📚 Données d'Exemple"]
    })
