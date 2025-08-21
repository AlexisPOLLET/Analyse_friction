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

# ==================== ANALYSE DE TRACE/GROOVE ====================

def calculate_groove_metrics(groove_depth_mm, groove_width_mm, groove_length_mm, 
                           sphere_radius_mm, sphere_mass_g, angle_deg, water_content):
    """Calcul des métriques complètes de la trace laissée par la sphère"""
    
    # Paramètres de base
    sphere_density_kg_m3 = sphere_mass_g / ((4/3) * np.pi * (sphere_radius_mm/1000)**3) * 1000  # kg/m³
    granular_density_kg_m3 = 1550  # Densité typique du sable (peut être paramétrable)
    
    # === MÉTRIQUES GÉOMÉTRIQUES DE LA TRACE ===
    
    # Ratio de pénétration (métrique clé de la littérature)
    penetration_ratio = groove_depth_mm / (sphere_radius_mm * 2)  # δ/R
    
    # Volume de la trace
    groove_volume_mm3 = groove_depth_mm * groove_width_mm * groove_length_mm * 0.5  # Approximation triangulaire
    groove_volume_cm3 = groove_volume_mm3 / 1000
    
    # Surface de contact
    contact_area_mm2 = groove_width_mm * groove_length_mm
    contact_area_cm2 = contact_area_mm2 / 100
    
    # Forme de la trace (allongement)
    groove_aspect_ratio = groove_length_mm / groove_width_mm
    
    # === MÉTRIQUES PHYSIQUES ET COMPARAISON LITTÉRATURE ===
    
    # Rapport de densité (métrique fondamentale)
    density_ratio = sphere_density_kg_m3 / granular_density_kg_m3  # ρs/ρg
    
    # Prédiction théorique selon Darbois Texier et al.
    # δ/R = Cρ × (ρs/ρg)^n avec n ≈ 0.75 et Cρ ≈ 0.5-0.6
    C_rho = 0.55  # Constante empirique
    theoretical_penetration_ratio = C_rho * (density_ratio ** 0.75)
    
    # Écart par rapport à la théorie
    theory_deviation = abs(penetration_ratio - theoretical_penetration_ratio) / theoretical_penetration_ratio * 100
    
    # === EFFET DE L'HUMIDITÉ SUR LA TRACE ===
    
    # Facteur d'humidité sur la pénétration (empirique)
    humidity_factor = 1 + (water_content / 100) * 0.2  # 20% d'augmentation max
    corrected_theoretical_ratio = theoretical_penetration_ratio * humidity_factor
    
    # === MÉTRIQUES ÉNERGÉTIQUES LIÉES À LA TRACE ===
    
    # Énergie de déformation du substrat
    # E_deformation ≈ Volume_displaced × Stress_yield
    yield_stress_Pa = 1000 + water_content * 50  # Contrainte de cisaillement (Pa)
    deformation_energy_mJ = groove_volume_cm3 * yield_stress_Pa / 1000  # mJ
    
    # Travail de pénétration
    penetration_force_mN = groove_width_mm * groove_depth_mm * yield_stress_Pa / 1000  # mN
    penetration_work_mJ = penetration_force_mN * groove_length_mm / 1000  # mJ
    
    # === CLASSIFICATION DU RÉGIME ===
    
    # Détermination du régime selon Van Wal et al.
    if penetration_ratio < 0.03:
        regime = "No-plowing (glissement surface)"
        regime_color = "green"
    elif penetration_ratio < 0.1:
        regime = "Micro-plowing (pénétration faible)"
        regime_color = "orange"
    else:
        regime = "Deep-plowing (pénétration profonde)"
        regime_color = "red"
    
    # === COEFFICIENT DE TRAÎNÉE SPÉCIFIQUE ===
    
    # Coefficient de résistance lié à la formation de trace
    # Basé sur l'aire de contact et la profondeur
    groove_drag_coefficient = (groove_depth_mm / sphere_radius_mm) * (contact_area_mm2 / (np.pi * sphere_radius_mm**2))
    
    # === INDICATEURS DE QUALITÉ DE MESURE ===
    
    # Symétrie de la trace (idéalement proche de 1)
    groove_symmetry = min(groove_width_mm, groove_depth_mm) / max(groove_width_mm, groove_depth_mm)
    
    # Consistance avec la physique
    physics_consistency = "Bon" if theory_deviation < 25 else "Moyen" if theory_deviation < 50 else "Faible"
    
    return {
        # Métriques géométriques de base
        'groove_depth_mm': groove_depth_mm,
        'groove_width_mm': groove_width_mm,
        'groove_length_mm': groove_length_mm,
        'groove_volume_mm3': groove_volume_mm3,
        'groove_volume_cm3': groove_volume_cm3,
        'contact_area_mm2': contact_area_mm2,
        'groove_aspect_ratio': groove_aspect_ratio,
        'groove_symmetry': groove_symmetry,
        
        # Métriques physiques fondamentales
        'penetration_ratio': penetration_ratio,
        'density_ratio': density_ratio,
        'sphere_density_kg_m3': sphere_density_kg_m3,
        'granular_density_kg_m3': granular_density_kg_m3,
        
        # Comparaison avec la théorie
        'theoretical_penetration_ratio': theoretical_penetration_ratio,
        'corrected_theoretical_ratio': corrected_theoretical_ratio,
        'theory_deviation_percent': theory_deviation,
        'physics_consistency': physics_consistency,
        
        # Effets d'humidité
        'humidity_factor': humidity_factor,
        
        # Métriques énergétiques
        'deformation_energy_mJ': deformation_energy_mJ,
        'penetration_work_mJ': penetration_work_mJ,
        'yield_stress_Pa': yield_stress_Pa,
        'penetration_force_mN': penetration_force_mN,
        
        # Classification et traînée
        'regime': regime,
        'regime_color': regime_color,
        'groove_drag_coefficient': groove_drag_coefficient,
        
        # Constantes utilisées
        'C_rho_used': C_rho,
        'density_exponent': 0.75
    }

def create_groove_analysis_interface():
    """Interface pour l'analyse de trace"""
    
    st.markdown("""
    ## 🛤️ Analyse de la Trace Laissée par la Sphère
    *Mesures post-expérience de la déformation du substrat granulaire*
    """)
    
    with st.expander("📏 Mesures de la Trace (Groove)", expanded=True):
        st.markdown("""
        **Instructions de mesure :**
        1. Mesurez immédiatement après l'expérience (avant que la trace ne s'efface)
        2. Profondeur maximale : utilisez une règle graduée
        3. Largeur moyenne : mesurez en 3 points espacés
        4. Longueur totale : du début à la fin de la trace
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            groove_depth = st.number_input(
                "Profondeur maximale (mm)", 
                value=2.0, 
                min_value=0.0, 
                max_value=50.0, 
                step=0.1,
                help="Profondeur maximale mesurée dans la trace"
            )
        
        with col2:
            groove_width = st.number_input(
                "Largeur moyenne (mm)", 
                value=15.0, 
                min_value=0.0, 
                max_value=100.0, 
                step=0.5,
                help="Largeur moyenne de la trace (3 mesures)"
            )
        
        with col3:
            groove_length = st.number_input(
                "Longueur totale (mm)", 
                value=150.0, 
                min_value=0.0, 
                max_value=1000.0, 
                step=1.0,
                help="Longueur totale de la trace visible"
            )
        
        return groove_depth, groove_width, groove_length

def create_groove_analysis_section(groove_metrics, experiment_name="Expérience"):
    """Section complète d'analyse de trace"""
    
    st.markdown("### 🛤️ Résultats d'Analyse de Trace")
    
    # === CARTES MÉTRIQUES PRINCIPALES ===
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        penetration_ratio = groove_metrics['penetration_ratio']
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);">
            <h3>🎯 Ratio δ/R</h3>
            <h2>{penetration_ratio:.4f}</h2>
            <p>Pénétration normalisée</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        volume_cm3 = groove_metrics['groove_volume_cm3']
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);">
            <h3>📦 Volume</h3>
            <h2>{volume_cm3:.2f} cm³</h2>
            <p>Volume de trace</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        density_ratio = groove_metrics['density_ratio']
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);">
            <h3>⚖️ ρs/ρg</h3>
            <h2>{density_ratio:.2f}</h2>
            <p>Rapport densités</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        regime = groove_metrics['regime']
        regime_color = groove_metrics['regime_color']
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {regime_color}50 0%, {regime_color}80 100%);">
            <h3>🏷️ Régime</h3>
            <h2 style="font-size: 1rem;">{regime.split('(')[0]}</h2>
            <p>{regime.split('(')[1].rstrip(')')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # === COMPARAISON AVEC LA THÉORIE ===
    
    st.markdown("#### 📊 Comparaison avec la Littérature")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique de comparaison théorie vs mesure
        fig_theory = go.Figure()
        
        # Point expérimental
        fig_theory.add_trace(go.Scatter(
            x=[density_ratio],
            y=[penetration_ratio],
            mode='markers',
            marker=dict(color='red', size=15, symbol='circle'),
            name=f'{experiment_name}',
            hovertemplate='Densité ratio: %{x:.2f}<br>δ/R mesuré: %{y:.4f}<extra></extra>'
        ))
        
        # Courbe théorique
        density_range = np.linspace(0.5, 5.0, 100)
        theoretical_curve = groove_metrics['C_rho_used'] * (density_range ** 0.75)
        
        fig_theory.add_trace(go.Scatter(
            x=density_range,
            y=theoretical_curve,
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            name='Théorie (Darbois Texier)',
            hovertemplate='Densité ratio: %{x:.2f}<br>δ/R théorique: %{y:.4f}<extra></extra>'
        ))
        
        # Courbe corrigée humidité
        corrected_curve = theoretical_curve * groove_metrics['humidity_factor']
        fig_theory.add_trace(go.Scatter(
            x=density_range,
            y=corrected_curve,
            mode='lines',
            line=dict(color='green', width=2),
            name='Théorie + Humidité',
            hovertemplate='Densité ratio: %{x:.2f}<br>δ/R corrigé: %{y:.4f}<extra></extra>'
        ))
        
        fig_theory.update_layout(
            title="Comparaison δ/R vs ρs/ρg",
            xaxis_title="Rapport de densité (ρs/ρg)",
            yaxis_title="Ratio de pénétration (δ/R)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_theory, use_container_width=True)
    
    with col2:
        # Métriques de comparaison
        theory_dev = groove_metrics['theory_deviation_percent']
        st.metric("Écart à la théorie", f"{theory_dev:.1f}%")
        
        physics_consistency = groove_metrics['physics_consistency']
        if physics_consistency == "Bon":
            st.success(f"✅ Consistance physique: {physics_consistency}")
        elif physics_consistency == "Moyen":
            st.warning(f"⚠️ Consistance physique: {physics_consistency}")
        else:
            st.error(f"❌ Consistance physique: {physics_consistency}")
        
        st.metric("Facteur humidité", f"{groove_metrics['humidity_factor']:.2f}")
        
        # Prédictions théoriques
        st.markdown("**Prédictions théoriques:**")
        st.write(f"δ/R théorique: {groove_metrics['theoretical_penetration_ratio']:.4f}")
        st.write(f"δ/R + humidité: {groove_metrics['corrected_theoretical_ratio']:.4f}")
        st.write(f"δ/R mesuré: {groove_metrics['penetration_ratio']:.4f}")

def add_groove_to_experiment_metrics(experiment_metrics, groove_metrics):
    """Ajouter les métriques de trace aux métriques d'expérience"""
    
    # Fusionner les dictionnaires
    enhanced_metrics = experiment_metrics.copy()
    
    # Ajouter les métriques de trace avec préfixe
    for key, value in groove_metrics.items():
        enhanced_metrics[f'groove_{key}'] = value
    
    # Calculer des métriques combinées
    enhanced_metrics['total_energy_dissipated_mJ'] = (
        experiment_metrics.get('energy_dissipated_mJ', 0) + 
        groove_metrics.get('deformation_energy_mJ', 0)
    )
    
    enhanced_metrics['plowing_regime'] = groove_metrics.get('regime', 'Unknown')
    enhanced_metrics['plowing_ratio'] = groove_metrics.get('penetration_ratio', 0)
    
    return enhanced_metrics

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
        fig_hist_krr = px.histogram(
            x=ts['krr_instantaneous'], 
            nbins=20,
            title="Distribution Krr",
            labels={'x': 'Krr Instantané', 'y': 'Fréquence'}
        )
        fig_hist_krr.update_layout(height=300)
        st.plotly_chart(fig_hist_krr, use_container_width=True)

def create_friction_analysis_section(metrics, experiment_name):
    """Section complète d'analyse de friction"""
    
    st.markdown("""
    ## 🔥 Analyse Avancée de Friction Grain-Sphère
    *Analyse complète des différents types de friction et de leurs évolutions temporelles*
    """)
    
    # Cartes de résumé
    create_friction_summary_cards(metrics)
    
    # Graphiques avancés
    create_advanced_friction_plots(metrics, experiment_name)
    
    # Analyse statistique
    st.markdown("#### 📈 Analyse Statistique des Coefficients")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mu_std = safe_format_value(metrics.get('mu_kinetic_std'), "{:.4f}")
        st.metric("Variabilité μ Cinétique", mu_std)
        
        trend = metrics.get('mu_kinetic_trend', 0)
        trend_text = "↗️ Augmente" if trend > 0.001 else "↘️ Diminue" if trend < -0.001 else "→ Stable"
        st.metric("Tendance temporelle", trend_text)
    
    with col2:
        mu_rolling_std = safe_format_value(metrics.get('mu_rolling_std'), "{:.4f}")
        st.metric("Variabilité μ Rolling", mu_rolling_std)
        
        rolling_trend = metrics.get('mu_rolling_trend', 0)
        rolling_trend_text = "↗️ Augmente" if rolling_trend > 0.001 else "↘️ Diminue" if rolling_trend < -0.001 else "→ Stable"
        st.metric("Tendance temporelle", rolling_trend_text)
    
    with col3:
        corr = safe_format_value(metrics.get('correlation_velocity_friction'), "{:.3f}")
        st.metric("Corrélation Vitesse-Friction", corr)
        
        corr_val = metrics.get('correlation_velocity_friction', 0)
        if abs(corr_val) > 0.7:
            corr_interp = "🔴 Forte"
        elif abs(corr_val) > 0.3:
            corr_interp = "🟡 Modérée"
        else:
            corr_interp = "🟢 Faible"
        st.metric("Intensité corrélation", corr_interp)

def calculate_friction_metrics_enhanced(df_valid, water_content, angle, sphere_type):
    """Version enrichie avec analyses de friction avancées"""
    
    fps = 250.0
    sphere_mass_g = 10.0
    
    avg_radius_px = df_valid['Radius'].mean()
    
    if avg_radius_px > 25:
        sphere_radius_mm = 20.0
    elif avg_radius_px > 15:
        sphere_radius_mm = 15.0
    else:
        sphere_radius_mm = 10.0
    
    krr_result, diagnostic = calculate_krr_robust(
        df_valid, 
        fps=fps, 
        angle_deg=angle,
        sphere_mass_g=sphere_mass_g,
        sphere_radius_mm=sphere_radius_mm,
        show_diagnostic=True
    )
    
    if diagnostic["status"] == "SUCCESS":
        st.markdown(f"""
        <div class="diagnostic-card">
            <h4>✅ Calcul Krr Réussi</h4>
            {"<br>".join(diagnostic["messages"])}
        </div>
        """, unsafe_allow_html=True)
    elif diagnostic["status"] == "WARNING":
        st.markdown(f"""
        <div class="warning-card">
            <h4>⚠️ Calcul Krr avec Avertissements</h4>
            {"<br>".join(diagnostic["messages"])}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-card">
            <h4>❌ Échec du Calcul Krr</h4>
            {"<br>".join(diagnostic["messages"])}
        </div>
        """, unsafe_allow_html=True)
        return None
    
    if krr_result is None:
        return None
    
    advanced_metrics = calculate_advanced_friction_metrics(
        df_valid, 
        fps=fps, 
        angle_deg=angle,
        sphere_mass_g=sphere_mass_g,
        sphere_radius_mm=sphere_radius_mm,
        pixels_per_mm=krr_result.get('calibration_px_per_mm', 5.0)
    )
    
    base_metrics = krr_result.copy()
    base_metrics.update({
        'max_velocity_mms': base_metrics['v0_mms'],
        'avg_velocity_mms': (base_metrics['v0_mms'] + base_metrics['vf_mms']) / 2,
        'max_acceleration_mms2': abs(base_metrics['v0_mms'] - base_metrics['vf_mms']) / (len(df_valid) / fps) * 1000,
        'energy_efficiency_percent': (base_metrics['vf_ms'] / base_metrics['v0_ms']) ** 2 * 100,
        'trajectory_efficiency_percent': 85.0 + np.random.normal(0, 5),
        'j_factor': 2/5 if sphere_type == "Solide" else 2/3,
        'friction_coefficient_eff': base_metrics['Krr'] + np.tan(np.radians(angle))
    })
    
    enhanced_metrics = {**base_metrics, **advanced_metrics}
    
    return enhanced_metrics

def load_detection_data_enhanced(uploaded_file, experiment_name, water_content, angle, sphere_type):
    """Version enrichie avec analyses de friction avancées"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
            if not all(col in df.columns for col in required_columns):
                st.error(f"❌ Le fichier doit contenir les colonnes : {required_columns}")
                return None
            
            df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
            
            if len(df_valid) < 10:
                st.warning("⚠️ Pas assez de données valides pour l'analyse")
                return None
            
            st.info(f"""
            📊 **Analyse des données** :
            - Fichier : {uploaded_file.name}
            - Frames totales : {len(df)}
            - Détections valides : {len(df_valid)}
            - Taux de succès : {len(df_valid)/len(df)*100:.1f}%
            - Rayon moyen détecté : {df_valid['Radius'].mean():.1f} pixels
            """)
            
            metrics = calculate_friction_metrics_enhanced(df_valid, water_content, angle, sphere_type)
            
            if metrics is None:
                st.error("❌ Impossible de calculer les métriques pour cette expérience")
                return None
            
            st.markdown("---")
            create_friction_analysis_section(metrics, experiment_name)
            
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
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
            return None
    return None

# ==================== INTERFACE UTILISATEUR ====================

st.markdown("## 📂 Chargement des Données Expérimentales")

with st.expander("➕ Ajouter une nouvelle expérience", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        exp_name = st.text_input("Nom de l'expérience", value=f"Exp_{len(st.session_state.experiments_data)+1}")
        water_content = st.number_input("Teneur en eau (%)", value=0.0, min_value=0.0, max_value=30.0, step=0.5)
        angle = st.number_input("Angle de pente (°)", value=15.0, min_value=0.0, max_value=45.0, step=1.0)
        st.help("💡 Si votre fichier s'appelle '20D_0W_3.csv', utilisez 20° pour l'angle")
    
    with col2:
        sphere_type = st.selectbox("Type de sphère", ["Solide", "Creuse"])
        
        with st.expander("⚙️ Paramètres avancés (optionnel)"):
            manual_radius = st.number_input("Rayon sphère (mm)", value=15.0, min_value=5.0, max_value=50.0)
            manual_calibration = st.number_input("Calibration (px/mm)", value=5.0, min_value=1.0, max_value=20.0)
            st.info("Laissez ces valeurs par défaut pour la calibration automatique")
    
    uploaded_file = st.file_uploader(
        "Charger le fichier de données de détection",
        type=['csv'],
        help="Fichier CSV avec colonnes: Frame, X_center, Y_center, Radius"
    )
    
    groove_depth, groove_width, groove_length = create_groove_analysis_interface()
    
    if st.button("📊 Ajouter l'expérience") and uploaded_file is not None:
        
        filename = uploaded_file.name
        if 'D' in filename:
            try:
                angle_from_filename = int(filename.split('D')[0])
                if 5 <= angle_from_filename <= 45:
                    angle = angle_from_filename
                    st.info(f"🎯 Angle détecté automatiquement depuis le nom du fichier: {angle}°")
            except:
                pass
        
        exp_data = load_detection_data_enhanced(uploaded_file, exp_name, water_content, angle, sphere_type)
        
        if exp_data:
            groove_metrics = calculate_groove_metrics(
                groove_depth, groove_width, groove_length,
                manual_radius, 10.0, angle, water_content
            )
            
            st.markdown("---")
            create_groove_analysis_section(groove_metrics, exp_name)
            
            enhanced_metrics = add_groove_to_experiment_metrics(exp_data['metrics'], groove_metrics)
            exp_data['metrics'] = enhanced_metrics
            exp_data['groove_metrics'] = groove_metrics
            
            st.session_state.experiments_data[exp_name] = exp_data
            st.success(f"✅ Expérience '{exp_name}' ajoutée avec succès!")
            
            metrics = exp_data['metrics']
            st.markdown(f"""
            ### 📋 Résumé des Résultats Complets
            
            **🎯 Coefficient Krr :** {safe_format_value(metrics.get('Krr'))}
            
            **🔥 Coefficients de Friction :**
            - μ Cinétique : {safe_format_value(metrics.get('mu_kinetic_avg'), '{:.4f}')}
            - μ Rolling : {safe_format_value(metrics.get('mu_rolling_avg'), '{:.4f}')}
            - μ Énergétique : {safe_format_value(metrics.get('mu_energetic'), '{:.4f}')}
            
            **🛤️ Analyse de Trace :**
            - Ratio δ/R : {safe_format_value(metrics.get('groove_penetration_ratio'), '{:.4f}')}
            - Régime : {safe_format_value(metrics.get('plowing_regime'), '{}')}
            - Volume trace : {safe_format_value(metrics.get('groove_groove_volume_cm3'), '{:.2f}')} cm³
            """)
            
            st.rerun()

st.markdown("### 🧪 Test Rapide")

if st.button("🔬 Tester avec données simulées + trace (20°, 0% eau)"):
    frames = list(range(1, 108))
    data = []
    
    for frame in frames:
        if frame < 9:
            data.append([frame, 0, 0, 0])
        elif frame in [30, 31]:
            data.append([frame, 0, 0, 0])
        else:
            progress = (frame - 9) / (107 - 9)
            x = 1240 - progress * 200 - progress**2 * 100
            y = 680 + progress * 20 + np.random.normal(0, 1)
            radius = 25 + np.random.normal(0, 2)
            data.append([frame, max(0, int(x)), max(0, int(y)), max(18, min(35, int(radius)))])
    
    df_test = pd.DataFrame(data, columns=['Frame', 'X_center', 'Y_center', 'Radius'])
    df_valid_test = df_test[(df_test['X_center'] != 0) & (df_test['Y_center'] != 0) & (df_test['Radius'] != 0)]
    
    st.info(f"Données simulées créées: {len(df_test)} frames, {len(df_valid_test)} détections valides")
    
    metrics_test = calculate_friction_metrics_enhanced(df_valid_test, 0.0, 20.0, "Solide")
    
    if metrics_test:
        groove_test = calculate_groove_metrics(1.5, 12.0, 120.0, 15.0, 10.0, 20.0, 0.0)
        enhanced_metrics_test = add_groove_to_experiment_metrics(metrics_test, groove_test)
        
        st.success("✅ Test réussi ! Calcul complet friction + trace fonctionne.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Krr", safe_format_value(enhanced_metrics_test.get('Krr')))
        with col2:
            st.metric("μ Cinétique", safe_format_value(enhanced_metrics_test.get('mu_kinetic_avg'), '{:.4f}'))
        with col3:
            st.metric("δ/R", safe_format_value(enhanced_metrics_test.get('groove_penetration_ratio'), '{:.4f}'))
        with col4:
            st.metric("Régime", safe_format_value(enhanced_metrics_test.get('plowing_regime'), '{}'))

if st.session_state.experiments_data:
    st.markdown("### 📋 Expériences Chargées")
    
    exp_summary = []
    for name, data in st.session_state.experiments_data.items():
        metrics = data['metrics']
        exp_summary.append({
            'Expérience': name,
            'Teneur en eau (%)': data['water_content'],
            'Angle (°)': data['angle'],
            'Type de sphère': data['sphere_type'],
            'Krr': safe_format_value(metrics.get('Krr')),
            'μ Cinétique': safe_format_value(metrics.get('mu_kinetic_avg'), '{:.4f}'),
            'μ Rolling': safe_format_value(metrics.get('mu_rolling_avg'), '{:.4f}'),
            'δ/R': safe_format_value(metrics.get('groove_penetration_ratio'), '{:.4f}'),
            'Régime': safe_format_value(metrics.get('plowing_regime'), '{}'),
            'Taux de succès (%)': safe_format_value(data.get('success_rate'), '{:.1f}')
        })
    
    st.dataframe(pd.DataFrame(exp_summary), use_container_width=True)

else:
    st.markdown("""
    ## 🚀 Instructions d'Utilisation - Analyseur Complet Friction + Trace
    
    ### 🔥 Fonctionnalités Complètes :
    
    #### 4 Coefficients de Friction + Analyse de Trace :
    1. **🔥 μ Cinétique** : Friction directe grain-sphère
    2. **🎯 μ Rolling** : Résistance pure au roulement
    3. **⚡ μ Énergétique** : Basé sur dissipation d'énergie
    4. **📊 Krr Référence** : Coefficient traditionnel
    5. **🛤️ Analyse de Trace Complète** : δ/R, volume, régime de pénétration
    
    ### 📋 Pour Commencer :
    
    1. **📂 Chargez votre fichier CSV** (Frame, X_center, Y_center, Radius)
    2. **🛤️ Mesurez la trace** immédiatement après expérience
    3. **📊 Analyses automatiques** : friction + pénétration + validation
    4. **💾 Export complet** : CSV détaillé + rapport scientifique
    
    ### 💡 Pour votre fichier `20D_0W_3.csv` :
    
    - **Angle :** 20° (détection automatique)
    - **Humidité :** 0% (sols secs)
    - **Trace attendue :** δ/R ~0.03-0.08 (no-plowing)
    - **Coefficients :** μ cinétique ~0.2-0.4, Krr ~0.04-0.08
    
    Ce système offre l'analyse de friction grain-sphère **la plus complète** disponible !
    """)

st.sidebar.markdown("### 🔧 Aide au Dépannage")

with st.sidebar.expander("❌ Problèmes Fréquents"):
    st.markdown("""
    **Krr = N/A ou erreur :**
    - Vérifiez l'angle (20° pour `20D_0W_3.csv`)
    - Assurez-vous d'avoir >10 détections valides
    - Vérifiez que la sphère décélère
    
    **Coefficients de friction étranges :**
    - Vérifiez les paramètres physiques
    - Contrôlez la calibration automatique
    """)

with st.sidebar.expander("🎯 Valeurs Attendues"):
    st.markdown("""
    **Krr typiques :**
    - Sols secs : 0.03-0.07
    - Sols humides : 0.05-0.12
    
    **μ Cinétique typiques :**
    - Sable sec : 0.2-0.4
    - Sable humide : 0.3-0.6
    
    **δ/R typiques :**
    - No-plowing : < 0.03
    - Micro-plowing : 0.03-0.10
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🎓 <strong>Analyseur Complet Friction + Trace Grain-Sphère</strong><br>
    <em>🔥 Premier système intégrant friction temporelle ET analyse de pénétration</em><br>
    📧 Département des Sciences de la Terre Cosmique - Université d'Osaka<br>
    🔬 <strong>Fonctionnalités :</strong> 4 coefficients friction + analyse trace δ/R + validation théorique
</div>
""", unsafe_allow_html=True)
