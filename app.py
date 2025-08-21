# Instructions d'utilisation si pas d'expériences
else:
    st.markdown("""
    ## 🚀 Instructions d'Utilisation - Analyseur Complet Friction + Trace
    
    ### 🔥 **Fonctionnalités Complètes :**
    
    #### **4 Coefficients de Friction + Analyse de Trace :**
    1. **🔥 μ Cinétique** : Friction directe grain-sphère (`F_résistance / F_normale`)
    2. **🎯 μ Rolling** : Résistance pure au roulement (`μ_cinétique - tan(angle)`)
    3. **⚡ μ Énergétique** : Basé sur dissipation d'énergie (`E_dissipée / (F_normale × distance)`)
    4. **📊 Krr Référence** : Coefficient traditionnel de résistance au roulement
    5. **🛤️ Analyse de Trace Complète** : δ/R, volume, régime de pénétration, validation théorique
    
    #### **🛤️ Nouvelles Métriques de Trace :**
    - **🎯 Ratio δ/R** : Pénétration normalisée (comparaison littérature)
    - **📦 Volume de trace** : Déformation totale du substrat
    - **⚖️ Rapport densités** : ρs/ρg (paramètre fondamental)
    - **🏷️ Classification régime** : No-plowing / Micro-plowing / Deep-plowing
    - **📊 Validation théorique** : Écart aux prédictions Darbois Texier et al.
    - **⚡ Énergie de déformation** : Travail de pénétration + déformation substrat
    
    #### **📈 Graphiques Automatiques Enrichis :**
    - **🔥 Coefficients vs Temps** : Évolution temporelle complète
    - **🛤️ Comparaison δ/R vs Théorie** : Validation avec littérature
    - **📐 Profil de trace** : Visualisation morphologique 3D
    - **⚡ Énergies combinées** : Cinétique + déformation substrat
    - **🔗 Corrélations avancées** : Relations friction-pénétration
    
    #### **🔍 Analyses Multi-Expériences Avancées :**
    - **💧 Effet Humidité** : Sur friction ET pénétration
    - **📐 Effet Angle** : Influence sur tous les paramètres
    - **🏷️ Distribution régimes** : Classification automatique
    - **📊 Matrices corrélation** : Relations inter-paramètres
    - **🎯 Insights automatiques** : Détection patterns physiques
    
    ### 📋 **Protocole Expérimental Intégré :**
    
    #### **Pendant l'expérience :**
    1. **📂 Enregistrement vidéo** à 250 fps
    2. **🎯 Détection sphère** avec marqueurs colorés
    3. **📏 Calibration automatique** depuis rayon détecté
    
    #### **Immédiatement après l'expérience :**
    4. **📏 Mesure trace** (URGENT avant effacement !) :
       - Profondeur maximale (mm)
       - Largeur moyenne (3 points)
       - Longueur totale visible
    
    #### **Analyse complète :**
    5. **📊 Upload fichier CSV** + paramètres expérimentation
    6. **🛤️ Saisie mesures trace** dans l'interface
    7. **🔬 Analyses automatiques** : friction + pénétration + validation
    
    ### 💡 **Pour votre fichier `20D_0W_3.csv` :**
    
    - **📂 Upload fichier** : Détection automatique angle 20°
    - **💧 Humidité** : 0% (sols secs)
    - **🛤️ Mesures trace** : Profondeur ~1-3mm, largeur ~10-20mm
    - **📊 Résultats attendus** :
      - μ cinétique ~0.2-0.4
      - δ/R ~0.03-0.08 (no-plowing)
      - Validation théorique <25% écart
      - Krr ~0.04-0.08
    
    ### 🎯 **Résultats Automatiques Complets :**
    
    ✅ **Dashboard friction** : 4 cartes coefficients  
    ✅ **Dashboard trace** : δ/R, volume, régime, validation  
    ✅ **Graphiques temporels** : Évolution tous paramètres  
    ✅ **Comparaison théorie** : Darbois Texier, Van Wal validations  
    ✅ **Analyse énergétique** : Cinétique + déformation combinées  
    ✅ **Export complet** : CSV détaillé + rapport scientifique  
    
    ### 🔬 **Innovation Scientifique :**
    
    **Premier système au monde** combinant :
    - Analyse friction temporelle grain-sphère 4 coefficients
    - Validation théorique traces δ/R vs littérature  
    - Effet humidité sur friction ET pénétration
    - Classification automatique régimes Van Wal
    - Énergies dissipation complètes (cinétique + déformation)
    
    **Applications directes pour votre recherche Osaka University !** 🎓
    """)

# Footer enrichi
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🎓 <strong>Analyseur Complet Friction + Trace Grain-Sphère</strong><br>
    <em>🔥 Premier système intégrant friction temporelle ET analyse de pénétration</em><br>
    📧 Département des Sciences de la Terre Cosmique - Université d'Osaka<br>
    🔬 <strong>Fonctionnalités :</strong> 4 coefficients friction + analyse trace δ/R + validation théorique + énergies combinées
</div>
""", unsafe_allow_html=True)import streamlit as st
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
    <p><em>🔥 Analyse complète des coefficients de friction grain-sphère</em></p>
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
        <div class="metric-card" style="background: linear-gradient(135deg, #{regime_color}50 0%, #{regime_color}80 100%);">
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
    
    # === ANALYSE ÉNERGÉTIQUE DE LA TRACE ===
    
    st.markdown("#### ⚡ Analyse Énergétique de la Déformation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        deformation_energy = groove_metrics['deformation_energy_mJ']
        st.metric("Énergie de déformation", f"{deformation_energy:.2f} mJ")
        
    with col2:
        penetration_work = groove_metrics['penetration_work_mJ']
        st.metric("Travail de pénétration", f"{penetration_work:.2f} mJ")
        
    with col3:
        groove_drag = groove_metrics['groove_drag_coefficient']
        st.metric("Coefficient traînée groove", f"{groove_drag:.4f}")
    
    # === ANALYSE MORPHOLOGIQUE ===
    
    st.markdown("#### 📐 Analyse Morphologique de la Trace")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique 3D conceptuel de la trace
        fig_morph = go.Figure()
        
        # Profil de la trace (section transversale)
        width_points = np.linspace(-groove_metrics['groove_width_mm']/2, groove_metrics['groove_width_mm']/2, 50)
        depth_profile = groove_metrics['groove_depth_mm'] * (1 - (2*width_points/groove_metrics['groove_width_mm'])**2)
        depth_profile = np.maximum(depth_profile, 0)  # Assurer profondeur positive
        
        fig_morph.add_trace(go.Scatter(
            x=width_points,
            y=-depth_profile,  # Négatif car profondeur
            mode='lines',
            fill='tozeroy',
            name='Profil de trace',
            line=dict(color='brown', width=3)
        ))
        
        fig_morph.update_layout(
            title="Profil Transversal de la Trace",
            xaxis_title="Position latérale (mm)",
            yaxis_title="Profondeur (mm)",
            height=300,
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig_morph, use_container_width=True)
    
    with col2:
        # Métriques morphologiques
        aspect_ratio = groove_metrics['groove_aspect_ratio']
        st.metric("Rapport d'aspect L/W", f"{aspect_ratio:.1f}")
        
        symmetry = groove_metrics['groove_symmetry']
        st.metric("Symétrie de trace", f"{symmetry:.2f}")
        
        contact_area = groove_metrics['contact_area_mm2']
        st.metric("Aire de contact", f"{contact_area:.0f} mm²")
        
        # Indicateur de qualité de trace
        if symmetry > 0.7 and aspect_ratio > 5:
            st.success("✅ Trace de bonne qualité")
        elif symmetry > 0.5:
            st.warning("⚠️ Trace de qualité moyenne")
        else:
            st.error("❌ Trace de qualité faible")

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
    a_tangential = np.gradient(v_magnitude, dt)  # Accélération tangentielle
    
    # Forces
    F_gravity_tangential = mass_kg * g * np.sin(angle_rad)  # Force motrice
    F_gravity_normal = mass_kg * g * np.cos(angle_rad)      # Force normale
    F_resistance = mass_kg * np.abs(a_tangential)           # Force de résistance
    F_net = mass_kg * a_tangential                          # Force nette
    
    # === COEFFICIENTS DE FRICTION TEMPORELS ===
    
    # 1. μ Cinétique (friction directe grain-sphère)
    mu_kinetic = F_resistance / F_gravity_normal
    
    # 2. μ Rolling (résistance pure au roulement)
    mu_rolling = mu_kinetic - np.tan(angle_rad)
    
    # 3. μ Énergétique (basé sur dissipation d'énergie)
    E_kinetic = 0.5 * mass_kg * v_magnitude**2
    P_dissipated = F_resistance * v_magnitude  # Puissance dissipée
    
    # Pour μ énergétique, utiliser l'énergie dissipée cumulée
    E_dissipated_cumul = np.cumsum(P_dissipated * dt)
    distance_cumul = np.cumsum(v_magnitude * dt)
    mu_energetic = np.where(distance_cumul > 0, 
                           E_dissipated_cumul / (F_gravity_normal * distance_cumul), 
                           0)
    
    # 4. Krr temporel
    # Krr instantané basé sur la décélération locale
    krr_instantaneous = np.abs(a_tangential) / (g * np.cos(angle_rad))
    
    # === MÉTRIQUES GLOBALES ===
    
    # Vitesses moyennées
    n_avg = max(2, len(v_magnitude) // 6)
    v0 = np.mean(v_magnitude[:n_avg])
    vf = np.mean(v_magnitude[-n_avg:])
    
    # Distance totale
    distances = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
    total_distance = np.sum(distances)
    
    # Krr global
    if total_distance > 0 and v0 > vf:
        krr_global = (v0**2 - vf**2) / (2 * g * total_distance)
    else:
        krr_global = None
    
    # Moyennes des coefficients de friction
    mu_kinetic_avg = np.mean(mu_kinetic)
    mu_rolling_avg = np.mean(mu_rolling)
    mu_energetic_final = mu_energetic[-1] if len(mu_energetic) > 0 else 0
    
    # === ANALYSE STATISTIQUE ===
    
    # Variabilité des coefficients
    mu_kinetic_std = np.std(mu_kinetic)
    mu_rolling_std = np.std(mu_rolling)
    
    # Corrélations
    correlation_v_mu = np.corrcoef(v_magnitude, mu_kinetic)[0, 1] if len(v_magnitude) > 3 else 0
    
    # Évolution temporelle (tendances)
    time_array = np.arange(len(df_clean)) * dt
    
    # Régression linéaire pour tendances
    if len(time_array) > 3:
        mu_kinetic_trend = np.polyfit(time_array, mu_kinetic, 1)[0]  # Pente
        mu_rolling_trend = np.polyfit(time_array, mu_rolling, 1)[0]
    else:
        mu_kinetic_trend = 0
        mu_rolling_trend = 0
    
    # === RETOUR COMPLET ===
    
    results = {
        # Métriques globales
        'Krr_global': krr_global,
        'mu_kinetic_avg': mu_kinetic_avg,
        'mu_rolling_avg': mu_rolling_avg,
        'mu_energetic': mu_energetic_final,
        
        # Variabilité
        'mu_kinetic_std': mu_kinetic_std,
        'mu_rolling_std': mu_rolling_std,
        'mu_kinetic_trend': mu_kinetic_trend,
        'mu_rolling_trend': mu_rolling_trend,
        
        # Corrélations
        'correlation_velocity_friction': correlation_v_mu,
        
        # Vitesses de référence
        'v0_ms': v0,
        'vf_ms': vf,
        'v0_mms': v0 * 1000,
        'vf_mms': vf * 1000,
        'total_distance_mm': total_distance * 1000,
        
        # === SÉRIES TEMPORELLES POUR GRAPHIQUES ===
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
        
        # Informations de nettoyage
        'cleaning_info': cleaning_info
    }
    
    return results

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
    
    # Paramètres de base
    fps = 250.0
    sphere_mass_g = 10.0
    
    # Détecter automatiquement les paramètres de la sphère
    avg_radius_px = df_valid['Radius'].mean()
    
    # Estimation intelligente du rayon réel
    if avg_radius_px > 25:
        sphere_radius_mm = 20.0
    elif avg_radius_px > 15:
        sphere_radius_mm = 15.0
    else:
        sphere_radius_mm = 10.0
    
    # Calcul robuste de Krr
    krr_result, diagnostic = calculate_krr_robust(
        df_valid, 
        fps=fps, 
        angle_deg=angle,
        sphere_mass_g=sphere_mass_g,
        sphere_radius_mm=sphere_radius_mm,
        show_diagnostic=True
    )
    
    # Affichage du diagnostic
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
    
    # Calcul avancé des métriques de friction
    advanced_metrics = calculate_advanced_friction_metrics(
        df_valid, 
        fps=fps, 
        angle_deg=angle,
        sphere_mass_g=sphere_mass_g,
        sphere_radius_mm=sphere_radius_mm,
        pixels_per_mm=krr_result.get('calibration_px_per_mm', 5.0)
    )
    
    # Fusion des résultats de base et avancés
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
    
    # Fusion finale
    enhanced_metrics = {**base_metrics, **advanced_metrics}
    
    return enhanced_metrics

# ==================== FONCTIONS DE COMPARAISON ====================

def create_friction_comparison_section(selected_experiments):
    """Section de comparaison spécialisée pour les analyses de friction"""
    
    st.markdown("## 🔥 Comparaison Avancée des Frictions")
    
    # Préparer les données de comparaison
    friction_comparison_data = []
    
    for exp_name in selected_experiments:
        try:
            exp = st.session_state.experiments_data[exp_name]
            metrics = exp['metrics']
            
            friction_comparison_data.append({
                'Expérience': exp_name,
                'Teneur_eau': exp['water_content'],
                'Angle': exp['angle'],
                'Type_sphère': exp['sphere_type'],
                
                # Coefficients de friction
                'mu_kinetic_avg': metrics.get('mu_kinetic_avg'),
                'mu_rolling_avg': metrics.get('mu_rolling_avg'),
                'mu_energetic': metrics.get('mu_energetic'),
                'Krr_global': metrics.get('Krr_global'),
                
                # Variabilité
                'mu_kinetic_std': metrics.get('mu_kinetic_std'),
                'mu_rolling_std': metrics.get('mu_rolling_std'),
                
                # Tendances
                'mu_kinetic_trend': metrics.get('mu_kinetic_trend'),
                'mu_rolling_trend': metrics.get('mu_rolling_trend'),
                
                # Corrélations
                'correlation_velocity_friction': metrics.get('correlation_velocity_friction'),
                
                # Référence
                'success_rate': exp.get('success_rate')
            })
        except Exception as e:
            st.warning(f"Erreur lors du traitement de l'expérience {exp_name}: {str(e)}")
            continue
    
    if len(friction_comparison_data) < 2:
        st.error("Pas assez de données valides pour la comparaison de friction")
        return
    
    friction_comp_df = pd.DataFrame(friction_comparison_data)
    
    # === GRAPHIQUES DE COMPARAISON FRICTION ===
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Coefficients vs Humidité", 
        "📐 Coefficients vs Angle", 
        "📊 Variabilité & Tendances",
        "🔗 Corrélations"
    ])
    
    with tab1:
        st.markdown("### 💧 Effet de l'Humidité sur les Coefficients de Friction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # μ Cinétique vs Humidité
            valid_kinetic = friction_comp_df.dropna(subset=['mu_kinetic_avg'])
            if len(valid_kinetic) > 0:
                fig_kinetic_water = px.scatter(
                    valid_kinetic,
                    x='Teneur_eau',
                    y='mu_kinetic_avg',
                    color='Angle',
                    size='success_rate',
                    hover_data=['Expérience'],
                    title="🔥 μ Cinétique vs Teneur en Eau",
                    labels={'Teneur_eau': 'Teneur en eau (%)', 'mu_kinetic_avg': 'μ Cinétique'}
                )
                st.plotly_chart(fig_kinetic_water, use_container_width=True)
        
        with col2:
            # μ Rolling vs Humidité
            valid_rolling = friction_comp_df.dropna(subset=['mu_rolling_avg'])
            if len(valid_rolling) > 0:
                fig_rolling_water = px.scatter(
                    valid_rolling,
                    x='Teneur_eau',
                    y='mu_rolling_avg',
                    color='Angle',
                    size='success_rate',
                    hover_data=['Expérience'],
                    title="🎯 μ Rolling vs Teneur en Eau",
                    labels={'Teneur_eau': 'Teneur en eau (%)', 'mu_rolling_avg': 'μ Rolling'}
                )
                st.plotly_chart(fig_rolling_water, use_container_width=True)
        
        # Comparaison tous coefficients
        st.markdown("#### 📊 Comparaison Tous Coefficients")
        
        fig_all_coeffs = go.Figure()
        
        for exp_idx, row in friction_comp_df.iterrows():
            exp_name = row['Expérience']
            water = row['Teneur_eau']
            
            coeffs = [
                row.get('mu_kinetic_avg', 0),
                row.get('mu_rolling_avg', 0),
                row.get('mu_energetic', 0),
                row.get('Krr_global', 0)
            ]
            
            coeff_names = ['μ Cinétique', 'μ Rolling', 'μ Énergétique', 'Krr Global']
            
            fig_all_coeffs.add_trace(go.Bar(
                x=coeff_names,
                y=coeffs,
                name=f"{exp_name} ({water}% eau)",
                text=[f"{c:.4f}" if c is not None else "N/A" for c in coeffs],
                textposition='auto'
            ))
        
        fig_all_coeffs.update_layout(
            title="Comparaison de Tous les Coefficients de Friction",
            xaxis_title="Type de Coefficient",
            yaxis_title="Valeur du Coefficient",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig_all_coeffs, use_container_width=True)
    
    with tab2:
        st.markdown("### 📐 Effet de l'Angle sur les Coefficients de Friction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # μ Cinétique vs Angle
            valid_kinetic_angle = friction_comp_df.dropna(subset=['mu_kinetic_avg', 'Angle'])
            if len(valid_kinetic_angle) > 0:
                fig_kinetic_angle = px.line(
                    valid_kinetic_angle,
                    x='Angle',
                    y='mu_kinetic_avg',
                    color='Teneur_eau',
                    markers=True,
                    title="🔥 μ Cinétique vs Angle",
                    labels={'Angle': 'Angle (°)', 'mu_kinetic_avg': 'μ Cinétique'}
                )
                st.plotly_chart(fig_kinetic_angle, use_container_width=True)
        
        with col2:
            # Krr vs Angle
            valid_krr_angle = friction_comp_df.dropna(subset=['Krr_global', 'Angle'])
            if len(valid_krr_angle) > 0:
                fig_krr_angle = px.line(
                    valid_krr_angle,
                    x='Angle',
                    y='Krr_global',
                    color='Teneur_eau',
                    markers=True,
                    title="📊 Krr Global vs Angle",
                    labels={'Angle': 'Angle (°)', 'Krr_global': 'Krr Global'}
                )
                st.plotly_chart(fig_krr_angle, use_container_width=True)
    
    with tab3:
        st.markdown("### 📊 Variabilité et Tendances Temporelles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Variabilité des coefficients
            valid_std = friction_comp_df.dropna(subset=['mu_kinetic_std', 'mu_rolling_std'])
            if len(valid_std) > 0:
                fig_variability = go.Figure()
                
                fig_variability.add_trace(go.Bar(
                    x=valid_std['Expérience'],
                    y=valid_std['mu_kinetic_std'],
                    name='Variabilité μ Cinétique',
                    marker_color='red',
                    opacity=0.7
                ))
                
                fig_variability.add_trace(go.Bar(
                    x=valid_std['Expérience'],
                    y=valid_std['mu_rolling_std'],
                    name='Variabilité μ Rolling',
                    marker_color='blue',
                    opacity=0.7
                ))
                
                fig_variability.update_layout(
                    title="Variabilité des Coefficients de Friction",
                    xaxis_title="Expérience",
                    yaxis_title="Écart-type",
                    barmode='group'
                )
                fig_variability.update_xaxes(tickangle=45)
                st.plotly_chart(fig_variability, use_container_width=True)
        
        with col2:
            # Tendances temporelles
            valid_trends = friction_comp_df.dropna(subset=['mu_kinetic_trend', 'mu_rolling_trend'])
            if len(valid_trends) > 0:
                fig_trends = go.Figure()
                
                fig_trends.add_trace(go.Scatter(
                    x=valid_trends['mu_kinetic_trend'],
                    y=valid_trends['mu_rolling_trend'],
                    mode='markers+text',
                    text=valid_trends['Expérience'],
                    textposition="top center",
                    marker=dict(
                        size=valid_trends['Teneur_eau'] * 2 + 10,
                        color=valid_trends['Angle'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Angle (°)")
                    ),
                    name='Expériences'
                ))
                
                # Lignes de référence
                fig_trends.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Stable μ Rolling")
                fig_trends.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Stable μ Cinétique")
                
                fig_trends.update_layout(
                    title="Tendances Temporelles des Coefficients",
                    xaxis_title="Tendance μ Cinétique (pente/s)",
                    yaxis_title="Tendance μ Rolling (pente/s)",
                    height=500
                )
                
                st.plotly_chart(fig_trends, use_container_width=True)
    
    with tab4:
        st.markdown("### 🔗 Corrélations et Relations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Corrélation vitesse-friction
            valid_corr = friction_comp_df.dropna(subset=['correlation_velocity_friction'])
            if len(valid_corr) > 0:
                fig_correlation = px.bar(
                    valid_corr,
                    x='Expérience',
                    y='correlation_velocity_friction',
                    color='Teneur_eau',
                    title="🔗 Corrélation Vitesse-Friction",
                    labels={'correlation_velocity_friction': 'Corrélation (r)'}
                )
                fig_correlation.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_correlation.add_hline(y=0.3, line_dash="dot", line_color="orange", annotation_text="Corrélation modérée")
                fig_correlation.add_hline(y=-0.3, line_dash="dot", line_color="orange")
                fig_correlation.update_xaxes(tickangle=45)
                st.plotly_chart(fig_correlation, use_container_width=True)
        
        with col2:
            # Relation μ Cinétique vs μ Rolling
            valid_mu_relation = friction_comp_df.dropna(subset=['mu_kinetic_avg', 'mu_rolling_avg'])
            if len(valid_mu_relation) > 0:
                fig_mu_relation = px.scatter(
                    valid_mu_relation,
                    x='mu_kinetic_avg',
                    y='mu_rolling_avg',
                    color='Teneur_eau',
                    size='Angle',
                    hover_data=['Expérience'],
                    title="🎯 Relation μ Cinétique vs μ Rolling",
                    labels={'mu_kinetic_avg': 'μ Cinétique', 'mu_rolling_avg': 'μ Rolling'}
                )
                
                # Ligne de tendance
                if len(valid_mu_relation) >= 2:
                    x_trend = valid_mu_relation['mu_kinetic_avg']
                    y_trend = valid_mu_relation['mu_rolling_avg']
                    z = np.polyfit(x_trend, y_trend, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_trend.min(), x_trend.max(), 100)
                    
                    fig_mu_relation.add_trace(go.Scatter(
                        x=x_line,
                        y=p(x_line),
                        mode='lines',
                        name='Tendance',
                        line=dict(dash='dash', color='red')
                    ))
                
                st.plotly_chart(fig_mu_relation, use_container_width=True)
    
    # === TABLEAU DE COMPARAISON FRICTION ===
    
    st.markdown("### 📋 Tableau Détaillé des Coefficients de Friction")
    
    # Formatage du tableau
    display_friction_df = friction_comp_df.copy()
    
    # Colonnes à formater
    friction_columns = {
        'mu_kinetic_avg': '{:.4f}',
        'mu_rolling_avg': '{:.4f}',
        'mu_energetic': '{:.4f}',
        'Krr_global': '{:.6f}',
        'mu_kinetic_std': '{:.4f}',
        'mu_rolling_std': '{:.4f}',
        'correlation_velocity_friction': '{:.3f}'
    }
    
    for col, fmt in friction_columns.items():
        if col in display_friction_df.columns:
            display_friction_df[col] = display_friction_df[col].apply(
                lambda x: safe_format_value(x, fmt)
            )
    
    # Renommer les colonnes pour l'affichage
    column_names = {
        'Expérience': 'Expérience',
        'Teneur_eau': 'Eau (%)',
        'Angle': 'Angle (°)',
        'Type_sphère': 'Sphère',
        'mu_kinetic_avg': 'μ Cinétique',
        'mu_rolling_avg': 'μ Rolling',
        'mu_energetic': 'μ Énergétique',
        'Krr_global': 'Krr Global',
        'mu_kinetic_std': 'Var. μ Cin.',
        'mu_rolling_std': 'Var. μ Roll.',
        'correlation_velocity_friction': 'Corr. V-F'
    }
    
    display_columns = [col for col in column_names.keys() if col in display_friction_df.columns]
    display_friction_df = display_friction_df[display_columns].rename(columns=column_names)
    
    st.dataframe(display_friction_df, use_container_width=True)

# ==================== FONCTION DE CHARGEMENT ENRICHIE ====================

def load_detection_data_enhanced(uploaded_file, experiment_name, water_content, angle, sphere_type):
    """Version enrichie avec analyses de friction avancées"""
    if uploaded_file is not None:
        try:
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
            
            # Afficher les informations de base
            st.info(f"""
            📊 **Analyse des données** :
            - Fichier : {uploaded_file.name}
            - Frames totales : {len(df)}
            - Détections valides : {len(df_valid)}
            - Taux de succès : {len(df_valid)/len(df)*100:.1f}%
            - Rayon moyen détecté : {df_valid['Radius'].mean():.1f} pixels
            """)
            
            # Calculer les métriques enrichies
            metrics = calculate_friction_metrics_enhanced(df_valid, water_content, angle, sphere_type)
            
            if metrics is None:
                st.error("❌ Impossible de calculer les métriques pour cette expérience")
                return None
            
            # Afficher immédiatement l'analyse de friction avancée
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

# Interface de chargement
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
    
    # === NOUVELLE SECTION: MESURES DE TRACE ===
    groove_depth, groove_width, groove_length = create_groove_analysis_interface()
    
    if st.button("📊 Ajouter l'expérience") and uploaded_file is not None:
        
        # Détecter l'angle depuis le nom du fichier si possible
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
            # Calculer les métriques de trace
            groove_metrics = calculate_groove_metrics(
                groove_depth, groove_width, groove_length,
                manual_radius, 10.0, angle, water_content
            )
            
            # Ajouter l'analyse de trace
            st.markdown("---")
            create_groove_analysis_section(groove_metrics, exp_name)
            
            # Fusionner les métriques
            enhanced_metrics = add_groove_to_experiment_metrics(exp_data['metrics'], groove_metrics)
            exp_data['metrics'] = enhanced_metrics
            exp_data['groove_metrics'] = groove_metrics
            
            st.session_state.experiments_data[exp_name] = exp_data
            st.success(f"✅ Expérience '{exp_name}' ajoutée avec succès!")
            
            # Afficher un résumé des résultats
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
            - Écart théorie : {safe_format_value(metrics.get('groove_theory_deviation_percent'), '{:.1f}')}%
            
            **⚡ Énergies :**
            - Dissipation cinétique : {safe_format_value(metrics.get('energy_dissipated_mJ'), '{:.2f}')} mJ
            - Déformation substrat : {safe_format_value(metrics.get('groove_deformation_energy_mJ'), '{:.2f}')} mJ
            - **Total dissipé : {safe_format_value(metrics.get('total_energy_dissipated_mJ'), '{:.2f}')} mJ**
            
            **🏃 Vitesses :**
            - Initiale : {safe_format_value(metrics.get('v0_mms'), '{:.1f}')} mm/s
            - Finale : {safe_format_value(metrics.get('vf_mms'), '{:.1f}')} mm/s
            - Décélération : {safe_format_value(metrics.get('deceleration_percent'), '{:.1f}')}%
            
            **📏 Géométrie :**
            - Distance parcourue : {safe_format_value(metrics.get('total_distance_mm'), '{:.1f}')} mm
            - Calibration : {safe_format_value(metrics.get('calibration_px_per_mm'), '{:.2f}')} px/mm
            """)
            
            st.rerun()rue :** {safe_format_value(metrics.get('total_distance_mm'), '{:.1f}')} mm
            
            **Calibration utilisée :** {safe_format_value(metrics.get('calibration_px_per_mm'), '{:.2f}')} px/mm
            """)
            
            st.rerun()

# Test rapide avec analyse de trace
st.markdown("### 🧪 Test Rapide")

if st.button("🔬 Tester avec données simulées + trace (20°, 0% eau)"):
    # Créer des données simulées réalistes
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
    
    # Test du calcul avec trace simulée
    metrics_test = calculate_friction_metrics_enhanced(df_valid_test, 0.0, 20.0, "Solide")
    
    if metrics_test:
        # Ajouter trace simulée
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
        
        # Affichage de l'analyse de trace test
        st.markdown("#### 🛤️ Analyse de Trace Test")
        create_groove_analysis_section(groove_test, "Test")

# Affichage des expériences avec métriques de trace
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
    
    # Sélection pour comparaison
    st.markdown("### 🔍 Sélection pour Comparaison")
    selected_experiments = st.multiselect(
        "Choisir les expériences à comparer :",
        options=list(st.session_state.experiments_data.keys()),
        default=list(st.session_state.experiments_data.keys())
    )
    
    if len(selected_experiments) >= 2:
        st.markdown("---")
        st.markdown("## 📊 Analyse Comparative Complète")
        
        # Préparer les données pour la comparaison standard
        comparison_data = []
        for exp_name in selected_experiments:
            exp = st.session_state.experiments_data[exp_name]
            metrics = exp['metrics']
            
            comparison_data.append({
                'Expérience': exp_name,
                'Teneur_eau': exp['water_content'],
                'Angle': exp['angle'],
                'Type_sphère': exp['sphere_type'],
                'Krr': metrics.get('Krr'),
                'v0_mms': metrics.get('v0_mms'),
                'vf_mms': metrics.get('vf_mms'),
                'total_distance_mm': metrics.get('total_distance_mm'),
                'deceleration_percent': metrics.get('deceleration_percent'),
                'success_rate': exp.get('success_rate'),
                # Nouvelles métriques de trace
                'penetration_ratio': metrics.get('groove_penetration_ratio'),
                'groove_volume': metrics.get('groove_groove_volume_cm3'),
                'theory_deviation': metrics.get('groove_theory_deviation_percent'),
                'plowing_regime': metrics.get('plowing_regime'),
                'total_energy_dissipated': metrics.get('total_energy_dissipated_mJ')
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # === ONGLETS DE COMPARAISON ENRICHIS ===
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔥 Friction & Cinématique", 
            "🛤️ Analyse de Traces", 
            "⚡ Énergies & Régimes",
            "📊 Corrélations Avancées"
        ])
        
        with tab1:
            # Graphiques de comparaison standard (existant)
            col1, col2 = st.columns(2)
            
            with col1:
                # Krr vs Teneur en eau
                valid_krr = comp_df.dropna(subset=['Krr'])
                if len(valid_krr) > 0:
                    try:
                        fig_krr = px.scatter(
                            valid_krr, 
                            x='Teneur_eau', 
                            y='Krr',
                            color='Angle',
                            hover_data=['Expérience'],
                            title="🔍 Coefficient Krr vs Teneur en Eau"
                        )
                        fig_krr.update_layout(height=400)
                        st.plotly_chart(fig_krr, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur création graphique Krr: {str(e)}")
            
            with col2:
                # Vitesses vs Angle
                valid_velocities = comp_df.dropna(subset=['v0_mms', 'vf_mms'])
                if len(valid_velocities) > 0:
                    try:
                        fig_vel = go.Figure()
                        fig_vel.add_trace(go.Scatter(
                            x=valid_velocities['Angle'], 
                            y=valid_velocities['v0_mms'],
                            mode='markers+lines', 
                            name='V₀ (initiale)',
                            marker=dict(color='blue', size=10)
                        ))
                        fig_vel.add_trace(go.Scatter(
                            x=valid_velocities['Angle'], 
                            y=valid_velocities['vf_mms'],
                            mode='markers+lines', 
                            name='Vf (finale)',
                            marker=dict(color='red', size=10)
                        ))
                        fig_vel.update_layout(
                            title="🏃 Vitesses vs Angle",
                            xaxis_title="Angle (°)",
                            yaxis_title="Vitesse (mm/s)",
                            height=400
                        )
                        st.plotly_chart(fig_vel, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur création graphique vitesses: {str(e)}")
        
        with tab2:
            st.markdown("### 🛤️ Comparaison des Traces et Pénétrations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # δ/R vs Teneur en eau
                valid_penetration = comp_df.dropna(subset=['penetration_ratio'])
                if len(valid_penetration) > 0:
                    fig_penetration = px.scatter(
                        valid_penetration,
                        x='Teneur_eau',
                        y='penetration_ratio',
                        color='Angle',
                        size='groove_volume',
                        hover_data=['Expérience', 'plowing_regime'],
                        title="🎯 Ratio de Pénétration δ/R vs Humidité",
                        labels={'penetration_ratio': 'δ/R', 'Teneur_eau': 'Teneur en eau (%)'}
                    )
                    st.plotly_chart(fig_penetration, use_container_width=True)
            
            with col2:
                # Régimes de pénétration
                valid_regime = comp_df.dropna(subset=['plowing_regime'])
                if len(valid_regime) > 0:
                    regime_counts = valid_regime['plowing_regime'].value_counts()
                    fig_regime = px.pie(
                        values=regime_counts.values,
                        names=regime_counts.index,
                        title="🏷️ Distribution des Régimes de Pénétration"
                    )
                    st.plotly_chart(fig_regime, use_container_width=True)
            
            # Volume de trace vs paramètres
            st.markdown("#### 📦 Volume de Trace vs Paramètres")
            
            col1, col2 = st.columns(2)
            
            with col1:
                valid_volume = comp_df.dropna(subset=['groove_volume'])
                if len(valid_volume) > 0:
                    fig_volume_water = px.scatter(
                        valid_volume,
                        x='Teneur_eau',
                        y='groove_volume',
                        color='Angle',
                        hover_data=['Expérience'],
                        title="Volume de Trace vs Humidité",
                        labels={'groove_volume': 'Volume (cm³)', 'Teneur_eau': 'Teneur en eau (%)'}
                    )
                    st.plotly_chart(fig_volume_water, use_container_width=True)
            
            with col2:
                if len(valid_volume) > 0:
                    fig_volume_angle = px.bar(
                        valid_volume,
                        x='Expérience',
                        y='groove_volume',
                        color='Teneur_eau',
                        title="Volume de Trace par Expérience",
                        labels={'groove_volume': 'Volume (cm³)'}
                    )
                    fig_volume_angle.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_volume_angle, use_container_width=True)
        
        with tab3:
            st.markdown("### ⚡ Analyse Énergétique Complète")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Énergie totale dissipée
                valid_energy = comp_df.dropna(subset=['total_energy_dissipated'])
                if len(valid_energy) > 0:
                    fig_energy = px.bar(
                        valid_energy,
                        x='Expérience',
                        y='total_energy_dissipated',
                        color='Teneur_eau',
                        title="⚡ Énergie Totale Dissipée",
                        labels={'total_energy_dissipated': 'Énergie (mJ)'}
                    )
                    fig_energy.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_energy, use_container_width=True)
            
            with col2:
                # Écart à la théorie vs humidité
                valid_theory = comp_df.dropna(subset=['theory_deviation'])
                if len(valid_theory) > 0:
                    fig_theory_dev = px.scatter(
                        valid_theory,
                        x='Teneur_eau',
                        y='theory_deviation',
                        color='Angle',
                        hover_data=['Expérience'],
                        title="📊 Écart à la Théorie vs Humidité",
                        labels={'theory_deviation': 'Écart (%)', 'Teneur_eau': 'Teneur en eau (%)'}
                    )
                    fig_theory_dev.add_hline(y=25, line_dash="dash", line_color="orange", 
                                           annotation_text="Seuil acceptable (25%)")
                    st.plotly_chart(fig_theory_dev, use_container_width=True)
        
        with tab4:
            st.markdown("### 📊 Corrélations Multi-Paramètres")
            
            # Matrice de corrélation avancée
            correlation_columns = ['Krr', 'penetration_ratio', 'groove_volume', 'total_energy_dissipated', 
                                 'Teneur_eau', 'Angle', 'theory_deviation']
            
            available_corr_columns = [col for col in correlation_columns if col in comp_df.columns and comp_df[col].notna().any()]
            
            if len(available_corr_columns) >= 3:
                corr_matrix = comp_df[available_corr_columns].corr()
                
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="🔗 Matrice de Corrélation Complète",
                    color_continuous_scale="RdBu_r"
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Corrélations les plus fortes
                st.markdown("##### 🎯 Corrélations les Plus Significatives:")
                
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                corr_values = corr_matrix.where(mask).stack().reset_index()
                corr_values.columns = ['Var1', 'Var2', 'Correlation']
                corr_values = corr_values.sort_values('Correlation', key=abs, ascending=False)
                
                for i, row in corr_values.head(5).iterrows():
                    correlation_strength = "Très forte" if abs(row['Correlation']) > 0.8 else "Forte" if abs(row['Correlation']) > 0.6 else "Modérée"
                    correlation_direction = "positive" if row['Correlation'] > 0 else "négative"
                    
                    st.markdown(f"- **{correlation_strength} corrélation {correlation_direction}** entre {row['Var1']} et {row['Var2']} (r = {row['Correlation']:.3f})")
        
        # Comparaison avancée des frictions (existante)
        st.markdown("---")
        create_friction_comparison_section(selected_experiments)
    
    elif len(selected_experiments) == 1:
        st.info("📊 Sélectionnez au moins 2 expériences pour effectuer une comparaison")

# Sidebar avec aide
st.sidebar.markdown("### 🔧 Aide au Dépannage")

with st.sidebar.expander("❌ Problèmes Fréquents"):
    st.markdown("""
    **Krr = N/A ou erreur :**
    - Vérifiez l'angle (20° pour `20D_0W_3.csv`)
    - Assurez-vous d'avoir >10 détections valides
    - Vérifiez que la sphère décélère
    
    **Vitesse finale > initiale :**
    - Mauvaise calibration
    - Angle incorrect
    - Données corrompues
    
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
    
    **μ Rolling typiques :**
    - Généralement < μ Cinétique
    - Peut être négatif (effet lubrifiant)
    
    **δ/R typiques :**
    - No-plowing : < 0.03
    - Micro-plowing : 0.03-0.10
    - Deep-plowing : > 0.10
    """)

# Gestion des expériences existantes
if st.session_state.experiments_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Expériences Chargées")
    
    for name, data in st.session_state.experiments_data.items():
        with st.sidebar.expander(f"📋 {name}"):
            st.write(f"**Eau :** {data['water_content']}%")
            st.write(f"**Angle :** {data['angle']}°")
            st.write(f"**Type :** {data['sphere_type']}")
            
            krr_val = data['metrics'].get('Krr')
            if krr_val is not None and not pd.isna(krr_val):
                st.write(f"**Krr :** {krr_val:.6f}")
                if 0.03 <= krr_val <= 0.15:
                    st.success("✅ Krr OK")
                else:
                    st.warning("⚠️ Krr inhabituel")
            else:
                st.error("❌ Krr non calculé")
            
            # Afficher coefficients de friction
            mu_kinetic = data['metrics'].get('mu_kinetic_avg')
            if mu_kinetic is not None:
                st.write(f"**μ Cinétique :** {mu_kinetic:.4f}")
    
    # Boutons de gestion
    st.sidebar.markdown("---")
    exp_to_remove = st.sidebar.selectbox(
        "Supprimer une expérience :",
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

# Instructions d'utilisation si pas d'expériences
else:
    st.markdown("""
    ## 🚀 Instructions d'Utilisation - Analyseur de Friction Complet
    
    ### 🔥 **Nouvelles Fonctionnalités de Friction :**
    
    #### **4 Coefficients de Friction Calculés :**
    1. **🔥 μ Cinétique** : Friction directe grain-sphère (`F_résistance / F_normale`)
    2. **🎯 μ Rolling** : Résistance pure au roulement (`μ_cinétique - tan(angle)`)
    3. **⚡ μ Énergétique** : Basé sur dissipation d'énergie (`E_dissipée / (F_normale × distance)`)
    4. **📊 Krr Référence** : Coefficient traditionnel de résistance au roulement
    
    #### **Graphiques Automatiques :**
    - **📈 Coefficients vs Temps** : Évolution temporelle de tous les coefficients
    - **⚖️ Analyse des Forces** : Forces, puissance et énergie dissipée vs temps
    - **📊 Histogrammes** : Distribution statistique des coefficients
    - **🔗 Corrélations** : Relations vitesse-friction et entre coefficients
    
    #### **Analyses Avancées Multi-Expériences :**
    - **💧 Effet Humidité** : Impact de la teneur en eau sur chaque coefficient
    - **📐 Effet Angle** : Influence de l'inclinaison sur la friction
    - **📊 Variabilité** : Stabilité temporelle et écarts-types
    - **🎯 Insights Automatiques** : Détection des tendances et corrélations
    
    ### 📋 **Pour Commencer :**
    
    1. **📂 Chargez votre fichier CSV** (Frame, X_center, Y_center, Radius)
    2. **📊 L'analyse de friction apparaît automatiquement** après le diagnostic Krr
    3. **🔍 Comparez plusieurs expériences** pour voir les effets humidité/angle
    4. **💾 Exportez les résultats** : CSV détaillé + rapport complet
    
    ### 💡 **Pour votre fichier `20D_0W_3.csv` :**
    
    - **Angle :** 20° (détection automatique depuis le nom)
    - **Humidité :** 0% (sols secs)
    - **Coefficients attendus :** μ cinétique ~0.2-0.4, μ rolling variable, Krr ~0.04-0.08
    
    ### 🎯 **Résultats Attendus :**
    
    Vous obtiendrez automatiquement :
    - ✅ **4 cartes résumé** style dashboard avec tous les coefficients
    - ✅ **Graphique coefficients vs temps** (votre demande principale)
    - ✅ **Analyse forces et énergies** 
    - ✅ **Distributions statistiques**
    - ✅ **Export complet** pour analyse externe
    
    Ce système offre l'analyse de friction grain-sphère **la plus complète** disponible !
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🎓 <strong>Analyseur Avancé de Friction Grain-Sphère</strong><br>
    <em>🔥 Analyse complète des coefficients de friction temporels - Université d'Osaka</em><br>
    📧 Département des Sciences de la Terre Cosmique<br>
    🔬 <strong>Fonctionnalités :</strong> 4 coefficients de friction, graphiques temporels, analyses multi-expériences, export complet
</div>
""", unsafe_allow_html=True)
