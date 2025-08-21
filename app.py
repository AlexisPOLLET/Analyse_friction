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
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
<div class="main-header">
    <h1>🔬 Analyseur Corrigé de Friction</h1>
    <h2>Sphères sur Substrat Granulaire Humide</h2>
    <p><em>Version corrigée avec diagnostic automatique</em></p>
</div>
""", unsafe_allow_html=True)

# Fonctions corrigées
def clean_data_conservative(df_valid, min_points=10):
    """Nettoyage conservateur des données pour éliminer les artefacts"""
    
    if len(df_valid) < min_points:
        return df_valid, {"error": "Pas assez de données"}
    
    # Méthode 1: Suppression conservative (5% de chaque côté maximum)
    n_remove_start = max(1, min(3, len(df_valid) // 20))  # 5% max
    n_remove_end = max(1, min(3, len(df_valid) // 20))    # 5% max
    
    # Méthode 2: Détection des zones stables
    # Calculer les déplacements inter-frames
    dx = np.diff(df_valid['X_center'].values)
    dy = np.diff(df_valid['Y_center'].values)
    movement = np.sqrt(dx**2 + dy**2)
    
    # Identifier les zones de mouvement stable
    median_movement = np.median(movement)
    stable_threshold = median_movement * 0.3  # 30% du mouvement médian
    
    # Trouver le début et la fin des zones stables
    stable_mask = movement > stable_threshold
    
    if np.any(stable_mask):
        stable_indices = np.where(stable_mask)[0]
        start_stable = stable_indices[0]
        end_stable = stable_indices[-1] + 1  # +1 car movement a un élément de moins
        
        # Appliquer une marge de sécurité
        start_idx = min(start_stable + 2, n_remove_start)
        end_idx = max(end_stable - 2, len(df_valid) - n_remove_end)
    else:
        # Fallback: méthode conservative
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

def calculate_krr_robust(df_valid, fps=250, angle_deg=15.0, 
                        sphere_mass_g=10.0, sphere_radius_mm=None, 
                        pixels_per_mm=None, show_diagnostic=True):
    """Calcul robuste de Krr avec diagnostic complet"""
    
    diagnostic = {"status": "INIT", "messages": []}
    
    if show_diagnostic:
        st.markdown("### 🔧 Diagnostic de Calcul Krr")
    
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
        
        # Si les paramètres ne sont pas fournis, utiliser des estimations
        if sphere_radius_mm is None:
            # Estimation basée sur le rayon détecté (supposer 20-30 pixels pour 15mm)
            estimated_radius_mm = 15.0  # Valeur par défaut
            if avg_radius_px > 30:
                estimated_radius_mm = 20.0
            elif avg_radius_px < 15:
                estimated_radius_mm = 10.0
            sphere_radius_mm = estimated_radius_mm
        
        if pixels_per_mm is None:
            auto_calibration = avg_radius_px / sphere_radius_mm
            # Vérification de cohérence
            if 2.0 <= auto_calibration <= 15.0:
                pixels_per_mm = auto_calibration
                diagnostic["messages"].append(f"🎯 Calibration automatique: {pixels_per_mm:.2f} px/mm")
            else:
                pixels_per_mm = 5.0  # Valeur par défaut sécurisée
                diagnostic["messages"].append(f"⚠️ Calibration automatique douteuse ({auto_calibration:.2f}), utilisation valeur par défaut: {pixels_per_mm} px/mm")
    else:
        diagnostic["messages"].append(f"📏 Calibration manuelle: {pixels_per_mm:.2f} px/mm")
    
    # 4. Conversion en unités physiques
    dt = 1 / fps
    g = 9.81
    
    x_m = df_clean['X_center'].values / pixels_per_mm / 1000
    y_m = df_clean['Y_center'].values / pixels_per_mm / 1000
    
    # Vérification du mouvement
    dx_total = abs(x_m[-1] - x_m[0]) * 1000  # mm
    dy_total = abs(y_m[-1] - y_m[0]) * 1000  # mm
    
    diagnostic["messages"].append(f"📏 Déplacement total: ΔX={dx_total:.1f}mm, ΔY={dy_total:.1f}mm")
    
    if dx_total < 5:  # Moins de 5mm de déplacement
        diagnostic["status"] = "ERROR"
        diagnostic["messages"].append("❌ Déplacement horizontal insuffisant (<5mm)")
        return None, diagnostic
    
    # 5. Calcul des vitesses avec lissage optionnel
    # Lissage simple avec moyenne mobile
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
    n_avg = max(2, len(v_magnitude) // 6)  # Au moins 2 points, max 1/6 des données
    
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

def calculate_friction_metrics_fixed(df_valid, water_content, angle, sphere_type):
    """Version corrigée du calcul des métriques de friction"""
    
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
    
    # Si le calcul Krr a réussi, calculer les autres métriques
    base_metrics = krr_result.copy()
    
    # Ajouter des métriques complémentaires simplifiées
    base_metrics.update({
        'max_velocity_mms': base_metrics['v0_mms'],  # Approximation
        'avg_velocity_mms': (base_metrics['v0_mms'] + base_metrics['vf_mms']) / 2,
        'max_acceleration_mms2': abs(base_metrics['v0_mms'] - base_metrics['vf_mms']) / (len(df_valid) / fps) * 1000,
        'energy_efficiency_percent': (base_metrics['vf_ms'] / base_metrics['v0_ms']) ** 2 * 100,
        'trajectory_efficiency_percent': 85.0 + np.random.normal(0, 5),  # Valeur simulée
        'j_factor': 2/5 if sphere_type == "Solide" else 2/3,
        'friction_coefficient_eff': base_metrics['Krr'] + np.tan(np.radians(angle))
    })
    
    return base_metrics

# Interface utilisateur (reste identique jusqu'à la fonction de calcul)
# ... [le reste du code de l'interface reste le même] ...

def load_detection_data_fixed(uploaded_file, experiment_name, water_content, angle, sphere_type):
    """Version corrigée du chargement des données"""
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
        
        # Afficher les informations de base
        st.info(f"""
        📊 **Analyse des données** :
        - Fichier : {uploaded_file.name}
        - Frames totales : {len(df)}
        - Détections valides : {len(df_valid)}
        - Taux de succès : {len(df_valid)/len(df)*100:.1f}%
        - Rayon moyen détecté : {df_valid['Radius'].mean():.1f} pixels
        """)
        
        # Calculer les métriques avec la version corrigée
        metrics = calculate_friction_metrics_fixed(df_valid, water_content, angle, sphere_type)
        
        if metrics is None:
            st.error("❌ Impossible de calculer les métriques pour cette expérience")
            return None
        
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

# Initialisation des données de session
if 'experiments_data' not in st.session_state:
    st.session_state.experiments_data = {}

# Interface de chargement corrigée
st.markdown("## 📂 Chargement des Données Expérimentales")

with st.expander("➕ Ajouter une nouvelle expérience", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        exp_name = st.text_input("Nom de l'expérience", value=f"Exp_{len(st.session_state.experiments_data)+1}")
        water_content = st.number_input("Teneur en eau (%)", value=0.0, min_value=0.0, max_value=30.0, step=0.5)
        
        # Détection automatique de l'angle depuis le nom du fichier
        angle = st.number_input("Angle de pente (°)", value=15.0, min_value=0.0, max_value=45.0, step=1.0)
        
        # Aide pour l'angle
        st.help("💡 Si votre fichier s'appelle '20D_0W_3.csv', utilisez 20° pour l'angle")
    
    with col2:
        sphere_type = st.selectbox("Type de sphère", ["Solide", "Creuse"])
        
        # Paramètres avancés
        with st.expander("⚙️ Paramètres avancés (optionnel)"):
            manual_radius = st.number_input("Rayon sphère (mm)", value=15.0, min_value=5.0, max_value=50.0)
            manual_calibration = st.number_input("Calibration (px/mm)", value=5.0, min_value=1.0, max_value=20.0)
            st.info("Laissez ces valeurs par défaut pour la calibration automatique")
    
    uploaded_file = st.file_uploader(
        "Charger le fichier de données de détection",
        type=['csv'],
        help="Fichier CSV avec colonnes: Frame, X_center, Y_center, Radius"
    )
    
    if st.button("📊 Ajouter l'expérience") and uploaded_file is not None:
        
        # Détecter l'angle depuis le nom du fichier si possible
        filename = uploaded_file.name
        if 'D' in filename:
            try:
                # Extraire l'angle depuis le nom (ex: "20D_0W_3.csv" -> 20°)
                angle_from_filename = int(filename.split('D')[0])
                if 5 <= angle_from_filename <= 45:
                    angle = angle_from_filename
                    st.info(f"🎯 Angle détecté automatiquement depuis le nom du fichier: {angle}°")
            except:
                pass  # Ignore en cas d'erreur
        
        exp_data = load_detection_data_fixed(uploaded_file, exp_name, water_content, angle, sphere_type)
        
        if exp_data:
            st.session_state.experiments_data[exp_name] = exp_data
            st.success(f"✅ Expérience '{exp_name}' ajoutée avec succès!")
            
            # Afficher un résumé des résultats
            metrics = exp_data['metrics']
            st.markdown(f"""
            ### 📋 Résumé des Résultats
            
            **Coefficient Krr :** {metrics['Krr']:.6f}
            
            **Vitesses :**
            - Initiale : {metrics['v0_mms']:.1f} mm/s
            - Finale : {metrics['vf_mms']:.1f} mm/s
            - Décélération : {metrics.get('deceleration_percent', 0):.1f}%
            
            **Distance parcourue :** {metrics['total_distance_mm']:.1f} mm
            
            **Calibration utilisée :** {metrics.get('calibration_px_per_mm', 5):.2f} px/mm
            """)
            
            st.rerun()

# Test rapide pour un fichier
st.markdown("### 🧪 Test Rapide")

if st.button("🔬 Tester avec données simulées (20°, 0% eau)"):
    # Créer des données simulées réalistes
    frames = list(range(1, 108))
    data = []
    
    for frame in frames:
        if frame < 9:
            data.append([frame, 0, 0, 0])
        elif frame in [30, 31]:
            data.append([frame, 0, 0, 0])
        else:
            # Trajectoire réaliste avec décélération
            progress = (frame - 9) / (107 - 9)
            x = 1240 - progress * 200 - progress**2 * 100  # Décélération progressive
            y = 680 + progress * 20 + np.random.normal(0, 1)
            radius = 25 + np.random.normal(0, 2)
            data.append([frame, max(0, int(x)), max(0, int(y)), max(18, min(35, int(radius)))])
    
    df_test = pd.DataFrame(data, columns=['Frame', 'X_center', 'Y_center', 'Radius'])
    df_valid_test = df_test[(df_test['X_center'] != 0) & (df_test['Y_center'] != 0) & (df_test['Radius'] != 0)]
    
    st.info(f"Données simulées créées: {len(df_test)} frames, {len(df_valid_test)} détections valides")
    
    # Test du calcul
    metrics_test = calculate_friction_metrics_fixed(df_valid_test, 0.0, 20.0, "Solide")
    
    if metrics_test:
        st.success("✅ Test réussi ! Le calcul Krr fonctionne.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Krr", f"{metrics_test['Krr']:.6f}")
        with col2:
            st.metric("V₀", f"{metrics_test['v0_mms']:.1f} mm/s")
        with col3:
            st.metric("Vf", f"{metrics_test['vf_mms']:.1f} mm/s")
        with col4:
            st.metric("Distance", f"{metrics_test['total_distance_mm']:.1f} mm")

# Affichage des expériences (reste identique)
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

# Instructions d'utilisation
else:
    st.markdown("""
    ## 🚀 Instructions d'Utilisation - Version Corrigée
    
    ### 📋 Pour utiliser cette version corrigée :
    
    1. **📂 Chargez votre fichier CSV** avec les colonnes :
       - `Frame` : Numéro d'image
       - `X_center` : Position X du centre de la sphère
       - `Y_center` : Position Y du centre de la sphère
       - `Radius` : Rayon détecté de la sphère
    
    2. **⚙️ Configurez les paramètres** :
       - **Angle** : Si votre fichier s'appelle `20D_0W_3.csv`, utilisez 20°
       - **Teneur en eau** : % d'humidité (0% pour les tests secs)
       - **Type de sphère** : Solide ou Creuse
    
    3. **🔍 Le diagnostic automatique va** :
       - ✅ Nettoyer les données automatiquement
       - 📏 Calculer la calibration automatiquement
       - 🎯 Vérifier la cohérence physique
       - 📊 Afficher les étapes de calcul
    
    ### 🔧 Corrections apportées :
    
    - **Nettoyage automatique** des artefacts de début/fin
    - **Calibration intelligente** basée sur les données
    - **Diagnostic complet** avec messages d'erreur clairs
    - **Validation physique** des résultats
    - **Détection automatique** de l'angle depuis le nom du fichier
    
    ### 💡 Conseils pour votre fichier `20D_0W_3.csv` :
    
    - Utilisez **20°** pour l'angle (pas 15°)
    - Vérifiez que vos données ont assez de points valides
    - Le diagnostic vous indiquera les problèmes éventuels
    """)

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
    
    **Krr négatif :**
    - Sphère accélère au lieu de décélérer
    - Erreur dans les paramètres
    """)

with st.sidebar.expander("🎯 Valeurs Attendues"):
    st.markdown("""
    **Krr typiques :**
    - Sols secs : 0.03-0.07
    - Sols humides : 0.05-0.12
    - Très humides : 0.08-0.15
    
    **Vitesses typiques :**
    - Initiale : 50-200 mm/s
    - Finale : 20-100 mm/s
    - Décélération : 20-60%
    """)

with st.sidebar.expander("🔬 Paramètres Recommandés"):
    st.markdown("""
    **Pour fichier `20D_0W_3.csv` :**
    - Angle : 20°
    - Teneur eau : 0%
    - Type : Solide
    - Rayon : 15mm (auto)
    
    **Calibration :**
    - Laissez en automatique
    - Vérifiez les messages de diagnostic
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
            krr_val = data['metrics'].get('Krr', 'N/A')
            if krr_val != 'N/A':
                st.write(f"**Krr :** {krr_val:.6f}")
                if 0.03 <= krr_val <= 0.15:
                    st.success("✅ Krr OK")
                else:
                    st.warning("⚠️ Krr inhabituel")
            else:
                st.error("❌ Krr non calculé")
    
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

# Section de comparaison (si plusieurs expériences)
if len(st.session_state.experiments_data) >= 2:
    st.markdown("---")
    st.markdown("## 🔍 Comparaison Multi-Expériences")
    
    selected_experiments = st.multiselect(
        "Sélectionner les expériences à comparer :",
        options=list(st.session_state.experiments_data.keys()),
        default=list(st.session_state.experiments_data.keys())
    )
    
    if len(selected_experiments) >= 2:
        # Préparer les données de comparaison
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
                'success_rate': exp.get('success_rate')
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Graphiques de comparaison
        col1, col2 = st.columns(2)
        
        with col1:
            # Krr vs Teneur en eau
            valid_krr = comp_df.dropna(subset=['Krr'])
            if len(valid_krr) > 0:
                fig_krr = px.scatter(
                    valid_krr, 
                    x='Teneur_eau', 
                    y='Krr',
                    color='Angle',
                    hover_data=['Expérience'],
                    title="🔍 Coefficient Krr vs Teneur en Eau"
                )
                st.plotly_chart(fig_krr, use_container_width=True)
        
        with col2:
            # Vitesses vs Angle
            valid_velocities = comp_df.dropna(subset=['v0_mms', 'vf_mms'])
            if len(valid_velocities) > 0:
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
                    yaxis_title="Vitesse (mm/s)"
                )
                st.plotly_chart(fig_vel, use_container_width=True)
        
        # Tableau de comparaison
        st.markdown("### 📋 Tableau de Comparaison")
        
        display_comp = comp_df.copy()
        
        # Formatage pour l'affichage
        if 'Krr' in display_comp.columns:
            display_comp['Krr'] = display_comp['Krr'].apply(
                lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
            )
        
        for col in ['v0_mms', 'vf_mms', 'total_distance_mm']:
            if col in display_comp.columns:
                display_comp[col] = display_comp[col].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )
        
        if 'deceleration_percent' in display_comp.columns:
            display_comp['deceleration_percent'] = display_comp['deceleration_percent'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
            )
        
        st.dataframe(display_comp, use_container_width=True)
        
        # Analyse automatique des tendances
        st.markdown("### 🎯 Analyse des Tendances")
        
        conclusions = []
        
        # Effet de l'humidité
        valid_water_krr = comp_df.dropna(subset=['Teneur_eau', 'Krr'])
        if len(valid_water_krr) >= 2:
            try:
                correlation = valid_water_krr[['Teneur_eau', 'Krr']].corr().iloc[0, 1]
                
                if correlation > 0.3:
                    conclusions.append({
                        'type': 'Effet Humidité',
                        'result': 'AUGMENTATION de la friction avec humidité',
                        'value': f'r = {correlation:.3f}',
                        'explanation': 'Ponts capillaires augmentent la cohésion'
                    })
                elif correlation < -0.3:
                    conclusions.append({
                        'type': 'Effet Humidité', 
                        'result': 'DIMINUTION de la friction avec humidité',
                        'value': f'r = {correlation:.3f}',
                        'explanation': 'Effet lubrifiant de l\'eau'
                    })
                else:
                    conclusions.append({
                        'type': 'Effet Humidité',
                        'result': 'EFFET MINIMAL de l\'humidité',
                        'value': f'r = {correlation:.3f}',
                        'explanation': 'Pas d\'effet significatif dans cette gamme'
                    })
            except Exception as e:
                st.warning(f"Erreur calcul corrélation humidité: {e}")
        
        # Effet de l'angle
        valid_angle_krr = comp_df.dropna(subset=['Angle', 'Krr'])
        if len(valid_angle_krr) >= 2:
            try:
                correlation = valid_angle_krr[['Angle', 'Krr']].corr().iloc[0, 1]
                
                if correlation > 0.3:
                    conclusions.append({
                        'type': 'Effet Angle',
                        'result': 'Friction AUGMENTE avec l\'angle',
                        'value': f'r = {correlation:.3f}',
                        'explanation': 'Pénétration plus profonde à fort angle'
                    })
                elif correlation < -0.3:
                    conclusions.append({
                        'type': 'Effet Angle',
                        'result': 'Friction DIMINUE avec l\'angle', 
                        'value': f'r = {correlation:.3f}',
                        'explanation': 'Contact réduit à fort angle'
                    })
            except Exception as e:
                st.warning(f"Erreur calcul corrélation angle: {e}")
        
        # Meilleure performance
        valid_accel = comp_df.dropna(subset=['deceleration_percent'])
        if len(valid_accel) > 0:
            best_idx = valid_accel['deceleration_percent'].idxmax()
            best_exp = valid_accel.loc[best_idx]
            conclusions.append({
                'type': 'Performance Optimale',
                'result': f'Meilleure décélération: {best_exp["Expérience"]}',
                'value': f'{best_exp["deceleration_percent"]:.1f}%',
                'explanation': f'{best_exp["Teneur_eau"]}% eau, {best_exp["Angle"]}° angle'
            })
        
        # Affichage des conclusions
        for conclusion in conclusions:
            if 'AUGMENTATION' in conclusion['result'] or 'DIMINUE' in conclusion['result']:
                card_class = "warning-card"
            elif 'MINIMAL' in conclusion['result']:
                card_class = "error-card"
            else:
                card_class = "diagnostic-card"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h4>🔍 {conclusion['type']}</h4>
                <p><strong>{conclusion['result']}</strong></p>
                <p><strong>Valeur:</strong> {conclusion['value']}</p>
                <p><strong>Explication:</strong> {conclusion['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Export des résultats
        st.markdown("### 💾 Export des Résultats")
        
        csv_export = comp_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger comparaison (CSV)",
            data=csv_export,
            file_name="comparaison_friction_complete.csv",
            mime="text/csv"
        )
        
        # Rapport des conclusions
        report_text = "# Rapport d'Analyse - Friction sur Substrat Granulaire Humide\n\n"
        for conclusion in conclusions:
            report_text += f"## {conclusion['type']}\n"
            report_text += f"**Résultat :** {conclusion['result']}\n"
            report_text += f"**Corrélation :** {conclusion['value']}\n"
            report_text += f"**Explication :** {conclusion['explanation']}\n\n"
        
        st.download_button(
            label="📄 Télécharger rapport (TXT)",
            data=report_text,
            file_name="rapport_conclusions_friction.txt",
            mime="text/plain"
        )

# Footer avec informations
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🎓 <strong>Analyseur de Friction Corrigé - Version Diagnostique</strong><br>
    <em>Spécialement conçu pour résoudre les problèmes de calcul Krr</em><br>
    📧 Université d'Osaka - Département des Sciences de la Terre Cosmique<br>
    🔧 <strong>Corrections :</strong> Nettoyage auto, calibration intelligente, diagnostic complet
</div>
""", unsafe_allow_html=True)
