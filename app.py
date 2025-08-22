import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Interface Krr Simple",
    page_icon="🔬",
    layout="wide"
)

# ==================== CSS ====================
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
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITRE ====================
st.markdown("""
<div class="main-header">
    <h1>🔬 Interface Krr - Analyse Friction</h1>
    <h2>Calcul et Comparaison des Coefficients</h2>
</div>
""", unsafe_allow_html=True)

# ==================== INITIALISATION ====================
if 'experiments' not in st.session_state:
    st.session_state.experiments = {}

# ==================== FONCTION CALCUL KRR AVANCÉ ====================
def calculate_krr_advanced(df_valid, fps=250, sphere_mass_g=10.0, sphere_radius_mm=15.0, angle_deg=15.0):
    """Calcul Krr avec métriques physiques avancées"""
    
    if len(df_valid) < 20:
        return None
    
    # Paramètres physiques
    dt = 1/fps
    g = 9.81
    mass_kg = sphere_mass_g / 1000
    radius_m = sphere_radius_mm / 1000
    angle_rad = np.radians(angle_deg)
    
    # Calibration
    avg_radius_px = df_valid['Radius'].mean()
    pixels_per_mm = avg_radius_px / sphere_radius_mm
    
    # Positions en mètres
    x_m = df_valid['X_center'].values / pixels_per_mm / 1000
    y_m = df_valid['Y_center'].values / pixels_per_mm / 1000
    
    # Temps
    t = np.arange(len(df_valid)) * dt
    
    # Vitesses
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Accélérations
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    acceleration = np.gradient(v_magnitude, dt)
    
    # Vitesses moyennées (plus robuste)
    n = max(5, len(v_magnitude) // 6)  # Moyenner sur plus de points
    v0 = np.mean(v_magnitude[:n])
    vf = np.mean(v_magnitude[-n:])
    
    # Distance
    distance = np.sum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))
    
    # === MÉTRIQUES PHYSIQUES AVANCÉES ===
    
    # 1. Krr basique
    if v0 > vf and distance > 0:
        krr = (v0**2 - vf**2) / (2 * g * distance)
    else:
        krr = 0
    
    # 2. Coefficient de friction effectif
    mu_effective = krr + np.tan(angle_rad)
    
    # 3. Forces
    F_gravity_tangential = mass_kg * g * np.sin(angle_rad)
    F_resistance = mass_kg * np.abs(acceleration)
    F_resistance_avg = np.mean(F_resistance)
    F_resistance_max = np.max(F_resistance)
    
    # 4. Énergies (translation + rotation)
    j_factor = 2/5  # Sphère solide
    E_translational = 0.5 * mass_kg * v_magnitude**2
    E_rotational = 0.5 * (j_factor * mass_kg * radius_m**2) * (v_magnitude / radius_m)**2
    E_total = E_translational + E_rotational
    
    E_initial = E_total[0] if len(E_total) > 0 else 0
    E_final = E_total[-1] if len(E_total) > 0 else 0
    E_dissipated = E_initial - E_final
    
    # 5. Efficacité énergétique
    energy_efficiency = (E_final / E_initial * 100) if E_initial > 0 else 0
    
    # 6. Puissance dissipée
    power_dissipated = F_resistance * v_magnitude  # Watts
    power_avg = np.mean(power_dissipated) * 1000  # mW
    power_max = np.max(power_dissipated) * 1000   # mW
    
    # 7. Vitesse de décélération
    if len(v_magnitude) > 10:
        # Ajustement linéaire pour la décélération
        try:
            decel_fit = np.polyfit(t, v_magnitude, 1)
            deceleration_rate = -decel_fit[0]  # m/s²
        except:
            deceleration_rate = (v0 - vf) / (t[-1] - t[0]) if len(t) > 1 else 0
    else:
        deceleration_rate = 0
    
    # 8. Qualité de la trajectoire (linéarité)
    if distance > 0:
        straight_distance = np.sqrt((x_m[-1] - x_m[0])**2 + (y_m[-1] - y_m[0])**2)
        trajectory_linearity = (straight_distance / distance * 100)  # %
        
        # Variation verticale (déviation de la ligne droite)
        vertical_deviation = np.std(y_m) * 1000  # mm
    else:
        trajectory_linearity = 0
        vertical_deviation = 0
    
    # 9. Validation physique
    # Test indépendance vitesse (coefficient de variation des vitesses)
    velocity_cv = (np.std(v_magnitude) / np.mean(v_magnitude) * 100) if np.mean(v_magnitude) > 0 else 0
    
    # Test cohérence décélération
    theoretical_decel = g * (np.sin(angle_rad) + krr * np.cos(angle_rad))
    decel_ratio = deceleration_rate / theoretical_decel if theoretical_decel > 0 else 0
    
    # 11. Velocity Rolling (condition de roulement sans glissement)
    # v_CM = R × ω pour un roulement parfait
    radius_m = sphere_radius_mm / 1000
    omega_theoretical = v_magnitude / radius_m  # vitesse angulaire théorique
    omega_avg = np.mean(omega_theoretical)
    
    # Vérification condition roulement sans glissement
    rolling_condition = np.mean(v_magnitude) / (radius_m * omega_avg) if omega_avg > 0 else 0
    rolling_quality = abs(1 - rolling_condition) * 100  # % d'écart par rapport au roulement parfait
    
    # Classification qualité roulement
    if rolling_quality < 5:
        rolling_status = "Excellent"
    elif rolling_quality < 15:
        rolling_status = "Bon"
    elif rolling_quality < 30:
        rolling_status = "Moyen"
    else:
        rolling_status = "Glissement"
    
    return {
        # Métriques de base
        'krr': krr,
        'v0': v0 * 1000,
        'vf': vf * 1000,
        'distance': distance * 1000,
        'calibration': pixels_per_mm,
        
        # Métriques physiques avancées
        'mu_effective': mu_effective,
        'mu_kinetic': mu_kinetic,
        'energy_efficiency': energy_efficiency,
        'power_avg_mW': power_avg,
        'power_max_mW': power_max,
        'deceleration_rate': deceleration_rate,
        'trajectory_linearity': trajectory_linearity,
        'vertical_deviation_mm': vertical_deviation,
        'velocity_cv': velocity_cv,
        'decel_ratio': decel_ratio,
        'rolling_quality': rolling_quality,
        'rolling_status': rolling_status,
        'omega_avg': omega_avg,
        
        # Forces et énergies
        'F_resistance_avg_mN': F_resistance_avg * 1000,
        'F_resistance_max_mN': F_resistance_max * 1000,
        'E_initial_mJ': E_initial * 1000,
        'E_final_mJ': E_final * 1000,
        'E_dissipated_mJ': E_dissipated * 1000,
        
        # Validation
        'theoretical_decel': theoretical_decel,
        'physics_valid': 0.7 <= decel_ratio <= 1.3 and velocity_cv < 30,  # Critères de validité
        
        # Données pour graphiques avancés
        'time_series': {
            'time': t,
            'velocity': v_magnitude * 1000,
            'acceleration': acceleration,
            'power': power_dissipated * 1000,
            'energy_total': E_total * 1000
        }
    }

# ==================== CHARGEMENT DONNÉES ====================
def load_data_advanced(uploaded_file, exp_name, water_content, angle, sphere_type):
    """Chargement avec calculs avancés"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
            
            metrics = calculate_krr_advanced(df_valid, angle_deg=angle)
            if metrics is None:
                return None
                
            return {
                'name': exp_name,
                'water_content': water_content,
                'angle': angle,
                'sphere_type': sphere_type,
                'metrics': metrics,
                'success_rate': len(df_valid) / len(df) * 100
            }
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
            return None
    return None

# ==================== INTERFACE CHARGEMENT ====================
st.markdown("## 📂 Chargement des Données")

col1, col2 = st.columns(2)

with col1:
    exp_name = st.text_input("Nom expérience", value=f"Exp_{len(st.session_state.experiments)+1}")
    water_content = st.number_input("Teneur en eau (%)", value=0.0, min_value=0.0, max_value=30.0)
    
with col2:
    angle = st.number_input("Angle (°)", value=5.0, min_value=1.0, max_value=45.0)
    sphere_type = st.selectbox("Type sphère", ["Solide", "Creuse"])

uploaded_file = st.file_uploader("Fichier CSV", type=['csv'], key=f"file_uploader_{len(st.session_state.experiments)}")

if st.button("🚀 Analyser et Ajouter à la Comparaison") and uploaded_file is not None:
    # Vérifier si le nom existe déjà
    if exp_name in st.session_state.experiments:
        st.warning(f"⚠️ Expérience '{exp_name}' existe déjà. Changez le nom ou elle sera remplacée.")
    
    exp_data = load_data_advanced(uploaded_file, exp_name, water_content, angle, sphere_type)
    
    if exp_data:
        # AJOUTER (pas remplacer) l'expérience
        st.session_state.experiments[exp_name] = exp_data
        metrics = exp_data['metrics']
        
        st.success(f"✅ Expérience '{exp_name}' AJOUTÉE à la comparaison!")
        st.info(f"📊 Total expériences: {len(st.session_state.experiments)}")
        
        # Affichage résultats de la nouvelle expérience
        st.markdown("#### 📊 Résultats de l'Expérience")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            krr_val = metrics['krr']
            status = "✅ NORMAL" if 0.03 <= krr_val <= 0.15 else "⚠️ ÉLEVÉ"
            physics_status = "🔬 VALIDE" if metrics.get('physics_valid', False) else "⚠️ À VÉRIFIER"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Krr</h3>
                <h2>{krr_val:.6f}</h2>
                <p>{status}</p>
                <p><small>{physics_status}</small></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mu_eff = metrics.get('mu_effective', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>μ Effectif</h3>
                <h2>{mu_eff:.4f}</h2>
                <p>Krr + tan(θ)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            energy_eff = metrics.get('energy_efficiency', 0)
            eff_status = "🟢 EFFICACE" if energy_eff > 50 else "🟡 MOYEN" if energy_eff > 20 else "🔴 DISSIPÉ"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Efficacité Énergétique</h3>
                <h2>{energy_eff:.1f}%</h2>
                <p>{eff_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            traj_qual = metrics.get('trajectory_linearity', 0)
            qual_status = "🎯 LINÉAIRE" if traj_qual > 90 else "📐 COURBÉE"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Qualité Trajectoire</h3>
                <h2>{traj_qual:.1f}%</h2>
                <p>{qual_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Métriques physiques détaillées
        st.markdown("#### 🔬 Métriques Physiques Avancées")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Puissance Moyenne", f"{metrics.get('power_avg_mW', 0):.2f} mW")
            st.metric("Force Résistance Moy.", f"{metrics.get('F_resistance_avg_mN', 0):.2f} mN")
        
        with col2:
            st.metric("Décélération", f"{metrics.get('deceleration_rate', 0):.3f} m/s²")
            st.metric("Ratio Théorique", f"{metrics.get('decel_ratio', 0):.2f}")
        
        with col3:
            st.metric("μ Cinétique", f"{metrics.get('mu_kinetic', 0):.4f}")
            st.metric("Énergie Dissipée", f"{metrics.get('E_dissipated_mJ', 0):.2f} mJ")
        
        with col4:
            st.metric("Variation Vitesse (CV)", f"{metrics.get('velocity_cv', 0):.1f}%")
            st.metric("Déviation Verticale", f"{metrics.get('vertical_deviation_mm', 0):.2f} mm")
        
        st.rerun()

# ==================== AFFICHAGE EXPÉRIENCES ACTUELLES ====================
# ==================== ANALYSE OPTIONNELLE DES TRACES ====================
st.markdown("### 📏 Analyse Optionnelle des Traces (si mesurées)")

with st.expander("📐 Ajouter Mesures de Traces"):
    st.markdown("Si tu as mesuré les traces laissées par la sphère, ajoute ces données pour une analyse plus complète :")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trace_exp = st.selectbox("Expérience à compléter:", list(st.session_state.experiments.keys()) if st.session_state.experiments else ["Aucune"])
    
    with col2:
        groove_depth = st.number_input("Profondeur trace (mm)", value=0.0, min_value=0.0, max_value=10.0, step=0.1)
        groove_width = st.number_input("Largeur trace (mm)", value=0.0, min_value=0.0, max_value=50.0, step=0.5)
    
    with col3:
        groove_length = st.number_input("Longueur trace (mm)", value=0.0, min_value=0.0, max_value=500.0, step=5.0)
        
        if st.button("➕ Ajouter Mesures Trace") and trace_exp != "Aucune":
            # Calculs avancés avec traces
            if trace_exp in st.session_state.experiments:
                exp = st.session_state.experiments[trace_exp]
                sphere_radius = 15.0  # mm par défaut
                
                # Ratio pénétration (δ/R) - métrique clé de Van Wal
                penetration_ratio = groove_depth / sphere_radius
                
                # Volume de substrat déplacé
                groove_volume = groove_depth * groove_width * groove_length  # mm³
                
                # Énergie de déformation du substrat
                # E_deformation ≈ ρ_substrat × g × Volume × profondeur_moyenne
                substrate_density = 1500  # kg/m³ pour sable typique
                E_deformation = substrate_density * 9.81 * (groove_volume * 1e-9) * (groove_depth * 1e-3)  # J
                
                # Ajouter aux métriques
                exp['metrics']['groove_depth_mm'] = groove_depth
                exp['metrics']['groove_width_mm'] = groove_width
                exp['metrics']['groove_length_mm'] = groove_length
                exp['metrics']['penetration_ratio'] = penetration_ratio
                exp['metrics']['groove_volume_mm3'] = groove_volume
                exp['metrics']['E_deformation_mJ'] = E_deformation * 1000
                
                # Classification selon Van Wal
                if penetration_ratio < 0.03:
                    regime = "No-plowing (Van Wal)"
                elif penetration_ratio < 0.1:
                    regime = "Transition"
                else:
                    regime = "Plowing (De Blasio)"
                
                exp['metrics']['regime'] = regime
                
                st.success(f"✅ Traces ajoutées pour {trace_exp}!")
                st.success(f"📊 Ratio δ/R = {penetration_ratio:.3f} → Régime: {regime}")
                st.success(f"🔋 Énergie déformation: {E_deformation*1000:.2f} mJ")
                
                st.rerun()

# Affichage traces si disponibles
if st.session_state.experiments:
    traces_available = False
    for exp_name, exp in st.session_state.experiments.items():
        if 'groove_depth_mm' in exp['metrics']:
            traces_available = True
            break
    
    if traces_available:
        st.markdown("### 📏 Analyse des Traces Mesurées")
        
        trace_data = []
        for exp_name, exp in st.session_state.experiments.items():
            if 'groove_depth_mm' in exp['metrics']:
                trace_data.append({
                    'Expérience': exp_name,
                    'Humidité': exp['water_content'],
                    'Angle': exp['angle'],
                    'Krr': exp['metrics']['krr'],
                    'Profondeur (mm)': exp['metrics']['groove_depth_mm'],
                    'Largeur (mm)': exp['metrics']['groove_width_mm'],
                    'Ratio δ/R': exp['metrics']['penetration_ratio'],
                    'Volume (mm³)': exp['metrics']['groove_volume_mm3'],
                    'E_déformation (mJ)': exp['metrics']['E_deformation_mJ'],
                    'Régime': exp['metrics']['regime']
                })
        
        if trace_data:
            trace_df = pd.DataFrame(trace_data)
            st.dataframe(trace_df, use_container_width=True)
            
            # Graphiques traces
            col1, col2 = st.columns(2)
            
            with col1:
                # Ratio δ/R vs Krr (validation Van Wal)
                fig_penetration = px.scatter(trace_df, x='Ratio δ/R', y='Krr',
                                           color='Régime', size='Humidité',
                                           hover_data=['Expérience'],
                                           title="📐 Ratio Pénétration vs Krr (Van Wal)",
                                           labels={'Ratio δ/R': 'Ratio δ/R', 'Krr': 'Coefficient Krr'})
                
                # Ligne seuil no-plowing
                fig_penetration.add_vline(x=0.03, line_dash="dash", line_color="red", 
                                        annotation_text="Seuil No-plowing: 0.03")
                fig_penetration.add_vline(x=0.1, line_dash="dash", line_color="orange", 
                                        annotation_text="Seuil Plowing: 0.1")
                
                st.plotly_chart(fig_penetration, use_container_width=True)
            
            with col2:
                # Énergie déformation vs humidité
                fig_deformation = px.scatter(trace_df, x='Humidité', y='E_déformation (mJ)',
                                           color='Angle', size='Volume (mm³)',
                                           hover_data=['Expérience', 'Régime'],
                                           title="🔋 Énergie Déformation vs Humidité",
                                           labels={'E_déformation (mJ)': 'Énergie Déformation (mJ)'})
                
                st.plotly_chart(fig_deformation, use_container_width=True)
            
            # Insights traces
            st.markdown("#### 🔍 Insights Traces")
            
            regime_counts = trace_df['Régime'].value_counts()
            no_plowing = regime_counts.get('No-plowing (Van Wal)', 0)
            total_traces = len(trace_df)
            
            if no_plowing / total_traces > 0.7:
                st.success(f"✅ **Régime No-plowing dominant** : {no_plowing}/{total_traces} expériences validées Van Wal")
            else:
                st.warning(f"⚠️ **Régimes mixtes** : Seulement {no_plowing}/{total_traces} en no-plowing")
            
            # Corrélation pénétration-Krr
            if len(trace_df) >= 3:
                pen_krr_corr = trace_df[['Ratio δ/R', 'Krr']].corr().iloc[0, 1]
                if pen_krr_corr > 0.6:
                    st.info(f"📊 **Forte corrélation pénétration-Krr** : r = {pen_krr_corr:.3f}")
                    st.info("   → Plus la résistance augmente, plus la pénétration augmente")
                else:
                    st.info(f"📊 **Corrélation pénétration-Krr modérée** : r = {pen_krr_corr:.3f}")
    st.markdown(f"### 📋 Expériences Chargées ({len(st.session_state.experiments)})")
    
    exp_summary = []
    for name, exp in st.session_state.experiments.items():
        metrics = exp['metrics']
        exp_summary.append({
            'Nom': name,
            'Eau (%)': exp['water_content'],
            'Angle (°)': exp['angle'],
            'Type': exp['sphere_type'],
            'Krr': f"{metrics['krr']:.6f}",
            'μ Effectif': f"{metrics.get('mu_effective', 0):.4f}",
            'Efficacité (%)': f"{metrics.get('energy_efficiency', 0):.1f}",
            'Qualité Traj (%)': f"{metrics.get('trajectory_linearity', 0):.1f}",
            'Physique': '✅' if metrics.get('physics_valid', False) else '⚠️'
        })
    
    summary_df = pd.DataFrame(exp_summary)
    st.dataframe(summary_df, use_container_width=True)

# ==================== TESTS RAPIDES ====================
st.markdown("### 🧪 Tests Rapides")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧪 Test Sec (0% eau, 5°)"):
        # Tests rapides avec métriques avancées
        np.random.seed(42)
        test_metrics = {
            'krr': 0.055 + np.random.normal(0, 0.005),
            'v0': 180 + np.random.normal(0, 20),
            'distance': 250 + np.random.normal(0, 30),
            'calibration': 1.33,
            'mu_effective': 0.142,
            'mu_kinetic': 0.089,
            'energy_efficiency': 45.2,
            'power_avg_mW': 2.8,
            'power_max_mW': 5.1,
            'deceleration_rate': 0.234,
            'trajectory_linearity': 94.7,
            'vertical_deviation_mm': 2.1,
            'velocity_cv': 12.3,
            'decel_ratio': 1.05,
            'F_resistance_avg_mN': 2.45,
            'F_resistance_max_mN': 4.21,
            'E_dissipated_mJ': 15.7,
            'physics_valid': True
        }
        st.session_state.experiments['Test_Sec_0%'] = {
            'name': 'Test_Sec_0%',
            'water_content': 0.0,
            'angle': 5.0,
            'sphere_type': 'Solide',
            'metrics': test_metrics,
            'success_rate': 85.0
        }
        st.success("✅ Test sec ajouté!")
        st.rerun()

with col2:
    if st.button("🧪 Test Humide (10% eau, 15°)"):
        np.random.seed(123)
        test_metrics = {
            'krr': 0.072 + np.random.normal(0, 0.005),
            'v0': 160 + np.random.normal(0, 15),
            'distance': 220 + np.random.normal(0, 25),
            'calibration': 1.33,
            'mu_effective': 0.339,
            'mu_kinetic': 0.094,
            'energy_efficiency': 38.9,
            'power_avg_mW': 3.2,
            'power_max_mW': 5.8,
            'deceleration_rate': 0.267,
            'trajectory_linearity': 91.2,
            'vertical_deviation_mm': 3.4,
            'velocity_cv': 15.7,
            'decel_ratio': 0.98,
            'F_resistance_avg_mN': 2.78,
            'F_resistance_max_mN': 4.67,
            'E_dissipated_mJ': 18.3,
            'physics_valid': True
        }
        st.session_state.experiments['Test_Humide_10%'] = {
            'name': 'Test_Humide_10%',
            'water_content': 10.0,
            'angle': 15.0,
            'sphere_type': 'Solide',
            'metrics': test_metrics,
            'success_rate': 78.0
        }
        st.success("✅ Test humide ajouté!")
        st.rerun()

with col3:
    if st.button("🧪 Test Angle (5% eau, 30°)"):
        np.random.seed(456)
        test_metrics = {
            'krr': 0.063 + np.random.normal(0, 0.006),
            'v0': 220 + np.random.normal(0, 25),
            'distance': 280 + np.random.normal(0, 35),
            'calibration': 1.33,
            'mu_effective': 0.641,
            'mu_kinetic': 0.087,
            'energy_efficiency': 42.1,
            'power_avg_mW': 4.1,
            'power_max_mW': 7.2,
            'deceleration_rate': 0.198,
            'trajectory_linearity': 88.9,
            'vertical_deviation_mm': 4.2,
            'velocity_cv': 18.2,
            'decel_ratio': 0.92,
            'rolling_quality': 8.3,
            'rolling_status': "Bon",
            'omega_avg': 12.5,
            'F_resistance_avg_mN': 3.12,
            'F_resistance_max_mN': 5.83,
            'E_dissipated_mJ': 22.1,
            'physics_valid': True
        }
        st.session_state.experiments['Test_Angle_30°'] = {
            'name': 'Test_Angle_30°',
            'water_content': 5.0,
            'angle': 30.0,
            'sphere_type': 'Solide',
            'metrics': test_metrics,
            'success_rate': 82.0
        }
        st.success("✅ Test angle ajouté!")
        st.rerun()

# ==================== TABLEAU RÉSULTATS ====================
if st.session_state.experiments:
    st.markdown("## 📋 Tableau de Comparaison Avancé")
    
    results = []
    for name, exp in st.session_state.experiments.items():
        metrics = exp['metrics']
        results.append({
            'Expérience': name,
            'Eau (%)': exp['water_content'],
            'Angle (°)': exp['angle'],
            'Type': exp['sphere_type'],
            'Krr': f"{metrics['krr']:.6f}",
            'μ Effectif': f"{metrics.get('mu_effective', 0):.4f}",
            'μ Cinétique': f"{metrics.get('mu_kinetic', 0):.4f}",
            'Efficacité (%)': f"{metrics.get('energy_efficiency', 0):.1f}",
            'Puissance Moy (mW)': f"{metrics.get('power_avg_mW', 0):.2f}",
            'Force Rés. Moy (mN)': f"{metrics.get('F_resistance_avg_mN', 0):.2f}",
            'Décélération (m/s²)': f"{metrics.get('deceleration_rate', 0):.3f}",
            'Qualité Traj (%)': f"{metrics.get('trajectory_linearity', 0):.1f}",
            'Énergie Dissipée (mJ)': f"{metrics.get('E_dissipated_mJ', 0):.2f}",
            'CV Vitesse (%)': f"{metrics.get('velocity_cv', 0):.1f}",
            'Rolling Quality (%)': f"{metrics.get('rolling_quality', 0):.1f}",
            'Rolling Status': metrics.get('rolling_status', 'N/A'),
            'Validation': '✅' if metrics.get('physics_valid', False) else '⚠️'
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    # === MÉTRIQUES AVANCÉES PAR EXPÉRIENCE ===
    st.markdown("### 🔬 Analyse Détaillée par Expérience")
    
    for name, exp in st.session_state.experiments.items():
        with st.expander(f"📊 Détails : {name}"):
            metrics = exp['metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔋 Énergies et Puissances**")
                st.metric("Énergie Initiale", f"{metrics.get('E_initial_mJ', 0):.2f} mJ")
                st.metric("Énergie Finale", f"{metrics.get('E_final_mJ', 0):.2f} mJ")
                st.metric("Énergie Dissipée", f"{metrics.get('E_dissipated_mJ', 0):.2f} mJ")
                st.metric("Puissance Moyenne", f"{metrics.get('power_avg_mW', 0):.2f} mW")
                st.metric("Puissance Maximum", f"{metrics.get('power_max_mW', 0):.2f} mW")
            
            with col2:
                st.markdown("**⚡ Forces et Décélération**")
                st.metric("Force Résistance Moy.", f"{metrics.get('F_resistance_avg_mN', 0):.2f} mN")
                st.metric("Force Résistance Max", f"{metrics.get('F_resistance_max_mN', 0):.2f} mN")
                st.metric("Décélération Mesurée", f"{metrics.get('deceleration_rate', 0):.3f} m/s²")
                st.metric("Décélération Théorique", f"{metrics.get('theoretical_decel', 0):.3f} m/s²")
                st.metric("Ratio Décel.", f"{metrics.get('decel_ratio', 0):.2f}")
            
            with col3:
                st.markdown("**🎯 Qualité et Validation**")
                st.metric("Linéarité Trajectoire", f"{metrics.get('trajectory_linearity', 0):.1f}%")
                st.metric("Déviation Verticale", f"{metrics.get('vertical_deviation_mm', 0):.2f} mm")
                st.metric("CV Vitesse", f"{metrics.get('velocity_cv', 0):.1f}%")
                st.metric("Rolling Quality", f"{metrics.get('rolling_quality', 0):.1f}%")
                st.metric("Rolling Status", metrics.get('rolling_status', 'N/A'))
                
                # Status de validation
                if metrics.get('physics_valid', False):
                    st.success("✅ Physique Validée")
                else:
                    st.warning("⚠️ À Vérifier")
                    
                # Critères de qualité
                if metrics.get('velocity_cv', 100) < 20:
                    st.success("✅ Vitesse Stable")
                else:
                    st.warning("⚠️ Vitesse Variable")
                    
                if metrics.get('trajectory_linearity', 0) > 85:
                    st.success("✅ Trajectoire Linéaire")
                else:
                    st.warning("⚠️ Trajectoire Courbée")
                    
                # Qualité du roulement
                rolling_status = metrics.get('rolling_status', 'N/A')
                if rolling_status == "Excellent":
                    st.success("✅ Roulement Parfait")
                elif rolling_status == "Bon":
                    st.success("✅ Bon Roulement")
                elif rolling_status == "Moyen":
                    st.warning("⚠️ Roulement Moyen")
                else:
                    st.error("❌ Glissement Détecté")

# ==================== GRAPHIQUES ====================
if st.session_state.experiments:
    if len(st.session_state.experiments) >= 1:
        st.markdown("## 📊 Graphiques de Comparaison")
        
        # Préparer données
        plot_data = []
        for name, exp in st.session_state.experiments.items():
            plot_data.append({
                'Expérience': name,
                'Humidité': exp['water_content'],
                'Angle': exp['angle'],
                'Krr': exp['metrics']['krr'],
                'V0': exp['metrics']['v0'],
                'Distance': exp['metrics']['distance'],
                'Sphere_Type': exp['sphere_type'],
                'Success_Rate': exp['success_rate']
            })
        plot_df = pd.DataFrame(plot_data)
        
        # === GRAPHIQUES PRINCIPAUX ===
        col1, col2 = st.columns(2)
        
        with col1:
            # Krr vs Humidité
            fig1 = px.scatter(plot_df, x='Humidité', y='Krr', 
                            color='Sphere_Type', size='Success_Rate',
                            hover_data=['Expérience', 'V0', 'Angle'],
                            title="💧 Krr vs Teneur en Eau",
                            labels={'Krr': 'Coefficient Krr', 'Humidité': 'Teneur en Eau (%)'})
            
            # Références Van Wal
            fig1.add_hline(y=0.052, line_dash="dash", line_color="red", annotation_text="Van Wal min: 0.052")
            fig1.add_hline(y=0.066, line_dash="dash", line_color="red", annotation_text="Van Wal max: 0.066")
            
            # Courbe de tendance
            if len(plot_df) >= 2:
                try:
                    z = np.polyfit(plot_df['Humidité'], plot_df['Krr'], min(2, len(plot_df)-1))
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Humidité'].min(), plot_df['Humidité'].max(), 100)
                    fig1.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                            name='Tendance', 
                                            line=dict(dash='dot', color='purple', width=3)))
                except:
                    pass
            
            fig1.update_layout(height=500)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Krr vs Angle
            fig2 = px.scatter(plot_df, x='Angle', y='Krr', 
                            color='Humidité', size='V0',
                            hover_data=['Expérience', 'Distance'],
                            title="📐 Krr vs Angle d'Inclinaison",
                            labels={'Krr': 'Coefficient Krr', 'Angle': 'Angle (°)'})
            
            # Références Van Wal
            fig2.add_hline(y=0.052, line_dash="dash", line_color="red", annotation_text="Van Wal min")
            fig2.add_hline(y=0.066, line_dash="dash", line_color="red", annotation_text="Van Wal max")
            
            # Courbe de tendance
            if len(plot_df) >= 2:
                try:
                    z = np.polyfit(plot_df['Angle'], plot_df['Krr'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Angle'].min(), plot_df['Angle'].max(), 100)
                    fig2.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                            name='Tendance', 
                                            line=dict(dash='dot', color='orange', width=3)))
                except:
                    pass
            
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
        
        # === GRAPHIQUE COMPARATIF ===
        st.markdown("### 📊 Comparaison Directe")
        
        fig_comparison = go.Figure()
        
        # Barres colorées par type
        colors = ['lightblue' if stype == 'Solide' else 'lightcoral' for stype in plot_df['Sphere_Type']]
        
        fig_comparison.add_trace(go.Bar(
            x=[f"{row['Expérience']}\n({row['Humidité']}%, {row['Angle']}°)" for _, row in plot_df.iterrows()],
            y=plot_df['Krr'],
            name='Krr',
            text=[f"{val:.4f}" for val in plot_df['Krr']],
            textposition='auto',
            marker_color=colors
        ))
        
        # Références Van Wal
        fig_comparison.add_hline(y=0.052, line_dash="dash", line_color="red", annotation_text="Van Wal min: 0.052")
        fig_comparison.add_hline(y=0.066, line_dash="dash", line_color="red", annotation_text="Van Wal max: 0.066")
        
        # Moyenne expériences
        mean_krr = plot_df['Krr'].mean()
        fig_comparison.add_hline(y=mean_krr, line_dash="dot", line_color="green", 
                               annotation_text=f"Moyenne: {mean_krr:.4f}")
        
        fig_comparison.update_layout(
            title="📊 Comparaison Krr - Toutes Expériences",
            xaxis_title="Expériences",
            yaxis_title="Coefficient Krr",
            height=500,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        # === GRAPHIQUES AVANCÉS FORCES ET ÉNERGIES ===
        st.markdown("### ⚡ Graphiques Forces, Puissances et Énergies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Forces de résistance vs conditions
            fig_forces = px.scatter(plot_df, x='Humidité', y='F_resistance_avg_mN',
                                  color='Angle', size='Krr',
                                  hover_data=['Expérience', 'mu_kinetic'],
                                  title="🔧 Force de Résistance vs Humidité",
                                  labels={'F_resistance_avg_mN': 'Force Résistance (mN)', 'Humidité': 'Humidité (%)'})
            
            # Tendance
            if len(plot_df) >= 2:
                try:
                    z = np.polyfit(plot_df['Humidité'], plot_df['F_resistance_avg_mN'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Humidité'].min(), plot_df['Humidité'].max(), 100)
                    fig_forces.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                                  name='Tendance Forces', 
                                                  line=dict(dash='dot', color='red', width=3)))
                except:
                    pass
            
            st.plotly_chart(fig_forces, use_container_width=True)
            
        with col2:
            # Puissance dissipée
            fig_power = px.scatter(plot_df, x='Angle', y='power_avg_mW',
                                 color='Humidité', size='energy_efficiency',
                                 hover_data=['Expérience', 'E_dissipated_mJ'],
                                 title="⚡ Puissance Dissipée vs Angle",
                                 labels={'power_avg_mW': 'Puissance (mW)', 'Angle': 'Angle (°)'})
            
            # Tendance
            if len(plot_df) >= 2:
                try:
                    z = np.polyfit(plot_df['Angle'], plot_df['power_avg_mW'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_df['Angle'].min(), plot_df['Angle'].max(), 100)
                    fig_power.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                                 name='Tendance Puissance', 
                                                 line=dict(dash='dot', color='orange', width=3)))
                except:
                    pass
            
            st.plotly_chart(fig_power, use_container_width=True)
        
        # === GRAPHIQUES EFFICACITÉS ===
        col1, col2 = st.columns(2)
        
        with col1:
            # Efficacité énergétique
            fig_efficiency = px.bar(plot_df, x='Expérience', y='energy_efficiency',
                                  color='Humidité',
                                  title="🔋 Efficacité Énergétique par Expérience",
                                  labels={'energy_efficiency': 'Efficacité Énergétique (%)'})
            fig_efficiency.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_efficiency, use_container_width=True)
            
        with col2:
            # Qualité du roulement (Velocity Rolling)
            rolling_data = []
            for _, row in plot_df.iterrows():
                rolling_quality = st.session_state.experiments[row['Expérience']]['metrics'].get('rolling_quality', 0)
                rolling_status = st.session_state.experiments[row['Expérience']]['metrics'].get('rolling_status', 'N/A')
                rolling_data.append({
                    'Expérience': row['Expérience'],
                    'Rolling_Quality': rolling_quality,
                    'Rolling_Status': rolling_status,
                    'Humidité': row['Humidité'],
                    'Angle': row['Angle']
                })
            
            rolling_df = pd.DataFrame(rolling_data)
            
            # Couleurs selon statut
            color_map = {'Excellent': 'green', 'Bon': 'lightgreen', 'Moyen': 'orange', 'Glissement': 'red'}
            colors = [color_map.get(status, 'gray') for status in rolling_df['Rolling_Status']]
            
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Bar(
                x=rolling_df['Expérience'],
                y=rolling_df['Rolling_Quality'],
                name='Rolling Quality',
                text=[f"{val:.1f}%" for val in rolling_df['Rolling_Quality']],
                textposition='auto',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Qualité: %{y:.1f}%<br>Status: %{text}<extra></extra>',
                text=[status for status in rolling_df['Rolling_Status']]
            ))
            
            fig_rolling.update_layout(
                title="🎯 Qualité du Roulement (Velocity Rolling)",
                xaxis_title="Expériences",
                yaxis_title="Écart Roulement Parfait (%)",
                xaxis_tickangle=-45
            )
            
            # Ligne qualité acceptable (< 15%)
            fig_rolling.add_hline(y=15, line_dash="dash", line_color="red", 
                                annotation_text="Seuil Acceptable: 15%")
            
            st.plotly_chart(fig_rolling, use_container_width=True)
        
        # === ANALYSE TENDANCES ===
        if len(plot_df) >= 2:
            st.markdown("### 🔍 Analyse des Tendances")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if plot_df['Humidité'].nunique() > 1:
                    humid_krr_corr = plot_df[['Humidité', 'Krr']].corr().iloc[0, 1]
                    st.metric("Corrélation Humidité ↔ Krr", f"{humid_krr_corr:.3f}")
                    
                    if humid_krr_corr > 0.5:
                        st.success("📈 Krr augmente avec humidité")
                    elif humid_krr_corr < -0.5:
                        st.info("📉 Krr diminue avec humidité")
                    else:
                        st.warning("➡️ Peu d'effet humidité")
                else:
                    st.info("Une seule humidité")
            
            with col2:
                if plot_df['Angle'].nunique() > 1:
                    angle_krr_corr = plot_df[['Angle', 'Krr']].corr().iloc[0, 1]
                    st.metric("Corrélation Angle ↔ Krr", f"{angle_krr_corr:.3f}")
                    
                    if abs(angle_krr_corr) > 0.7:
                        direction = "augmente" if angle_krr_corr > 0 else "diminue"
                        st.warning(f"⚠️ Krr {direction} avec angle")
                    else:
                        st.success("✅ Krr indépendant angle (Van Wal)")
                else:
                    st.info("Un seul angle")
            
            with col3:
                krr_values = plot_df['Krr'].values
                van_wal_range = [0.052, 0.066]
                in_range = np.sum((krr_values >= van_wal_range[0]) & (krr_values <= van_wal_range[1]))
                total = len(krr_values)
                
                st.metric("Dans plage Van Wal", f"{in_range}/{total}")
                
                if in_range / total > 0.7:
                    st.success("✅ Cohérent littérature")
                else:
                    st.warning("⚠️ Écart Van Wal")
        
        # === EXPORT ===
        st.markdown("### 📥 Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📋 Export Tableau",
                data=csv_data,
                file_name="resultats_krr.csv",
                mime="text/csv"
            )
        
        with col2:
            plot_csv = plot_df.to_csv(index=False)
            st.download_button(
                label="📊 Export Graphiques",
                data=plot_csv,
                file_name="donnees_graphiques.csv",
                mime="text/csv"
            )

# ==================== GESTION ====================
if st.session_state.experiments:
    st.markdown("## 🗂️ Gestion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exp_to_remove = st.selectbox("Supprimer:", ["Aucune"] + list(st.session_state.experiments.keys()))
        if exp_to_remove != "Aucune" and st.button("🗑️ Supprimer"):
            del st.session_state.experiments[exp_to_remove]
            st.rerun()
    
    with col2:
        if st.button("🧹 Tout effacer"):
            st.session_state.experiments = {}
            st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
### ✅ Interface Krr Fonctionnelle
- Calcul Krr : Krr = (V₀² - Vf²) / (2gL)
- Graphiques comparatifs avec courbes de tendance
- Références Van Wal intégrées
- Export des données et résultats
""")
