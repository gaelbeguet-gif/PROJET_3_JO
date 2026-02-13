import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import os
from PIL import Image

# ==============================================================================
# CONFIGURATION DE LA PAGE & STYLE LA28
# ==============================================================================
st.set_page_config(
    page_title="Oracle Olympique LA28",
    page_icon="🥇",
    layout="centered"
)

# Injection de CSS pour le thème Noir & Rose (LA28 Style)
st.markdown("""
    <style>
    /* Fond principal Noir */
    .stApp {
        background-color: #1A1A1A;
        color: #FFFFFF;
    }
    
    /* Titres en Rose LA28 */
    h1, h2, h3 {
        color: #FA009C !important;
    }
    
    /* Inputs et Textes */
    .stSelectbox label {
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
    }
    
    /* Métriques (KPIs) */
    div[data-testid="stMetricLabel"] p {
        color: #DDDDDD !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #FA009C !important;
        font-size: 2rem !important;
    }
    
    /* Bouton Principal */
    div.stButton > button {
        background-color: #FA009C;
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: bold;
        width: 100%;
        padding: 10px;
        margin-top: 15px;
    }
    div.stButton > button:hover {
        background-color: #D60085;
        color: white;
    }
    
    /* Tableaux */
    div[data-testid="stDataFrame"] {
        background-color: #1E1E1E;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CHARGEMENT ET ENTRAÎNEMENT
# ==============================================================================
@st.cache_resource
def load_and_train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_name = 'athlete_events_ml_ready.csv'
    csv_path = os.path.join(current_dir, csv_name)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"❌ ERREUR : Fichier '{csv_name}' introuvable dans : {current_dir}")
        return None, None, None

    # --- Traitement ---
    df['NOC'] = df['NOC'].replace({'URS': 'RUS'})
    df['Team'] = df['Team'].replace({'Soviet Union': 'Russia'})
    df = df[df['Season'] == 'Summer'].copy()

    noc_team_map = df.groupby('NOC')['Team'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])

    # Target : On groupe par Event pour avoir la granularité fine
    df_medals = df[df['Medal'] != 'No Medal'].copy()
    target = df_medals.groupby(['Year', 'NOC', 'Sport', 'Event'])['Medal'].count().reset_index()
    target.rename(columns={'Medal': 'Medal_Count'}, inplace=True)

    # Features
    features = df.groupby(['Year', 'NOC', 'Sport', 'Event']).agg({
        'ID': 'nunique', 'Age': 'mean', 'Height': 'mean', 'Weight': 'mean'
    }).reset_index()
    features.rename(columns={'ID': 'Nb_Athletes', 'Age': 'Avg_Age', 
                             'Height': 'Avg_Height', 'Weight': 'Avg_Weight'}, inplace=True)

    data = pd.merge(features, target, on=['Year', 'NOC', 'Sport', 'Event'], how='left')
    data['Medal_Count'] = data['Medal_Count'].fillna(0)

    # Feature Engineering (Lag)
    data_prev = data[['Year', 'NOC', 'Sport', 'Event', 'Medal_Count']].copy()
    data_prev['Year'] = data_prev['Year'] + 4 
    data_prev.rename(columns={'Medal_Count': 'Prev_Medals'}, inplace=True)
    
    data = pd.merge(data, data_prev, on=['Year', 'NOC', 'Sport', 'Event'], how='left')
    data['Prev_Medals'] = data['Prev_Medals'].fillna(0)
    
    data['Team'] = data['NOC'].map(noc_team_map)
    data['Is_Host'] = 0 
    
    data = data.dropna(subset=['Avg_Age', 'Avg_Height', 'Avg_Weight'])

    X = data[['Nb_Athletes', 'Avg_Age', 'Avg_Height', 'Avg_Weight', 'Prev_Medals', 'Is_Host']]
    y = data['Medal_Count']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, data, df

# Chargement
model, data_ml, df_raw = load_and_train_model()

if data_ml is None:
    st.stop()

# ==============================================================================
# FONCTION UTILITAIRE : PRÉDICTION BATCH
# ==============================================================================
def predict_batch(df_subset, team_name):
    """
    Prédit les médailles pour un sous-ensemble de données (ex: tout un sport ou tout un pays).
    """
    if df_subset.empty:
        return 0, pd.DataFrame()
        
    # On prépare les features pour la prédiction (Moyennes historiques)
    # Pour faire simple et rapide, on prend la dernière année connue pour chaque épreuve
    last_known = df_subset.sort_values('Year', ascending=False).drop_duplicates(['Event'])
    
    input_batch = pd.DataFrame({
        'Nb_Athletes': last_known['Nb_Athletes'],
        'Avg_Age': last_known['Avg_Age'],
        'Avg_Height': last_known['Avg_Height'],
        'Avg_Weight': last_known['Avg_Weight'],
        'Prev_Medals': last_known['Medal_Count'], # Le 'Medal_Count' de la dernière année devient le 'Prev_Medals' de 2028
        'Is_Host': [1 if "United States" in team_name else 0] * len(last_known)
    })
    
    # Prédiction
    predictions = model.predict(input_batch)
    
    # Création du DataFrame de résultats
    results = last_known[['Sport', 'Event']].copy()
    results['Prediction'] = predictions
    results['Prediction'] = results['Prediction'].apply(lambda x: round(x, 1)) # Arrondi
    
    total_medals = results['Prediction'].sum()
    
    return total_medals, results

# ==============================================================================
# 2. INTERFACE UTILISATEUR
# ==============================================================================

# --- Header ---
col_left, col_center, col_right = st.columns([1, 10, 1])
with col_center:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Olympic_rings_without_rims.svg/1200px-Olympic_rings_without_rims.svg.png", width=600)

st.write("") 

# --- Titre et Logo ---
current_dir = os.path.dirname(os.path.abspath(__file__))
image_filename = 'LA28GamesPlanMainPage.webp'
image_path = os.path.join(current_dir, image_filename)

col_logo, col_title = st.columns([1, 3])
with col_logo:
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Image introuvable : {image_filename}")

with col_title:
    st.markdown("# L.A. 28 PREDICTOR")
    st.markdown("### L'Oracle Olympique")

st.markdown("---")

# --- SÉLECTION ET KPI TOTAL (Row 1) ---
col1, col2, col3 = st.columns([2, 2, 1.5])

with col1:
    pays_list = sorted(data_ml['Team'].astype(str).unique())
    default_idx = pays_list.index("France") if "France" in pays_list else 0
    selected_team = st.selectbox("🏳️ Pays", pays_list, index=default_idx)

with col2:
    sports_available = sorted(data_ml[data_ml['Team'] == selected_team]['Sport'].unique())
    default_sport_idx = 0
    if 'Swimming' in sports_available:
        default_sport_idx = sports_available.index('Swimming')
    elif 'Athletics' in sports_available:
        default_sport_idx = sports_available.index('Athletics')
    selected_sport = st.selectbox("🏅 Discipline", sports_available, index=default_sport_idx)

# --- CALCUL DU TOTAL PAYS (En temps réel) ---
with col3:
    # On récupère toutes les données de ce pays pour simuler son total 2028
    country_data = data_ml[data_ml['Team'] == selected_team]
    total_country_pred, _ = predict_batch(country_data, selected_team)
    
    st.metric(
        label="Total Pays (Est.)",
        value=f"{int(total_country_pred)}",
        help="Estimation totale des médailles pour ce pays à LA2028, tous sports confondus."
    )

# ==============================================================================
# 3. RÉSULTATS PAR SPORT
# ==============================================================================

if st.button('🔮 ANALYSER LA DISCIPLINE'):
    
    # 1. Prédiction globale pour le sport choisi
    sport_data = data_ml[
        (data_ml['Team'] == selected_team) & 
        (data_ml['Sport'] == selected_sport)
    ]
    
    if sport_data.empty:
        st.warning(f"Pas de données historiques pour {selected_team} en {selected_sport}.")
    else:
        total_sport_pred, df_details = predict_batch(sport_data, selected_team)
        
        # --- Affichage Résultat Principal ---
        st.markdown("---")
        st.subheader(f"Résultats : {selected_team} - {selected_sport}")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Médailles prévues (Total Discipline)", f"{total_sport_pred:.1f}")
        with m_col2:
            # On prend le max historique comme référence
            best_past = sport_data.groupby('Year')['Medal_Count'].sum().max()
            st.metric("Record historique", int(best_past))

        # --- Commentaire ---
        if total_sport_pred >= 1:
            st.success(f"🔥 C'est une discipline forte pour **{selected_team}**.")
        else:
            st.info(f"Difficile de décrocher une médaille en **{selected_sport}** selon les stats.")

        # --- TABLEAU DÉTAILLÉ (Alternative au 3ème menu) ---
        st.markdown("#### 📋 Détail par Épreuve / Catégorie")
        
        # Mise en forme du tableau pour l'affichage
        if not df_details.empty:
            # On trie par les meilleures chances
            df_display = df_details[['Event', 'Prediction']].sort_values(by='Prediction', ascending=False)
            
            # Configuration de l'affichage du tableau
            st.dataframe(
                df_display,
                column_config={
                    "Event": "Épreuve / Catégorie",
                    "Prediction": st.column_config.ProgressColumn(
                        "Chances de Médaille",
                        help="Score prédit (proche de 1 = médaille probable)",
                        format="%.1f",
                        min_value=0,
                        max_value=max(1.5, df_display['Prediction'].max()), # Echelle dynamique
                    ),
                },
                use_container_width=True,
                hide_index=True
            )

# ==============================================================================
# 4. COMPTE A REBOURS
# ==============================================================================
st.write("")
st.write("")

target_date = datetime(2028, 7, 14)
remaining = target_date - datetime.now()

st.markdown(
    f"""
    <div style="
        background-color: #1E1E1E; 
        border: 2px solid #FA009C; 
        padding: 20px; 
        border-radius: 15px; 
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 0 15px rgba(250, 0, 156, 0.3);">
        <h3 style="color:white !important; margin-bottom:0px;">Cérémonie d'Ouverture</h3>
        <h1 style="color:#FA009C !important; font-size: 3em; margin: 10px 0;">J - {remaining.days}</h1>
        <p style="color:#E5E5E5; font-size: 1.2em;">Los Angeles 2028</p>
    </div>
    """, 
    unsafe_allow_html=True
)