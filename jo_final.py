import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime
import os
import random
import time
from PIL import Image, ImageOps
import google.generativeai as genai
import pydeck as pdk
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie

# ==============================================================================
# CONFIGURATION & STYLE (CSS)
# ==============================================================================
st.set_page_config(page_title="Oracle Olympique LA28", page_icon="🥇", layout="centered")

st.markdown("""
    <style>
    /* --- FOND GLOBAL & SIDEBAR --- */
    .stApp { background-color: #1A1A1A; color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #1E1E1E !important; }
    
    /* --- TITRE NAVIGATION & TEXTES MENU --- */
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }

    /* --- TITRES PRINCIPAUX --- */
    h1, h2, h3 { color: #FA009C !important; }
    
    /* --- INPUTS & SELECTBOX --- */
    .stSelectbox label, .stSelectbox div[data-baseweb="select"] { color: #FFFFFF !important; }
    .stSelectbox div[data-baseweb="select"] > div { background-color: #1E1E1E !important; color: #FFFFFF !important; }
    
    /* --- BOUTONS --- */
    div.stButton > button {
        background-color: #FA009C; color: white; border: none;
        border-radius: 20px; font-weight: bold; width: 100%; padding: 10px; margin-top: 15px;
    }
    div.stButton > button:hover { background-color: #D60085; color: white; }
    
    /* --- KPIS (MÉTRIQUES) --- */
    [data-testid="stMetric"] label, [data-testid="stMetric"] div,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #FFFFFF !important; }
    [data-testid="stMetric"] svg { fill: #AAAAAA !important; }

    /* --- CHATBOT (DESIGN AMÉLIORÉ) --- */
    /* Couleur du texte dans les bulles */
    .stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage td { color: #FFFFFF !important; }
    /* Titres dans les réponses IA (ex: gras, listes) */
    .stChatMessage h1, .stChatMessage h2, .stChatMessage h3, .stChatMessage strong, .stChatMessage th { color: #FA009C !important; }
    /* Fond de la bulle de chat */
    div[data-testid="stChatMessage"] { background-color: #2D2D2D !important; border: 1px solid #444444; border-radius: 10px; }
    /* Zone de saisie (Input) */
    .stChatInput textarea { color: #FFFFFF !important; background-color: #2D2D2D !important; border-color: #444444 !important; }
    /* Couleur de l'icône 'Envoyer' dans le chat */
    button[data-testid="stChatInputSubmitButton"] { color: #FA009C !important; }
    
    /* --- DIVERS --- */
    .official-link a { color: #AAAAAA; border: 1px solid #333; }
    div[data-testid="stDataFrame"] { background-color: #1E1E1E; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. INITIALISATION IA & FONCTIONS CACHÉES
# ==============================================================================

if "chat_session" not in st.session_state:
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ Clé API Google manquante. Ajoutez-la dans .streamlit/secrets.toml")
        st.stop()
    
    try:
        genai.configure(api_key=api_key)
        model_ia = genai.GenerativeModel("gemini-flash-latest")
        
        system_prompt = "Tu es le Coach Olympique LA28. Réponds de manière concise, sportive et encourageante."
        
        st.session_state.chat_session = model_ia.start_chat(history=[
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["Compris coach ! Je suis prêt."]}
        ])
        
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis ton Coach Olympique. Pose-moi une question sur les JO ! 🏋️‍♂️"}]
        
    except Exception as e:
        st.session_state.ia_error = str(e)

@st.cache_data(show_spinner=False)
def obtenir_reponse_ia(question):
    if "ia_error" in st.session_state:
        return f"Erreur de configuration IA : {st.session_state.ia_error}"

    if "chat_session" not in st.session_state:
        return "Session expirée. Cliquez sur 'Effacer l'historique'."

    max_retries = 3
    wait_time = 2
    
    for attempt in range(max_retries):
        try:
            response = st.session_state.chat_session.send_message(question)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if attempt == max_retries - 1:
                return f"🛑 Échec définitif après 3 essais. Erreur : {error_msg}"
            
            if "429" in error_msg or "503" in error_msg:
                time.sleep(wait_time)
                wait_time += 3
                continue
            
            if "404" in error_msg:
                return "Erreur technique : Modèle introuvable avec cette clé API."
    
    return "Service indisponible."

# ==============================================================================
# 1. FONCTIONS UTILITAIRES (VISUEL)
# ==============================================================================

def afficher_header():
    """Affiche les anneaux olympiques centrés."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Olympic_rings_without_rims.svg/1200px-Olympic_rings_without_rims.svg.png", use_container_width=True)
    st.write("")

def afficher_banniere_photos():
    assets_dir = "assets"
    if not os.path.exists(assets_dir): return

    # On accepte toutes les extensions listées dans votre inventaire
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    all_files = [f for f in os.listdir(assets_dir) if f.lower().endswith(valid_extensions)]
    
    # Exclure les images système
    files_to_show = [f for f in all_files if all(x not in f for x in ["default", "LA28", "master_"])]

    nb_photos = min(len(files_to_show), 5)
    if nb_photos > 0:
        selected_photos = random.sample(files_to_show, nb_photos)
        cols = st.columns(nb_photos)
        for idx, col in enumerate(cols):
            img_path = os.path.join(assets_dir, selected_photos[idx])
            with col:
                try:
                    img = Image.open(img_path)
                    # Conversion en RGB pour éviter les erreurs avec les PNG/WEBP transparents
                    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
                    img_resized = ImageOps.fit(img, (600, 400), method=Image.Resampling.LANCZOS)
                    st.image(img_resized, use_container_width=True)
                except Exception:
                    st.empty()
        st.markdown("---")

# ==============================================================================
# 3. LOGIQUE MÉTIER (Model & Utils)
# ==============================================================================

@st.cache_resource
def load_and_train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'athlete_events_ml_ready.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None, None, None, None

    df['NOC'] = df['NOC'].replace({'URS': 'RUS'})
    df['Team'] = df['Team'].replace({'Soviet Union': 'Russia'})
    df = df[df['Season'] == 'Summer'].copy()
    recent_events = df[df['Year'] >= 2012]['Event'].unique()

    for m in ['Gold', 'Silver', 'Bronze']:
        df[m] = (df['Medal'] == m).astype(int)

    noc_team_map = df.groupby('NOC')['Team'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    aggregation_rules = {'ID': 'nunique', 'Age': 'mean', 'Height': 'mean', 'Weight': 'mean', 'Gold': 'max', 'Silver': 'max', 'Bronze': 'max'}
    data = df.groupby(['Year', 'NOC', 'Sport', 'Event']).agg(aggregation_rules).reset_index()
    data = data.rename(columns={'ID': 'Nb_Athletes', 'Age': 'Avg_Age', 'Height': 'Avg_Height', 'Weight': 'Avg_Weight'})

    data = data.sort_values(['NOC', 'Event', 'Year'])
    for col in ['Gold', 'Silver', 'Bronze']:
        data[col] = data[col].fillna(0)
        data[f'Prev_{col}'] = data.groupby(['NOC', 'Event'])[col].shift(1).fillna(0)
    
    data['Team'] = data['NOC'].map(noc_team_map)
    host_map = {1996: 'USA', 2000: 'AUS', 2004: 'GRE', 2008: 'CHN', 2012: 'GBR', 2016: 'BRA', 2020: 'JPN', 2024: 'FRA'}
    data['Is_Host'] = data.apply(lambda row: 1 if host_map.get(row['Year']) == row['NOC'] else 0, axis=1)
    data = data.dropna(subset=['Avg_Age', 'Avg_Height', 'Avg_Weight'])

    X = data[['Nb_Athletes', 'Avg_Age', 'Avg_Height', 'Avg_Weight', 'Prev_Gold', 'Prev_Silver', 'Prev_Bronze', 'Is_Host']]
    y = data[['Gold', 'Silver', 'Bronze']]
    model_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model_rf.fit(X, y)
    return model_rf, data, df, recent_events

model, data_ml, df_raw, recent_events = load_and_train_model()

def predict_batch(df_subset, team_name):
    if df_subset.empty: return 0, 0, 0, pd.DataFrame()
    last_known = df_subset.sort_values('Year', ascending=False).drop_duplicates(['Event'])
    last_known = last_known[last_known['Event'].isin(recent_events)]
    if last_known.empty: return 0, 0, 0, pd.DataFrame()

    input_batch = pd.DataFrame({
        'Nb_Athletes': last_known['Nb_Athletes'], 'Avg_Age': last_known['Avg_Age'],
        'Avg_Height': last_known['Avg_Height'], 'Avg_Weight': last_known['Avg_Weight'],
        'Prev_Gold': last_known['Gold'], 'Prev_Silver': last_known['Silver'],
        'Prev_Bronze': last_known['Bronze'], 'Is_Host': [1 if "United States" in team_name else 0] * len(last_known)
    })
    raw_preds = model.predict(input_batch)
    preds = (raw_preds > 0.5).astype(int)
    results = last_known[['Sport', 'Event']].copy()
    results['Or'], results['Argent'], results['Bronze'] = preds[:, 0], preds[:, 1], preds[:, 2]
    results['Total_Score'] = results[['Or', 'Argent', 'Bronze']].sum(axis=1)
    return preds[:, 0].sum(), preds[:, 1].sum(), preds[:, 2].sum(), results

# ==============================================================================
# 4. PAGES DE L'APPLICATION
# ==============================================================================

def page_home():
    st.markdown("<h1 style='text-align: center; color: #FA009C;'>ORACLE OLYMPIQUE LA28</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>L'Intelligence Artificielle au service de la performance</h3>", unsafe_allow_html=True)
    if os.path.exists("assets/LA28GamesPlanMainPage.webp"):
        st.image("assets/LA28GamesPlanMainPage.webp", caption="SoFi Stadium - Cérémonie d'Ouverture 2028", use_container_width=True)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### 🏆 Prédictions")
        st.info("Machine Learning analysant 120 ans d'histoire.")
    with c2:
        st.markdown("### 🗺️ Carte")
        st.warning("60+ sites olympiques en Californie.")
    with c3:
        st.markdown("### 🌟 Athlètes")
        st.error("Annuaire et biographies IA des stars.")
    with c4:
        st.markdown("### 🤖 Coach IA")
        st.success("Chatbot connecté à Gemini.")

def page_prediction():
    if data_ml is None:
        st.error("Données ML non chargées.")
        return
    col_logo, col_title = st.columns([1, 3])
    image_path = "assets/LA28GamesPlanMainPage.webp"
    with col_logo:
        if os.path.exists(image_path): st.image(Image.open(image_path), use_container_width=True)
    with col_title:
        st.markdown("# L.A. 28 PREDICTOR\n### L'Oracle Olympique")
        st.markdown('<div class="official-link"><a href="https://la28.org/" target="_blank">🔗 Site Officiel LA28.org</a></div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    pays_list = sorted(data_ml['Team'].astype(str).unique())
    selected_team = col1.selectbox("🏳️ Pays", pays_list, index=pays_list.index("France") if "France" in pays_list else 0)
    sports = sorted(data_ml[data_ml['Team'] == selected_team]['Sport'].unique())
    if not sports:
        st.warning(f"Aucune donnée pour {selected_team}.")
        return
    default_sport = 'Swimming' if 'Swimming' in sports else sports[0]
    selected_sport = col2.selectbox("🏅 Discipline", sports, index=sports.index(default_sport))

    if st.button('🔮 ANALYSER LA DISCIPLINE'):
        sport_data = data_ml[(data_ml['Team'] == selected_team) & (data_ml['Sport'] == selected_sport)]
        country_data = data_ml[data_ml['Team'] == selected_team]
        c_gold, c_silver, c_bronze, _ = predict_batch(country_data, selected_team)
        if sport_data.empty:
            st.warning("Pas de données historiques.")
        else:
            s_gold, s_silver, s_bronze, df_details = predict_batch(sport_data, selected_team)
            st.markdown("---")
            k1, k2, k3 = st.columns(3)
            k1.metric(f"Total {selected_sport}", int(s_gold+s_silver+s_bronze))
            k2.metric(f"Total {selected_team}", int(c_gold+c_silver+c_bronze))
            history_sum = sport_data.groupby('Year')[['Gold', 'Silver', 'Bronze']].sum().sum(axis=1)
            k3.metric("Record Historique", int(history_sum.max() if not history_sum.empty else 0))
            st.dataframe(df_details[['Event', 'Or', 'Argent', 'Bronze']].sort_values('Or', ascending=False), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# 4.3 PAGE CARTE INTERACTIVE
# ------------------------------------------------------------------------------
def page_map():
    st.markdown("## 🗺️ Carte Interactive des Sites")
    
    # Dictionnaire COMPLET basé sur votre inventaire de fichiers
    VENUE_FILES = {
        # Sites majeurs
        "sofi": "sofi.jpg", 
        "coliseum": "coliseum.jpg", 
        "rose bowl": "rose_bowl.jpg", 
        "crypto": "crypto.jpg", 
        "dignity": "dignity.jpg", 
        "intuit": "intuit.jpg", 
        "bmo": "bmo.jpg", 
        "inglewood": "inglewood.jpg",
        "hollywood": "hollywood.webp",
        "peacock": "peacock.webp",
        "convention": "convention.jpg",
        "usc": "usc.jpg",
        "galen": "galen.jpg",
        "grand": "grand.webp", # Grand Park
        
        # Long Beach & Côtes
        "long beach": "long beach.jpg", 
        "belmont": "belmont.jpg",  # Ajouté !
        "marine stadium": "marine_stadium.jpg", 
        "santa monica": "santa_monica.jpg", 
        "venice": "venice.jpg", # Ajouté !
        
        # Vallées & Extérieur
        "sepulveda": "sepulveda.jpg", 
        "riviera": "riviera.jpg", 
        "universal": "universal.jpg", 
        "pauley": "pauley.jpg", 
        "ucla": "UCLA.webp", # Ajouté (Attention majuscules)
        "dodger": "dodger.jpg", 
        "angel": "angel.jpg", 
        "honda": "honda.jpg", 
        "trestles": "trestles.jpeg",
        "santa anita": "santa anita.jpg", 
        "fairplex": "fairplex.webp", 
        "industry": "industry.png",
        "alamitos": "alamitos.jpg",
        "clays": "clays.jpg", # Ajouté
        "dome": "dome.jpg",   # Ajouté
        
        # Hors Californie
        "okc": "okc.webp"
    }

    def get_venue_image(venue_name):
        # On met tout en minuscule pour la recherche
        v_lower = str(venue_name).lower()
        
        for key, filename in VENUE_FILES.items():
            # Si le mot clé (ex: "belmont") est dans le nom du site (ex: "Belmont Shore")
            if key in v_lower:
                path = os.path.join("assets", filename)
                if os.path.exists(path): return path
        
        # Image par défaut si aucune correspondance
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Olympic_flag.svg/640px-Olympic_flag.svg.png"

    try:
        df_map = pd.read_csv("la28_venues_geocoded.csv")
    except FileNotFoundError:
        st.error("Fichier CSV introuvable.")
        return

    all_sports = sorted(df_map['Sports'].astype(str).unique().tolist())
    selected_sports = st.multiselect("🔍 Filtrer par discipline", options=all_sports)
    
    if selected_sports:
        df_display = df_map[df_map['Sports'].isin(selected_sports)]
    else:
        df_display = df_map
    
    if not df_display.empty:
        df_display = df_display.groupby(['Venue', 'latitude', 'longitude'], as_index=False).agg({'Sports': lambda x: ', '.join(sorted(x.unique()))})

    top_image_container = st.container()
    col_map, col_table = st.columns([2, 1])
    view_state = pdk.ViewState(latitude=34.0522, longitude=-118.2437, zoom=9, pitch=0)
    
    with col_table:
        event = st.dataframe(df_display[['Venue', 'Sports']], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=400)
        if len(event.selection.rows) > 0:
            idx = event.selection.rows[0]
            row = df_display.iloc[idx]
            view_state = pdk.ViewState(latitude=row['latitude'], longitude=row['longitude'], zoom=13, pitch=45)
            with top_image_container:
                # Appel de la fonction corrigée
                img_path = get_venue_image(row['Venue'])
                st.image(img_path, caption=row['Venue'], use_container_width=True)

    with col_map:
        layer = pdk.Layer("ScatterplotLayer", data=df_display, get_position='[longitude, latitude]', get_color='[250, 0, 156, 200]', get_radius=800, pickable=True)
        st.pydeck_chart(pdk.Deck(initial_view_state=view_state, layers=[layer], tooltip={"html": "<b>{Venue}</b><br/>{Sports}"}))
def page_athletes():
    st.markdown("# 🌟 Annuaire & Biographies IA")
    if df_raw is None: return
    country_map = df_raw.groupby('NOC')['Team'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Inconnu").to_dict()
    
    c1, c2 = st.columns(2)
    countries = sorted(list(set(country_map.values())))
    selected_country = c1.selectbox("🏳️ Pays", countries, index=countries.index("France") if "France" in countries else 0)
    sports = sorted(df_raw[df_raw['NOC'].map(country_map) == selected_country]['Sport'].dropna().unique())
    selected_sport = c2.selectbox("🏅 Discipline", sports)

    athletes_df = df_raw[(df_raw['NOC'].map(country_map) == selected_country) & (df_raw['Sport'] == selected_sport)][['Name', 'Sex', 'Age', 'Height', 'Weight', 'Event', 'Medal']].drop_duplicates()
    
    selection = st.dataframe(athletes_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
    if len(selection.selection.rows) > 0:
        athlete_name = athletes_df.iloc[selection.selection.rows[0]]['Name']
        with st.spinner("Rédaction de la bio..."):
            bio = obtenir_reponse_ia(f"Rédige une bio courte et inspirante de l'athlète {athlete_name} en {selected_sport}.")
            st.markdown(f'<div style="background-color: #2D2D2D; padding: 20px; border-radius: 15px; border-left: 5px solid #FA009C;">{bio}</div>', unsafe_allow_html=True)

def page_chatbot():
    st.title("💬 Coach Olympique (IA)")
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        if "chat_session" in st.session_state: del st.session_state.chat_session
        st.rerun()

    if len(st.session_state.messages) <= 1:
        lottie_url = "https://lottie.host/02c38ed4-9584-4786-9051-2df880a6498e/O7W8Q3d2Ww.json"
        res = requests.get(lottie_url)
        if res.status_code == 200: st_lottie(res.json(), height=200)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    if question := st.chat_input("Pose ta question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"): st.markdown(question)
        with st.chat_message("assistant"):
            resp = obtenir_reponse_ia(question)
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})

# ==============================================================================
# 5. ROUTAGE & NAVIGATION
# ==============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/3/36/Logo_JO_d%27%C3%A9t%C3%A9_-_Los_Angeles_2028.svg/1200px-Logo_JO_d%27%C3%A9t%C3%A9_-_Los_Angeles_2028.svg.png", use_container_width=True)
    selection = option_menu(
        menu_title="Navigation",
        options=["Accueil", "Prédictions", "Carte Interactive", "Athlètes", "Chatbot IA"],
        icons=["house", "trophy", "map", "star", "robot"],
        styles={"container": {"padding": "5px", "background-color": "#1E1E1E"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},
            "nav-link-selected": {"background-color": "#FA009C"},
        }
    )
    st.markdown("---")

# --- Affichage du Header et du Bandeau Photo ---
afficher_header()
if selection != "Carte Interactive":
    afficher_banniere_photos()

# --- Affichage de la Page ---
if selection == "Accueil": page_home()
elif selection == "Prédictions": page_prediction()
elif selection == "Carte Interactive": page_map()
elif selection == "Athlètes": page_athletes()
elif selection == "Chatbot IA": page_chatbot()