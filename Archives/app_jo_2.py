import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime
import os
import random
from PIL import Image
from google import genai
import pydeck as pdk
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
from PIL import Image, ImageOps

# ==============================================================================
#  INITIALISATION & CONFIGURATION
# ==============================================================================
# --- INITIALISATION UNIQUE ---
if "chat_session" not in st.session_state:
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Clé API manquante dans .streamlit/secrets.toml")
        st.stop()
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    st.session_state.chat_session = model.start_chat(history=[])
    st.session_state.messages = []

# --- LA FONCTION MANQUANTE (AVEC CACHE) ---
@st.cache_data(show_spinner=False)
def obtenir_reponse_ia(question):
    """Envoie la question à l'IA et mémorise la réponse."""
    # Cette ligne utilise la session créée juste au-dessus
    response = st.session_state.chat_session.send_message(question)
    return response.text 

# ==============================================================================
# CONFIGURATION & STYLE (CSS)
# ==============================================================================
st.set_page_config(page_title="Oracle Olympique LA28", page_icon="🥇", layout="centered")

st.markdown("""
    <style>
    /* --- FOND GLOBAL --- */
    .stApp { background-color: #1A1A1A; color: #FFFFFF; }
    
    /* --- TITRES --- */
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

    /* --- CHATBOT --- */
    .stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage td { color: #FFFFFF !important; }
    .stChatMessage h1, .stChatMessage h2, .stChatMessage h3, .stChatMessage strong, .stChatMessage th { color: #FA009C !important; }
    div[data-testid="stChatMessage"] { background-color: #2D2D2D !important; border: 1px solid #444444; border-radius: 10px; }
    .stChatInput textarea { color: #FFFFFF !important; background-color: #2D2D2D !important; }
    
    /* --- DIVERS --- */
    .official-link a { color: #AAAAAA; border: 1px solid #333; }
    div[data-testid="stDataFrame"] { background-color: #1E1E1E; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. FONCTION BANDEAU PHOTO ALEATOIRE
# ==============================================================================
# N'oublie pas d'ajouter : from PIL import Image, ImageOps tout en haut

def afficher_banniere_photos():
    """Affiche une frise de 5 photos recadrées uniformément (600x400)."""
    assets_dir = "assets"
    
    # Si le dossier n'existe pas, on ne fait rien
    if not os.path.exists(assets_dir):
        return

    # 1. Liste des fichiers images valides
    all_files = [f for f in os.listdir(assets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    # On exclut les images systèmes pour ne garder que les photos de sport
    files_to_show = [f for f in all_files if "default" not in f and "LA28" not in f and "master_" not in f]

    # 2. Sélection aléatoire de 5 images (ou moins si pas assez)
    nb_photos = min(len(files_to_show), 5)
    
    if nb_photos > 0:
        selected_photos = random.sample(files_to_show, nb_photos)
        
        # 3. Création des colonnes
        cols = st.columns(nb_photos)
        target_size = (600, 400) # Taille standardisée

        for idx, col in enumerate(cols):
            img_path = os.path.join(assets_dir, selected_photos[idx])
            with col:
                try:
                    with Image.open(img_path) as img:
                        # On redimensionne et recadre proprement
                        img_resized = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)
                        st.image(img_resized, use_container_width=True)
                except Exception:
                    # Si une image est illisible, on laisse un vide propre
                    st.empty()
        
        st.write("")
        st.markdown("---")

# ==============================================================================
# 2. AFFICHAGE ANNEAUX (Fonction réutilisable)
# ==============================================================================
def afficher_header():
    """Affiche les anneaux olympiques centrés en haut de page."""
    # On utilise 3 colonnes pour centrer l'image (1/4 vide, 1/2 image, 1/4 vide)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Olympic_rings_without_rims.svg/1200px-Olympic_rings_without_rims.svg.png", use_container_width=True)
    
    # Un petit espace pour aérer avant le contenu de la page
    st.write("")

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
        st.error(f"❌ Fichier introuvable : {csv_path}")
        return None, None, None, None

    # Préparation des données
    df['NOC'] = df['NOC'].replace({'URS': 'RUS'})
    df['Team'] = df['Team'].replace({'Soviet Union': 'Russia'})
    df = df[df['Season'] == 'Summer'].copy()
    
    # On filtre les épreuves qui ont existé depuis 2012
    recent_events = df[df['Year'] >= 2012]['Event'].unique()

    # Encodage One-Hot manuel pour les médailles
    for m in ['Gold', 'Silver', 'Bronze']:
        df[m] = (df['Medal'] == m).astype(int)

    noc_team_map = df.groupby('NOC')['Team'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])

    # Aggregations Target & Features
    aggregation_rules = {
        'ID': 'nunique',      
        'Age': 'mean', 
        'Height': 'mean', 
        'Weight': 'mean',
        'Gold': 'max',        
        'Silver': 'max',      
        'Bronze': 'max'       
    }
    
    data = df.groupby(['Year', 'NOC', 'Sport', 'Event']).agg(aggregation_rules).reset_index()

    # Renommage des colonnes
    data = data.rename(columns={
        'ID': 'Nb_Athletes', 
        'Age': 'Avg_Age', 
        'Height': 'Avg_Height', 
        'Weight': 'Avg_Weight'
    })

    # Lag Features
    data = data.sort_values(['NOC', 'Event', 'Year'])
    for col in ['Gold', 'Silver', 'Bronze']:
        data[col] = data[col].fillna(0)
        data[f'Prev_{col}'] = data.groupby(['NOC', 'Event'])[col].shift(1).fillna(0)
    
    data['Team'] = data['NOC'].map(noc_team_map)
    
    # Gestion du Pays Hôte (Correction Historique)
    host_map = {
        1996: 'USA', 2000: 'AUS', 2004: 'GRE', 2008: 'CHN', 
        2012: 'GBR', 2016: 'BRA', 2020: 'JPN', 2024: 'FRA'
    }
    data['Is_Host'] = data.apply(lambda row: 1 if host_map.get(row['Year']) == row['NOC'] else 0, axis=1)

    data = data.dropna(subset=['Avg_Age', 'Avg_Height', 'Avg_Weight'])

    # Training
    X = data[['Nb_Athletes', 'Avg_Age', 'Avg_Height', 'Avg_Weight', 'Prev_Gold', 'Prev_Silver', 'Prev_Bronze', 'Is_Host']]
    y = data[['Gold', 'Silver', 'Bronze']]
    
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, y)

    return model, data, df, recent_events

# Chargement global
model, data_ml, df_raw, recent_events = load_and_train_model()

def predict_batch(df_subset, team_name):
    """Prédit les médailles pour un sous-ensemble donné."""
    if df_subset.empty: return 0, 0, 0, pd.DataFrame()

    last_known = df_subset.sort_values('Year', ascending=False).drop_duplicates(['Event'])
    last_known = last_known[last_known['Event'].isin(recent_events)]
    
    if last_known.empty: return 0, 0, 0, pd.DataFrame()

    input_batch = pd.DataFrame({
        'Nb_Athletes': last_known['Nb_Athletes'],
        'Avg_Age': last_known['Avg_Age'],
        'Avg_Height': last_known['Avg_Height'],
        'Avg_Weight': last_known['Avg_Weight'],
        'Prev_Gold': last_known['Gold'],
        'Prev_Silver': last_known['Silver'],
        'Prev_Bronze': last_known['Bronze'],
        'Is_Host': [1 if "United States" in team_name else 0] * len(last_known)
    })
    
    raw_preds = model.predict(input_batch)
    preds = (raw_preds > 0.7).astype(int)
    
    results = last_known[['Sport', 'Event']].copy()
    results['Or'], results['Argent'], results['Bronze'] = preds[:, 0], preds[:, 1], preds[:, 2]
    results['Total_Score'] = results[['Or', 'Argent', 'Bronze']].sum(axis=1)
    
    return preds[:, 0].sum(), preds[:, 1].sum(), preds[:, 2].sum(), results

if data_ml is None: st.stop()

# ==============================================================================
# 4. PAGES DE L'APPLICATION
# ==============================================================================

# ==============================================================================
# 4.1   PAGE ACCUEIL
# ==============================================================================

def page_home():
    # --- ON A SUPPRIMÉ LE BLOC IMAGE ICI ---
    
    # On commence directement par le Titre
    st.markdown("<h1 style='text-align: center; color: #FA009C;'>ORACLE OLYMPIQUE LA28</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>L'Intelligence Artificielle au service de la performance</h3>", unsafe_allow_html=True)
    
    st.write("") # Espace

    # HERO IMAGE (Le Stade en Grand)
    st.image("assets/LA28GamesPlanMainPage.webp", 
             caption="SoFi Stadium - Cérémonie d'Ouverture 2028", 
             use_container_width=True)

    st.markdown("---")

    # EXPLICATIF DU PROJET (3 Colonnes)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### 🏆 Prédictions")
        st.info("Algorithme de **Machine Learning** analysant 120 ans d'histoire pour prédire les médailles.")

    with c2:
        st.markdown("### 🗺️ Carte")
        st.warning("Visualisation interactive des **60+ sites olympiques** en Californie via PyDeck.")

    with c3:
        st.markdown("### 🌟 Athlètes")
        st.error("Annuaire complet des **stars mondiales** à suivre pour les Jeux de 2028.")

    with c4:
        st.markdown("### 🤖 Coach IA")
        st.success("Chatbot intelligent connecté à **Gemini** pour répondre à toutes vos questions.")

    st.write("")
    st.markdown("""
    <div style='text-align: center; background-color: #1E1E1E; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 20px;'>
        <h4>🚀 Prêt à explorer ?</h4>
        <p style='color: #888;'>Utilisez le menu latéral pour naviguer dans l'application.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4.2   PAGE PREDICTION
# ==============================================================================

def page_prediction():
    # --- ON A SUPPRIMÉ LE BLOC IMAGE ICI ---
    
    st.write("") 
    
    col_logo, col_title = st.columns([1, 3])
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LA28GamesPlanMainPage.webp')
    
    with col_logo:
        if os.path.exists(image_path): st.image(Image.open(image_path), use_container_width=True)
    with col_title:
        st.markdown("# L.A. 28 PREDICTOR\n### L'Oracle Olympique")
        st.markdown('<div class="official-link"><a href="https://la28.org/" target="_blank">🔗 Site Officiel LA28.org</a></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Filtres
    col1, col2 = st.columns(2)
    pays_list = sorted(data_ml['Team'].astype(str).unique())
    selected_team = col1.selectbox("🏳️ Pays", pays_list, index=pays_list.index("France") if "France" in pays_list else 0)
    
    sports = sorted(data_ml[data_ml['Team'] == selected_team]['Sport'].unique())
    
    # Sécurité si la liste de sports est vide
    if not sports:
        st.warning(f"Aucune donnée de sport pour {selected_team}.")
        return

    default_sport = 'Swimming' if 'Swimming' in sports else ('Athletics' if 'Athletics' in sports else sports[0])
    selected_sport = col2.selectbox("🏅 Discipline", sports, index=sports.index(default_sport))

    # Analyse
    if st.button('🔮 ANALYSER LA DISCIPLINE'):
        sport_data = data_ml[(data_ml['Team'] == selected_team) & (data_ml['Sport'] == selected_sport)]
        country_data = data_ml[data_ml['Team'] == selected_team]
        
        c_gold, c_silver, c_bronze, _ = predict_batch(country_data, selected_team)
        total_country = c_gold + c_silver + c_bronze
        
        if sport_data.empty:
            st.warning(f"Pas de données historiques pour {selected_team} en {selected_sport}.")
        else:
            s_gold, s_silver, s_bronze, df_details = predict_batch(sport_data, selected_team)
            total_sport = s_gold + s_silver + s_bronze
            
            st.markdown("---")
            k1, k2, k3 = st.columns(3)
            k1.metric(f"Total {selected_sport}", int(total_sport))
            k2.metric(f"Total {selected_team}", int(total_country))
            
            # Calcul du record historique
            history_sum = sport_data.groupby('Year')[['Gold', 'Silver', 'Bronze']].sum().sum(axis=1)
            best_past = history_sum.max() if not history_sum.empty else 0
            
            k3.metric("Record Historique", int(best_past))

            if total_sport >= 1:
                st.success(f"🔥 **{selected_team}** devrait performer en **{selected_sport}** ({int(total_sport)} médailles).")
            else:
                st.info(f"Difficile de décrocher une médaille en **{selected_sport}**.")

            st.markdown("#### 📋 Détail par Épreuve")
            if not df_details.empty:
                st.dataframe(
                    df_details[['Event', 'Or', 'Argent', 'Bronze', 'Total_Score']].sort_values('Total_Score', ascending=False).drop(columns='Total_Score'),
                    column_config={"Or": st.column_config.NumberColumn("🥇", format="%.0f"), "Argent": st.column_config.NumberColumn("🥈", format="%.0f"), "Bronze": st.column_config.NumberColumn("🥉", format="%.0f")},
                    use_container_width=True, hide_index=True
                )

    days = (datetime(2028, 7, 14) - datetime.now()).days
    st.markdown(f"""<div style="background-color:#1E1E1E; border:2px solid #FA009C; padding:20px; border-radius:15px; text-align:center; margin-top:30px;">
        <h3 style="color:white !important; margin:0;">Cérémonie d'Ouverture</h3>
        <h1 style="color:#FA009C !important; font-size:3em; margin:10px 0;">J - {days}</h1>
        <p style="color:#E5E5E5;">Los Angeles 2028</p></div>""", unsafe_allow_html=True)

# ==============================================================================
# 4.3    PAGE CARTE INTERACTIVE
# ==============================================================================

def page_map():
    st.markdown("## 🗺️ Carte Interactive des Sites")
    st.markdown("Explorez les sites olympiques en 3D.")

    # --- 1. DÉFINITION DU DICTIONNAIRE ---
    VENUE_FILES = {
        "sofi": "sofi",
        "coliseum": "coliseum",
        "memorial": "coliseum",
        "rose bowl": "rose_bowl",
        "crypto": "crypto",
        "dignity": "dignity",
        "intuit": "intuit",
        "bmo": "bmo",
        "long beach": "long_beach",
        "sepulveda": "sepulveda",
        "santa monica": "santa_monica",
        "riviera": "riviera",
        "usc": "usc",
        "convention": "convention",
        "dodger": "dodger",
        "angel": "angel",
        "honda": "honda",
        "universal": "universal",
        "pauley": "pauley",
        "peacock": "peacock",
        "galen": "galen",
        "inglewood": "inglewood"
    }

    # --- 2. FONCTION UTILITAIRE ---
    def get_venue_image(venue_name):
        v_lower = str(venue_name).lower()
        match_file = None
        for key, filename in VENUE_FILES.items():
            if key in v_lower:
                match_file = f"{filename}.jpg"
                break
        
        if match_file:
            image_path = os.path.join("assets", match_file)
            if os.path.exists(image_path):
                return image_path
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Olympic_flag.svg/640px-Olympic_flag.svg.png"

    # --- 3. CHARGEMENT DONNÉES ---
    try:
        df_map = pd.read_csv("la28_venues_geocoded.csv")
    except FileNotFoundError:
        st.error("Fichier CSV introuvable.")
        return

    # Filtres
    all_sports = sorted(df_map['Sports'].astype(str).unique().tolist())
    selected_sports = st.multiselect("🔍 Filtrer par discipline", options=all_sports)

    if selected_sports:
        df_display = df_map[df_map['Sports'].isin(selected_sports)]
    else:
        df_display = df_map

    if not df_display.empty:
        df_display = df_display.groupby(['Venue', 'latitude', 'longitude'], as_index=False).agg({
            'Sports': lambda x: ', '.join(sorted(x.unique()))
        })
        df_display = df_display.reset_index(drop=True)

    st.markdown("---")

    # ==========================================================================
    # MISE EN PAGE
    # ==========================================================================

    # 1. ZONE DU HAUT : PHOTO LARGE
    # On crée un emplacement vide qu'on remplira après avoir déterminé l'image
    top_image_container = st.container()

    # 2. ZONE DU BAS : CARTE (2/3) + TABLEAU (1/3)
    col_map, col_table = st.columns([2, 1])

    # Variables par défaut
    view_state = pdk.ViewState(latitude=34.0522, longitude=-118.2437, zoom=9, pitch=0)
    current_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Olympic_flag.svg/640px-Olympic_flag.svg.png"
    current_caption = "Sélectionnez un site dans le tableau ci-dessous"
    street_view_url = None

    # --- LOGIQUE DE SÉLECTION (DANS LA COLONNE DE DROITE) ---
    with col_table:
        st.markdown(f"### 📍 Liste des Sites ({len(df_display)})")
        event = st.dataframe(
            df_display[['Venue', 'Sports']],
            column_config={"Venue": "Site", "Sports": "Sports"},
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=400 # On limite la hauteur du tableau pour qu'il ne soit pas trop long
        )

        if len(event.selection.rows) > 0:
            idx = event.selection.rows[0]
            
            selected_lat = df_display.at[idx, 'latitude']
            selected_lon = df_display.at[idx, 'longitude']
            selected_venue = str(df_display.at[idx, 'Venue'])

            # Zoom sur la carte
            view_state = pdk.ViewState(
                latitude=selected_lat, longitude=selected_lon, zoom=13, pitch=45
            )
            
            # Mise à jour Image
            current_caption = selected_venue
            current_image = get_venue_image(selected_venue)
            street_view_url = f"https://www.google.com/maps/search/?api=1&query={selected_lat},{selected_lon}"
            
            st.toast(f"📸 Vue sur : {selected_venue}")

    # --- AFFICHAGE DE L'IMAGE (EN HAUT) ---
    with top_image_container:
        # On utilise une colonne centrale pour ne pas que l'image soit trop écrasée si elle est en format portrait
        # ou on laisse en full width. Ici, full width :
        
        # Astuce : On utilise ImageOps.fit si tu as importé PIL pour avoir une bannière propre
        # Sinon on affiche l'image standard
        try:
            # Si c'est un fichier local, on le recadre façon "Bannière Cinéma" (1200x400)
            if os.path.exists(current_image) and "http" not in current_image:
                with Image.open(current_image) as img:
                    # Format très large pour le haut de page
                    img_banner = ImageOps.fit(img, (1200, 500), method=Image.Resampling.LANCZOS)
                    st.image(img_banner, caption=current_caption, use_container_width=True)
            else:
                # Si c'est une URL (drapeau par défaut), on l'affiche telle quelle
                st.image(current_image, caption=current_caption, width=200) # Plus petit si c'est le drapeau
        except:
             st.image(current_image, caption=current_caption, use_container_width=True)

        if street_view_url:
            st.link_button("👀 Voir sur Google Street View", street_view_url)
        
        st.write("") # Espace

    # --- AFFICHAGE DE LA CARTE (EN BAS A GAUCHE) ---
    with col_map:
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_display,
            get_position='[longitude, latitude]',
            get_color='[250, 0, 156, 200]',
            get_radius=800,
            pickable=True,
            stroked=True,
            filled=True,
            radius_min_pixels=5, radius_max_pixels=30, line_width_min_pixels=1,
            get_line_color=[255, 255, 255],
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style=None, 
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"html": "<b>{Venue}</b><br/>🏅 <i>{Sports}</i>", "style": {"backgroundColor": "#1E1E1E", "color": "white", "border": "1px solid #FA009C"}}
            )
        )

# ==============================================================================
# 4.4   PAGE ANNuaire des Athlètes (placeholder)
# ==============================================================================

def page_athletes():
    st.title("🌟 Annuaire des Stars LA28")
    st.markdown("Découvrez les athlètes qui marqueront l'histoire en 2028.")

    # Barre de recherche (Fictive pour le moment)
    st.text_input("🔍 Rechercher un athlète (ex: Simone Biles, Teddy Riner...)", placeholder="Tapez un nom...")
    st.write("")

    # --- FICHE PROFIL EXEMPLE (Léon Marchand) ---
    st.markdown("### 🔥 Athlète à la Une")
    
    # On crée un conteneur stylisé
    with st.container():
        col_img, col_info, col_stats = st.columns([1, 2, 1])
        
        with col_img:
            # Photo de Léon Marchand
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/L%C3%A9on_Marchand_2023.jpg/800px-L%C3%A9on_Marchand_2023.jpg", use_container_width=True)
        
        with col_info:
            st.markdown("## 🇫🇷 Léon MARCHAND")
            st.markdown("**Discipline :** Natation")
            st.markdown("**Spécialité :** 400m 4 Nages / Papillon")
            st.markdown("""
            Léon Marchand est le phénomène de la natation mondiale. Après avoir écrasé la concurrence à Paris 2024, il est le grand favori pour devenir la légende de Los Angeles 2028.
            """)
            st.button("Voir la fiche complète")

        with col_stats:
            st.markdown("##### Palmarès JO")
            st.metric("🥇 Or", "4")
            st.metric("🥈 Argent", "1")
            st.metric("🥉 Bronze", "0")

    st.markdown("---")
    
    # --- GRILLE D'AUTRES ATHLÈTES ---
    st.markdown("### 👥 À suivre aussi...")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Simone_Biles_at_the_2016_Olympics_all-around_gold_medal_podium_%2828987361732%29_%28cropped%29.jpg/640px-Simone_Biles_at_the_2016_Olympics_all-around_gold_medal_podium_%2828987361732%29_%28cropped%29.jpg", caption="Simone Biles (USA)")
    with c2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Eliud_Kipchoge_in_Berlin_-_2015.jpg/640px-Eliud_Kipchoge_in_Berlin_-_2015.jpg", caption="Eliud Kipchoge (KEN)")
    with c3:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Mondo_Duplantis_2020.jpg/640px-Mondo_Duplantis_2020.jpg", caption="Mondo Duplantis (SWE)")

# ==============================================================================
# 4.5  PAGE Chatbot (IA)
# ==============================================================================

if st.button("🗑️ Effacer l'historique"):
    # On vide les messages et on réinitialise la session de chat
    st.session_state.messages = []
    # On force la recréation de la session au prochain passage
    if "chat_session" in st.session_state:
        del st.session_state.chat_session
    st.rerun()

def load_lottieurl(url):
    """Fonction pour charger l'animation Lottie depuis une URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None

def page_chatbot():
    st.title("💬 Coach Olympique (IA)")
    
    # 1. Configuration de l'IA (Google Gemini)
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ Clé API Google manquante. Ajoutez-la dans .streamlit/secrets.toml")
        st.stop()

    if "chat_session" not in st.session_state:
        try:
            st.session_state.client = genai.Client(api_key=api_key)
            # Prompt Système : On donne une personnalité à l'IA
            system_prompt = """
            Tu es le 'Coach Olympique LA28', un assistant expert et enthousiaste.
            Ton but est d'aider l'utilisateur à comprendre les Jeux Olympiques, l'histoire, les règles des sports et les athlètes.
            Réponds de manière concise, sportive et encourageante. Utilise des émojis.
            Si la question ne concerne pas le sport ou les JO, refuse poliment de répondre.
            """
            st.session_state.chat_session = st.session_state.client.chats.create(
                model="gemini-2.0-flash",
                history=[
                    {"role": "user", "parts": [{"text": system_prompt}]},
                    {"role": "model", "parts": [{"text": "Compris coach ! Je suis prêt."}]}
                ]
            )
            # Message de bienvenue par défaut
            st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis ton Coach Olympique. Pose-moi une question sur les JO de 2028 ! 🏋️‍♂️"}]
        except Exception as e:
            st.error(f"Erreur connexion IA : {e}")
            st.stop()

    # 2. GESTION DE L'AFFICHAGE "VIDE" (ANIMATION)
    # Si l'historique ne contient que le message de bienvenue (donc l'utilisateur n'a rien dit)
    if len(st.session_state.messages) <= 1:
        
        # Chargement de l'animation (Robot sympa)
        lottie_robot = load_lottieurl("https://lottie.host/02c38ed4-9584-4786-9051-2df880a6498e/O7W8Q3d2Ww.json")
        
        col_anim, col_text = st.columns([1, 2])
        
        with col_anim:
            # Affichage de l'animation
            if lottie_robot:
                st_lottie(lottie_robot, height=250, key="robot_anim")
            else:
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Robot_icon.svg/1024px-Robot_icon.svg.png", width=150)

        with col_text:
            # Explication du projet et invitation à poser une question
            st.markdown("### 👋 Comment ça marche ?")
            st.markdown("""
            Je suis une intelligence artificielle de pointe.
            
            **Je peux répondre à des questions passionnantes comme :**
            * 📜 *Pourquoi le Curling est le meilleur sport de 2026 ?*
            * 🏊‍♂️ *Quelles sont les règles du plongeon synchronisé masculin ?*
            * 🥇 *Qui a gagné le plus de médailles en chocolat en 2024 ?*
            * 📅 *Quand aura lieu la cérémonie d'ouverture des bénévoles?*
            """)
            st.info("👇 Écris ta question dans la barre ci-dessous pour commencer !")
        
        st.markdown("---")

    # 3. AFFICHAGE DES MESSAGES (Boucle standard)
    for msg in st.session_state.messages:
        # On n'affiche pas le prompt système caché
        if msg.get("parts"): continue 
        
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 4. ZONE DE SAISIE
    if question := st.chat_input("Ex: Qui est Léon Marchand ?"):
        # Affiche la question de l'utilisateur
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Réponse de l'IA
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyse tactique en cours..."):
                    # --- APPLICATION DU CACHE ICI ---
                    full_resp = obtenir_reponse_ia(question)
                    st.markdown(full_resp)
                
                st.session_state.messages.append({"role": "assistant", "content": full_resp})
            except Exception as e:
                st.error(f"Oups, petite faute technique : {e}")

# ==============================================================================
# 5. ROUTAGE & NAVIGATION (Mis à jour avec Accueil)
# ==============================================================================

# ==============================================================================
# 5.1 NAVIGATION (Sidebar) 
# ==============================================================================

# On utilise le package streamlit-option-menu pour une navigation plus stylée et intuitive
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/3/36/Logo_JO_d%27%C3%A9t%C3%A9_-_Los_Angeles_2028.svg/1200px-Logo_JO_d%27%C3%A9t%C3%A9_-_Los_Angeles_2028.svg.png", use_container_width=True)
    st.write("") 

    selection = option_menu(
        menu_title="Navigation",
        options=["Accueil", "Prédictions", "Carte Interactive", "Athlètes", "Chatbot IA"],
        # J'ai mis "star" pour Athlètes, c'est une valeur sûre !
        icons=["house", "trophy", "map", "star", "robot"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#1E1E1E"},
            "icon": {"color": "#FA009C", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},
            "nav-link-selected": {"background-color": "#FA009C"},
        }
    )
    
    # Petit espace avant le footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'><small>Projet Data Analyst<br>Promo 2026</small></div>", unsafe_allow_html=True)

# ==============================================================================
# 5.2 ROUTAGE DES PAGES
# ==============================================================================

# 1. --- On affiche le Header (Anneaux) ---
afficher_header()

# 2. --- On affiche le bandeau photo ---
# On choisit de ne pas l'afficher sur la page Carte pour ne pas gêner la lisibilité de la carte (trop d'images + carte déjà visuelle)
if selection != "Carte Interactive":
    afficher_banniere_photos()

# 3. --- Ensuite, on charge la page demandée ---
if selection == "Accueil":
    page_home()
elif selection == "Prédictions":
    page_prediction()
elif selection == "Carte Interactive":
    page_map()
elif selection == "Athlètes":
    page_athletes()
elif selection == "Chatbot IA":
    page_chatbot()