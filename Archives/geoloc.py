import pandas as pd
import os

# --- DICTIONNAIRE COMPLET (Hardcoded) ---
# L'enfer des requete API sur des sites pourris qui restreignent à mort le nombre de requetes.
# Au lieu de faire 1000 requetes pour geocoder les sites, on va faire du matching de mots-clés dans les noms de sites.
# On complète le dictionnaire ALL_VENUES_DB au fur et à mesure des essais.

ALL_VENUES_DB = {
    # --- LES GRANDS STADES ---
    "Dignity Health Sports Park": (33.8644, -118.2611),
    "SoFi Stadium": (33.9535, -118.3392),
    "Memorial Coliseum": (34.0141, -118.2879),
    "Crypto.com Arena": (34.0430, -118.2673),
    "Rose Bowl": (34.1613, -118.1676),
    "Intuit Dome": (33.9456, -118.3414),
    "BMO Stadium": (34.0112, -118.2856), 
    "Galen Center": (34.0210, -118.2801),
    "Peacock Theater": (34.0416, -118.2647),
    "Dodger Stadium": (34.0738, -118.2400),
    "Angel Stadium": (33.8003, -117.8827),
    "Honda Center": (33.8078, -117.8765),
    "Hollywood Park": (33.9534, -118.3390),
    
    # --- LONG BEACH ---
    "Long Beach Convention Center": (33.7651, -118.1924),
    "Long Beach Arena": (33.7644, -118.1906),
    "Long Beach Waterfront": (33.7610, -118.1970),
    "Alamitos Beach": (33.7656, -118.1793),
    "Belmont Veterans Memorial Pier": (33.7576, -118.1457),
    "Marine Stadium": (33.7592, -118.1232),
    
    # --- DOWNTOWN & USC ---
    "Grand Park": (34.0564, -118.2461),
    "Convention Center": (34.0403, -118.2696),
    "USC Sports Center": (34.0242, -118.2863),
    
    # --- AUTRES SITES ---
    "Santa Monica State Beach": (34.0118, -118.5028),
    "Riviera Country Club": (34.0506, -118.5005),
    "Sepulveda Basin": (34.1728, -118.4746),
    "Lake Perris": (33.8608, -117.2289),
    "Galway Downs": (33.5222, -117.0375),
    "Universal Studios": (34.1381, -118.3534),
    "Veloway": (34.1728, -118.4746),
    "Frank G. Bonelli": (34.0934, -117.8072),
    "Temecula": (33.4936, -117.1484),
    "Fairplex": (34.0921, -117.7699),
    "Industry Hills": (34.0186, -117.9388),
    "Whittier Narrows": (34.0207, -118.0673),
    "LA Clays": (34.0336, -118.0514),
    "Santa Anita Park": (34.1394, -118.0418),
    "Trestles": (33.3853, -117.5939),
    "UCLA": (34.0689, -118.4452),
    "Pauley Pavilion": (34.0703, -118.4468),
    "Walter Pyramid": (33.7876, -118.1141),
    "Venice Beach": (33.9850, -118.4695),
    "San Dimas": (34.1067, -117.8067)
}

# --- FONCTION DE GEOCODAGE INSTANTANÉ ---
def run_geocoding_instant():
    input_file = "la28_venues_clean.csv"
    output_file = "la28_venues_geocoded.csv"

    if not os.path.exists(input_file):
        print(f"❌ Fichier '{input_file}' introuvable.")
        return

    print("📡 Chargement des données...")
    df = pd.read_csv(input_file)
    df['Venue'] = df['Venue'].astype(str)

    # Initialisation
    df['latitude'] = None
    df['longitude'] = None

    print(f"📍 Mapping immédiat de {len(df)} sites...")

    found_count = 0
    
    for index, row in df.iterrows():
        venue_text = row['Venue']
        
        # Recherche par mots-clés dans notre super-dictionnaire
        for key, coords in ALL_VENUES_DB.items():
            # Si le mot clé (ex: "SoFi") est dans le nom complet (ex: "Inglewood SoFi Stadium")
            if key.lower() in venue_text.lower():
                df.at[index, 'latitude'] = coords[0]
                df.at[index, 'longitude'] = coords[1]
                found_count += 1
                # On arrête de chercher pour cette ligne
                break
    
    # On sauvegarde tout (même ceux non trouvés, on les filtrera dans Streamlit)
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print(f"🚀 TERMINE EN 0.1 SECONDE !")
    print(f"✅ {found_count} sites localisés sur {len(df)}.")
    print(f"📁 Fichier généré : {output_file}")
    print("👉 Tu peux lancer ton Streamlit maintenant.")

if __name__ == "__main__":
    run_geocoding_instant()