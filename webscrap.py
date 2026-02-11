import pandas as pd
import requests
from io import StringIO
import re

def scraper_sites_la28_v7():
    """
    Version V7 : Correctif Anti-Doublons.
    Gère le cas où plusieurs colonnes seraient renommées 'Venue' par erreur,
    ce qui faisait planter le nettoyage de texte.
    """
    url = "https://en.wikipedia.org/wiki/Venues_of_the_2028_Summer_Olympics_and_Paralympics"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"🚀 Connexion à Wikipedia (V7)...")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        html_content = StringIO(response.text)
        
        dfs = pd.read_html(html_content, match="Venue")
        print(f"🔎 {len(dfs)} tableaux candidats trouvés.")
        
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        return pd.DataFrame()

    all_sites = []

    for i, df in enumerate(dfs):
        # Gestion des MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        
        # Renommage intelligent
        rename_map = {}
        for col in df.columns:
            col_str = str(col).lower()
            if any(x in col_str for x in ['venue', 'site', 'asset', 'arena', 'stadium']):
                rename_map[col] = 'Venue'
            elif any(x in col_str for x in ['sport', 'event', 'discipline']):
                rename_map[col] = 'Sports'

        df = df.rename(columns=rename_map)

        # --- CORRECTIF CRUCIAL ICI ---
        # Si deux colonnes s'appellent 'Venue', on ne garde que la première
        df = df.loc[:, ~df.columns.duplicated()]
        # -----------------------------

        if 'Venue' in df.columns and 'Sports' in df.columns:
            clean_df = df[['Venue', 'Sports']].copy()
            
            # Pour éviter le crash, on s'assure qu'on travaille bien sur des Strings
            clean_df['Venue'] = clean_df['Venue'].astype(str)
            clean_df['Sports'] = clean_df['Sports'].astype(str)

            # Nettoyage texte sécurisé
            clean_df['Venue'] = clean_df['Venue'].apply(lambda x: re.sub(r'\[.*?\]', '', x))
            clean_df['Sports'] = clean_df['Sports'].apply(lambda x: re.sub(r'\[.*?\]', '', x))
            
            # Filtres
            clean_df = clean_df[~clean_df['Venue'].str.contains("Total", case=False, na=False)]
            clean_df = clean_df[clean_df['Venue'] != ""]
            clean_df = clean_df[clean_df['Sports'].str.len() > 2] # Ignore les sports vides

            all_sites.append(clean_df)
            print(f"   ✅ Tableau {i} intégré ({len(clean_df)} lignes)")

    if not all_sites:
        return pd.DataFrame()

    final_df = pd.concat(all_sites, ignore_index=True)
    return final_df

if __name__ == "__main__":
    df = scraper_sites_la28_v7()
    
    if not df.empty:
        csv_name = "la28_venues_clean.csv"
        df.to_csv(csv_name, index=False)
        print("="*40)
        print(f"🎉 SUCCÈS V7 ! {len(df)} sites récupérés.")
        print(df.head())
        print(f"\n📁 Fichier sauvegardé : {csv_name}")
    else:
        print("❌ Échec : Aucun tableau valide.")