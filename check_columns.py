import pandas as pd
import os

# Probleme de lecture du fichier CSV : les colonnes sont toutes regroupées dans une seule colonne
fichier = "olympic_athletes_ml_ready.csv"
chemin = os.path.join(os.path.dirname(os.path.abspath(__file__)), fichier)

print(f"--- ANALYSE DE : {fichier} ---")

try:
    # Essai 1 : Lecture standard (virgule)
    df = pd.read_csv(chemin)
    print(f"\n[Lecture avec virgules standard]")
    print(f"Colonnes trouvées : {list(df.columns)}")
    
    # Vérification du séparateur
    if len(df.columns) < 2:
        print("\n⚠️ ALERTE : Une seule colonne détectée ! C'est probablement un problème de séparateur (;).")
        
        # Essai 2 : Lecture avec point-virgule
        df_sep = pd.read_csv(chemin, sep=';')
        print(f"\n[Tentative avec point-virgule ';']")
        print(f"Colonnes trouvées : {list(df_sep.columns)}")
        
except Exception as e:
    print(e)