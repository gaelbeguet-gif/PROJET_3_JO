import os

dossier = r"C:\Users\gawal\Documents\DATA_ANALYST\Visual_studio_code\PROJET_3"

print(f"--- ANALYSE DU DOSSIER : {dossier} ---")

if os.path.exists(dossier):
    fichiers = os.listdir(dossier)
    found = False
    for f in fichiers:
        print(f"• '{f}'")  # J'ai mis des quotes '' pour voir s'il y a des espaces
        if "athlete_events_ml_ready" in f:
            found = True
            print(f"   >>> POTENTIEL CANDIDAT TROUVÉ : Copiez ce nom exact -> {f}")
    
    if not found:
        print(">>> AUCUN fichier ressemblant trouvé.")
else:
    print(">>> Le dossier lui-même n'existe pas !")