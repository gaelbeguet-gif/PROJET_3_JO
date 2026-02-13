import os
import requests
import shutil
import time

# 1. Création du dossier assets
if not os.path.exists("assets"):
    os.makedirs("assets")

# 2. Liste simple des noms de fichiers (sans URLs compliquées)
VENUE_NAMES = [
    "sofi", "coliseum", "rose_bowl", "crypto", "dignity", "intuit", "bmo",
    "long_beach", "sepulveda", "santa_monica", "riviera", "usc", "convention",
    "dodger", "angel", "honda", "universal", "pauley", "peacock", "galen", 
    "inglewood"
]

print("⬇️  Démarrage du téléchargement (Méthode Infaillible)...")

for name in VENUE_NAMES:
    filename = f"assets/{name}.jpg"
    
    # URL MAGIQUE : Picsum génère une image unique basée sur le nom ("seed")
    # Cela garantit que l'image est toujours la même pour un nom donné, et jamais une 404.
    url = f"https://picsum.photos/seed/{name}/800/600"

    try:
        print(f"⏳ Téléchargement pour '{name}'...", end=" ")
        
        # On ajoute un timeout pour ne pas bloquer indéfiniment
        r = requests.get(url, stream=True, timeout=10)
        
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            print("✅ OK")
        else:
            # Si jamais Picsum échoue (très rare), on copie l'image par défaut si elle existe
            print(f"⚠️ Erreur {r.status_code}. Utilisation de l'image par défaut.")
            if os.path.exists("assets/default.png"):
                shutil.copy("assets/default.png", filename)

    except Exception as e:
        print(f"❌ Échec : {e}")
    
    # Petite pause pour être poli avec le serveur
    time.sleep(0.5)

# 3. Vérification de l'image par défaut
if not os.path.exists("assets/default.png"):
    print("⚠️ Téléchargement de l'image de secours...")
    try:
        r = requests.get("https://picsum.photos/seed/default/800/600")
        with open("assets/default.png", 'wb') as f:
            f.write(r.content)
    except:
        pass

print("\n🎉 TOUT EST TÉLÉCHARGÉ ! Lance ton app maintenant :")
print("👉 streamlit run app_jo_2.py")