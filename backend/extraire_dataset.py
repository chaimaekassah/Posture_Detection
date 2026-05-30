import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# 1. Charger le modèle YOLOv8
print("⏳ Chargement du modèle d'IA YOLOv8...")
model = YOLO('yolov8n-pose.pt')

# Chemin vers ton dossier racine de données
DATASET_PATH = "dataset"
DOSSIERS = {"good": 1, "bad": 0}

donnees_posture = []

print("⏳ Début de l'analyse récursive de ton dataset local...\n")

# 2. Explorer chaque catégorie (good / bad)
for categorie, label in DOSSIERS.items():
    chemin_categorie = os.path.join(DATASET_PATH, categorie)
    
    if not os.path.exists(chemin_categorie):
        print(f"⚠️ Le dossier {chemin_categorie} est introuvable.")
        continue

    # os.walk scanne automatiquement le dossier et ses sous-dossiers (1, 2, 3)
    for racine, sous_dossiers, fichiers in os.walk(chemin_categorie):
        # Filtrer uniquement les images
        images = [f for f in fichiers if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) > 0:
            print(f"📁 Analyse de : {racine} ({len(images)} images trouvées)")
            
            for nom_image in images:
                chemin_image = os.path.join(racine, nom_image)
                
                # Traitement par YOLOv8-Pose
                results = model(chemin_image, verbose=False)
                
                if len(results[0].keypoints) > 0:
                    points = results[0].keypoints.xy[0].cpu().numpy()
                    
                    # Extraction des points du profil visible
                    if points[4][0] != 0 and points[6][0] != 0:     # Profil droit
                        oreille, epaule = points[4], points[6]
                        hanche = points[12]
                    elif points[3][0] != 0 and points[5][0] != 0:   # Profil gauche
                        oreille, epaule = points[3], points[5]
                        hanche = points[11]
                    else:
                        oreille, epaule, hanche = None, None, None

                    # Si l'oreille et l'épaule sont visibles, on calcule
                    if os.path.exists(chemin_image) and oreille is not None and epaule is not None:
                        
                        # --- 1. CALCUL DE L'ANGLE ET DISTANCE DU COU ---
                        dx_cou = oreille[0] - epaule[0]
                        dy_cou = oreille[1] - epaule[1]
                        angle_cou = np.degrees(np.arctan2(abs(dx_cou), abs(dy_cou)))
                        dist_cou = np.sqrt(dx_cou**2 + dy_cou**2) # Théorème de Pythagore

                        # --- 2. CALCUL DE L'ANGLE ET DISTANCE DU DOS ---
                        if hanche is not None and hanche[0] != 0:
                            dx_dos = epaule[0] - hanche[0]
                            dy_dos = epaule[1] - hanche[1]
                            angle_dos = np.degrees(np.arctan2(abs(dx_dos), abs(dy_dos)))
                            dist_dos = np.sqrt(dx_dos**2 + dy_dos**2)
                        else:
                            angle_dos = 0.0
                            dist_dos = 0.001 # Éviter la division par zéro pour le ratio

                        # --- 3. NOUVEAU : CALCUL DU RATIO DE POSTURE ---
                        ratio_posture = dist_cou / dist_dos if dist_dos > 0 else 0.0

                        # Enregistrer les TROIS caractéristiques
                        donnees_posture.append({
                            "chemin_relatif": os.path.relpath(chemin_image, DATASET_PATH),
                            "angle_cou": angle_cou,
                            "angle_dos": angle_dos,
                            "ratio_posture": ratio_posture,
                            "label": label
                        })

# 3. Sauvegarder les résultats dans le fichier CSV du Backend
if donnees_posture:
    df = pd.DataFrame(donnees_posture)
    # S'assurer que le dossier backend existe
    os.makedirs("backend", exist_ok=True)
    df.to_csv("backend/donnees_calibrees.csv", index=False)
    print("\n✅ Extraction terminée avec succès !")
    print(f"📊 Le fichier CSV 'backend/donnees_calibrees.csv' contient désormais {len(df)} lignes avec 3 caractéristiques (Cou, Dos, Ratio).")
else:
    print("\n❌ Échec : Aucune coordonnée n'a pu être extraite. Vérifie la visibilité des corps sur les images.")