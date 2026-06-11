import os
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from collections import deque
import pandas as pd

# 1. Charger les modèles
print("⏳ Chargement du modèle de squelette YOLOv8-Pose...")
yolo_model = YOLO('yolov8n-pose.pt')

modele_ml_path = "backend/modele_posture.pkl"
print("⏳ Chargement du classifieur (IA Elite)...")
with open(modele_ml_path, 'rb') as f:
    classifier = pickle.load(f)

# --- NOUVEAU : FILTRES DE STABILISATION POUR LES 4 VARIABLES ---
historique_cou = deque(maxlen=15)
historique_dos = deque(maxlen=15)
historique_ratio = deque(maxlen=15)
historique_epaules = deque(maxlen=15)

# 2. Démarrer la webcam
cap = cv2.VideoCapture(0)

print("\n🚀 Système de Posture IA Ultime (4 Variables) Activé !")
print("-> Appuyez sur 'q' pour quitter.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = yolo_model(frame, verbose=False)

    if len(results[0].keypoints) > 0:
        points = results[0].keypoints.xy[0].cpu().numpy()
        
        # Détection du profil pour le cou et le dos
        if points[4][0] != 0 and points[6][0] != 0:     # Côté droit
            oreille, epaule = points[4], points[6]
            hanche = points[12]
        elif points[3][0] != 0 and points[5][0] != 0:   # Côté gauche
            oreille, epaule = points[3], points[5]
            hanche = points[11]
        else:
            oreille, epaule, hanche = None, None, None

        if oreille is not None and epaule is not None:
            # --- 1 & 2. CALCUL COU ET DOS ---
            dx_cou = oreille[0] - epaule[0]
            dy_cou = oreille[1] - epaule[1]
            angle_cou_brut = np.degrees(np.arctan2(abs(dx_cou), abs(dy_cou)))
            dist_cou = np.sqrt(dx_cou**2 + dy_cou**2)

            if hanche is not None and hanche[0] != 0:
                dx_dos = epaule[0] - hanche[0]
                dy_dos = epaule[1] - hanche[1]
                angle_dos_brut = np.degrees(np.arctan2(abs(dx_dos), abs(dy_dos)))
                dist_dos = np.sqrt(dx_dos**2 + dy_dos**2)
            else:
                angle_dos_brut = 0.0
                dist_dos = 0.001

            # --- 3. CALCUL RATIO ---
            ratio_brut = dist_cou / dist_dos if dist_dos > 0 else 0.0

            # --- 4. CALCUL DISTANCE ÉPAULES ---
            epaule_gauche = points[5]
            epaule_droite = points[6]
            if epaule_gauche[0] != 0 and epaule_droite[0] != 0:
                dx_epaules = epaule_gauche[0] - epaule_droite[0]
                dy_epaules = epaule_gauche[1] - epaule_droite[1]
                distance_epaules_brut = np.sqrt(dx_epaules**2 + dy_epaules**2)
            else:
                distance_epaules_brut = 0.0

            # --- AJOUTER LES 4 ANGLES/DISTANCES AU BUFFER ---
            historique_cou.append(angle_cou_brut)
            historique_dos.append(angle_dos_brut)
            historique_ratio.append(ratio_brut)
            historique_epaules.append(distance_epaules_brut)

            # --- CALCULER LA MOYENNE GLISSANTE (LISSAGE) ---
            angle_cou_lisse = np.mean(historique_cou)
            angle_dos_lisse = np.mean(historique_dos)
            ratio_lisse = np.mean(historique_ratio)
            epaules_lisse = np.mean(historique_epaules)

            # --- PRÉDICTION MULTI-VARIABLE (4 Variables) ---
            donnees_webcam = pd.DataFrame(
                [[angle_cou_lisse, angle_dos_lisse, ratio_lisse, epaules_lisse]], 
                columns=['angle_cou', 'angle_dos', 'ratio_posture', 'distance_epaules']
            )

            prediction = classifier.predict(donnees_webcam)[0]

            if prediction == 1:
                statut = "BONNE POSTURE"
                couleur = (0, 255, 0)
            else:
                statut = "⚠️ MAUVAISE POSTURE ⚠️"
                couleur = (0, 0, 255)

            # --- DESSIN ET AFFICHAGE ---
            cv2.line(frame, (int(oreille[0]), int(oreille[1])), (int(epaule[0]), int(epaule[1])), (255, 255, 0), 2)
            if hanche is not None and hanche[0] != 0:
                cv2.line(frame, (int(epaule[0]), int(epaule[1])), (int(hanche[0]), int(hanche[1])), (255, 100, 0), 2)

            cv2.putText(frame, f"Cou: {angle_cou_lisse:.1f} | Dos: {angle_dos_lisse:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)
            cv2.putText(frame, f"Ratio: {ratio_lisse:.2f} | Epaules: {epaules_lisse:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)
            cv2.putText(frame, statut, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)

    cv2.imshow("Smart Posture - Mode IA Elite", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()