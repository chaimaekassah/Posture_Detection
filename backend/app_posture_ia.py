import os
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from collections import deque # Import pour créer une file d'attente fixe
import pandas as pd
# 1. Charger les modèles
print("⏳ Chargement du modèle de squelette YOLOv8-Pose...")
yolo_model = YOLO('yolov8n-pose.pt')

modele_ml_path = "backend/modele_posture.pkl"
print("⏳ Chargement du classifieur Random Forest optimisé...")
with open(modele_ml_path, 'rb') as f:
    classifier = pickle.load(f)

# --- NOUVEAU : FILTRES DE STABILISATION (Moyenne glissante sur 15 frames) ---
historique_cou = deque(maxlen=15)
historique_dos = deque(maxlen=15)

# 2. Démarrer la webcam
cap = cv2.VideoCapture(0)

print("\n🚀 Système de Posture Stabilisé Activé !")
print("-> Appuyez sur 'q' pour quitter.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = yolo_model(frame, verbose=False)

    if len(results[0].keypoints) > 0:
        points = results[0].keypoints.xy[0].cpu().numpy()
        
        if points[4][0] != 0 and points[6][0] != 0:     # Côté droit
            oreille, epaule = points[4], points[6]
            hanche = points[12]
        elif points[3][0] != 0 and points[5][0] != 0:   # Côté gauche
            oreille, epaule = points[3], points[5]
            hanche = points[11]
        else:
            oreille, epaule, hanche = None, None, None

        if oreille is not None and epaule is not None:
            # Calcul des angles bruts
            dx_cou = oreille[0] - epaule[0]
            dy_cou = oreille[1] - epaule[1]
            angle_cou_brut = np.degrees(np.arctan2(abs(dx_cou), abs(dy_cou)))

            if hanche is not None and hanche[0] != 0:
                dx_dos = epaule[0] - hanche[0]
                dy_dos = epaule[1] - hanche[1]
                angle_dos_brut = np.degrees(np.arctan2(abs(dx_dos), abs(dy_dos)))
            else:
                angle_dos_brut = 0.0

            # --- AJOUTER LES ANGLES AU BUFFER ---
            historique_cou.append(angle_cou_brut)
            historique_dos.append(angle_dos_brut)

            # --- CALCULER LA MOYENNE GLISSANTE ---
            angle_cou_lisse = np.mean(historique_cou)
            angle_dos_lisse = np.mean(historique_dos)

            # --- PRÉDICTION MULTI-VARIABLE SUR LES ANGLES LISSÉS ---
            # 1. On emballe les données lissées dans un mini DataFrame avec les bons noms de colonnes
            donnees_webcam = pd.DataFrame([[angle_cou_lisse, angle_dos_lisse]], columns=['angle_cou', 'angle_dos'])

            # 2. On l'envoie au classifieur
            prediction = classifier.predict(donnees_webcam)[0]

            if prediction == 1:
                statut = "BONNE POSTURE"
                couleur = (0, 255, 0)
            else:
                statut = "⚠️ MAUVAISE POSTURE ⚠️"
                couleur = (0, 0, 255)

            # Dessin du squelette
            cv2.line(frame, (int(oreille[0]), int(oreille[1])), (int(epaule[0]), int(epaule[1])), (255, 255, 0), 2)
            if hanche is not None and hanche[0] != 0:
                cv2.line(frame, (int(epaule[0]), int(epaule[1])), (int(hanche[0]), int(hanche[1])), (255, 100, 0), 2)

            # Interface graphique (On affiche les valeurs lissées)
            cv2.putText(frame, f"Angle Cou: {angle_cou_lisse:.1f} deg", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)
            cv2.putText(frame, f"Angle Dos: {angle_dos_lisse:.1f} deg", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)
            cv2.putText(frame, statut, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)

    cv2.imshow("Smart Posture - Mode IA Filtre Glissant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()