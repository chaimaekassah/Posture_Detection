import os
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Charger le modèle
model = YOLO('yolov8n-pose.pt')
image_path = "dataset/bad/2/2.jpg" # Ton image actuelle

if not os.path.exists(image_path):
    print(f"❌ Image introuvable : {image_path}")
else:
    img = cv2.imread(image_path)
    results = model(image_path, verbose=False)
    
    if len(results[0].keypoints) > 0:
        points = results[0].keypoints.xy[0].cpu().numpy()
        oreille = points[4]
        epaule = points[6]

        if epaule[0] != 0 and oreille[0] != 0:
            # Calcul de l'angle réel de l'image
            dx_cou = oreille[0] - epaule[0]
            dy_cou = oreille[1] - epaule[1]
            angle_cou = np.degrees(np.arctan2(abs(dx_cou), abs(dy_cou)))

            # --- SIMULATION DE CALIBRATION ---
            # Supposons que l'étudiant s'est calibré et que sa bonne posture est à 40°
            angle_reference = 40.0 
            tolerance = 12.0 # On s'autorise une variation de 12 degrés avant d'alerter

            # Calcul de l'écart par rapport à la bonne posture
            ecart = abs(angle_cou - angle_reference)

            if ecart > tolerance:
                statut = f"MAUVAISE POSTURE (Ecart: {ecart:.1f} deg)"
                couleur = (0, 0, 255) # Rouge
            else:
                statut = "BONNE POSTURE (Calibre)"
                couleur = (0, 255, 0) # Vert

            # Dessin des repères
            cv2.line(img, (int(oreille[0]), int(oreille[1])), (int(epaule[0]), int(epaule[1])), (255, 255, 0), 3)
            cv2.circle(img, (int(oreille[0]), int(oreille[1])), 8, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (int(epaule[0]), int(epaule[1])), 8, (0, 255, 255), cv2.FILLED)
            
            # Affichage des textes
            cv2.putText(img, f"Angle Actuel: {angle_cou:.1f} deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)
            cv2.putText(img, statut, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)
        else:
            print("Points non visibles.")
    else:
        print("Aucune personne détectée.")

    cv2.imshow("Phase 3 - Calcul avec Seuil Calibre", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()