import cv2
import numpy as np
from ultralytics import YOLO

# 1. Initialiser le modèle YOLOv8-Pose
print("⏳ Chargement du modèle d'IA...")
model = YOLO('yolov8n-pose.pt')

# 2. Ouvrir le flux de la webcam (0 est l'index de la caméra intégrée)
cap = cv2.VideoCapture(0)

# Variables de calibration globales
angle_reference = None
tolerance = 12.0  # Tolérance de 12 degrés d'écart maximum

print("\n🚀 Système de Posture Activé !")
print("-> Installez-vous bien droit et appuyez sur 'c' pour calibrer.")
print("-> Appuyez sur 'q' pour quitter.\n")

while cap.isOpened():
    # Lire l'image actuelle de la webcam
    success, frame = cap.read()
    if not success:
        print("❌ Impossible d'accéder à la webcam.")
        break

    # Lancer la détection YOLOv8 sur la frame en direct
    results = model(frame, verbose=False)

    # Vérifier si une personne est visible
    if len(results[0].keypoints) > 0:
        points = results[0].keypoints.xy[0].cpu().numpy()
        
        # Par défaut, on essaie d'utiliser le profil droit (Points 4 et 6)
        # Si le profil droit n'est pas visible, on bascule sur le profil gauche (Points 3 Choisi et 5)
        if points[4][0] != 0 and points[6][0] != 0:  # Côté droit visible
            oreille = points[4]
            epaule = points[6]
        elif points[3][0] != 0 and points[5][0] != 0:  # Côté gauche visible
            oreille = points[3]
            epaule = points[5]
        else:
            oreille, epaule = None, None

        # Si l'oreille et l'épaule sont détectées, on fait le calcul
        if oreille is not None and epaule is not None:
            dx = oreille[0] - epaule[0]
            dy = oreille[1] - epaule[1]
            angle_actuel = np.degrees(np.arctan2(abs(dx), abs(dy)))

            # Dessiner la ligne et les points sur le flux vidéo
            cv2.line(frame, (int(oreille[0]), int(oreille[1])), (int(epaule[0]), int(epaule[1])), (255, 255, 0), 2)
            cv2.circle(frame, (int(oreille[0]), int(oreille[1])), 6, (0, 255, 255), cv2.FILLED)
            cv2.circle(frame, (int(epaule[0]), int(epaule[1])), 6, (0, 255, 255), cv2.FILLED)

            # --- LOGIQUE DE DIAGNOSTIC ---
            if angle_reference is None:
                statut = "NON CALIBRE (Appuyez sur 'c')"
                couleur = (255, 165, 0)  # Orange en BGR
            else:
                # Calcul de l'écart par rapport à la bonne posture enregistrée
                ecart = abs(angle_actuel - angle_reference)
                
                if ecart > tolerance:
                    statut = f"MAUVAISE POSTURE ! Ecart: {ecart:.1f} deg"
                    couleur = (0, 0, 255)  # Rouge
                else:
                    statut = "BONNE POSTURE"
                    couleur = (0, 255, 0)  # Vert

            # Afficher les informations en temps réel sur la vidéo
            cv2.putText(frame, f"Angle: {angle_actuel:.1f} deg", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
            cv2.putText(frame, statut, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
            if angle_reference is not None:
                cv2.putText(frame, f"Ref: {angle_reference:.1f} deg", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Afficher la fenêtre vidéo en direct
    cv2.imshow("Smart Posture - Flux Webcam", frame)

    # --- GESTION DES TOUCHES CLAVIER ---
    key = cv2.waitKey(1) & 0xFF
    
    # Si l'utilisateur appuie sur 'c', on enregistre l'angle actuel comme référence
    if key == ord('c') and 'angle_actuel' in locals():
        angle_reference = angle_actuel
        print(f"🎯 Calibration réussie ! Angle de référence enregistré : {angle_reference:.1f}°")
        
    # Si l'utilisateur appuie sur 'q', on ferme la fenêtre
    elif key == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres proprement
cap.release()
cv2.destroyAllWindows()
print("👋 Programme arrêté avec succès.")