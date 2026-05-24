import os
import cv2
from ultralytics import YOLO


print("⏳ Chargement du modèle YOLOv8-Pose...")
model = YOLO('yolov8n-pose.pt')

# 2. Indiquer le chemin de ton image Kaggle
image_path = "dataset/good/2/2.jpg"

if not os.path.exists(image_path):
    print(f"❌ Erreur : L'image '{image_path}' est introuvable.")
else:
    print("🧠 L'IA YOLOv8 analyse la posture...")
    
    # 3. Lancer la détection sur l'image
    results = model(image_path)
    
    # 4. Récupérer l'image sur laquelle YOLO a dessiné le squelette automatiquement
    annotated_frame = results[0].plot()
    
    # Optionnel pour la suite : récupérer les coordonnées des points clés si besoin
    # keypoints = results[0].keypoints.xy  # Contient les coordonnées (x,y) de chaque articulation
    
    # 5. Afficher le résultat dans une fenêtre OpenCV
    cv2.imshow("Objectif Jour 1 - Reussi avec YOLOv8", annotated_frame)
    print("⌨️ Clique sur l'image et appuie sur une touche pour quitter.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()