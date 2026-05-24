import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger le fichier CSV contenant les deux angles
csv_path = "backend/donnees_calibrees.csv"

if not os.path.exists(csv_path):
    print(f"❌ Impossible de trouver le fichier {csv_path}. Lance extraire_dataset.py d'abord.")
else:
    df = pd.read_csv(csv_path)
    print(f"📊 Données chargées : {len(df)} lignes disponibles.")

    # 2. SÉLECTION DES DEUX VARIABLES (Cou + Dos)
    X = df[['angle_cou', 'angle_dos']]
    y = df['label']

    # 3. Séparation Entraînement (80%) / Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"🧠 Entraînement multi-variable sur {len(X_train)} images...")
    print(f"🧪 Test de validation sur {len(X_test)} images...")

    # Un excellent compromis pour stabiliser et maximiser la précision
    clf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=4, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Évaluation du nouveau modèle
    y_pred = clf.predict(X_test)
    precision = accuracy_score(y_test, y_pred) * 100
    
    print("\n=============================================")
    print(f"🎯 NOUVELLE PRÉCISION DU MODÈLE : {precision:.2f}%")
    print("=============================================\n")

    # Afficher le rapport détaillé pour le jury (Precision, Recall par classe)
    print("📋 Rapport de classification détaillé :")
    print(classification_report(y_test, y_pred, target_names=['Mauvaise Posture', 'Bonne Posture']))

    # 6. Sauvegarder le nouveau cerveau robuste
    modele_path = "backend/modele_posture.pkl"
    with open(modele_path, 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"💾 Modèle Random Forest sauvegardé avec succès dans '{modele_path}' !")