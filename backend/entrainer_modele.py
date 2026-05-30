import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger le fichier CSV mis à jour
csv_path = "backend/donnees_calibrees.csv"

if not os.path.exists(csv_path):
    print(f"❌ Impossible de trouver le fichier {csv_path}.")
    exit()

df = pd.read_csv(csv_path)
print(f"📊 Données chargées : {len(df)} lignes.")

# 2. SÉLECTION DES 3 VARIABLES (Cou + Dos + Ratio)
X = df[['angle_cou', 'angle_dos', 'ratio_posture']]
y = df['label']

# 3. Séparation Entraînement / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⏳ Lancement de l'Auto-Optimisation (GridSearch)... (Cela peut prendre quelques secondes)")

# 4. Définir la grille des paramètres à tester (36 combinaisons différentes !)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10, 15],
    'min_samples_split': [2, 4, 6]
}

# 5. Création et exécution du chercheur d'hyperparamètres
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Récupération du MEILLEUR modèle trouvé
meilleur_modele = grid_search.best_estimator_

print("\n🏆 OPTIMISATION TERMINÉE !")
print(f"Meilleurs paramètres trouvés par l'IA : {grid_search.best_params_}")

# 6. Évaluation finale
y_pred = meilleur_modele.predict(X_test)
precision = accuracy_score(y_test, y_pred) * 100

print("\n=============================================")
print(f"🚀 PRÉCISION MAXIMALE ATTEINTE : {precision:.2f}%")
print("=============================================\n")
print(classification_report(y_test, y_pred, target_names=['Mauvaise Posture', 'Bonne Posture']))

# 7. Sauvegarder ce super-modèle
modele_path = "backend/modele_posture.pkl"
with open(modele_path, 'wb') as f:
    pickle.dump(meilleur_modele, f)
print(f"💾 Modèle d'élite sauvegardé dans '{modele_path}' !")