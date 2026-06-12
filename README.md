# 🧘‍♀️ Smart Posture AI | Real-Time Posture Correction System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Deep_Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-red)
![Accuracy](https://img.shields.io/badge/Accuracy-97.5%25-brightgreen)

> **Système intelligent et non-intrusif de détection et de correction de la posture assise en temps réel, propulsé par la vision par ordinateur et le Machine Learning.**

---

## 📖 À propos du projet

Avec l'augmentation du temps passé devant les écrans, les troubles musculosquelettiques (TMS) sont devenus un problème de santé majeur. **Smart Posture AI** est une solution logicielle conçue pour analyser la posture de l'utilisateur à travers une simple webcam, sans nécessiter d'équipement spécialisé.

L'application combine un modèle d'extraction de squelette **YOLOv8-Pose** avec un classifieur d'élite **XGBoost** pour détecter instantanément si l'utilisateur est bien positionné ou s'il s'affaisse (dos voûté, tête en avant, épaules enroulées).

🎥 **[Voir la vidéo de démonstration ici](#) *(Remplacer par ton lien vidéo)***

---

## ✨ Fonctionnalités Clés

* **⚡ Analyse Temps Réel :** Traitement fluide du flux vidéo de la webcam.
* **🧠 Feature Engineering Avancé :** L'IA ne regarde pas de simples pixels, mais analyse 4 caractéristiques géométriques invariantes :
  * Angle d'inclinaison du cou.
  * Angle de courbure du dos.
  * Ratio morphologique (Distance Cou / Distance Dos).
  * Distance inter-épaules (pour détecter l'enroulement).
* **🎯 Précision d'Élite (97.52%) :** Modèle XGBoost optimisé par `GridSearchCV`.
* **🛡️ Stabilité Anti-Jittering :** Implémentation d'un filtre temporel (moyenne glissante sur 15 frames) garantissant une interface sans clignotement.

---

## 🏗️ Architecture du Pipeline Logiciel

1. **Computer Vision (Extraction) :** `YOLOv8n-pose` identifie les points clés du corps humain (oreilles, épaules, hanches).
2. **Data Processing :** Calcul trigonométrique des angles et distances (Théorème de Pythagore).
3. **Machine Learning (Classification) :** Le classifieur binaire (XGBoost) analyse le vecteur de données lissées et prend une décision.
4. **Interface Graphique :** OpenCV dessine le squelette et affiche les alertes visuelles (Vert = OK, Rouge = Danger).

---

## 🚀 Installation & Démarrage

### 1. Cloner le dépôt
```bash
git clone [https://github.com/chaimaekassah/Posture_Detection.git](https://github.com/chaimaekassah/Posture_Detection.git)
cd Posture_Detection
