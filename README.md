# Classificateur d'Images CIFAR-10

Ce projet est un classificateur d'images basé sur un réseau de neurones convolutif (CNN) entraîné sur le dataset CIFAR-10.

## Fonctionnalités

- Classification d'images en 10 catégories : avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion
- Interface utilisateur web avec Streamlit
- Modèle CNN personnalisé avec PyTorch
- Précision du modèle : ~71.5%

## Installation

1. Cloner le repository
2. Créer un environnement virtuel :
```bash
python3 -m venv venv
source venv/bin/activate  # Sur Unix/macOS
```
3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Préparer les données :
```bash
python3 src/data_processing.py
```

2. Entraîner le modèle :
```bash
python3 src/train.py
```

3. Lancer l'application :
```bash
streamlit run src/app.py
```

## Structure du Projet

- `src/data_processing.py` : Traitement des données CIFAR-10
- `src/model.py` : Architecture du CNN
- `src/train.py` : Script d'entraînement
- `src/app.py` : Interface Streamlit
- `requirements.txt` : Dépendances du projet

## Modèle

Le modèle utilise une architecture CNN avec :
- 3 couches convolutives
- Couches de pooling
- Dropout pour la régularisation
- 2 couches fully connected 