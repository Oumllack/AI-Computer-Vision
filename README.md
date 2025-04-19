# Classification d'Images CIFAR-10 avec Deep Learning

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![Accuracy](https://img.shields.io/badge/accuracy-91.18%25-brightgreen)

Une application de classification d'images utilisant un réseau de neurones convolutif (CNN) entraîné sur le dataset CIFAR-10, avec une interface utilisateur moderne et intuitive.

## 🚀 Fonctionnalités

- **Modèle CNN avancé** avec Batch Normalization et Dropout
- **Précision de 91.18%** sur le dataset de test
- **Interface utilisateur moderne** avec animations et design responsive
- **Visualisation des probabilités** avec barres de progression interactives
- **Support multi-classes** (10 catégories d'images)
- **Optimisation des performances** avec cache et chargement intelligent

## 📸 Catégories supportées

- ✈️ Avion
- 🚗 Automobile
- 🐦 Oiseau
- 🐱 Chat
- 🦌 Cerf
- 🐕 Chien
- 🐸 Grenouille
- 🐴 Cheval
- ⛵ Bateau
- 🚚 Camion

## 🛠️ Technologies utilisées

- **PyTorch** : Framework de deep learning
- **Streamlit** : Interface utilisateur web
- **NumPy** : Calculs numériques
- **PIL** : Traitement d'images
- **Matplotlib** : Visualisation des résultats

## 🏗️ Architecture du modèle

Le modèle utilise une architecture CNN améliorée avec :
- 3 blocs de couches convolutives
- Batch Normalization après chaque couche
- Dropout pour la régularisation
- Couches fully connected avec activation ReLU
- Optimisation avec Adam et learning rate scheduling

## 🚀 Installation

1. Clonez le repository :
```bash
git clone https://github.com/Oumllack/AI-Computer-Vision.git
cd AI-Computer-Vision
```

2. Créez un environnement virtuel et installez les dépendances :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## 💻 Utilisation

1. Lancez l'application Streamlit :
```bash
streamlit run src/app.py
```

2. Ouvrez votre navigateur à l'adresse indiquée (généralement http://localhost:8501)

3. Téléchargez une image et observez les résultats de la classification

## 📊 Résultats

Le modèle atteint une précision de 91.18% sur le dataset de test CIFAR-10, ce qui le place parmi les meilleures performances pour cette tâche.

## 🎨 Interface utilisateur

L'application propose une interface moderne avec :
- Design responsive et animations fluides
- Visualisation claire des probabilités
- Cartes d'information interactives
- Support du glisser-déposer pour les images
- Retour visuel immédiat sur les prédictions

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- Dataset CIFAR-10
- Communauté PyTorch
- Équipe Streamlit
- Tous les contributeurs 