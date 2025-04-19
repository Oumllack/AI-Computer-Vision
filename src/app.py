import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import ImprovedCNN, load_model, create_model
import os

# Configuration de la page
st.set_page_config(
    page_title="Classification d'Images CIFAR-10",
    page_icon="🖼️",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title {
        color: #2E86C1;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subtitle {
        color: #5D6D7E;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .upload-section {
        background-color: white;
        padding: 2em;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2em;
    }
    .result-section {
        background-color: white;
        padding: 2em;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .confidence-bar {
        height: 20px;
        background-color: #E5E8E8;
        border-radius: 10px;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        background-color: #2E86C1;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre et description
st.markdown('<h1 class="title">Classification d\'Images CIFAR-10</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Téléchargez une image pour la classifier parmi 10 catégories</p>', unsafe_allow_html=True)

# Classes CIFAR-10
classes = ('avion', 'automobile', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion')

# Chargement du modèle
@st.cache_resource
def load_cached_model():
    try:
        model = load_model()
    except FileNotFoundError:
        st.warning("Modèle non trouvé. Création d'un nouveau modèle...")
        model = create_model()
        torch.save(model.state_dict(), './models/cifar10_cnn.pt')
    model.eval()
    return model

model = load_cached_model()

# Section de téléchargement
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📤 Téléchargez votre image")
uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])
st.markdown('</div>', unsafe_allow_html=True)

# Section des résultats
st.markdown('<div class="result-section">', unsafe_allow_html=True)
if uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.markdown("### 📷 Image téléchargée")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption='Image téléchargée', width=300)
    
    with col2:
        # Prétraitement de l'image
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        # Afficher les résultats
        st.markdown("### 📊 Résultats de la classification")
        st.markdown(f"#### 🏆 Classe prédite: **{classes[predicted_class]}**")
        
        # Afficher les probabilités
        st.markdown("#### 📈 Probabilités par classe:")
        for i, prob in enumerate(probabilities):
            prob_value = prob.item() * 100
            color = "#2E86C1" if i == predicted_class else "#5D6D7E"
            
            st.markdown(f"""
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>{classes[i]}</span>
                        <span>{prob_value:.2f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {prob_value}%; background-color: {color};"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Informations supplémentaires
st.markdown("""
    ### ℹ️ À propos
    Ce modèle a été entraîné sur le dataset CIFAR-10 et atteint une précision de 91.18%.
    Les images doivent être de taille 32x32 pixels en format RGB.
    
    ### 📝 Instructions
    1. Téléchargez une image de l'une des 10 classes suivantes :
       - Avion ✈️
       - Automobile 🚗
       - Oiseau 🐦
       - Chat 🐱
       - Cerf 🦌
       - Chien 🐕
       - Grenouille 🐸
       - Cheval 🐴
       - Bateau ⛵
       - Camion 🚚
    2. Attendez que le modèle classe l'image
    3. Consultez les résultats et les probabilités
""") 