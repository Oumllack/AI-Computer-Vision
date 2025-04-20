import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import ImprovedCNN, load_model, create_model
import os
import torch.nn as nn

# Configuration de la page
st.set_page_config(
    page_title="Classification d'Images CIFAR-10",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé amélioré
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-in;
    }
    .subtitle {
        color: #34495e;
        text-align: center;
        font-size: 1.4em;
        margin-bottom: 2em;
        animation: slideIn 1s ease-out;
    }
    .upload-section {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2.5em;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 2em;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideUp 0.5s ease-out;
    }
    .result-section {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2.5em;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideUp 0.5s ease-out;
    }
    .confidence-bar {
        height: 25px;
        background-color: #ecf0f1;
        border-radius: 12px;
        margin: 8px 0;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        border-radius: 12px;
        transition: width 1s ease-in-out;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .class-label {
        font-weight: 600;
        color: #2c3e50;
        font-size: 1.1em;
    }
    .probability-value {
        font-weight: 600;
        color: #2c3e50;
        font-size: 1.1em;
    }
    .info-card {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1.5em;
        border-radius: 12px;
        margin: 1em 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .prediction-card {
        background: linear-gradient(135deg, #3498db, #2ecc71);
        color: white;
        padding: 1.5em;
        border-radius: 12px;
        margin: 1em 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: pulse 2s infinite;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .stFileUploader > div {
        border: 2px dashed #3498db;
        border-radius: 12px;
        padding: 2em;
        background-color: rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
    }
    .stFileUploader > div:hover {
        background-color: rgba(255, 255, 255, 0.8);
        border-color: #2ecc71;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre et description
st.markdown('<h1 class="title">Classification d\'Images CIFAR-10</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Découvrez la puissance de l\'IA avec notre classificateur d\'images</p>', unsafe_allow_html=True)

# Classes CIFAR-10 avec emojis
classes = {
    'avion': '✈️',
    'automobile': '🚗',
    'oiseau': '🐦',
    'chat': '🐱',
    'cerf': '🦌',
    'chien': '🐕',
    'grenouille': '🐸',
    'cheval': '🐴',
    'bateau': '⛵',
    'camion': '🚚'
}

# Chargement du modèle
@st.cache_resource
def load_cached_model():
    try:
        st.info("Tentative de chargement du modèle existant...")
        # Vérifier si le fichier existe
        if not os.path.exists('./models/cifar10_cnn.pt'):
            st.error(f"Le fichier du modèle n'existe pas dans {os.path.abspath('./models/cifar10_cnn.pt')}")
            raise FileNotFoundError
            
        # Afficher la taille du fichier
        file_size = os.path.getsize('./models/cifar10_cnn.pt') / (1024 * 1024)  # Taille en MB
        st.info(f"Taille du fichier modèle : {file_size:.2f} MB")
        
        model = load_model()
        
        # Vérifier l'état du modèle
        st.info("État du modèle :")
        st.info(f"- Mode : {'train' if model.training else 'eval'}")
        st.info(f"- Nombre de paramètres : {sum(p.numel() for p in model.parameters())}")
        st.success("Modèle chargé avec succès !")
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        st.warning("Création d'un nouveau modèle simple...")
        os.makedirs('./models', exist_ok=True)
        model = create_model()
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        torch.save(model.state_dict(), './models/cifar10_cnn.pt')
        st.success("Nouveau modèle créé et sauvegardé avec succès !")
    model.eval()
    return model

model = load_cached_model()

# Section de téléchargement
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📤 Téléchargez votre image")
uploaded_file = st.file_uploader("Glissez-déposez votre image ici ou cliquez pour sélectionner", 
                                type=['png', 'jpg', 'jpeg'],
                                help="L'image doit être de taille 32x32 pixels en format RGB")
st.markdown('</div>', unsafe_allow_html=True)

# Section des résultats
if uploaded_file is not None:
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.markdown("### 📷 Image téléchargée")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption='Votre image', width=300, use_column_width=True)
    
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
        st.markdown(f"""
            <div class="prediction-card">
                <h3>🏆 Prédiction</h3>
                <h2>{classes[list(classes.keys())[predicted_class]]} {list(classes.keys())[predicted_class].capitalize()}</h2>
                <p>Confiance: {probabilities[predicted_class].item()*100:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Afficher les probabilités
        st.markdown("#### 📈 Détail des probabilités")
        for i, prob in enumerate(probabilities):
            prob_value = prob.item() * 100
            color = "#2ecc71" if i == predicted_class else "#3498db"
            
            st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span class="class-label">{classes[list(classes.keys())[i]]} {list(classes.keys())[i].capitalize()}</span>
                        <span class="probability-value">{prob_value:.2f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {prob_value}%; background: linear-gradient(90deg, {color}, {color});"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Informations supplémentaires
st.markdown("""
    <div class="info-card">
        <h3>ℹ️ À propos du modèle</h3>
        <p>Ce modèle de deep learning a été entraîné sur le dataset CIFAR-10 et atteint une précision de 91.18%.</p>
        <p>Architecture : CNN avec Batch Normalization et Dropout</p>
    </div>
    
    <div class="info-card">
        <h3>📝 Guide d'utilisation</h3>
        <ol>
            <li>Téléchargez une image de l'une des 10 classes suivantes :</li>
            <ul>
                <li>✈️ Avion</li>
                <li>🚗 Automobile</li>
                <li>🐦 Oiseau</li>
                <li>🐱 Chat</li>
                <li>🦌 Cerf</li>
                <li>🐕 Chien</li>
                <li>🐸 Grenouille</li>
                <li>🐴 Cheval</li>
                <li>⛵ Bateau</li>
                <li>🚚 Camion</li>
            </ul>
            <li>Attendez que le modèle analyse l'image</li>
            <li>Consultez les résultats et les probabilités détaillées</li>
        </ol>
    </div>
    
    <div class="info-card">
        <h3>💡 Conseils</h3>
        <ul>
            <li>Utilisez des images claires et bien centrées</li>
            <li>Évitez les images trop complexes ou floues</li>
            <li>Pour de meilleurs résultats, utilisez des images de 32x32 pixels</li>
        </ul>
    </div>
""", unsafe_allow_html=True) 