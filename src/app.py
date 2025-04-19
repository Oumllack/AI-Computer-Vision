import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import load_model

# Titre de l'application
st.title("Classification d'Images CIFAR-10")

# Charger le modèle
@st.cache_resource
def load_cached_model():
    model = load_model()
    model.eval()
    return model

model = load_cached_model()

# Classes CIFAR-10
classes = ['avion', 'automobile', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# Interface utilisateur
st.write("""
### Téléchargez une image pour la classification
L'image doit être de taille 32x32 pixels et en couleur (RGB).
""")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    
    # Prétraiter l'image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Faire la prédiction
    if st.button("Classifier"):
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_class].item()
            
            st.write(f"### Résultat : {classes[predicted_class]}")
            st.write(f"Confiance : {confidence:.2%}")
            
            # Afficher les probabilités pour toutes les classes
            st.write("### Probabilités pour chaque classe :")
            for i, prob in enumerate(probabilities[0]):
                st.write(f"{classes[i]}: {prob.item():.2%}") 