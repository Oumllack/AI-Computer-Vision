import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import create_model, save_model
from data_processing import load_and_preprocess_data

def train_model(epochs=10, batch_size=64):
    """
    Entraîne le modèle CNN sur le dataset CIFAR-10
    """
    # Vérifier si GPU est disponible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    # Charger les données
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Créer les dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Créer le modèle
    model = create_model()
    model = model.to(device)
    
    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Afficher le résumé du modèle
    print(model)
    
    # Entraîner le modèle
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # Évaluer le modèle
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Précision sur le jeu de test à l\'époque {epoch + 1}: {100 * correct / total:.2f}%')
    
    # Sauvegarder le modèle
    save_model(model)
    
    return model

if __name__ == "__main__":
    # Démarrer l'entraînement
    train_model() 