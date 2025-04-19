import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

def load_and_preprocess_data():
    """
    Charge et prétraite le dataset CIFAR-10
    """
    # Définir les transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Charger les données
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    
    # Convertir en numpy arrays
    x_train = torch.stack([sample[0] for sample in trainset])
    y_train = torch.tensor([sample[1] for sample in trainset])
    x_test = torch.stack([sample[0] for sample in testset])
    y_test = torch.tensor([sample[1] for sample in testset])
    
    # Convertir les labels en one-hot encoding
    y_train_one_hot = torch.zeros(y_train.size(0), 10)
    y_train_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    y_test_one_hot = torch.zeros(y_test.size(0), 10)
    y_test_one_hot.scatter_(1, y_test.unsqueeze(1), 1)
    
    return (x_train, y_train_one_hot), (x_test, y_test_one_hot)

def save_processed_data(x_train, y_train, x_test, y_test):
    """
    Sauvegarde les données prétraitées
    """
    # Créer le dossier data s'il n'existe pas
    os.makedirs('./data', exist_ok=True)
    
    # Sauvegarder les données
    torch.save(x_train, './data/x_train.pt')
    torch.save(y_train, './data/y_train.pt')
    torch.save(x_test, './data/x_test.pt')
    torch.save(y_test, './data/y_test.pt')

if __name__ == "__main__":
    # Charger et prétraiter les données
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Sauvegarder les données prétraitées
    save_processed_data(x_train, y_train, x_test, y_test)
    
    print("Données prétraitées et sauvegardées avec succès !")
    print(f"Taille du dataset d'entraînement : {x_train.shape[0]} images")
    print(f"Taille du dataset de test : {x_test.shape[0]} images") 