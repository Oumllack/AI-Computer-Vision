import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_model():
    """
    Crée un modèle CNN pour la classification d'images
    """
    model = CNN()
    return model

def save_model(model, path='./models/cifar10_cnn.pt'):
    """
    Sauvegarde le modèle entraîné
    """
    torch.save(model.state_dict(), path)
    print(f"Modèle sauvegardé à {path}")

def load_model(path='./models/cifar10_cnn.pt'):
    """
    Charge un modèle sauvegardé
    """
    model = CNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model 