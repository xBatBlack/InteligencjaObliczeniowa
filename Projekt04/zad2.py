import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# a) Wczytywanie i obróbka danych z iris_big.csv
df = pd.read_csv("iris_big.csv")
X = df.iloc[:, 0:4].values
y_text = df.iloc[:, 4].values

# Ponieważ Pytorch chce klas w postaci liczb (0, 1, 2), a nie tekstu, konwertujemy je
le = LabelEncoder()
y = le.fit_transform(y_text)

# Podział na zbiory
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizacja (skalowanie cech jest krytycznie ważne w sieciach neuronowych!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Konwersja na Tensory PyTorcha
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train) # LongTensor bo klasy to indeksy całkowite
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

# Utworzenie Datasetów i DataLoaderów (batchowanie)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# Uczymy sieć podając po 32 przykłady na raz (batch_size) i losowo je mieszamy (shuffle)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# b) Budowa modelu PyTorch
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        # Sieć dwuwarstwowa: Wejście (4) -> Ukryta (16) -> Wyjście (3 klasy)
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU() # Funkcja aktywacji warstwy ukrytej
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Przy stosowaniu CrossEntropyLoss nie podpinamy ręcznie warstwy Softmax, 
        # bo PyTorch ma ją już zaszytą w kalkulatorze straty!
        return x

model = IrisNet()

# CrossEntropyLoss to domyślny standard przy klasyfikacji wielu klas
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# c) Pętla Trenująca
epochs = 50
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Rozpoczynam trening")
for epoch in range(epochs):
    model.train() # Przejście w tryb uczenia
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad() # Zerowanie gradientów przed każdym krokiem!
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # Propagacja wstecz
        optimizer.step() # Aktualizacja wag
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    # Zapamiętujemy statystyki treningowe by narysować wykres
    train_losses.append(running_loss/len(train_loader))
    train_accs.append(correct/total)
    
    # Przejście w tryb ewaluacji (bez uczenia, z wyłączonym gradientem)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())
        
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_accs.append(accuracy_score(y_val, val_predicted.numpy()))

# d) Wizualizacja
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Trening')
plt.plot(val_losses, label='Walidacja')
plt.title('Krzywa straty (Loss)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Trening')
plt.plot(val_accs, label='Walidacja')
plt.title('Krzywa dokładności (Accuracy)')
plt.legend()
plt.show()

# e & f) Końcowe statystyki i Macierz
final_acc = val_accs[-1]
print(f"\nKońcowe Accuracy walidacyjne: {final_acc * 100:.2f}%")

cm = confusion_matrix(y_val, val_predicted.numpy())
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Macierz błędów - Irysy (PyTorch)')
plt.show()

"""
b) Topologia: model typu Multi-layer Perceptron (MLP). Posiada on 4 neurony wejściowe (na 4 cechy), jedną warstwę ukrytą składającą się z 16 neuronów 
     (z funkcją aktywacji ReLU, powszechnie stosowaną dla wydajnego uczenia) oraz warstwę wyjściową liczącą 3 neurony (po jednym dla każdej klasy). Użyto Adam Optimizera.
f) Interpretacja wyników: sieć szybko i bez większych problemów zbiega do optymalnego rozwiązania, co obrazuje stromo opadająca krzywa funkcji Loss oraz wygładzona,
     wysoka dokładność. Brak nagłego wzrostu błędu walidacyjnego na krzywych świadczy o tym, że ustrzegliśmy się zjawiska przetrenowania (overfittingu).
"""