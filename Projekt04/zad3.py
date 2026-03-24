import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

df_diag = pd.read_csv("diagnosis.csv")
X = df_diag.iloc[:, :-1].values
y = df_diag.iloc[:, -1].values

# Podział i Normalizacja
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

# budowa modelu
class DiagnosisNet(nn.Module):
    def __init__(self):
        super(DiagnosisNet, self).__init__()
        self.fc1 = nn.Linear(3, 16)   # 3 parametry medyczne
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)   # Jedno wyjście podsumowujące (Chory/Zdrowy)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) 
        return x

model = DiagnosisNet()

# BCELoss
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
train_losses, val_losses, train_accs, val_accs = [], [], [], []

print("Rozpoczynam trening")
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_losses.append(running_loss/len(train_loader))
    train_accs.append(correct/total)
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())
        
        val_predicted = (val_outputs >= 0.5).float()
        val_accs.append(accuracy_score(y_val, val_predicted.numpy()))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Trening')
plt.plot(val_losses, label='Walidacja')
plt.title('Krzywa straty (Diagnoza)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Trening')
plt.plot(val_accs, label='Walidacja')
plt.title('Krzywa dokładności (Diagnoza)')
plt.legend()
plt.show()

y_val_numpy = y_val_tensor.numpy()
val_pred_numpy = val_predicted.numpy()

acc = accuracy_score(y_val_numpy, val_pred_numpy)
prec = precision_score(y_val_numpy, val_pred_numpy, zero_division=0)
rec = recall_score(y_val_numpy, val_pred_numpy, zero_division=0)

print(f"\nOstateczne statystyki:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")

# Macierz
cm = confusion_matrix(y_val_numpy, val_pred_numpy)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Zdrowy', 'Chory'], yticklabels=['Zdrowy', 'Chory'])
plt.title('Macierz błędów - Diagnoza')
plt.show()

"""
Interpretacja miar w kontekście medycznym:
- Precision w tym eksperymencie pokazuje procent ludzi z postawioną przez nasz model diagnozą "chory", którzy rzeczywiście są chorzy. 
- Recall określa, jaki ułamek wszystkich faktycznie chorych z całego zbioru walidacyjnego został wychwycony.
Dla diagnoz medycznych najczęściej chcemy zmaksymalizować Recall, ponieważ przepuszczenie pacjenta chorego weryfikując go jako "zdrowy" (False Negative) ma zazwyczaj o wiele 
groźniejsze skutki niż wysłanie zdrowej osoby na niepotrzebne drugie badanie (False Positive).
"""