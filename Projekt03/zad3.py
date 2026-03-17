import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r"x:\programowanie\intob\Projekt03\iris_big.csv")

# Podział na cechy (X) i klasy (y)
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# Podział na zbiór treningowy i testowy w proporcjach 70% / 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=300852)

# Słownik z klasyfikatorami do przetestowania
classifiers = {
    "k-NN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "MLP (Sieć Neuronowa)": MLPClassifier(max_iter=1500, random_state=300852)
}

print("--- ZADANIE 3: PORÓWNANIE KLASYFIKATORÓW ---")
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    
    # Przewidywanie na zbiorze testowym
    y_pred = clf.predict(X_test)
    
    # Obliczenie metryk
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nKlasyfikator: {name}")
    print(f"Dokładność (Accuracy): {acc * 100:.2f}%")
    print(f"Macierz błędów:\n{cm}")