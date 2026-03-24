import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df_diag = pd.read_csv(r"c:\Programowanie\InteligencjaObliczeniowa\Projekt03\diagnosis.csv")

X = df_diag.iloc[:, :-1].values
y = df_diag.iloc[:, -1].values

# a) Wykres 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Rysowanie punktów - zdrowe (0) na niebiesko, chore (1) na czerwono
ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='blue', label='Zdrowy (0)', alpha=0.6)
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='red', label='Chory (1)', alpha=0.6)

ax.set_title("Wykres parametrów medycznych (3D)")
ax.legend()
plt.show()

# Podział zbioru na treningowy i testowy 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=300852)

#b) Ewaluacja wszystkich klasyfikatorów
# Słownik ze wszystkimi klasyfikatorami z zadania 2 i 3
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=300852),
    "k-NN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "MLP (Sieć Neuronowa)": MLPClassifier(max_iter=1500, random_state=300852)
}

print("\nZADANIE 4: WYNIKI DIAGNOZY")

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"\nModel: {name}")
    print(f"Accuracy (Dokładność): {acc:.4f}")
    print(f"Precision (Precyzja):  {prec:.4f}")
    print(f"Recall (Czułość):      {rec:.4f}")

    # Graficzna macierz błędów z wykorzystaniem Seaborn
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Zdrowy (0)', 'Chory (1)'], 
                yticklabels=['Zdrowy (0)', 'Chory (1)'])
    plt.xlabel('Przewidywania Modelu (AI)')
    plt.ylabel('Rzeczywistość (Prawda)')
    plt.title(f'Macierz błędów - {name}')
    plt.show()

"""
- Accuracy (Dokładność): Odsetek poprawnych klasyfikacji (zarówno chorych, jak i zdrowych) w stosunku do wszystkich przypadków.
- Precision (Precyzja): Zdolność modelu do unikania błędnego oznaczania osób zdrowych jako chore (False Positives). 
- Recall (Czułość): Zdolność modelu do wykrywania wszystkich faktycznie chorych pacjentów. Minimalizuje sytuacje, w których chory uznany jest za zdrowego (False Negatives).

* Aby nie klasyfikować zdrowych jako chorych, istotna jest wysoka Precision (Precyzja).
* Aby nie przeoczyć osoby chorej i nie oznaczyć jej jako zdrowej, istotna jest wysoka Recall (Czułość).

Czy Accuracy jest bezpieczną miarą przy niezbalansowanym zbiorze?
- Zdecydowanie NIE. Jeśli w zbiorze mamy 99% osób zdrowych i 1% chorych, to model, który zawsze bezmyślnie odpowiada "zdrowy", uzyska 99% Accuracy.
   Miara ta nie oddaje faktu, że model w ogóle nie potrafi rozpoznać choroby.
   W takich przypadkach znacznie bezpieczniej jest używać Precision, Recall lub miary F1-Score.
"""