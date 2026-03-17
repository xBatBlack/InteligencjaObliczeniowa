import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df_diag = pd.read_csv(r"x:\programowanie\intob\Projekt03\diagnosis.csv")

# Zakładamy, że 3 pierwsze kolumny to wyniki badań (parametry), a ostatnia to diagnoza (0 lub 1)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=12345)

# Trenujemy przykładowy klasyfikator (np. k-NN dla k=5)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# b) Obliczenie miar oceny
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("\n--- ZADANIE 4: WYNIKI DIAGNOZY ---")
print(f"Accuracy (Dokładność): {acc:.4f}")
print(f"Precision (Precyzja): {prec:.4f}")
print(f"Recall (Czułość):     {rec:.4f}")

# Graficzna macierz błędów z wykorzystaniem Seaborn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Zdrowy (0)', 'Chory (1)'], 
            yticklabels=['Zdrowy (0)', 'Chory (1)'])
plt.xlabel('Przewidywania Modelu (AI)')
plt.ylabel('Rzeczywistość (Prawda)')
plt.title('Macierz błędów - Diagnoza Choroby')
plt.show()

"""

- Accuracy (Dokładność): Odsetek poprawnych klasyfikacji (zarówno chorych, jak i zdrowych) w stosunku do wszystkich przypadków.
- Precision (Precyzja): Zdolność modelu do unikania błędnego oznaczania osób zdrowych jako chore (False Positives). 
- Recall (Czułość): Zdolność modelu do wykrywania wszystkich faktycznie chorych pacjentów. Minimalizuje sytuacje, w których chory uznany jest za zdrowego (False Negatives).

Aby nie klasyfikowac zdrowych jako chorych, istotna jest wysoka Precision.
Aby nie przeoczyc osoby chorej i nie oznaczyc jej jako zdrowej, istotna jest wysoka Recall.

Czy Accuracy jest bezpieczną miarą przy niezbalansowanym zbiorze?
   - Zdecydowanie NIE. Jeśli w zbiorze mamy 99% osób zdrowych i 1% chorych, to model, który zawsze bezmyślnie odpowiada "zdrowy", 
     uzyska 99% Accuracy. Miara ta nie oddaje faktu, że model w ogóle nie potrafi rozpoznać choroby. W takich przypadkach 
     znacznie bezpieczniej jest używać Precision, Recall lub tzw. miary F1-Score.
"""