import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv(r"c:\Programowanie\InteligencjaObliczeniowa\Projekt03\iris_big.csv") 

# a) Podział na inputy i klasy
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=300852)

# b & c) Inicjalizacja i trening drzewa decyzyjnego
dtc = DecisionTreeClassifier(random_state=300852)
dtc.fit(X_train, y_train)

# d) Wyświetlenie drzewa w formie graficznej
plt.figure(figsize=(12, 8))
plot_tree(dtc, filled=True, feature_names=df.columns[:4], class_names=dtc.classes_)
plt.title("Drzewo Decyzyjne")
plt.show()

# e & f) Ewaluacja i macierz błędów
y_pred_dtc = dtc.predict(X_test)
acc_dtc = accuracy_score(y_test, y_pred_dtc)
cm_dtc = confusion_matrix(y_test, y_pred_dtc)

print(f"Dokładność Drzewa Decyzyjnego: {acc_dtc * 100:.2f}%")
print("Macierz błędów (Confusion Matrix):\n", cm_dtc)