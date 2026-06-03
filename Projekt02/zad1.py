import pandas as pd
import numpy as np
import re

# 1. ominięcie błędów
with open("iris_big_with_errors.csv", "r", encoding="utf-8", errors="ignore") as file:
    lines = file.readlines()

cleaned_data = []

for line in lines:
    #usuwanie cudzysłowów
    line = line.replace('"', '').replace("'", "").strip()
    
    #zmieniamy wszystkie separatory na przeciniki
    line = re.sub(r'[;\t]+', ',', line)
    
    # Dzielimy linijkę po przecinku
    row = line.split(',')
    
    if len(row) >= 5:
        cleaned_data.append(row[:5])

# 2. Tworzymy czysty DataFrame narzucając własne, poprawne nazwy kolumn
columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target_name']
df = pd.DataFrame(cleaned_data[1:], columns=columns)

print(f"Struktura po naprawieniu: {df.shape}")

# 3. Naprawa danych numerycznych 1c
num_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

for col in num_cols:
    # Wymuszamy liczby jeśli błędy to zamieniamy na NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Wartości spoza zakresu (0; 15) zamieniamy na NaN
    df.loc[(df[col] <= 0) | (df[col] >= 15), col] = np.nan
    
    # Podstawiamy średnią w miejsce wszystkich NaN
    df[col] = df[col].fillna(df[col].mean())

# 4. Naprawa nazw gatunków (Zadanie 1d)
# Zmieniamy na małe litery i usuwamy białe znaki na końcach
df['target_name'] = df['target_name'].astype(str).str.lower().str.strip()

# Słownik z typowymi literówkami, które trzeba podmienić
corrections = {
    'iris-setosa': 'setosa', 'setos': 'setosa', 'seotsa': 'setosa',
    'iris-versicolor': 'versicolor', 'versicolour': 'versicolor', 
    'iris-virginica': 'virginica', 'virgnica': 'virginica'
}
df['target_name'] = df['target_name'].replace(corrections)

# Upewniamy się, że w bazie zostają tylko 3 poprawne gatunki
valid_species = ['setosa', 'versicolor', 'virginica']
df = df[df['target_name'].isin(valid_species)]

print("\nBrakujące wartości")
print(df.isnull().sum()) # Powinno wyjść 0 wszędzie

print("\nUnikalne gatunki")
print(df['target_name'].unique()) # Powinny być tylko 3 poprawne
