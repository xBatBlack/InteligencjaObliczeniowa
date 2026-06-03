import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

df = pd.read_csv("iris_big.csv")

# Pobieramy dokładne nazwy kolumn na podstawie ich pozycji
col_sepal_length = df.columns[0] 
col_sepal_width = df.columns[1] 
col_target = df.columns[-1]      

# Wybieramy tylko interesujące nas dane
X = df[[col_sepal_length, col_sepal_width]]
y = df[col_target]

print("normalizacja i skalowanie")
# 1. Normalizacja Min-Max
min_max_scaler = MinMaxScaler()
X_minmax = pd.DataFrame(min_max_scaler.fit_transform(X), columns=[col_sepal_length, col_sepal_width])
X_minmax['Gatunek'] = y.values

# 2. Skalowanie Z-Score (Standaryzacja)
z_scaler = StandardScaler()
X_zscore = pd.DataFrame(z_scaler.fit_transform(X), columns=[col_sepal_length, col_sepal_width])
X_zscore['Gatunek'] = y.values

df['Gatunek'] = y.values

print("Wykresy")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Wykres 1: Oryginalne
sns.scatterplot(ax=axes[0], x=col_sepal_length, y=col_sepal_width, hue='Gatunek', data=df)
axes[0].set_title('Original Dataset')

# Wykres 2: Min-Max
sns.scatterplot(ax=axes[1], x=col_sepal_length, y=col_sepal_width, hue='Gatunek', data=X_minmax)
axes[1].set_title('Min-Max Normalised Dataset')

# Wykres 3: Z-Score
sns.scatterplot(ax=axes[2], x=col_sepal_length, y=col_sepal_width, hue='Gatunek', data=X_zscore)
axes[2].set_title('Z-Score Scaled Dataset')

plt.tight_layout()
plt.show()
#do zrobienia 2 5 8 10, przynajmniej jedno z nich