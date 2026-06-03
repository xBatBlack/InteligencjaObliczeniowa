import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv("iris_big.csv")

X = df.iloc[:, :-1] #wszystkie oprócz ostatniej
y = df.iloc[:, -1]

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nAnaliza PCA")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Wyświetlenie wariancji dla poszczególnych składowych
explained_variance = pca.explained_variance_ratio_
print(f"Wariancja wyjaśniona przez kolejne składowe: {explained_variance}")
print(f"Skumulowana wariancja: {explained_variance.cumsum()}")


print("\nWykres")
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

#PC1 i PC2 dają 95% wariancji, więc wystarczą do wizualizacji

df_pca = pd.DataFrame(data = X_pca_2d, columns = ['PC1', 'PC2'])
df_pca['Gatunek'] = y.values

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Gatunek', data=df_pca, palette=['blue', 'turquoise', 'orange'])
plt.title('PCA of IRIS dataset')
plt.show()