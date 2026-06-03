import math
import numpy as np
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

# Oryginalna funkcja wytrzymałości [cite: 55, 57]
def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)

# Funkcja adaptująca (wrapper) dla całego roju [cite: 92, 97]
def f(particles):
    n_particles = particles.shape[0]
    results = np.zeros(n_particles)
    
    for i in range(n_particles):
        # Pobieramy 6 współrzędnych dla każdej cząstki [cite: 96]
        x, y, z, u, v, w = particles[i]
        # Dodajemy znak minus, aby funkcja szukająca minimum znalazła maksimum 
        results[i] = -endurance(x, y, z, u, v, w)
        
    return results

# Ustawienie limitów: min 0, max 1 dla sześciu zmiennych [cite: 87]
min_bound = np.zeros(6) # [cite: 88]
max_bound = np.ones(6)  # [cite: 88]
bounds = (min_bound, max_bound) # [cite: 82]

# Hiperparametry roju [cite: 72]
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Inicjalizacja optymalizatora (6 wymiarów) [cite: 87]
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=6, options=options, bounds=bounds)

# Uruchomienie algorytmu [cite: 98]
cost, pos = optimizer.optimize(f, iters=1000)

print(f"Najlepszy koszt (z minusem): {cost}") # [cite: 109]
print(f"Najlepsza pozycja: {pos}") # [cite: 109]

# Wykres historii kosztu [cite: 110]
plot_cost_history(optimizer.cost_history) # [cite: 112]
plt.show()