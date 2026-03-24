import numpy as np

# Funkcja aktywacji (sigmoidalna) i jej pochodna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Dane wejściowe zgodne ze skryptem:
x1, x2 = 0.6, 0.1
y_true = 0.8
eta = 0.1

# Początkowe wagi i biasy wyciągnięte ze skryptu z PDF-a
w1, w2, b1 = 0.2, -0.3, 0.4
w3, w4, b2 = -0.5, 0.1, -0.2
w5, w6, b3 = 0.3, -0.4, 0.2

# a,b
def forward_propagation(x1, x2):
    # neuron ukryty nr 1
    z1 = w1*x1 + w2*x2 + b1
    h1 = sigmoid(z1)
    
    # neuron ukryty nr 2
    z2 = w3*x1 + w4*x2 + b2
    h2 = sigmoid(z2)
    
    # neuron wyjściowy
    z3 = w5*h1 + w6*h2 + b3
    y_pred = z3
    
    return z1, h1, z2, h2, z3, y_pred

z1, h1, z2, h2, z3, y_pred = forward_propagation(x1, x2)
print(f"Predykcja (y_pred): {y_pred:.4f} (Spodziewane: ~0.2341)")

# Backpropagation
# Błąd średniokwadratowy (MSE)
loss = 0.5 * (y_pred - y_true)**2
print(f"Strata (Loss): {loss:.4f} (Spodziewane: ~0.1601)")

# Krok 1: sygnał błędu na wyjściu
delta3 = y_pred - y_true

# Krok 2: Gradienty wag warstwy wyjściowej
dL_dw5 = delta3 * h1
dL_dw6 = delta3 * h2
dL_db3 = delta3

# Krok 3: propagacja błędu do warstwy ukrytej z regułą łańcuchową i pochodną sigmoidy
delta1 = delta3 * w5 * sigmoid_derivative(z1)
delta2 = delta3 * w6 * sigmoid_derivative(z2)

# Krok 4: Gradienty wag warstwy ukrytej
dL_dw1 = delta1 * x1
dL_dw2 = delta1 * x2
dL_db1 = delta1

dL_dw3 = delta2 * x1
dL_dw4 = delta2 * x2
dL_db2 = delta2

# aktualizacja wag
w5_new = w5 - eta * dL_dw5
w6_new = w6 - eta * dL_dw6

print(f"\nZaktualizowana waga w5: {w5_new:.4f} (Spodziewane : 0.3351)")
print(f"Zaktualizowana waga w6: {w6_new:.4f} (Spodziewane : -0.3785)")