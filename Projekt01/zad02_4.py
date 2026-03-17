import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Dane stałe z zadania [cite: 85, 103]
V0 = 50      # prędkość początkowa m/s
H = 100      # wysokość trebusza m
G = 9.81     # przyspieszenie ziemskie m/s^2

def oblicz_odleglosc(alpha_deg):
    # Konwersja kąta na radiany
    alpha = math.radians(alpha_deg)
    # Wzór z aneksu 
    term1 = V0 * math.sin(alpha) + math.sqrt((V0 * math.sin(alpha))**2 + 2 * G * H)
    term2 = (V0 * math.cos(alpha)) / G
    return term1 * term2

# 1) Wybór celu [cite: 100]
cel = random.uniform(50, 340)
print(f"Cel znajduje się w odległości: {cel} m")

proby = 0
trafiony = False
ostatni_kat = 0

# 2) Pętla strzelania [cite: 104]
while not trafiony:
    try:
        kat = float(input("Podaj kąt strzału (w stopniach): "))
        proby += 1
        d = oblicz_odleglosc(kat)
        
        print(f"Pocisk upadł w odległości: {d:.2f} m")
        
        # Sprawdzenie trafienia z marginesem 5m [cite: 101, 102]
        if cel - 5 <= d <= cel + 5:
            print(f"Cel trafiony! Liczba prób: {proby}")
            trafiony = True
            ostatni_kat = kat
        else:
            if d < cel:
                print("Za blisko!")
            else:
                print("Za daleko!")
    except ValueError:
        print("Proszę podać poprawną liczbę.")

# 3) Rysowanie trajektorii [cite: 118, 120]
alpha_rad = math.radians(ostatni_kat)
d_final = oblicz_odleglosc(ostatni_kat)

# Tworzenie punktów x od 0 do miejsca upadku
x = np.linspace(0, d_final, 500)

# Wzór na trajektorię y(x) z aneksu [cite: 127]
y = (-G / (2 * V0**2 * math.cos(alpha_rad)**2)) * x**2 + \
    (math.sin(alpha_rad) / math.cos(alpha_rad)) * x + H

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Trajektoria pocisku', color='blue')
plt.axhline(0, color='black', linewidth=1) # Linia gruntu
plt.grid(True)
plt.title("Projectile Motion for the Trebuchet")
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.ylim(bottom=0)

# Zapis do pliku 
plt.savefig('trajektoria.png')
print("Wykres trajektorii został zapisany jako 'trajektoria.png'.")
plt.show()