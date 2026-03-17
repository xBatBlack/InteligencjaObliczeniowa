import math
import random
import matplotlib.pyplot as plt
import numpy as np


def rysuj_trajektorie(v0, h, g, alpha_rad, dystans_koncowy):
    # obliczenie czasu lotu
    vx = v0 * math.cos(alpha_rad)
    vy = v0 * math.sin(alpha_rad)
    t_total = dystans_koncowy / vx

    # punkty czasu
    t = np.linspace(0, t_total, num=100)

    x = vx * t
    y = h + vy * t - 0.5 * g * t**2

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Trajektoria pocisku')
    
    plt.title("Trajektoria lotu pocisku z trebusza Warwolf")
    plt.xlabel("Dystans (m)")
    plt.ylabel("Wysokość (m)")
    plt.grid(True)
    
    plt.axhline(0, color='black', linewidth=1)

    # zapisanie do pliku
    plt.savefig('trajektoria.png')
    print("Wykres zapisano jako 'trajektoria.png'")
    plt.show()




g = 9.81 
v0 = 50.0 
h = 100.0

cel_dystans = random.randint(50, 340)
print(f"Cel znajduje się w odległości: {cel_dystans} metrów.")
print(f"Musisz trafić w przedział [{cel_dystans - 5}, {cel_dystans + 5}] metrów.")

trafiony = False
proby = 0
kat_trafienia = 0

while not trafiony:
    try:
        kat_str = input("Podaj kąt strzału (w stopniach): ")
        alpha_deg = float(kat_str)
        proby += 1
        alpha_rad = math.radians(alpha_deg)

        czlon_pierwiastek = math.sqrt((v0 * math.sin(alpha_rad))**2 + 2 * g * h)
        dystans = (v0 * math.sin(alpha_rad) + czlon_pierwiastek) * (v0 * math.cos(alpha_rad) / g)

        print(f"Pocisk poleciał na odległość: {dystans:.2f} metrów.")

        #  warunek trafienia (margines 5 metrów)
        if cel_dystans - 5 <= dystans <= cel_dystans + 5:
            print(f"Cel trafiony! Liczba prób: {proby}")
            trafiony = True
            kat_trafienia = alpha_rad
        else:
            print("Pudło! Spróbuj ponownie.")
                
    except ValueError:
        print("Proszę podać poprawną liczbę.")

# rysowanie trajektorii
rysuj_trajektorie(v0, h, g, kat_trafienia, dystans)

