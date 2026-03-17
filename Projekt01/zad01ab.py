import math
from datetime import date

def oblicz_sinus(dni, dlugosc_cyklu):
    return math.sin((2 * math.pi * dni) / dlugosc_cyklu)

def analizuj_i_wyswietl(nazwa_cyklu, wynik, dni_zycia, dlugosc_cyklu):
    print(f"--- Biorytm {nazwa_cyklu} ---")
    print(f"Wynik: {wynik:.2f}") 

    if wynik > 0.5:
        print(f"Gratulacje! Twój potencjał {nazwa_cyklu} jest dzisiaj bardzo wysoki!")
    
    elif wynik < -0.5:
        print(f"Masz gorszy dzień pod względem {nazwa_cyklu} (czuj się pocieszony/a).")
        wynik_jutro = oblicz_sinus(dni_zycia + 1, dlugosc_cyklu)
        
        if wynik_jutro > wynik:
            print("Nie martw się. Jutro będzie lepiej!")
        else:
            print("Jutro wynik może być podobny lub niższy, zadbaj o odpoczynek.")
    
    else:
        print(f"Twój stan {nazwa_cyklu} jest umiarkowany/neutralny.")
    print("")


print("Witaj w kalkulatorze biorytmów!")
    
imie = input("Podaj swoje imię: ")
rok = int(input("Podaj rok urodzenia: "))
miesiac = int(input("Podaj miesiąc urodzenia: "))
dzien = int(input("Podaj dzień urodzenia: "))

data_urodzenia = date(rok, miesiac, dzien)
dzis = date.today()
        
delta = dzis - data_urodzenia
t = delta.days
        
print(f"\nWitaj, {imie}!")
print(f"Dzisiaj jest {t}. dzień Twojego życia.")
print("-" * 40)

cykle = [
    ("FIZYCZNY", 23),
    ("EMOCJONALNY", 28),
    ("INTELEKTUALNY", 33)
]

for nazwa, dlugosc in cykle:
    wynik_dzis = oblicz_sinus(t, dlugosc)
    analizuj_i_wyswietl(nazwa, wynik_dzis, t, dlugosc)


#15-20min