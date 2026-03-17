import math
import datetime

def main():
    # --- CZĘŚĆ A: Pobieranie danych i obliczenia ---
    
    print("--- Kalkulator Biorytmów ---")
    imie = input("Podaj swoje imię: ")
    
    # Pobieranie daty urodzenia
    try:
        rok = int(input("Podaj rok urodzenia (np. 1990): "))
        miesiac = int(input("Podaj miesiąc urodzenia (1-12): "))
        dzien = int(input("Podaj dzień urodzenia (1-31): "))
        
        data_urodzenia = datetime.date(rok, miesiac, dzien)
    except ValueError:
        print("Błąd: Wprowadzono niepoprawną datę!")
        return

    # Obliczanie dni życia (t)
    dzisiaj = datetime.date.today()
    delta = dzisiaj - data_urodzenia
    t = delta.days

    print(f"\nWitaj, {imie}!")
    print(f"Dzisiaj jest Twój {t}. dzień życia.")
    print("-" * 40)

    # Definicja cykli (nazwa, długość cyklu)
    cykle = [
        ("Fizyczny", 23),
        ("Emocjonalny", 28),
        ("Intelektualny", 33)
    ]

    # Pętla iterująca przez każdy rodzaj biorytmu
    for nazwa, okres in cykle:
        # Obliczanie wyniku dla dnia dzisiejszego (t)
        # Wzór: sin(2 * pi * t / okres)
        wynik = math.sin(2 * math.pi * t / okres)
        
        # Wyświetlenie wyniku (formatowanie do 2 miejsc po przecinku)
        print(f"Biorytm {nazwa}: {wynik:.2f}")

        # --- CZĘŚĆ B: Interpretacja i prognoza ---
        
        if wynik > 0.5:
            print(f"-> Gratulacje! Twój stan {nazwa.lower()} jest świetny! 💪")
        
        elif wynik < -0.5:
            print(f"-> Trudny czas dla sfery: {nazwa.lower()}. Głowa do góry.")
            
            # Sprawdzanie trendu na jutro (t + 1)
            wynik_jutro = math.sin(2 * math.pi * (t + 1) / okres)
            
            if wynik_jutro > wynik:
                print("   Nie martw się. Jutro będzie lepiej! 📈")
            else:
                print("   Jutro wynik może być jeszcze niższy, oszczędzaj siły.")
        
        # Separator dla czytelności
        print("-" * 40)

if __name__ == "__main__":
    main()

#2min