import math
from datetime import date, datetime

def oblicz_sinus(dni, dlugosc_cyklu):
    """Oblicza wartość biorytu w zakresie od -1 do 1."""
    # Używamy wzoru na falę sinusoidalną
    return math.sin((2 * math.pi * dni) / dlugosc_cyklu)

def rysuj_pasek(wynik, szerokosc=20):
    """Tworzy prosty pasek wizualizujący wynik."""
    # Skalujemy wynik z zakresu -1..1 na 0..szerokosc
    pozycja = int((wynik + 1) / 2 * szerokosc)
    # Zabezpieczenie zakresu
    pozycja = max(0, min(pozycja, szerokosc))
    
    pasek = ['-'] * (szerokosc + 1)
    pasek[szerokosc // 2] = '|'  # Oznaczenie środka (0)
    
    znak = '●' if wynik >= 0 else '○'
    pasek[pozycja] = znak
    
    return "".join(pasek)

def analizuj_i_wyswietl(nazwa_cyklu, wynik, dni_zycia, dlugosc_cyklu):
    print(f"\n--- BIORYTM {nazwa_cyklu.upper()} (Cykl: {dlugosc_cyklu} dni) ---")
    
    # Wizualizacja
    pasek = rysuj_pasek(wynik)
    print(f"Poziom: [{pasek}] {wynik * 100:+.1f}%")

    if wynik > 0.5:
        print(f"Świetnie! Twój potencjał {nazwa_cyklu.lower()} jest dzisiaj bardzo wysoki.")
        print("To dobry czas na działanie w tej sferze!")
    
    elif wynik < -0.5:
        print(f"Dziś masz trudniejszy dzień pod względem {nazwa_cyklu.lower()}.")
        print("Głowa do góry, nie przemęczaj się.")
        
        # Sprawdzamy trend na jutro
        wynik_jutro = oblicz_sinus(dni_zycia + 1, dlugosc_cyklu)
        
        if wynik_jutro > wynik:
            print("Trend jest rosnący – jutro będzie lepiej!")
        else:
            print("Jutro wciąż możesz czuć spadek formy, zaplanuj odpoczynek.")
    
    else:
        trend = "rosnący" if oblicz_sinus(dni_zycia + 1, dlugosc_cyklu) > wynik else "spadkowy"
        print(f"Twój stan {nazwa_cyklu.lower()} jest stabilny (strefa neutralna).")
        print(f"Trend: {trend}.")

def pobierz_date_urodzenia():
    """Pętla wymuszająca podanie poprawnej daty."""
    while True:
        try:
            data_str = input("Podaj datę urodzenia (RRRR-MM-DD): ")
            # Parsowanie daty z jednego ciągu znaków jest wygodniejsze
            return datetime.strptime(data_str, "%Y-%m-%d").date()
        except ValueError:
            print("Błąd: Nieprawidłowy format daty lub data nie istnieje. Spróbuj ponownie (np. 1990-05-25).")

def main():
    print("=" * 50)
    print(" WITAJ W KALKULATORZE BIORYTMÓW ".center(50, "="))
    print("=" * 50)
    
    imie = input("\nJak masz na imię? ").strip()
    data_urodzenia = pobierz_date_urodzenia()
    
    dzis = date.today()
    
    # Zabezpieczenie przed datą z przyszłości
    if data_urodzenia > dzis:
        print("\nWygląda na to, że pochodzisz z przyszłości! Nie mogę obliczyć biorytmu.")
        return

    delta = dzis - data_urodzenia
    t = delta.days
        
    print(f"\nWitaj, {imie}!")
    print(f"Żyjesz już {t} dni.")
    
    cykle = [
        ("FIZYCZNY", 23),
        ("EMOCJONALNY", 28),
        ("INTELEKTUALNY", 33)
    ]

    for nazwa, dlugosc in cykle:
        wynik_dzis = oblicz_sinus(t, dlugosc)
        analizuj_i_wyswietl(nazwa, wynik_dzis, t, dlugosc)

    print("\n" + "=" * 50)
    input("Naciśnij ENTER, aby zakończyć...")

if __name__ == "__main__":
    main()