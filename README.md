# AI_project_for_classes
A project of simple neural network based on https://nnfs.io/

# Wstęp
Celem projektu jest stworzenie i wytrenowanie sieci neuronowej, która na podstawie cen kupna i sprzedaży walut określi kod waluty. Projekt wykorzystuje techniki takie jak propagacja wsteczna, propagacja w przód, aktywacja ReLU, Softmax, optymalizator Adam oraz funkcja straty kategorycznej entropii krzyżowej. Sieć została zaimplementowana w Pythonie 3.12.4 przy użyciu Visual Studio Code. Dokładniejszy opis działania sieci znajduje się w pliku projekt_AI.py w formie komentarzy do kodu.

# Dane
Dane zostały pobrane za pomocą pliku index.html, który został wykonany do projektu na inny przedmiot. A następnie przetworzone, aby dostosować je do wymagań modelu:
•	Normalizacja danych
•	Usunięcie niepotrzebnych kolumn
•	Przekształcenie kolumny code z kodami walut na unikalne liczby

Zestawy danych użyte w projekcie obejmują różne okresy czasowe i różną ilość próbek:
•	testing_2024-06-01_2024-06-14_c_data: 130 próbek, 13 walut
•	Usd_NotUsd_2014-01-01_2024-06-18_c_data: 34,177 próbek, dwie klasy (USD i inne)
•	training_2014-01-01_2024-06-18_c_data: 34,177 próbek, 13 walut
•	training_2004-01-01_2024-06-16_c_data: 672,100 próbek, 13 walut

# Struktura Sieci
## Warstwa Wejściowa: Przyjmuje dwa wejścia - cenę kupna i cenę sprzedaży waluty.

## Warstwy Ukryte:
Trzy warstwy ukryte, każda z funkcją aktywacji ReLU, aby wprowadzić nieliniowość.
Zestawienie parametrów warstw zostało dobrane metodą prób i błędów, aby uzyskać optymalne wyniki.

## Warstwa Wyjściowa:
Funkcja aktywacji Softmax, która przekształca wyjścia na prawdopodobieństwa przynależności do poszczególnych klas walut.
Proces Treningowy

## Propagacja w Przód:
Dane wejściowe przechodzą przez kolejne warstwy, stosując funkcję aktywacji ReLU.
Warstwa wyjściowa z Softmax produkuje prawdopodobieństwa dla każdej z 13 klas walut.(lub 2 klas dla pliku UsdNotUsd)

## Funkcja Straty:
Kategoryczna entropia krzyżowa mierzy różnicę między przewidywanymi prawdopodobieństwami a rzeczywistymi etykietami klas.

## Optymalizacja:
Optymalizator Adam dostosowuje wagi sieci na podstawie gradientów, aby minimalizować funkcję straty.

## Propagacja wsteczna:
Obliczanie gradientów funkcji straty względem wag sieci i aktualizacja wag poprzez optymalizator Adam.

# Wyniki
Model był testowany na danych testowych na początku, bez wyuczenia sieci, co pokazało bazową skuteczność.
Sieć była trenowana przez określoną liczbę epok na danych treningowych.
Po treningu model był testowany ponownie na danych testowych, aby ocenić jego wydajność.

# Wnioski
Najlepsze wyniki uzyskane dla zestawu danych z 13 klasami walut to około 75% dokładności i 50% straty.
Dla tak samo licznego zestawu danych z 2 klasami wyjściowymi uzyskałem około 90% dokładności oraz 15% straty.
Model działa lepiej dla zestawów danych z mniejszą liczbą klas, co jest widoczne na przykładzie Usd_NotUsd, gdzie osiągnięto lepsze wyniki.
Aby poprawić wyniki, można rozważyć dodanie dodatkowych warstw ukrytych i więcej neuronów oraz eksperymentowanie z innymi parametrami pomocniczymi dla danych z 13 klasami.

# Podsumowanie
Projekt pokazuje, że sieć neuronowa może skutecznie klasyfikować waluty na podstawie cen kupna i sprzedaży, choć dokładność może się różnić w zależności od liczby klas w danych. Dalsza optymalizacja modelu i eksperymentowanie z jego architekturą może prowadzić do lepszych wyników. Projekt demonstruje również znaczenie przetwarzania danych i odpowiedniego doboru hiperparametrów dla skutecznego trenowania modeli sieci neuronowych.
