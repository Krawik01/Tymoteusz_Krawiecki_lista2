import ssl
import pandas as pd
from sklearn import preprocessing
import sqlite3
import os
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

ssl._create_default_https_context = ssl._create_unverified_context

#Wczytaj dane z adresu podanego w pliku tekstowym: pliktextowy.txt
# do ramki danych.
#Użyj reszty wierszy jako nagłówków ramki danych.
#Uwaga! Zobacz która zmienna jest zmienną objaśnianą, będzie to potrzebne do dalszych zadań.


with open('pliktextowy.txt', 'r') as file:
    header_rows = file.read().splitlines()

data_url = header_rows[0]

df = pd.read_csv(data_url, header=None, names=header_rows[1:])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#Zadanie1 przypisz nazwy kolumn z df w jednej linii:   (2pkt)

wynik1 = df.columns.tolist()
print(wynik1)

#Zadanie 2: Wypisz liczbę wierszy oraz kolumn ramki danych w jednej linii.  (2pkt)
wynik2 = f"Liczba wierszy: {df.shape[0]}, Liczba kolumn: {df.shape[1]}"
print(wynik2)



#Zadanie Utwórz klasę Wine na podstawie wczytanego zbioru:
#wszystkie zmienne objaśniające powinny być w liscie.
#Zmienna objaśniana jako odrębne pole.
# metoda __init__ powinna posiadać 2 parametry:
#listę (zmienne objaśniające) oraz liczbę(zmienna objaśniana).
#nazwy mogą być dowolne.

# Klasa powinna umożliwiać stworzenie nowego obiektu na podstawie
# już istniejącego obiektu jak w pdf z lekcji lab6.
# podpowiedź: metoda magiczna __repr__
#Nie pisz metody __str__.

class Wine:
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __repr__(self):
        return f"Wine(features={self.features}, target={self.target})"

#Zadanie 3 Utwórz przykładowy obiekt:   (3pkt)
wynik3 = Wine(df.columns.tolist()[:-1], df.columns.tolist()[-1]) #do podmiany. Pamiętaj - ilość elementów, jak w zbiorze danych.
#Uwaga! Pamiętaj, która zmienna jest zmienną objaśnianą
print(wynik3)

#Zadanie 4.                             (3pkt)
#Zapisz wszystkie dane z ramki danych do listy obiektów typu Wine.
#Nie podmieniaj listy, dodawaj elementy.
##Uwaga! zobacz w jakiej kolejności podawane są zmienne objaśniające i objaśniana.
# Podpowiedź zobacz w pliktextowy.txt
wineList = []
for _, row in df.iterrows():
    wine = Wine(row.tolist()[:-1], row.tolist()[-1])
    wineList.append(wine)

wynik4 = len(wineList)
print(wynik4)


#Zadanie5 - Weź ostatni element z listy i na podstawie         (3pkt)
#wyniku funkcji repr utwórz nowy obiekt - eval(repr(obiekt))
#do wyniku przypisz zmienną objaśnianą z tego obiektu:
last_wine = wineList[-1]
new_wine = eval(repr(last_wine))
wynik5 = new_wine.target
print(wynik5)


#Zadanie 6:                                                          (3pkt)
#Zapisz ramkę danych  do bazy SQLite nazwa bazy(dopisz swoje imię i nazwisko):
# wines_imie_nazwisko, nazwa tabeli: wines.
#Następnie wczytaj dane z tabeli wybierając z bazy danych tylko wiersze z typem wina nr 3
# i zapisz je do nowego data frame:

if os.path.exists('wines_Tymoteusz_Krawiecki.db'):
    os.remove('wines_Tymoteusz_Krawiecki.db')

conn = sqlite3.connect('wines_Tymoteusz_Krawiecki.db')

conn.execute('''
    CREATE TABLE wines (
        TypeOf INTEGER,
        Alcohol REAL,
        Malic_acid REAL,
        Ash REAL,
        Alcalinity_of_ash REAL,
        Magnesium REAL,
        Total_phenols REAL,
        Flavanoids REAL,
        Nonflavanoid_phenols REAL,
        Proanthocyanins REAL,
        Color_intensity REAL,
        Hue REAL,
        OD280_OD315_of_diluted_wines REAL,
        Proline INTEGER
    )
''')

df.to_sql('wines', conn, if_exists='replace', index=False)

query = "SELECT * FROM wines WHERE TypeOf = 3"
df_filtered = pd.read_sql_query(query, conn)

conn.close()

wynik6 = df_filtered

print(wynik6.shape)


#Zadanie 7                                                          (1pkt)
#Utwórz model regresji Logistycznej z domyślnymi ustawieniami:

model = LogisticRegression()

wynik7 = model.__class__.__name__
print(wynik7)
# Zadanie 8:                                                        (3pkt)
#Dokonaj podziału ramki danych na dane objaśniające i  do klasyfikacji.
#Znormalizuj dane objaśniające za pomocą:
#X = preprocessing.normalize(X)
# Wytenuj model na wszystkich danych bez podziału na zbiór treningowy i testowy.
# Wykonaj sprawdzian krzyżowy, używając LeaveOneOut() zamiast KFold (Parametr cv)
#  Podaj średnią dokładność (accuracy)

X = df.drop("TypeOf", axis=1)
y = df["TypeOf"]  # Target variable
X = preprocessing.normalize(X)
model = LogisticRegression()
loocv = LeaveOneOut()
accuracy_scores = cross_val_score(model, X, y, cv=loocv, scoring="accuracy")
mean_accuracy = accuracy_scores.mean()
wynik8 = mean_accuracy
print(wynik8)