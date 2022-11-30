# Travelling salesman problem CPU vs GPU

Rozpoczynając ten projekt przyjąłem podejście, w którym chciałbym wyznaczyć wszystkie możliwe permutacje dla danych n węzów, które reprezentują n miast, które nasz komiwojażer miałby odwiedzić. Ze względu na to, że istotny jest punkt startu, to sztucznie dokładając pierwsze miasto oraz obliczając n! możliwości uzyskujemy łącznie rozwiązania problemu dla n+1 miast, gdzie n! to liczba rozwiązań.

## Progress - kradzione
Na ten moment mam znaleziony działający algorytm, który oblicza wszystkie możliwe permutacje maksymalnie dla 12 węzłów. Dodając stały początkowy węzeł otrzymujemy rozwiązanie TSP dla 13 miast. Jednak nie do końca rozumiem jak działa to co gość, od którego mam kod zrobił. 

<a href="https://www.codeproject.com/Articles/380399/Permutations-with-CUDA-and-OpenCL#cudahttps://www.quickperm.org/01example.php#menu">Link do podjebanego kodu, który liczy do max 12!<a/>

## Progress - mój własny program
Staram się też jednak samemu napisać od podstaw taki algorytm, który zadziała na CUDA. W tym jednak pojawia się problem, bo mając klasyczne algorytmy wyznaczające wszystkie permutacje trochę ciężko wyznaczyć drogę, która pozwoli na zrównoleglenie obliczeń - a w zasadzie ja tego nie potafię zrobić. Rekurencja jest trudna w zrównolegleniu więc też odpada. Z tego powodu odrzuciłem kilka algorytmów:
 * prosty algorytm z rekurencją - zagnieżdżone n pętli
 * algorytm Heaps'a oraz QuickPerm są efektywne, jednak podczas działania nie widzę możliwości sensownego zrównoleglenia - nie zwracają wyników w leksykograficznej kolejności

Wkłaśnie dwa ostatnie słowa to według mnie klucz do obliczeń równoległych.

### Kolejność leksykograficzna
Jeśli będziemy generować wszystkie możliwe permutacje w takiej właśnie kolejności zaczynając od najmniejszej oraz najbardziej intuicyjnej permutacji początkowej, którą będzie {1, 2, 3, ..., n}, wówczas mamy pewność, że wygenerujemy wszystkie możliwe permutacje realizując je po kolei aż do ostatniej, którą jest inwersja permutacji początkowej: {n, n-1, n-2, ..., 1}.

#### Wszystko fajnie, tylko jak generować permutacje w kolejności leksykograficznej?

Ważne jest jeszcze to, że leksykograficznie w naszym przypadku oznacza po prostu rosnąco, czyli tak jakbyśmy posortowali wyniki dla jakiegoś algorytmu, tak żeby liczba tworząca następną permutację była najmniejszą z wszystkich możliwych następnych permutacji. Przykład dla 3 elementów, 3! = 6. 
<div align="center">
<table>
  <tr>
    <td>i</td> <td>arr[0]</td> <td>arr[1]</td> <td>arr[2]</td>
  </tr>
  <tr> <td>1</td> <td>1</td> <td>2</td> <td>3</td> </tr>
  <tr> <td>2</td> <td>1</td> <td>3</td> <td>2</td> </tr>
  <tr> <td>3</td> <td>2</td> <td>1</td> <td>3</td> </tr>
  <tr> <td>4</td> <td>2</td> <td>3</td> <td>1</td> </tr>
  <tr> <td>5</td> <td>3</td> <td>1</td> <td>2</td> </tr>
  <tr> <td>6</td> <td>3</td> <td>2</td> <td>1</td> </tr>
</table>
</div>

Znalazłem gościa, który bardzo fajnie wyjaśnił o co biega na <a href="https://www.youtube.com/watch?v=6qXO72FkqwM">tym filmiku na YT</a> ale postaram się wyjaśnić to też tutaj.

#### Wyznaczanie kolejności leksykograficznej
  
