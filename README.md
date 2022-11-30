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
  
Wyznaczmy następną permutację dla n = 7 elementowej tablicy: {3, 2, 6, 7, 5, 4, 1}.
Indeksując kazdy element od 0 do n-1, możemy przedstawić tę tablicę za pomocą prostego wykresu:
<p align="center">
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204880214-34f5ae3b-4f54-4092-bc07-e7654744fe97.png">
</p>

Następnym krokiem jest znalezienie wierzchołka - patrząc z prawej strony. W kodzie wykonane zostanie to przez iterowanie się od końca tablicy do początku - sprawdzając przy tym czy aktualny element jest mniejszy od poprzedniego.
<p align="center">
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204880283-f61e3987-2b03-473f-8efa-696a38244d11.png">
</p>

W tym przypadku jest to element o indeksie i = 3, został oznaczony na czerwono. W celu wyznaczenia kolejnej permutacji musimy zamienić miejscami wyznaczony właśnie element z poprzednim (o indeksie i - 1 = 2). Uzyskana wówczas tablica jest następująca:
<p align="center">
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204880342-f13e1f36-ed44-4215-b275-454d29927dc9.png">
</p>

Nie jest to jednak ostateczny układ liczb, oznaczający końcową permutację. Aby ją uzyskać musimy posortować rosnąco wszystkie elementy znajdujące się na prawo od nowego miejsca w którym znajduje się wierzchołek (bez niego samego). Zostało to przedstawione poniżej - wierzchołki oznaczone na niebiesko.
<p align="center">
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204880976-cfa160f1-47a5-47ba-acc7-711018741f37.png">
</p>

Uzyskana w ten sposób tablica będzie najmniejszą z wszystkich możliwych kolejnych permutacji tablicy początkowej. Należy jednak dodatkowo rozpatrzyć jeden przypadek, w którym algorytm działa inaczej - pokazano go na poniższej tablicy.
<p align="center">
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204882807-776e3a7a-1a2e-4c46-a3e2-53f6329b6865.png">
</p>  
Jak widać na powyższym rysunku, znaleziony został pierwszy wierzchołek (maksimum lokalne) oraz znajduje się on pod indeksem i = 3. Zamieniając kolejnością elementy i = 3 oraz i = 2, oraz sortując elementy dla i > 2, nie uzyskalibyśmy kolejnej permutacji w kolejności leksykograficznej. 
  
Jest to spowodowane przez fakt, że na prawo od wierzchołka znajdują się liczby, które są mniejsze od samego wierzchołka ale są również większe od elementu o indeksie i = 2. W takim przypadku należy wybrać najmniejszy z elementów znajdujących się na prawo od miejsca w którym chcemy dokonać zmian - w tym przypadku indeks i = 2. 

