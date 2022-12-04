# Travelling salesman problem CPU vs GPU

Rozpoczynając ten projekt przyjąłem podejście, w którym chciałbym wyznaczyć wszystkie możliwe permutacje dla danych n węzów, które reprezentują n miast, które nasz komiwojażer miałby odwiedzić. Ze względu na to, że istotny jest punkt startu, to sztucznie dokładając pierwsze miasto oraz obliczając n! możliwości uzyskujemy łącznie rozwiązania problemu dla n+1 miast, gdzie n! to liczba rozwiązań.

## Znaleziony kod
Na ten moment mam znaleziony działający algorytm, który oblicza wszystkie możliwe permutacje maksymalnie dla 12 węzłów. Dodając stały początkowy węzeł otrzymujemy rozwiązanie TSP dla 13 miast. Jednak nie do końca rozumiem jak działa to co gość, od którego mam kod zrobił. 

<a href="https://www.codeproject.com/Articles/380399/Permutations-with-CUDA-and-OpenCL#cudahttps://www.quickperm.org/01example.php#menu">Link do znalezionego kodu, który liczy do max 12!<a/>

## Progress - mój własny program
Staram się też jednak samemu napisać od podstaw taki algorytm, który zadziała na CUDA. W tym jednak pojawia się problem, bo mając klasyczne algorytmy wyznaczające wszystkie permutacje trochę ciężko wyznaczyć drogę, która pozwoli na zrównoleglenie obliczeń - a w zasadzie ja tego nie potafię zrobić. Rekurencja jest trudna w zrównolegleniu więc też odpada. Z tego powodu odrzuciłem kilka algorytmów:
 * prosty algorytm z rekurencją - zagnieżdżone n pętli
 * algorytm Heaps'a oraz QuickPerm są efektywne, jednak podczas działania nie widzę możliwości sensownego zrównoleglenia - nie zwracają wyników w leksykograficznej kolejności

Właśnie dwa ostatnie słowa to według mnie klucz do obliczeń równoległych.

## Kolejność leksykograficzna
Jeśli będziemy generować wszystkie możliwe permutacje w takiej właśnie kolejności zaczynając od najmniejszej oraz najbardziej intuicyjnej permutacji początkowej, którą będzie {1, 2, 3, ..., n}, wówczas mamy pewność, że wygenerujemy wszystkie możliwe permutacje realizując je po kolei aż do ostatniej, którą jest inwersja permutacji początkowej: {n, n-1, n-2, ..., 1}.

### Wszystko fajnie, tylko jak generować permutacje w kolejności leksykograficznej?

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

### Wyznaczanie kolejności leksykograficznej
  
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
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204884766-bc5bc282-b9d3-4a04-b1e5-cdf925437960.png">
</p>  
Jak widać na powyższym rysunku, znaleziony został pierwszy wierzchołek (maksimum lokalne) oraz znajduje się on pod indeksem i = 3. Zamieniając kolejnością elementy i = 3 oraz i = 2, oraz sortując elementy dla i > 2, nie uzyskalibyśmy kolejnej permutacji w kolejności leksykograficznej. 
  
Jest to spowodowane przez fakt, że na prawo od wierzchołka znajdują się liczby, które są mniejsze od samego wierzchołka i są również większe od elementu o indeksie i = 2. W takim przypadku należy wybrać najmniejszy z elementów znajdujących się na prawo od miejsca w którym chcemy dokonać zmian (indeks i = 2). W tym przypadku najmniejszy z dostępnych elementów znajduje się pod indeksem i = 5. Tak więc następuje zamiana elementów o indeksach i = 2 oraz i = 5 tak jak pokazano na poniższym rysunku.
<p align="center">
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204915369-ff3bc994-efc6-481d-b688-087433ae5616.png">
</p>  
Kolejnym krokiem tak jak poprzednio jest posortowanie poszczególnych elementów - w tym przypadku dla indeksów i > 2. Zostało to przedstawione poniżej.
<p align="center">
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204915678-480d8a46-c81e-405a-94b4-7ded8f4bf117.png">
</p>
W ten sposób możemy uzyskać kolejną permutację dla dowolnego przypadku - ograniczeniem jest brak możliwości występowania elementów o jednakowej wartości pod różnymi adresami. W TSP jednak nie występuje taka zależność.
  
## Wyznaczanie i-tej permutacji
Aby móc równolegle obliczać kolejne permutacje potrzebujemy znać zbiór permutacji początkowych, od których poszczególne wątki będą przeprowadzać obliczenia związane z wyznaczeniem kolejnych permutacji.
<p align="center">
    <img width="1000" src="https://user-images.githubusercontent.com/96399051/204917619-1807a01b-0ced-4703-9b83-02ab8098e2a3.png">
</p>
Powyższa grafika przedstawia przestrzeń do wyznaczenia wszystkich iteracji dla zbioru składającego się z 4 elementów. Wszystkich permutacji jest 4! = 24. Chcąc realizować te peramutacje równolegle korzystając z uprzednio przedstawionego algorytmu musimy poznać pierwszą permutację dla każdego podzbioru permutacji - zostały one rozróżnione kolorami. Elementy o indeksach i = 0, 6, 12, 18 są pierwszymi w każdym podzbiorze. Musimy więc wyznaczyć permutacje początkowe dla tych elementów.
  
Korzystając z tego, że wyznaczamy permutacje w kolejności leksykograficznej możemy obliczać jaka będzie i-ta permutacja za pomocą sprytnego algorytmu opartego na silniowym systemie pozycyjnym (Factorial number system)
  
### Factorial number system
Zeby zrozumieć o co chodzi chcę przywołać analogię do systemu dziesiętnego. W systemie dziesiętnym liczby zapisywane są za pomocą ciągu cyfr, gdzie docelowa wartość liczby stworzonej przez ten ciąg jest liczona na bazie dziesiętnej:
<p align="center">
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204919697-e17cbef9-447d-4314-9ab7-90da436b5ebf.png">
</p>

W przypadku systemu silniowego bazą zapisu jest silnia. Liczbę z systemu dziesiętnego możemy przenieść do systemu silniowego realizując operację rozkładu za pomocą dzielenia z resztą. Dla przykładu rozważmy znów 123:
<p align="center">
    <img width="350" src="https://user-images.githubusercontent.com/96399051/204921271-e0923c85-2c2c-4fa0-b1fc-b12d09e9a266.png">
</p>
Wykonane operacje:
  
1. W pierwszym kroku dzielimy liczbę 123 przez 1, efektem tego jest liczba 123 oraz 0 reszty,
2. Następnie dzielimy wyżej uzyskany rezultat przez kolejną liczbę jaką jest 2 - uzyskujemy 61 całości oraz 1 reszty,
3. Liczbę 61 dzielimy przez kolejny dzielnik jakim jest 3 co daje 20 całości i 1 reszty,
4. Liczbę 20 dzielimy przez 4 co daje 5 całości i 0 reszty,
5. Liczbę 5 dzielimy przez 5 co daje 1 całości i 0 reszty,
6. Liczbę 1 dzielimy przez 6 co daje 0 całości i 1 reszty.
  
W powyższym zestawie operacji istotny jest zapis reszty. Reprezentacją liczby 123 (system dziesiętny) w systemie silniowym jest ciąg 1:0:0:1:1:0!
  
### Co to ma wspólnego z i-tą permutacją?
  
To jest ciekawe ponieważ, chcąc wyznaczyć i-tą permutację na zbiorze n elementów ułożonych w kolejności leksykograficznej w początkowej permutacji np: dla n = 4   :   1, 2, 3, 4 możemy obliczyć reprezentację silniową liczby która określa numer permutacji, który chcemy uzyskać. Obliczmy np 4 permutację na powyższym zbiorze. 

Na początek zrobimy to ręcznie:
  
<div align="center">
<table>
  <tr> <td>Zbiór początkowy</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> </tr>
  <tr> <td>Permutacja 1</td> <td>1</td> <td>2</td> <td>4</td> <td>3</td> </tr>
  <tr> <td>Permutacja 2</td> <td>1</td> <td>3</td> <td>2</td> <td>4</td> </tr>
  <tr> <td>Permutacja 3</td> <td>1</td> <td>3</td> <td>4</td> <td>3</td> </tr>
  <tr> <td>Permutacja 4</td> <td>1</td> <td>4</td> <td>2</td> <td>3</td> </tr>
</table>
</div>

Zapiszmy teraz numer permutacji którą chcemy uzyskać w reprezentacji silniowej.
<p align="center">
    <img width="350" src="https://user-images.githubusercontent.com/96399051/204923629-fed2c10c-3923-4da0-83cd-afe7c4d6290a.png">
</p>

Wiemy teraz, że reprezentacja silniowa liczby 4 wynosi:
<div align="center">
<table>
  <tr> <td>2</td> <td>0</td> <td>0</td> </tr>
</table>
</div>
  
Ten ciąg liczb jest bardzo ważny. Musimy jednak go zmodyfikować ponieważ każda z cyfr w tym ciągu odnosi się do cyfry w permutacji początkowej, w której mamy 4 cyfry, tak więc, żeby nasz ciąg miał 4 cyfry, dopisujemy z jego lewej strony zera, tak by dopełnić do odpowiedniej liczby cyfr. Tak więc docelowy ciąg będzie w postaci:
<div align="center">
<table>
  <tr> <td>0</td> <td>2</td> <td>0</td> <td>0</td> </tr>
</table>
</div>

Na tej podstawie możemy obliczyć jakie elementy należy poddać permutacji. Zabawa polega na tym, że iterujemy się przez poszczególne cyfry powyższego ciągu od lewej do prawej. Każda kolejna cyfra oznacza jaką liczbę ze zbioru cyfr wchodzących w skład permutacji początkowej ułożonej leksykograficznie:
<div align="center">
<table>
  <tr> <td>liczba</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> </tr>
  <tr> <td>indeks</td> <td>0</td> <td>1</td> <td>2</td> <td>3</td> </tr>
</table>
</div>

Pierwsza liczba w ciągu silniowym to 0, oznacza to, że jako pierwszą cyfrę do docelowej permutacji należy wybrać element na zerowej pozycji z początkowego zbioru liczb. Będzie to 1. Następnie ze zbioru dostępnych elementów usuwamy jedynkę oraz aktualizujemy indeksy. Kolejna cyfra to 2. Oznacza to, że na drugiej pozycji w docelowej permutacji znajdzie się liczba o indeksie 2 ze zbioru dostępnych elementów - jest nią 4. Realizujemy ten algorytm dla pozostałych elementów, tak jak pokazano na poniższym rysunku:
<p align="center">
    <img width="700" src="https://user-images.githubusercontent.com/96399051/204927376-e64af8b1-f0ea-43e4-b918-132c75ee2723.png">
</p>

Jak widać, uzyskana permutacja jest dokładnie taka sama jak ręcznie wyznaczona wcześniej.
  
## Obliczenia równoległe

Podejście jakie planuję przyjąć jest dwojakie:
 * każdy z wątków oblicza kolejne permutacje dla swojej grupy, która dostała permutację początkową wyliczoną na podstawie rozkładu silniowego - w jednym podejściu, możemy obliczyć wszystkie możliwe permutacje.
 * każdy i-ty wątek liczy i-tą permutację. Jednak nie mamy nieskończoność wątków, tylko jakieś 1024 * 1024 * 64 = 67 108 864 (67 baniek). Oznacza to, że na strzała będziemy mogli obliczyć nawet 11! permutacji, ponieważ 11! = 39 916 800 (39 baniek). Jednak chcąc obliczyć pozostałe permutacje, będziemy musieli wszystkie wątki ponownie zaprzęc do roboty, tak, żeby obliczyły pozostałe permutacje gdy ich łączna ilość przekracza ~67 baniek.

Na ten moment zająłem się drugą wersją, i nawet działa. Jednak jest ona dość upośledzona ponieważ nie ma adaptacyjnie dobieranych rozmiarów grid'ów i block'ów, do tego nie ma sprawdzania czy nie musimy liczyć więcej niż 1 raz za pomocą wszystkich wątków (dla permutacji powyżej 11!).

Na ten moment to co działa składa się z wyznaczania i-tej iteracji za pomocą GPU. Adapatacyjnie - odpowiednio do ilości koniecznych obliczeń do wykonania uruchamiam wątki, które obliczają iterację o numerze odpowiadającym ich indeksie. Dodatkowo procesor wykonuje sekwencyjnie operacje wyznaczenia i-tej operacji. Operacje wykonywane są w jednymn i drugim miejscu w celu weryfikacji poprawności obliczeń na GPU oraz porównania czasów jakie są konieczne na wykonanie tych obliczeń.

### Uzyskane rezultaty
Niestety efekty obliczeń nie są satysfakcjonujące. GPU jest przy aktualnym algorytmie wolniejsze od CPU...
Tak prezentują się czasy obliczeń dla poszczególnej liczby węzłów:
<p align="center">
    <img width="1000" src="https://user-images.githubusercontent.com/96399051/205522830-a8f08199-d766-4063-8ca1-fb1d87e33834.png">
</p>
 
 Łatwiej jednak porównać wyniki na wykresach:
 
<p align="center">
    <img width="1000" src="https://user-images.githubusercontent.com/96399051/205522892-42c28dea-0836-44b8-8a8c-3c05044dbb0c.png">
    <img width="1000" src="https://user-images.githubusercontent.com/96399051/205522898-b0b6440f-f338-40ab-aa0f-e620c4b28628.png">
</p>


