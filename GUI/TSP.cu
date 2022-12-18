#include "TSP.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include <time.h>
#include <stdlib.h>

#include <chrono>


inline char gpuErrorCheck(cudaError_t code) {
	// sprawdzenie czy działanie funkcji zakończyło się błędem
	if (code != cudaSuccess) {
		// printowanie błędu, pliku w którym wystąpił oraz linii kodu
		const char* e = cudaGetErrorString(code);
		return e[0];
	}
	return 'o';
}


unsigned long long factorial(int n) {
	unsigned long long resoult = 1;
	while (n) resoult *= n--;
	return resoult;
}


void swap(int& a, int& b) {
	int temp = b;
	b = a;
	a = temp;
}


__device__ void swapGPU(int& a, int& b) {
	int temp = b;
	b = a;
	a = temp;
}


int partition(int arr[], int start, int end) {

	int pivot = arr[start];

	int count = 0;
	for (int i = start + 1; i <= end; i++) {
		if (arr[i] <= pivot)
			count++;
	}

	// Giving pivot element its correct position
	int pivotIndex = start + count;
	swap(arr[pivotIndex], arr[start]);

	// Sorting left and right parts of the pivot element
	int i = start, j = end;

	while (i < pivotIndex && j > pivotIndex) {

		while (arr[i] <= pivot) {
			i++;
		}

		while (arr[j] > pivot) {
			j--;
		}

		if (i < pivotIndex && j > pivotIndex) {
			swap(arr[i++], arr[j--]);
		}
	}

	return pivotIndex;
}


__device__ int partitionGPU(int arr[], int start, int end) {

	int pivot = arr[start];

	int count = 0;
	for (int i = start + 1; i <= end; i++) {
		if (arr[i] <= pivot)
			count++;
	}

	// Giving pivot element its correct position
	int pivotIndex = start + count;
	swapGPU(arr[pivotIndex], arr[start]);

	// Sorting left and right parts of the pivot element
	int i = start, j = end;

	while (i < pivotIndex && j > pivotIndex) {

		while (arr[i] <= pivot) {
			i++;
		}

		while (arr[j] > pivot) {
			j--;
		}

		if (i < pivotIndex && j > pivotIndex) {
			swapGPU(arr[i++], arr[j--]);
		}
	}

	return pivotIndex;
}


void quickSort(int arr[], int start, int end) {

	// base case
	if (start >= end)
		return;

	// partitioning the array
	int p = partition(arr, start, end);

	// Sorting the left part
	quickSort(arr, start, p - 1);

	// Sorting the right part
	quickSort(arr, p + 1, end);
}


__device__ void quickSortGPU(int arr[], int start, int end) {

	// base case
	if (start >= end)
		return;

	// partitioning the array
	int p = partitionGPU(arr, start, end);

	// Sorting the left part
	quickSortGPU(arr, start, p - 1);

	// Sorting the right part
	quickSortGPU(arr, p + 1, end);
}


void find_ith_permutation(int arr[], int n, int index, int* sol) {

	// stworzenie tablicy, na której będą przekształcenia
	int* _arr = (int*)malloc(n * sizeof(int));
	for (int j = 0; j < n; j++) {
		_arr[j] = arr[j];
	}

	// create array with known size equal to n
	int* factoradic = (int*)malloc(n * sizeof(int));

	// factorial decomposition with modulo function
	int rest = index;
	for (int j = 1; j <= n; j++) {
		factoradic[n - j] = rest % j;
		rest /= j;
	}

	// array to contain target permutation
	int* permutation_arr = (int*)malloc(n * sizeof(int));
	int _n = n - 1;

	// iteration over all elements in factoradic
	for (int j = 0; j < n; j++) {
		// Assigning factoradic[j]-th element of array to target array 
		permutation_arr[j] = _arr[factoradic[j]];

		// instead of creating new array I am moving all elements which will
		// still be in my factoradic to the left and I am decreasing size of
		// this array to assure that I am only using proper part of it
		for (int k = 0; k < (_n - factoradic[j]); k++) {
			swap(_arr[factoradic[j] + k], _arr[factoradic[j] + k + 1]);
		}
		_n--;
	}
	// zapis permutacji do pamięci
	for (int o = 0; o < n; o++) {
		sol[o] = permutation_arr[o];
	}

	std::free(factoradic);
	std::free(permutation_arr);
	std::free(_arr);

	return;
}


__device__ void find_ith_permutationGPU(int* sol, int* arr, int n, int index) {
	// int *sol - place for i-th permutation
	// int *arr - place for 0-th permutation
	// int n - length of permutation (nodes count)
	// int index - number of wanted permutation

	// stworzenie tablicy, na której będą przekształcenia
	int* _arr = new int(n);
	if (_arr == NULL) {
		// std::printf("Memory not allocated at: _arr \t\t\t gid: %d\n", index);
	}

	for (int i = 0; i < n; i++) {
		_arr[i] = arr[i];
	}

	// create array with known size equal to n
	int* factoradic = new int(n);

	if (factoradic == NULL) {
		// std::printf("Memory not allocated at: factoradic \t\t gid: %d\n", index);
	}

	// factorial decomposition with modulo function
	int rest = index;
	for (int j = 1; j <= n; j++) {
		factoradic[n - j] = rest % j;
		rest /= j;
	}

	// array to contain target permutation
	int* permutation_arr = new int(n);

	if (permutation_arr == NULL) {
		// std::printf("Memory not allocated at: permutation_arr \t gid: %d\n", index);
	}

	int _n = n - 1;

	// iteration over all elements in factoradic
	for (int j = 0; j < n; j++) {
		// Assigning factoradic[j]-th element of array to target array 
		permutation_arr[j] = _arr[factoradic[j]];

		// instead of creating new array I am moving all elements which will
		// still be in my factoradic to the left and I am decreasing size of
		// this array to assure that I am only using proper part of it
		for (int k = 0; k < (_n - factoradic[j]); k++) {
			swapGPU(_arr[factoradic[j] + k], _arr[factoradic[j] + k + 1]);
		}
		_n--;
	}
	// put proper element into target array
	for (int o = 0; o < n; o++) {
		sol[o] = permutation_arr[o];
	}
	delete[] _arr;
	delete[] factoradic;
	delete[] permutation_arr;
	return;
}


__device__ float calculate_length_of_permutation(float* distances, int* arr, int n, int source) {

	float len = 0.0;
	int from = 0;
	int to = 0;
	int m = n + 1;

	for (int i = 0; i <= n; i++) {
		if (i == 0) {
			// uwzględnienie odległości od punktu source do pierwszego
			from = source;
			to = arr[i];
		}
		else if (i == n) {
			// uwzględnienie odległości od ostatniego punktu do source 
			from = arr[n - 1];
			to = source;
		}
		else {
			// pozostałe przypadki
			from = arr[i - 1];
			to = arr[i];
		}

		// sumowanie odległości
		len += distances[from * m + to];
	}
	return len;
}


float calculate_length_of_permutation_CPU(float* distances, int* arr, int n, int source) {

	float len = 0.0;
	int from = 0;
	int to = 0;
	int m = n + 1;

	for (int i = 0; i <= n; i++) {
		if (i == 0) {
			// uwzględnienie odległości od punktu source do pierwszego
			from = source;
			to = arr[i];
		}
		else if (i == n) {
			// uwzględnienie odległości od ostatniego punktu do source 
			from = arr[n - 1];
			to = source;
		}
		else {
			// pozostałe przypadki
			from = arr[i - 1];
			to = arr[i];
		}

		// sumowanie odległości
		len += distances[from * m + to];
	}
	return len;
}


float next_permutation(int n, int index, int current_permutation[], float* distances, int source) {
	// create array same as current_permutation
	/*int* target_permutation = new int(n);
	for (int i = 0; i < n; i++) {
		target_permutation[i] = current_permutation[i];
	}*/

	// set max_index to index of last value
	int max_index = n - 1;

	// iterate over elements in array to find peak
	for (int i = n - 2; i >= 0; i--) {
		// checking if current element is peak
		if (current_permutation[i] < current_permutation[max_index]) {
			// checking if after peak - from right side there are elements smaller 
			// than peak but bigger than element on very left of peak
			// done by iterating and finding element which fulfills conditions
			int min_index = max_index;
			for (int j = i + 2; j <= n - 1; j++) {
				if (current_permutation[j] < current_permutation[min_index] &&
					current_permutation[j] > current_permutation[i]) {
					min_index = j;
				}
			}

			// now time to swap two elements - one on the very left of the peak
			// second is defined as min_index
			swap(current_permutation[i], current_permutation[min_index]);

			// if swap took place on elements beyond n-2 there is need to sort those elements 
			if (i < n - 2) {
				quickSort(current_permutation, i + 1, n - 1);
			}
			break;
		}
		else {
			max_index = i;
		}
	}
	float res = calculate_length_of_permutation_CPU(distances, current_permutation, n, source);
	return res;
}


__device__ void next_permutationGPU(int n, int current_permutation[]) {
	// set max_index to index of last value
	int max_index = n - 1;

	// iterate over elements in array to find peak
	for (int i = n - 2; i >= 0; i--) {
		// checking if current element is peak
		if (current_permutation[i] < current_permutation[max_index]) {
			// checking if after peak - from right side there are elements smaller 
			// than peak but bigger than element on very left of peak
			// done by iterating and finding element which fulfills conditions
			int min_index = max_index;
			for (int j = i + 2; j <= n - 1; j++) {
				if (current_permutation[j] < current_permutation[min_index] &&
					current_permutation[j] > current_permutation[i]) {
					min_index = j;
				}
			}

			// now time to swap two elements - one on the very left of the peak
			// second is defined as min_index
			swapGPU(current_permutation[i], current_permutation[min_index]);

			// if swap took place on elements beyond n-2 there is need to sort those elements 
			if (i < n - 2) {
				quickSortGPU(current_permutation, i + 1, n - 1);
			}
			break;
		}
		else {
			max_index = i;
		}
	}
	return;
}


__global__ void find_permutation_combined(int n, int sol_num, int solutions_per_thread, int* index_of_min_permutation, float* distance, float* lenth_of_min_permutation, int source) {

	// calculating id for each thread
	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int gid = offset + tid;

	int start_permutation_number = gid * solutions_per_thread;

	if (start_permutation_number > sol_num) {
		return;
	}


	// 0-th permutation
	int* zero_permutation = new int(n);

	// filling array with values of 0-th permutation
	for (int i = 0; i < n; i++) {
		// uzupełniamy tak, żeby pominąć źródło
		if (i < source) {
			zero_permutation[i] = i;
		}
		else {
			zero_permutation[i] = i + 1;
		}
	}

	// place for i-th permutation
	int* ith_permutation = new int(n);
	if (ith_permutation == NULL) {
	}

	// znalezienie pierwszej permutacji dla danego wątka
	find_ith_permutationGPU(ith_permutation, zero_permutation, n, start_permutation_number);

	// deklaracja zmiennych z indeksem oraz minimalną długością
	int min_index = start_permutation_number;
	float min_length = calculate_length_of_permutation(distance, ith_permutation, n, source);

	// zmienna do przechowywania aktualnej długości permutacji
	float current_length = min_length;

	// szukanie wszystkich kolejnych permutacji dla tego wątka
	for (int ind = 1; ind <= solutions_per_thread; ind++) {
		next_permutationGPU(n, ith_permutation);

		// sprawdzenie czy dana permutacja jest krótsza od aktualnie najkrószej:
		current_length = calculate_length_of_permutation(distance, ith_permutation, n, source);
		if (current_length < min_length) {
			min_length = current_length;
			min_index = start_permutation_number + ind;
		}
	}
	// zapisanie indeksu z minimalna permutacją
	index_of_min_permutation[gid] = min_index;
	lenth_of_min_permutation[gid] = min_length;

	return;
}


bool checkValidity(int* GPU, int* CPU, int sol_num, int n) {

	for (int i = 0; i < sol_num; i++) {
		for (int j = 0; j < n; j++) {
			if (GPU[i * n + j] != CPU[i * n + j]) {

				// pokazanie kilku rozwiązań z miejsca wystąpienia błędu
				for (int p = i - 1; p < i + 5; p++) {
					for (int r = 0; r < n; r++) {
						// std::printf("%d\t", GPU[n * p + r]);
					}
					// std::printf("\t\t");
					for (int r = 0; r < n; r++) {
						// std::printf("%d \t", CPU[n * p + r]);
					}
					// std::printf("\n");
				}
				return false;
			}
		}
	}
	return true;
}


struct dimensions {
	int block;
	int grid_x;
	int solutions_per_thread;
};


dimensions get_dimensions(int n, int sol_num) {
	// dopasowanie na czuja
	int solutions_per_thread[13]{ 1, 1, 1, 2, 6, 24, 42 ,56, 72, 360, 1980, 23760, 95040 };

	// deklaracja struktury do zwracania danych
	dimensions dim;

	// obliczenie ile wątków musimy aktywować
	int threads_to_activate = sol_num / solutions_per_thread[n - 1];

	dim.solutions_per_thread = solutions_per_thread[n - 1];

	// dopasowanie grida oraz block-ów
	if (threads_to_activate < 1024) {
		// polecane wartości to 128 / 256 - zawsze wielokrotność 32
		int minimal_block_size = 32;

		// how many minimal_block_size fits in sol_num
		int help = threads_to_activate / minimal_block_size;

		// make it minimal value that covers all solutions
		dim.block = minimal_block_size + help * minimal_block_size;
		dim.grid_x = 1;
		return dim;
	}
	int max_threads_per_block = 1024;
	dim.block = max_threads_per_block;
	dim.grid_x = 1 + threads_to_activate / max_threads_per_block;

	return dim;
}


namespace Wrapper {
	fromCUDA wrapper(int m, int * location, int _source)	{

		fromCUDA _solution_;

		int n = m - 1;

		// punkt startowy 
		int source = _source;

		unsigned long long solutions_number = factorial(n);
		// -------------------------------------------------- GRAF -----------------------------------------------

		size_t distance_size = static_cast<size_t>(m * m) * sizeof(float);
		float* distance = (float*)malloc(distance_size);

		if (distance == NULL) {
			_solution_.e = 'a';
			return _solution_;
		}


		// uzupełnianie macierzy odległości from i to j
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				// obliczenie odległości w osi X
				int delta_x = abs(location[2 * i] - location[2 * j]);

				// obliczenie odległości w osi Y
				int delta_y = abs(location[2 * i + 1] - location[2 * j + 1]);

				// obliczenie odległości - pitagoras
				float dist = sqrt(delta_x * delta_x + delta_y * delta_y);

				// zapisanie odpowiednych wartości
				distance[i * m + j] = dist;
			}
		}

		// -------------------------------------------------- GRAF -----------------------------------------------

		// alokacja pamięci w GPU z grafem
		float* distanceGPU;
		cudaMalloc((void**)&distanceGPU, distance_size);
		_solution_.e = gpuErrorCheck(cudaMemcpy(distanceGPU, distance, distance_size, cudaMemcpyHostToDevice));
		if (_solution_.e != 'o') {
			return _solution_;
		}

		if (distanceGPU == NULL) {
			_solution_.e = 'b';
			return _solution_;
		}

		// obliczenie rozmiaru w bajtach tablicy pojedynczej permutacji
		size_t size_in_bytes = static_cast<size_t>(n * sizeof(int));

		int* first_permutation = (int*)malloc(size_in_bytes);
		if (first_permutation == NULL) {
			_solution_.e = 'c';
			return _solution_;
		}

		// filling array with values of 0-th permutation
		for (int i = 0; i < n; i++) {
			// uzupełniamy tak, żeby pominąć źródło
			if (i < source) {
				first_permutation[i] = i;
			}
			else {
				first_permutation[i] = i + 1;
			}
		}

		
		// ---------------------------------------------OBLICZENIA NA CPU--------------------------------------------------------
		auto CPU_start = std::chrono::high_resolution_clock::now();

		int min_index_CPU = 0;
		float min_length_CPU = calculate_length_of_permutation_CPU(distance, first_permutation, n, source);
		float current_length_CPU;

		for (int o = 1; o < solutions_number; o++) {
			current_length_CPU = next_permutation(n, o, first_permutation, distance, source);
			if (current_length_CPU < min_length_CPU) {
				min_index_CPU = o;
				min_length_CPU = current_length_CPU;
			}
		}

		auto CPU_finish = std::chrono::high_resolution_clock::now();
		auto CPU_duration = std::chrono::duration_cast<std::chrono::microseconds>(CPU_finish - CPU_start);
		// ---------------------------------------------------------------------------------------------------------------------

		_solution_.CPU_time = static_cast<unsigned long>(CPU_duration.count());

		// ponowne uzupełnienie first_permutation pierwotnym ciągiem
		for (int i = 0; i < n; i++) {
			// uzupełniamy tak, żeby pominąć źródło
			if (i < source) {
				first_permutation[i] = i;
			}
			else {
				first_permutation[i] = i + 1;
			}
		}

		// obliczenie rozmiaru w bajtach tablicy rozwiązań -> każde rozwiązanie to n intów więc n * sizeof(int) * ilość rozwiązań
		unsigned long long size_in_bytes_of_solutions = n * solutions_number * sizeof(int);

		// stworzenie wskaźnika na tablicę z permutacją początkową
		int* first_permutationGPU;
		cudaMalloc((void**)&first_permutationGPU, static_cast<size_t>(n * sizeof(int)));

		// kopiowanie pamięci z CPU do GPU
		_solution_.e = gpuErrorCheck(cudaMemcpy(first_permutationGPU, first_permutation, size_in_bytes, cudaMemcpyHostToDevice));
		if (_solution_.e != 'o') {
			return _solution_;
		}

		dimensions dims = get_dimensions(n, solutions_number);
		dim3 block(dims.block, 1, 1);
		dim3 grid(dims.grid_x, 1, 1);

		int active_threads = solutions_number / dims.solutions_per_thread;

		// tablice do przechowywania indeksu permutacji dla najmniejszej ścieżki oraz jej długości dla każdego z threadów
		int* index_of_min_permutation_GPU;
		float* length_of_min_permutation_GPU;

		// obliczenie rozmiarów tablic w bajtach - rozmiar = liczba potrzebnych wątków
		size_t min_index_permutation_array_size = static_cast<size_t>(active_threads * sizeof(int));
		size_t min_length_permutation_array_size = static_cast<size_t>(active_threads * sizeof(float));

		// alokacja pamięci w GPU
		cudaMalloc((void**)&index_of_min_permutation_GPU, min_index_permutation_array_size);
		cudaMalloc((void**)&length_of_min_permutation_GPU, min_length_permutation_array_size);

		// ustawienie każdego z elementów na wartość INT_MAX
		cudaMemset(index_of_min_permutation_GPU, INT_MAX, n);
		cudaMemset(length_of_min_permutation_GPU, 9999.9, n);

		// zrobienie analogicznych tablic dla CPU:
		int* index_of_min_permutation_CPU = (int*)malloc(min_index_permutation_array_size);
		float* length_of_min_permutation_CPU = (float*)malloc(min_length_permutation_array_size);

		// gpuErrorCheck(cudaDeviceReset());

		// timer start
		auto GPU_start = std::chrono::high_resolution_clock::now();

		// calling kernel
		//find_ith_permutationGPU <<< grid, block >>> (d_solutionsGPU, first_permutationGPU, n, solutions_number);
		find_permutation_combined <<< grid, block >>> (n, solutions_number, dims.solutions_per_thread,
			index_of_min_permutation_GPU, distanceGPU, length_of_min_permutation_GPU, source);

		// wait till device sunc
		_solution_.e = gpuErrorCheck(cudaDeviceSynchronize());
		if (_solution_.e != 'o') {
			return _solution_;
		}

		auto GPU_finish = std::chrono::high_resolution_clock::now();
		auto GPU_duration = std::chrono::duration_cast<std::chrono::microseconds>(GPU_finish - GPU_start);

		_solution_.GPU_time = static_cast<unsigned long>(GPU_duration.count());


		_solution_.e = gpuErrorCheck(cudaMemcpy(index_of_min_permutation_CPU, index_of_min_permutation_GPU, min_index_permutation_array_size, cudaMemcpyDeviceToHost));
		if (_solution_.e != 'o') {
			return _solution_;
		}

		_solution_.e = gpuErrorCheck(cudaMemcpy(length_of_min_permutation_CPU, length_of_min_permutation_GPU, min_length_permutation_array_size, cudaMemcpyDeviceToHost));
		if (_solution_.e != 'o') {
			return _solution_;
		}

		// szukanie minimalnej permutacji w tym, co zrealizowało GPU:
		int min_index = index_of_min_permutation_CPU[0];
		float min_length = length_of_min_permutation_CPU[0];

		for (int i = 1; i < active_threads; i++) {
			if (length_of_min_permutation_CPU[i] < min_length) {
				min_index = index_of_min_permutation_CPU[i];
				min_length = length_of_min_permutation_CPU[i];
			}
		}
						
		_solution_.cites[0] = source;

		// ponowne uzupełnienie first_permutation pierwotnym ciągiem
		for (int i = 0; i < n; i++) {
			// uzupełniamy tak, żeby pominąć źródło
			if (i < source) {
				first_permutation[i] = i;
			}
			else {
				first_permutation[i] = i + 1;
			}
		}

		
		int* target = (int*)malloc(size_in_bytes);
		if (target == NULL) {
			_solution_.e = 'c';
			return _solution_;
		}
		find_ith_permutation(first_permutation, n, min_index_CPU, target);

		for (int i = 1; i <= n; i++) {
			_solution_.cites[i] = target[i-1];
		}

		_solution_.cites[m] = source;

		//zwolnienie pamięci w GPU
		cudaFree(first_permutationGPU);
		cudaFree(distanceGPU);
		cudaFree(index_of_min_permutation_GPU);
		cudaFree(length_of_min_permutation_GPU);
		free(distance);
		free(first_permutation);
		free(index_of_min_permutation_CPU);
		free(length_of_min_permutation_CPU);
		free(target);

		_solution_.e = gpuErrorCheck(cudaDeviceReset());
		if (_solution_.e != 'o') {
			return _solution_;
		}

		return _solution_;
	}
}
