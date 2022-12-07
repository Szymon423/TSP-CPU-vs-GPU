#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <time.h>
#include <stdlib.h>

//#include <stdio.h>
//#include <algorithm>
//#include "permutate.cuh"
//#pragma comment(lib, "winmm.lib")


using namespace std;


// definiowanie makra do obs�ugi b��d�w
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	// sprawdzenie czy dzia�anie funkcji zako�czy�o si� b��dem
	if (code != cudaSuccess) {
		// printowanie b��du, pliku w kt�rym wyst�pi� oraz linii kodu
		fprintf(stderr, "GPUassert : %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) {
			// je�li przekazujemy argument true to exitujemy program
			exit(code);
		}
	}
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


void next_permutation(int n, int index, int current_permutation[], int *solutions) {
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

	for (int i = 0; i < n; i++) {
		solutions[n * index + i] = current_permutation[i];
		// printf("%d\t", current_permutation[i]);
	}
    // printf("\n");
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

    /*for (int i = 0; i < n; i++) printf("%d\t", current_permutation[i]);
    printf("\n");*/
}


void find_ith_permutation(int arr[], int n, int index, int* sol) {

	// stworzenie tablicy, na kt�rej b�d� przekszta�cenia
	// int* _arr = new int(n);
	int* _arr = (int*)malloc(n * sizeof(int));
	for (int j = 0; j < n; j++) {
		_arr[j] = arr[j];
	}

	// create array with known size equal to n
	// int* factoradic = new int(n);
	int* factoradic = (int*)malloc(n * sizeof(int));

	// factorial decomposition with modulo function
	int rest = index;
	for (int j = 1; j <= n; j++) {
		factoradic[n - j] = rest % j;
		rest /= j;
		// printf("factoradic[%d] = %d\n", n - j, factoradic[n - j]);
	}

	// array to contain target permutation
	// int* permutation_arr = new int(n);
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
	// just simple print of permutation
	for (int o = 0; o < n; o++) {
		sol[index * n + o] = permutation_arr[o];
		// printf("%d\t", permutation_arr[o]);
	}
	// printf("\n");

	std::free(factoradic);
	std::free(permutation_arr);
	std::free(_arr);

	return;
}

//
//__global__ void find_ith_permutationGPU(int* sol, int* arr, int n, int sol_num) {
//
//	// calculating id for each thread
//	int tid = threadIdx.x;
//	int offset = blockIdx.x * blockDim.x;
//	int gid = offset + tid;
//
//	if (gid > sol_num) {
//		return;
//	}
//
//	// stworzenie tablicy, na kt�rej b�d� przekszta�cenia
//	int* _arr = new int(n);
//	if (_arr == NULL) {
//		printf("Memory not allocated at: _arr \t\t\t gid: %d\n", gid);
//	}
//
//	for (int i = 0; i < n; i++) {
//		_arr[i] = arr[i];
//	}
//
//	// create array with known size equal to n
//	int* factoradic = new int(n);
//
//	if (factoradic == NULL) {
//		printf("Memory not allocated at: factoradic \t\t gid: %d\n", gid);
//	}
//
//	// factorial decomposition with modulo function
//	int rest = gid;
//	for (int j = 1; j <= n; j++) {
//		factoradic[n - j] = rest % j;
//		rest /= j;
//	}
//
//	// array to contain target permutation
//	int* permutation_arr = new int(n);
//
//	if (permutation_arr == NULL) {
//		printf("Memory not allocated at: permutation_arr \t gid: %d\n", gid);
//	}
//
//	int _n = n - 1;
//
//	// iteration over all elements in factoradic
//	for (int j = 0; j < n; j++) {
//		// Assigning factoradic[j]-th element of array to target array 
//		permutation_arr[j] = _arr[factoradic[j]];
//
//		// instead of creating new array I am moving all elements which will
//		// still be in my factoradic to the left and I am decreasing size of
//		// this array to assure that I am only using proper part of it
//		for (int k = 0; k < (_n - factoradic[j]); k++) {
//			swapGPU(_arr[factoradic[j] + k], _arr[factoradic[j] + k + 1]);
//		}
//		_n--;
//	}
//	// put proper element into target array
//	for (int o = 0; o < n; o++) {
//		sol[gid * n + o] = permutation_arr[o];
//	}
//	delete[] _arr;
//	delete[] factoradic;
//	delete[] permutation_arr;
//	return;
//}
//

__device__ void find_ith_permutationGPU(int* sol, int* arr, int n, int index) {
	// int *sol - place for i-th permutation
	// int *arr - place for 0-th permutation
	// int n - length of permutation (nodes count)
	// int index - number of wanted permutation

	// stworzenie tablicy, na kt�rej b�d� przekszta�cenia
	int* _arr = new int(n);
	if (_arr == NULL) {
		std::printf("Memory not allocated at: _arr \t\t\t gid: %d\n", index);
	}

	for (int i = 0; i < n; i++) {
		_arr[i] = arr[i];
	}

	// create array with known size equal to n
	int* factoradic = new int(n);

	if (factoradic == NULL) {
		std::printf("Memory not allocated at: factoradic \t\t gid: %d\n", index);
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
		std::printf("Memory not allocated at: permutation_arr \t gid: %d\n", index);
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

	for (int i = 0; i <= n; i++) {
		if (i == 0) {
			// uwzgl�dnienie odleg�o�ci od punktu source do pierwszego
			from = source;
			to = arr[i];
		}
		else if (i == n) {
			// uwzgl�dnienie odleg�o�ci od ostatniego punktu do source 
			from = arr[n-1];
			to = source;
		}
		else {
			// pozosta�e przypadki
			from = arr[i-1];
			to = arr[i];
		}
		// dopasowanie do indeksacji w grafie
		from--;
		to--;

		// sumowanie odleg�o�ci
		len += distances[from * (n + 1) + to];
	}

	return len;
}


__global__ void find_permutation_combined(int n, int sol_num, int solutions_per_thread, int *solutions, int *index_of_min_permutation, float *distance, float *lenth_of_min_permutation, int source) {
	
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
	if (zero_permutation == NULL) {
		std::printf("Memory not allocated at: zero_permutation \t\t\t gid: %d\n", gid);
	}

	// filling array with values of 0-th permutation
	for (int i = 0; i < n; i++) {
		// uzupe�niamy tak, �eby pomin�� �r�d�o
		if (i + 1 < source) {
			zero_permutation[i] = i + 1;
		}
		else {
			zero_permutation[i] = i + 2;
		}
	}
	
	// place for i-th permutation
	int* ith_permutation = new int(n);
	if (ith_permutation == NULL) {
		std::printf("Memory not allocated at: ith_permutation \t\t\t gid: %d\n", gid);
	}
	
	// znalezienie pierwszej permutacji dla danego w�tka
	find_ith_permutationGPU(ith_permutation, zero_permutation, n, start_permutation_number);
	
	// deklaracja zmiennych z indeksem oraz minimaln� d�ugo�ci�
	int min_index = start_permutation_number;
	float min_length = calculate_length_of_permutation(distance, ith_permutation, n, source);

	// zapisanie tej permutacji do solutions
	for (int i = 0; i < n; i++) {
		solutions[start_permutation_number * n + i] = ith_permutation[i];
		// printf("%d\t", ith_permutation[i]);
	}

	// zmienna do przechowywania aktualnej d�ugo�ci permutacji
	float current_length = min_length;

	// szukanie wszystkich kolejnych permutacji dla tego w�tka
	for (int ind = 1; ind <= solutions_per_thread; ind++) {
		next_permutationGPU(n, ith_permutation);
		for (int i = 0; i < n; i++) {
			solutions[start_permutation_number * n + i + n * ind] = ith_permutation[i];
			// printf("%d\t", ith_permutation[i]);
		}
		// sprawdzenie czy dana permutacja jest kr�tsza od aktualnie najkr�szej:
		current_length = calculate_length_of_permutation(distance, ith_permutation, n, source);
		if (current_length < min_length) {
			min_length = current_length;
			min_index = start_permutation_number + ind;
		}
	}
	// zapisanie indeksu z minimalna permutacj�
	index_of_min_permutation[gid] = min_index;
	lenth_of_min_permutation[gid] = min_length;

	// printf("\n");	
	
	return;
}


bool checkValidity(int* GPU, int* CPU, int sol_num, int n) {
	
	for (int i = 0; i < sol_num; i++) {
		for (int j = 0; j < n; j++) {
			if (GPU[i * n + j] != CPU[i * n + j]) {
				std::printf("\nNot valid data at index : %d\n", i);
				
				// pokazanie kilku rozwi�za� z miejsca wyst�pienia b��du
				std::printf("\t\tGPU:\t\t\t\tCPU:\n");
				for (int p = i-1 ; p < i + 5; p++) {
					std::printf("%d\t\t", p);
					for (int r = 0; r < n; r++) {
						std::printf("%d\t", GPU[n * p + r]);
					}
					std::printf("\t\t");
					for (int r = 0; r < n; r++) {
						std::printf("%d \t", CPU[n * p + r]);
					}
					std::printf("\n");
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
	int solutions_per_thread[11] {1, 1, 1, 2, 6, 24, 42 ,56, 72, 360, 1980};

	// deklaracja struktury do zwracania danych
	dimensions dim;

	// obliczenie ile w�tk�w musimy aktywowa�
	int threads_to_activate = sol_num / solutions_per_thread[n - 1];

	dim.solutions_per_thread = solutions_per_thread[n - 1];

	// dopasowanie grida oraz block-�w
	if (threads_to_activate < 1024) {
		// polecane warto�ci to 128 / 256 - zawsze wielokrotno�� 32
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


// int main(int argc, char **argv) {
int main() {

	//int m = atoi(argv[1]);
	int m = 5;

	int n = m - 1;

	// punkt startowy 
	int source = 3;

	unsigned long long solutions_number = factorial(n);
	// -------------------------------------------------- GRAF -----------------------------------------------
		
	// macierz po�o�e� - m x m, w przestrzeni kartezja�skiej 2d ale zapisana w formie tablicy 1d, zamiast 2d
	// kazdy punkt (miasto) ma wsp�rz�dne x oraz y
	int* location = (int*)malloc(static_cast<size_t>(2 * m * sizeof(int)));

	// losowanie unikatowych punkt�w w przestrzeni
	srand(time(NULL));

	bool point_allready_exist = false;
	int x, y;

	for (int i = 0; i < 2 * m; i += 2) {
		// losowanie punktu, kt�rego jeszcze nie by�o
		point_allready_exist = false;

		// realizuj tak d�ugo, a� wylosowany punkt b�dzie inny od istniej�cych
		do {
			// przyjmujemy, �e punkt jest unikatowy
			point_allready_exist = false;

			// losujemy wsp�rz�dne
			x = rand() % (m + 1);
			y = rand() % (m + 1);

			// sprawdzenie, czy nie istnieje ju� punkt o tych wsp�rz�dnych
			for (int j = 0; j < i; j += 2) {
				// sprawdzenie wsp�rz�dnej X
				if (x == location[i - j]) {
					// sprawdzenie wsp�rz�dnej Y
					if (y == location[i - j + 1]) {
						point_allready_exist = true;
					}
				}
			}

		} while (point_allready_exist);	

		location[i] = x;
		location[i + 1] = y;
		std::printf("node: %d\t\tx: %d\ty: %d\n", i/2, location[i], location[i + 1]);
	}

	// macierz odleg�o�ci - w formie tablicy 1d, zamiast 2d (pogl�dowo 2d:)
	//					to
	//	        | 0   1   2   3  
	//	    ----|----------------
	//   f    0	|
	//   r    1	|
	//   o    2	|
	//   m    3	|
	size_t distance_size = static_cast<size_t>(m * m * sizeof(float));
	float* distance = (float*)malloc(distance_size);


	// uzupe�nianie macierzy odleg�o�ci from i to j
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			// obliczenie odleg�o�ci w osi X
			int delta_x = abs(location[2 * i] - location[2 * j]);

			// obliczenie odleg�o�ci w osi Y
			int delta_y = abs(location[2 * i + 1] - location[2 * j + 1]);

			// obliczenie odleg�o�ci - pitagoras
			float dist = sqrt(delta_x * delta_x + delta_y * delta_y);

			// zapisanie odpowiednych warto�ci
			distance[i * m + j] = dist;

			// std::printf("from: %d\tto: %d\tdist: %4.2f\n", i, j, distance[i * m + j]);
		}
	}	

	// -------------------------------------------------- GRAF -----------------------------------------------

	// alokacja pami�ci w GPU z grafem
	float* distanceGPU;
	cudaMalloc((void**)&distanceGPU, distance_size);
	gpuErrorCheck(cudaMemcpy(distanceGPU, distance, distance_size, cudaMemcpyHostToDevice));

	int* first_permutation = 0;

	// obliczenie rozmiaru w bajtach tablicy pojedynczej permutacji
	size_t size_in_bytes = static_cast<size_t>(n * sizeof(int));

	first_permutation = (int*)malloc(size_in_bytes);

	// filling array with values of 0-th permutation
	for (int i = 0; i < n; i++) {
		// uzupe�niamy tak, �eby pomin�� �r�d�o
		if (i + 1 < source) {
			first_permutation[i] = i + 1;
		}
		else {
			first_permutation[i] = i + 2;
		}
	}

	// wska�nik na rozwi�zania z CPU w RAMie
	int* h_solutionsCPU;
	h_solutionsCPU = (int*)malloc(static_cast<size_t>(n * solutions_number * sizeof(int)));

	if (h_solutionsCPU == NULL) {
		std::printf("Memory not allocated.\n");
	}

	// ---------------------------------------------NEXT PERMUTATION--------------------------------------------------------
	auto CPU_start = chrono::high_resolution_clock::now();

	for (int o = 1; o < solutions_number; o++) {
	    next_permutation(n, o, first_permutation, h_solutionsCPU);
	}

	auto CPU_finish = chrono::high_resolution_clock::now();
	auto CPU_duration = chrono::duration_cast<chrono::microseconds>(CPU_finish - CPU_start);
	// ---------------------------------------------------------------------------------------------------------------------

	 	 
	// ponowne uzupe�nienie first_permutation pierwotnym ci�giem
	for (int i = 0; i < n; i++) {
		// uzupe�niamy tak, �eby pomin�� �r�d�o
		if (i + 1 < source) {
			first_permutation[i] = i + 1;
		}
		else {
			first_permutation[i] = i + 2;
		}
	}

	// obliczenie rozmiaru w bajtach tablicy rozwi�za� -> ka�de rozwi�zanie to n int�w wi�c n * sizeof(int) * ilo�� rozwi�za�
	unsigned long long size_in_bytes_of_solutions = n * solutions_number * sizeof(int);

	// wska�nik na rozwi�zania pochodz�ce z GPU w RAMie
	int* h_solutionsGPU = (int*)malloc(static_cast<size_t>(n * solutions_number * sizeof(int)));
	// int* h_solutionsGPU = new int(static_cast<unsigned long long>(n) * solutions_number);
	if (h_solutionsGPU == NULL) {
		std::printf("Memory not allocated.\n");
	}

	// alokacja pami�ci na GPU oraz na CPU

	// stworzenie wska�nika na tablic� rozwi�za� z GPU znajduj�c� si� w VRAMie
	int* d_solutionsGPU;
	gpuErrorCheck(cudaMalloc((void**)&d_solutionsGPU, static_cast<size_t>(n * solutions_number * sizeof(int))));
	if (d_solutionsGPU == NULL) {
		std::printf("Memory not allocated.\n");
	}

	// stworzenie wska�nika na tablic� z permutacj� pocz�tkow�
	int* first_permutationGPU;
	cudaMalloc((void**)&first_permutationGPU, static_cast<size_t>(n * sizeof(int)));

	// kopiowanie pami�ci z CPU do GPU
	gpuErrorCheck(cudaMemcpy(first_permutationGPU, first_permutation, size_in_bytes, cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_solutionsGPU, h_solutionsGPU, size_in_bytes_of_solutions, cudaMemcpyHostToDevice));

	// ka�dy blok mo�e mie� maksymalnie 1024 thread-y dlatego daj� tam max warto��
	// dim3 block(1024);

	// grid to zestawienie obok siebie blok�w thread-�w, mo�e by� w osi x maksymalnie 2^32 - 1 thread�w
	// dim3 grid(1024, 1024);

	dimensions dims = get_dimensions(n, solutions_number);
	dim3 block(dims.block, 1, 1);
	dim3 grid(dims.grid_x, 1, 1);
	std::printf("block.x : %d\n", block.x);
	std::printf("grid.x : %d\n", grid.x);
	std::printf("solutions_per_thread : %d\n", dims.solutions_per_thread);
	int active_threads = solutions_number / dims.solutions_per_thread;



	// tablice do przechowywania indeksu permutacji dla najmniejszej �cie�ki oraz jej d�ugo�ci dla ka�dego z thread�w
	int* index_of_min_permutation_GPU;
	float* length_of_min_permutation_GPU;

	// obliczenie rozmiar�w tablic w bajtach - rozmiar = liczba potrzebnych w�tk�w
	size_t min_index_permutation_array_size = static_cast<size_t>(active_threads * sizeof(int));
	size_t min_length_permutation_array_size = static_cast<size_t>(active_threads * sizeof(float));

	// alokacja pami�ci w GPU
	cudaMalloc((void**)&index_of_min_permutation_GPU, min_index_permutation_array_size);
	cudaMalloc((void**)&length_of_min_permutation_GPU, min_length_permutation_array_size);

	// ustawienie ka�dego z element�w na warto�� INT_MAX
	cudaMemset(index_of_min_permutation_GPU, INT_MAX, n);

	// zrobienie analogicznych tablic dla CPU:
	int* index_of_min_permutation_CPU = (int*)malloc(min_index_permutation_array_size);
	float* length_of_min_permutation_CPU = (float*)malloc(min_length_permutation_array_size);



	// timer start
	auto GPU_start = chrono::high_resolution_clock::now();

	// calling kernel
	//find_ith_permutationGPU <<< grid, block >>> (d_solutionsGPU, first_permutationGPU, n, solutions_number);
	find_permutation_combined <<< grid, block >>> (n, solutions_number, dims.solutions_per_thread, 
												   d_solutionsGPU, index_of_min_permutation_GPU, 
												   distanceGPU, length_of_min_permutation_GPU, source);

	// wait till device sunc
	gpuErrorCheck(cudaDeviceSynchronize());

	auto GPU_finish = chrono::high_resolution_clock::now();
	auto GPU_duration = chrono::duration_cast<chrono::microseconds>(GPU_finish - GPU_start);

	// kopiowanie obliczonych danych spowrotem do CPU
	gpuErrorCheck(cudaMemcpy(h_solutionsGPU, d_solutionsGPU, size_in_bytes_of_solutions, cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(index_of_min_permutation_CPU, index_of_min_permutation_GPU, min_index_permutation_array_size, cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(length_of_min_permutation_CPU, length_of_min_permutation_GPU, min_length_permutation_array_size, cudaMemcpyDeviceToHost));


	// ----------------------------------TO G�WNO PONI�EJ wed�ug mnie wcze�niej powodowa�o b��dy--------------------------------------
	// https://stackoverflow.com/questions/28289312/illegal-memory-access-on-cudadevicesynchronize  - fajnie opisane co w zasadzie si� odpierdala - wykraczam poza pamie�
	// auto CPU_start = chrono::high_resolution_clock::now();
	// for (int o = 0; o < solutions_number; o++) {
	// 	find_ith_permutation(first_permutation, n, o + 1, h_solutionsCPU);
	// }
	// 	auto CPU_finish = chrono::high_resolution_clock::now();
	// 	auto CPU_duration = chrono::duration_cast<chrono::microseconds>(CPU_finish - CPU_start);
	// ------------------------------------------------------------------------------------------------------------------------------

	// filling up solutions with first permutation
	for (int i = 0; i < n; i++) {
		// uzupe�niamy tak, �eby pomin�� �r�d�o
		if (i + 1 < source) {
			h_solutionsGPU[i] = i+1;
			h_solutionsCPU[i] = i+1;
		}
		else {
			h_solutionsGPU[i] = i+2;
			h_solutionsCPU[i] = i+2;
		}
	}

	// szukanie minimalnej permutacji w tym, co zrealizowa�o GPU:
	int min_index = 0;
	float min_length = length_of_min_permutation_CPU[0];
	for (int i = 1; i < active_threads; i++) {
		if (length_of_min_permutation_CPU[i] < min_length) {
			min_index = i;
			min_length = length_of_min_permutation_CPU[i];
		}
	}



	/*for (int p = 0; p < solutions_number; p++) {
		printf("%d\t\t", p);
		for (int r = 0; r < n; r++) {
			printf("%d\t", h_solutionsGPU[n * p + r]);
		}
		printf("\t\t");
		for (int r = 0; r < n; r++) {
			printf("%d\t", h_solutionsCPU[n * p + r]);
		}
		printf("\n");
	}*/

	// sprawdzenie czy dane z GPU s� jednokowe jak te z CPU
	bool data_equality = checkValidity(h_solutionsGPU, h_solutionsCPU, solutions_number, n);
	
	// printowanie czas�w oblicze�
	std::printf("Obliczenia dla %d!\n", n);
	std::printf("Liczba wynikow %d\n", solutions_number);
	if (data_equality) std::printf("Obliczenia sa poprawne\n");
	else std::printf("Niepoprawne obliczenia\n");
	std::printf("CPU time:\t%lld us\n", CPU_duration.count());
	std::printf("GPU time:\t%lld us\n", GPU_duration.count()); 
	std::printf("Najkrotsza sciezka:\t %d\t", source);
	for (int i = 0; i < n; i++) {
		std::printf("%d\t", h_solutionsGPU[min_index * n + i]);
	}
	std::printf("%d\n", source);
	std::printf("Najkrotsza sciezka ma indeks permutacji: %d, a jej dlugosc to: %4.2f\n", min_index, min_length);


	//zwolnienie pami�ci w GPU
	cudaFree(first_permutationGPU);
	cudaFree(d_solutionsGPU);
	std::free(h_solutionsCPU);
	std::free(h_solutionsGPU);

	cudaDeviceReset();

	return 0;
}
