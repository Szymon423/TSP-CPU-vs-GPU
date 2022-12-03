#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include "permutate.cuh"

#pragma comment(lib, "winmm.lib")


using namespace std;


// definiowanie makra do obs³ugi b³êdów
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    // sprawdzenie czy dzia³anie funkcji zakoñczy³o siê b³êdem
    if (code != cudaSuccess) {
        // printowanie b³êdu, pliku w którym wyst¹pi³ oraz linii kodu
        fprintf(stderr, "GPUassert : %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            // jeœli przekazujemy argument true to exitujemy program
            exit(code);
        }
    }
}



long long factorial(int n) {
    int resoult = 1;
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



void next_permutation(int n, int current_permutation[]) {
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

    /*for (int i = 0; i < n; i++) printf("%d\t", current_permutation[i]);
    printf("\n");*/
}


__global__ void next_permutationGPU(int n, int current_permutation[]) {
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


void find_ith_permutation(int arr[], int n, int index, int *sol) {

    // stworzenie tablicy, na której bêd¹ przekszta³cenia
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

    free(factoradic);
    free(permutation_arr);
    free(_arr);

    return;
}




__global__ void find_ith_permutationGPU(int *sol, int *arr, int n, int sol_num) {

    // calculating id for each thread
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int row_offset = blockDim.x * gridDim.x * blockIdx.y;
    int gid = row_offset + block_offset + tid;

    if (gid > sol_num) {
        return;
    }

    // stworzenie tablicy, na której bêd¹ przekszta³cenia
    int* _arr = new int(n);
    for (int i = 0; i < n; i++) {
        _arr[i] = arr[i];
    }
       
    // create array with known size equal to n
    int* factoradic = new int(n);

    // factorial decomposition with modulo function
    int rest = gid;
    for (int j = 1; j <= n; j++) {
        factoradic[n - j] = rest % j;
        rest /= j;
    }

    // array to contain target permutation
    int* permutation_arr = new int(n);
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
        sol[gid * n + o] = permutation_arr[o];
    }

    return;
}



int main() {

    int n = 5;
    int solutions_number = factorial(n);

    int* first_permutation;

    // obliczenie rozmiaru w bajtach tablicy pojedynczej permutacji
    int size_in_bytes = n * sizeof(int);
   
    first_permutation = (int*)malloc(size_in_bytes);

    // stworzenie wskaŸnika na tablicê z permutacj¹ pocz¹tkow¹
    int* first_permutationGPU;

    // stworzenie wskaŸnika na tablicê rozwi¹zañ z GPU znajduj¹c¹ siê w RAMie CPU
    int* d_solutionsGPU;

    // wskaŸnik na rozwi¹zania z GPU w VRAMie
    int* h_solutionsGPU;

    // wskaŸnik na rozwi¹zania z CPU w RAMie
    int* h_solutionsCPU;

    // obliczenie rozmiaru w bajtach tablicy rozwi¹zañ -> ka¿de rozwi¹zanie to n intów wiêc n * sizeof(int) * iloœæ rozwi¹zañ
    int size_in_bytes_of_solutions = n * solutions_number * sizeof(int);
    h_solutionsGPU = (int*)malloc(size_in_bytes_of_solutions);
    h_solutionsCPU = (int*)malloc(size_in_bytes_of_solutions);

    // filling array with values of 0-th permutation
    for (int i = 0; i < n; i++) {
        first_permutation[i] = i + 1;
    }

    auto CPU_start = chrono::high_resolution_clock::now();

    // for (int o = 0; o < solutions_number; o++) {
    //     next_permutation(n, first_permutation);
    // }
    for (int o = 0; o < solutions_number; o++) {
        find_ith_permutation(first_permutation, n, o+1, h_solutionsCPU);
    }

    auto CPU_finish = chrono::high_resolution_clock::now();
    auto CPU_duration = chrono::duration_cast<chrono::microseconds>(CPU_finish - CPU_start);

    /*int excel_row = 98;
    find_ith_permutation(first_permutation, n, excel_row - 2)*/;

    // ponowne uzupe³nienie first_permutation pierwotnym ci¹giem
    for (int i = 0; i < n; i++) {
        first_permutation[i] = i + 1;
    }

    // alokacja pamiêci na GPU oraz na CPU
    gpuErrorCheck(cudaMalloc((void**)&first_permutationGPU, size_in_bytes));
    gpuErrorCheck(cudaMalloc((void**)&d_solutionsGPU, size_in_bytes_of_solutions));

    // kopiowanie pamiêci z CPU do GPU
    cudaMemcpy(first_permutationGPU, first_permutation, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_solutionsGPU, h_solutionsGPU, size_in_bytes_of_solutions, cudaMemcpyHostToDevice);

    // dla 10! :
    dim3 block(10);
    dim3 grid(10, 10);

    // timer start
    auto GPU_start = chrono::high_resolution_clock::now();

    // calling kernel
    find_ith_permutationGPU <<< grid, block >>> (d_solutionsGPU, first_permutationGPU, n, solutions_number);
    
    // wait till device sunc
    cudaDeviceSynchronize();

    auto GPU_finish = chrono::high_resolution_clock::now();
    auto GPU_duration = chrono::duration_cast<chrono::microseconds>(GPU_finish - GPU_start);

    // kopiowanie obliczonych danych spowrotem do CPU
    cudaMemcpy(h_solutionsGPU, d_solutionsGPU, size_in_bytes_of_solutions, cudaMemcpyDeviceToHost);

    // filling up solutions with first permutation
    for (int i = 1; i <= n; i++) {
        h_solutionsGPU[i-1] = i;
        h_solutionsCPU[i-1] = i;
    }

    for (int p = 0; p < solutions_number; p++) {
        for (int r = 0; r < n; r++) {
            printf("%d\t", h_solutionsGPU[n*p + r]);
        }
        printf("\t\t");
        for (int r = 0; r < n; r++) {
            printf("%d\t", h_solutionsCPU[n * p + r]);
        }
        printf("\n");
    }
    // printowanie czasów obliczeñ
    printf("Obliczenia dla %d!\n", n);
    printf("CPU time:\t%lld us\n", CPU_duration.count());
    printf("GPU time:\t%lld us\n", GPU_duration.count()); 

    
    // zwolnienie pamiêci w GPU
    cudaFree(first_permutationGPU);
    cudaFree(d_solutionsGPU);
    free(h_solutionsCPU);
    free(h_solutionsGPU);

    cudaDeviceReset();

    return 0;
}
