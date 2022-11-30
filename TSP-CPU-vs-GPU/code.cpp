#include <iostream>
#include "permutate.cuh"


using namespace std;


long long factorial(int n) {
    int resoult = 1;
    while (n) resoult *= n--;
    return resoult;
}



void swap(int* a, int* b) {
    int* temp = b;
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

    for (int i = 0; i < n; i++) printf("%d\t", current_permutation[i]);
    printf("\n");
}


void find_ith_permutation(int arr[], int n, int i) {

    // create array with known size equal to n
    int* factoradic = new int(n);

    // factorial decomposition with modulo function
    int rest = i;
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
        permutation_arr[j] = arr[factoradic[j]];

        // instead of creating new array I am moving all elements which will
        // still be in my factoradic to the left and I am decreasing size of
        // this array to assure that I am only using proper part of it
        for (int k = 0; k < (_n - factoradic[j]); k++) {
            swap(arr[factoradic[j] + k], arr[factoradic[j] + k + 1]);
        }
        _n--;
    }
    // just simple print of permutation
    for (int o = 0; o < n; o++) {
        printf("%d\t", permutation_arr[o]);
    }
    printf("\n");

    return;
}



int main()
{
    const int n = 5;
    int solutions_number = factorial(n) - 1;

    int first_permutation[] = { 5, 4, 3, 2, 1 };
    for (int i = 0; i < n; i++) {
        first_permutation[i] = i + 1;
        printf("%d\t", first_permutation[i]);
    }
    printf("\n");

    // for (int o = 0; o < solutions_number; o++) {
    //     next_permutation(n, first_permutation);    
    // }
    int excel_row = 98;
    find_ith_permutation(first_permutation, n, excel_row - 2);



    return 0;
}
