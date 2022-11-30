#include <iostream>

using namespace std;




long long factorial(int n) {
    int resoult = 1;
    while (n) resoult *= n--;
    return resoult;
}



void swap(int *a, int *b) {
    int *temp = b;
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
    int max_index = n-1;
    for (int i = n-2; i >= 0; i--) {
        if (current_permutation[i] < current_permutation[max_index]) {
            int min_index = max_index;
            for (int j = i+2; j <= n-1; j++) {
                if (current_permutation[j] < current_permutation[min_index] and current_permutation[j] > current_permutation[i]) {
                    min_index = j;
                }
            }
            swap(current_permutation[i], current_permutation[min_index]);
            if (i < n-2) {
               quickSort(current_permutation, i+1, n-1);
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



int main()
{
    const int n = 5;
    int solutions_number = factorial(n) - 1;
   
    int first_permutation[] = {5, 4, 3, 2, 1};
    for (int i = 0; i < n; i++) {
        first_permutation[i] = i + 1;
        printf("%d\t", first_permutation[i]); 
    }
    printf("\n");
    
    for (int o = 0; o < solutions_number; o++) {
        next_permutation(n, first_permutation);    
    }
    
    
    
    return 0;
}
