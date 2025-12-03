#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "../include/other_operating_parallel.h"

double** transpose_parallel(double** A, int rows, int cols, int num_threads) {
    double** T = (double**)malloc(cols * sizeof(double*));

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < cols; ++i) {
        T[i] = (double*)malloc(rows * sizeof(double));
    }
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T[j][i] = A[i][j];
        }
    }

    return T;
}

typedef struct{
    int index;
    double value;
} pair;

static int comparator(const void* a, const void* b) {
    double diff = ((pair*)a)->value - ((pair*)b)->value;
    if (diff < 0) return -1;
    else if (diff > 0) return 1;
    else return 0;
}

// Helper function to merge two sorted subarrays
void merge_arrays(pair* arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    pair* L = (pair*)malloc(n1 * sizeof(pair));
    pair* R = (pair*)malloc(n2 * sizeof(pair));

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0; 
    j = 0; 
    k = l; 
    while (i < n1 && j < n2) {
        if (comparator(&L[i], &R[j]) <= 0) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

double* argsort_parallel(double* A, int n, int num_threads) {
    pair* flat_array = (pair*)malloc(n * sizeof(pair));

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; ++i) {
        flat_array[i].index = i;
        flat_array[i].value = A[i];
    }  

    // 1. Local Sort
    int* starts = (int*)malloc((num_threads + 1) * sizeof(int));
    
    #pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num();
        int num_t = omp_get_num_threads();
        if (id < num_threads) {
            int remainder = n % num_threads;
            int chunk = n / num_threads;
            
            int start = id * chunk + (id < remainder ? id : remainder);
            int end = start + chunk + (id < remainder ? 1 : 0);
            
            starts[id] = start;
            
            if (start < end) {
                qsort(flat_array + start, end - start, sizeof(pair), comparator);
            }
        }
    }
    starts[num_threads] = n;

    // 2. Merge Sorted Chunks (Pairwise)
    int step = 1;
    while (step < num_threads) {
        #pragma omp parallel for num_threads(num_threads) 
        for (int i = 0; i < num_threads; i += 2 * step) {
            if (i + step < num_threads) {
                int l = starts[i];
                int m = starts[i + step] - 1;
                
                int r_idx = i + 2 * step;
                if (r_idx > num_threads) r_idx = num_threads;
                int r = starts[r_idx] - 1;
                
                merge_arrays(flat_array, l, m, r);
            }
        }
        step *= 2;
    }

    double* sorted_indices = (double*)malloc(n * sizeof(double));
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; ++i) {
        sorted_indices[i] = (double)flat_array[i].index;
    }
    
    free(starts);
    free(flat_array);

    return sorted_indices;
}


double* square_parallel(double* A, int n, int num_threads) {
    double* B = (double*)malloc(n * sizeof(double));
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; ++i) {
        B[i] = A[i] * A[i];
    }
    return B;
}

/*
int main(int argc, char** argv) {
    // Example usage of the functions
    int n = 5;
    int num_threads = 4;
    double* A = (double*)malloc(n * sizeof(double));
    printf("Original array A:\n");
    for (int i = 0; i < n; ++i) {
        A[i] = (double)(n - i); // 5, 4, 3, 2, 1
        printf("%.2f ", A[i]);
    }
    printf("\n");

    // Test square
    double* squared_A = square_parallel(A, n, num_threads);
    printf("Squared A:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", squared_A[i]);
    }
    printf("\n");

    // Test argsort
    double* sorted_indices = argsort_parallel(A, n, num_threads);
    printf("Argsorted indices of A:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.0f ", sorted_indices[i]);
    }
    printf("\n");
    
    // Verify sort
    printf("Values at sorted indices:\n");
    for (int i = 0; i < n; ++i) {
        int idx = (int)sorted_indices[i];
        printf("%.2f ", A[idx]);
    }
    printf("\n");

    free(A);
    free(squared_A);
    free(sorted_indices);
    return 0;
}
*/
