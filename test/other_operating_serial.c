#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../include/other_operating_serial.h"

double** transpose_serial(double** A, int rows, int cols) {
    double** T = (double**)malloc(cols * sizeof(double*));
    for (int i = 0; i < cols; ++i) {
        T[i] = (double*)malloc(rows * sizeof(double));
    }
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

double* argsort_serial(double* A, int n) {
    pair* flat_array = (pair*)malloc(n * sizeof(pair));
    for (int i = 0; i < n; ++i) {
        flat_array[i].index = i;
        flat_array[i].value = A[i];
    }  

    qsort(flat_array, n, sizeof(pair), comparator);
    double* sorted_indices = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        sorted_indices[i] = flat_array[i].index;
    }
    free(flat_array);

    return sorted_indices;
}

double* square_serial(double* A, int n) {
    double* B = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        B[i] = A[i] * A[i];
    }
    return B;
}

/*
int main(int argc, char** argv) {
    // Example usage of the functions
    int n = 5;
    double* A = (double*)malloc(n * sizeof(double));
    printf("Original array A:\n");
    for (int i = 0; i < n; ++i) {
        A[i] = (double)(n - i); // 5, 4, 3, 2, 1
        printf("%.2f ", A[i]);
    }
    printf("\n");

    // Test square_serial
    double* squared_A = square_serial(A, n);
    printf("Squared A:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", squared_A[i]);
    }
    printf("\n");

    // Test argsort
    double* sorted_indices = argsort_serial(A, n);
    printf("Argsort indices:\n");
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

    // Test transpose_serial
    int rows = 2, cols = 3;
    double** Mat = (double**)malloc(rows * sizeof(double*));
    printf("Original Matrix (%dx%d):\n", rows, cols);
    int count = 0;
    for (int i = 0; i < rows; ++i) {
        Mat[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; ++j) {
            Mat[i][j] = (double)(++count);
            printf("%.2f ", Mat[i][j]);
        }
        printf("\n");
    }

    double** T = transpose_serial(Mat, rows, cols);
    printf("Transposed Matrix (%dx%d):\n", cols, rows);
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            printf("%.2f ", T[i][j]);
        }
        printf("\n");
    }

    // Cleanup
    free(A);
    free(squared_A);
    free(sorted_indices);
    for(int i=0; i<rows; ++i) free(Mat[i]);
    free(Mat);
    for(int i=0; i<cols; ++i) free(T[i]);
    free(T);

    return 0;
}
*/