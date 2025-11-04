#include <stdio.h>
#include <stdlib.h>
#include "../include/n2array.h"
#include "../include/numc.h"

/* forward declarations for stream.c functions (defined in test/stream.c) */
void stream_matrix_reader(const char* filename, double*** data, int* rows, int* cols);
void print_matrix(double** data, int rows, int cols);
void free_matrix(double** data, int rows);

N2Array* normalize(N2Array* A) {
    if (!A) return NULL;
    // Compute mean and stdev (returns N2Array* with shape [1,1])
    N2Array* mean_arr = mean(A, -1);
    N2Array* std_arr = stdev(A, -1);

    N2Array_free(mean_arr);
    N2Array_free(std_arr);
    
    // Normalize: (A - mean) / std
    N2Array* A_c = N2Array_sub(A, mean_arr);
    N2Array* A_norm = N2Array_div(A_c, std_arr);
    N2Array_free(A_c);
    
    return A_norm;
}

void compute(double** data, int rows, int cols) {
    // Example computation: compute mean and stdev, then normalize
    N2Array* X = N2Array_from_2d(data, (int[]){rows, cols});
    N2Array* X_norm = normalize(X);
    N2Array* z = ones(10, 1);
    N2Array_free(X);
    N2Array* d = diag(z);
    // Print normalized result
    if (X_norm && X_norm->n2array) {
        // printf("Normalized matrix:\n");
        // print_matrix(X_norm->n2array, X_norm->shape[0], X_norm->shape[1]);
        print_matrix(d->n2array, d->shape[0], d->shape[1]);
    }
    
    N2Array_free(X_norm);
    N2Array_free(z);
    N2Array_free(d);
}


int main() {
    const char* filename = "./data/matrix.txt";
    double** data = NULL;
    int rows = 0, cols = 0;

    // Đọc ma trận từ file
    stream_matrix_reader(filename, &data, &rows, &cols);
    // print_matrix(data, rows, cols);
    compute(data, rows, cols);
    free_matrix(data, rows);
    return 0;
}