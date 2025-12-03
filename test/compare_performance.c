#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "../include/other_operating_serial.h"
#include "../include/other_operating_parallel.h"

// ================= MAIN =================
int main() {
    int start_n = 1000;
    int end_n = 10000001;
    int step_n = 100000; 
    
    int max_threads = omp_get_max_threads();

    // Open files
    FILE *fp_sq = fopen("data/output/square.csv", "w");
    FILE *fp_arg = fopen("data/output/argsort.csv", "w");
    FILE *fp_tr = fopen("data/output/transpose.csv", "w");
    
    if (!fp_sq || !fp_arg || !fp_tr) {
        fprintf(stderr, "Error opening output files.\n");
        return 1;
    }

    // Write Headers: size, 1, 2, ..., max_threads
    fprintf(fp_sq, "size"); fprintf(fp_arg, "size"); fprintf(fp_tr, "size");
    for(int t = 1; t <= max_threads; ++t) {
        fprintf(fp_sq, ",%d", t); fprintf(fp_arg, ",%d", t); fprintf(fp_tr, ",%d", t);
    }
    fprintf(fp_sq, "\n"); fprintf(fp_arg, "\n"); fprintf(fp_tr, "\n");
    
    printf("Benchmarking Parallel Functions...\n");
    printf("Range: [%d, %d], Step: %d\n", start_n, end_n, step_n);
    printf("Max Threads: %d\n", max_threads);

    for(int n = start_n; n < end_n; n += step_n) {
        int rows = (int)sqrt(n);
        int cols = rows;
        if (rows < 1) { rows = 1; cols = 1; }

        for (int op = 0; op < 10; op++) {
            // Alloc Data
        double* A = (double*)malloc(n * sizeof(double));
        for(int i=0; i<n; ++i) A[i] = (double)rand() / RAND_MAX;
        
        double** Mat = (double**)malloc(rows * sizeof(double*));
        for(int i=0; i<rows; ++i) {
            Mat[i] = (double*)malloc(cols * sizeof(double));
            for(int j=0; j<cols; ++j) Mat[i][j] = (double)rand() / RAND_MAX;
        }
        
        // Write Size
        fprintf(fp_sq, "%d", n); fprintf(fp_arg, "%d", n); fprintf(fp_tr, "%d", n);

        // Iterate Threads
        for(int t = 1; t <= max_threads; ++t) {
            double start, end;

            if (t == 1) {
                // Use Serial Functions for 1 thread
                // Square
                start = omp_get_wtime();
                double* sq = square_serial(A, n);
                end = omp_get_wtime();
                fprintf(fp_sq, ",%.6f", end - start);
                free(sq);

                // Argsort
                start = omp_get_wtime();
                double* arg = argsort_serial(A, n);
                end = omp_get_wtime();
                fprintf(fp_arg, ",%.6f", end - start);
                free(arg);

                // Transpose
                start = omp_get_wtime();
                double** tr = transpose_serial(Mat, rows, cols);
                end = omp_get_wtime();
                fprintf(fp_tr, ",%.6f", end - start);
                for(int i=0; i<cols; ++i) free(tr[i]);
                free(tr);
            } else {
                // Use Parallel Functions for > 1 threads
                // Square
                start = omp_get_wtime();
                double* sq = square_parallel(A, n, t);
                end = omp_get_wtime();
                fprintf(fp_sq, ",%.6f", end - start);
                free(sq);

                // Argsort
                start = omp_get_wtime();
                double* arg = argsort_parallel(A, n, t);
                end = omp_get_wtime();
                fprintf(fp_arg, ",%.6f", end - start);
                free(arg);

                // Transpose
                start = omp_get_wtime();
                double** tr = transpose_parallel(Mat, rows, cols, t);
                end = omp_get_wtime();
                fprintf(fp_tr, ",%.6f", end - start);
                for(int i=0; i<cols; ++i) free(tr[i]);
                free(tr);
            }
        }

        // Newline
        fprintf(fp_sq, "\n"); fprintf(fp_arg, "\n"); fprintf(fp_tr, "\n");
        
        if (n % 1000000 < step_n) { 
            printf("Processed N=%d\n", n);
        }
        
        // Cleanup
        free(A);
        for(int i=0; i<rows; ++i) free(Mat[i]);
        free(Mat);
        }
    }
    
    fclose(fp_sq); fclose(fp_arg); fclose(fp_tr);
    
    printf("Done. Results saved to data/output/\n");
    
    return 0;
}