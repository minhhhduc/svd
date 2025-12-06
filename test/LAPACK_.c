#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../include/stream.h"

// LAPACK SVD prototype
// dgesvd: Computes the singular value decomposition (SVD) of a real M-by-N matrix A
// A = U * SIGMA * V^T
// Singular values are the square roots of eigenvalues of A^T*A
extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda,
                    double *s, double *u, int *ldu, double *vt, int *ldvt,
                    double *work, int *lwork, int *info,
                    size_t len_jobu, size_t len_jobvt);

int main() {
    int num_files = 21;
    char filename[256];
    const char* output_filename = "data/output/lapack_output.txt";
    
    // Try to open output file
    FILE* fout = fopen(output_filename, "w");
    if (!fout) {
        fout = fopen("../data/output/lapack_output.txt", "w");
    }
    
    if (fout) {
        fprintf(fout, "rows,cols,svd_time,singular_values\n");
    } else {
        printf("Warning: Could not open output file for writing.\n");
    }

    printf("=== LAPACK SVD Test (dgesvd) ===\n");
    printf("Singular values of A (NOT eigenvalues of A^T*A)\n");
    printf("Note: sigma_i = sqrt(lambda_i) where lambda_i are eigenvalues of A^T*A\n\n");

    for (int f_idx = 0; f_idx < num_files; f_idx++) {
        sprintf(filename, "data/input/matrix_%d.txt", f_idx);
        
        double** A_rows = NULL;
        int rows, cols;
        
        // Check if file exists
        FILE* f = fopen(filename, "r");
        if (!f) {
            char temp[256];
            sprintf(temp, "../%s", filename);
            f = fopen(temp, "r");
            if (!f) {
                printf("Skipping %s (not found)\n", filename);
                continue;
            }
            strcpy(filename, temp);
        }
        fclose(f);

        printf("\nProcessing %s...\n", filename);
        stream_matrix_reader(filename, &A_rows, &rows, &cols);
        printf("Dimensions: %d x %d\n", rows, cols);

        // Convert A to Column-Major for LAPACK
        // A_rows is [rows][cols].
        // We want a flat array A_col where A_col[i + j*rows] = A_rows[i][j]
        double *A_col = (double*)malloc(rows * cols * sizeof(double));
        if (!A_col) {
            printf("Memory allocation failed for A_col\n");
            free_matrix(A_rows, rows);
            continue;
        }

        for(int j = 0; j < cols; j++) {
            for(int i = 0; i < rows; i++) {
                A_col[j * rows + i] = A_rows[i][j];
            }
        }

        // SVD: A = U * S * V^T
        // We only need singular values (S), not U or V^T
        char jobu = 'N';   // Don't compute U
        char jobvt = 'N';  // Don't compute V^T
        int m = rows;
        int n = cols;
        int lda = rows;
        int min_mn = (m < n) ? m : n;
        
        // Allocate singular values array
        double *s = (double*)malloc(min_mn * sizeof(double));
        if (!s) {
            printf("Memory allocation failed for singular values\n");
            free(A_col);
            free_matrix(A_rows, rows);
            continue;
        }
        
        // Dummy arrays for U and VT (not computed)
        double u_dummy, vt_dummy;
        int ldu = 1, ldvt = 1;
        
        // Query optimal workspace size
        int lwork = -1;
        double wkopt;
        int info;
        
        dgesvd_(&jobu, &jobvt, &m, &n, A_col, &lda, s, &u_dummy, &ldu, &vt_dummy, &ldvt,
                &wkopt, &lwork, &info, 1, 1);
        
        lwork = (int)wkopt;
        double *work = (double*)malloc(lwork * sizeof(double));
        if (!work) {
            printf("Memory allocation failed for work array\n");
            free(s);
            free(A_col);
            free_matrix(A_rows, rows);
            continue;
        }
        
        clock_t start = clock();
        
        // Compute SVD - only singular values
        dgesvd_(&jobu, &jobvt, &m, &n, A_col, &lda, s, &u_dummy, &ldu, &vt_dummy, &ldvt,
                work, &lwork, &info, 1, 1);
        
        clock_t end = clock();
        double svd_time = (double)(end - start) / CLOCKS_PER_SEC;

        if (info != 0) {
            printf("Error in dgesvd: %d\n", info);
        } else {
            printf("SVD Time: %.4f s\n", svd_time);
            printf("Singular values (first 5, descending): ");
            // dgesvd returns singular values in descending order
            for(int k = 0; k < 5 && k < min_mn; k++) {
                printf("%e ", s[k]); 
            }
            printf("\n");
            
            // Also show corresponding eigenvalues of A^T*A for comparison
            printf("Eigenvalues of A^T*A (sigma^2): ");
            for(int k = 0; k < 5 && k < min_mn; k++) {
                printf("%e ", s[k] * s[k]); 
            }
            printf("\n");
        }

        if (fout) {
            fprintf(fout, "%d,%d,%.6f", rows, cols, svd_time);
            // Write first 5 singular values
            for(int k = 0; k < 5 && k < min_mn; k++) {
                fprintf(fout, ",%.6e", s[k]);
            }
            fprintf(fout, "\n");
            fflush(fout);
        }

        free(A_col);
        free(s);
        free(work);
        free_matrix(A_rows, rows);
    }

    if (fout) fclose(fout);
    return 0;
}
