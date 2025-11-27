#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define NUM_THREADS 8

void matmul_mono(double** A, double** B, double** C, int n, int m, int p){
    for(int i=0;i<n;i++){
        for(int j=0;j<p;j++){
            C[i][j] = 0;
            for(int k=0;k<m;k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

void matmul_dns(double** A, double** B, double** C, int n, int m, int p_dim){
    int p = (int)cbrt(NUM_THREADS);
    if(p*p*p != NUM_THREADS) p = 1;
    
    if(n != p_dim || p < 2) {
        #pragma omp parallel for collapse(2) num_threads(NUM_THREADS) schedule(static)
        for(int i=0; i<n; i++) {
            for(int j=0; j<p_dim; j++) {
                double sum = 0;
                for(int k=0; k<m; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return;
    }
    
    int n_padded = p * ((n + p - 1) / p);
    int m_padded = p * ((m + p - 1) / p);
    int p_dim_padded = p * ((p_dim + p - 1) / p);
    
    double** A_work = A;
    double** B_work = B;
    int need_padding = (n != n_padded || m != m_padded || p_dim != p_dim_padded);
    
    printf("DNS: n=%d m=%d p_dim=%d, padded: %d %d %d, need_padding=%d\n", 
           n, m, p_dim, n_padded, m_padded, p_dim_padded, need_padding);
    
    if(need_padding) {
        A_work = (double**)malloc(n_padded * sizeof(double*));
        for(int i = 0; i < n_padded; i++) {
            A_work[i] = (double*)calloc(m_padded, sizeof(double));
            if(i < n) {
                for(int j = 0; j < m; j++) {
                    A_work[i][j] = A[i][j];
                }
            }
        }
        
        B_work = (double**)malloc(m_padded * sizeof(double*));
        for(int i = 0; i < m_padded; i++) {
            B_work[i] = (double*)calloc(p_dim_padded, sizeof(double));
            if(i < m) {
                for(int j = 0; j < p_dim; j++) {
                    B_work[i][j] = B[i][j];
                }
            }
        }
    }
    
    int block_size_n = n_padded / p;
    int block_size_m = m_padded / p;
    int p_squared = p * p;
    
    double* A_local = (double*)malloc(NUM_THREADS * block_size_n * block_size_m * sizeof(double));
    double* B_local = (double*)malloc(NUM_THREADS * block_size_m * block_size_n * sizeof(double));
    double* temp_results = (double*)calloc(NUM_THREADS * block_size_n * block_size_n, sizeof(double));

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        int i = tid / p_squared;
        int j = (tid / p) % p;
        int k = tid % p;

        int A_buf_offset = tid * block_size_n * block_size_m;
        int B_buf_offset = tid * block_size_m * block_size_n;
        int result_buf_offset = tid * block_size_n * block_size_n;
        
        int A_row_start = i * block_size_n;
        int A_col_start = k * block_size_m;
        int B_row_start = k * block_size_m;
        int B_col_start = j * block_size_n;
        
        for(int r = 0; r < block_size_n; r++) {
            for(int c = 0; c < block_size_m; c++) {
                A_local[A_buf_offset + r * block_size_m + c] = A_work[A_row_start + r][A_col_start + c];
            }
        }
        
        for(int r = 0; r < block_size_m; r++) {
            for(int c = 0; c < block_size_n; c++) {
                B_local[B_buf_offset + r * block_size_n + c] = B_work[B_row_start + r][B_col_start + c];
            }
        }

        #pragma omp barrier
        
        for(int r = 0; r < block_size_n; r++) {
            for(int kk = 0; kk < block_size_m; kk++) {
                double a_val = A_local[A_buf_offset + r * block_size_m + kk];
                for(int c = 0; c < block_size_n; c++) {
                    temp_results[result_buf_offset + r * block_size_n + c] += 
                        a_val * B_local[B_buf_offset + kk * block_size_n + c];
                }
            }
        }

        #pragma omp barrier
        
        if(k == 0) {
            int C_row_start = i * block_size_n;
            int C_col_start = j * block_size_n;
            int plane_base_tid = i * p_squared + j * p;
            
            for(int r = 0; r < block_size_n; r++) {
                for(int c = 0; c < block_size_n; c++) {
                    double sum = 0;
                    for(int kk = 0; kk < p; kk++) {
                        int thread_id = plane_base_tid + kk;
                        sum += temp_results[thread_id * block_size_n * block_size_n + r * block_size_n + c];
                    }
                    int global_row = C_row_start + r;
                    int global_col = C_col_start + c;
                    if(global_row < n && global_col < p_dim) {
                        C[global_row][global_col] = sum;
                    }
                }
            }
        }
    }
    
    if(need_padding) {
        for(int i = 0; i < n_padded; i++) {
            free(A_work[i]);
        }
        free(A_work);
        
        for(int i = 0; i < m_padded; i++) {
            free(B_work[i]);
        }
        free(B_work);
    }
    
    free(A_local);
    free(B_local);
    free(temp_results);
}

int main() {
    // Test với ma trận nhỏ 5×3
    int n = 5, m = 3, p_dim = 5;
    
    double** A = (double**)malloc(n * sizeof(double*));
    double** B = (double**)malloc(m * sizeof(double*));
    double** C_dns = (double**)malloc(n * sizeof(double*));
    double** C_mono = (double**)malloc(n * sizeof(double*));
    
    for(int i=0; i<n; i++) {
        A[i] = (double*)malloc(m * sizeof(double));
        C_dns[i] = (double*)calloc(p_dim, sizeof(double));
        C_mono[i] = (double*)calloc(p_dim, sizeof(double));
        for(int j=0; j<m; j++) {
            A[i][j] = i * m + j + 1;
        }
    }
    
    for(int i=0; i<m; i++) {
        B[i] = (double*)malloc(p_dim * sizeof(double));
        for(int j=0; j<p_dim; j++) {
            B[i][j] = i * p_dim + j + 1;
        }
    }
    
    printf("A (5×3):\n");
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) printf("%.0f ", A[i][j]);
        printf("\n");
    }
    
    printf("\nB (3×5):\n");
    for(int i=0; i<m; i++) {
        for(int j=0; j<p_dim; j++) printf("%.0f ", B[i][j]);
        printf("\n");
    }
    
    matmul_mono(A, B, C_mono, n, m, p_dim);
    matmul_dns(A, B, C_dns, n, m, p_dim);
    
    printf("\nC_mono (5×5):\n");
    for(int i=0; i<n; i++) {
        for(int j=0; j<p_dim; j++) printf("%.1f ", C_mono[i][j]);
        printf("\n");
    }
    
    printf("\nC_dns (5×5):\n");
    for(int i=0; i<n; i++) {
        for(int j=0; j<p_dim; j++) printf("%.1f ", C_dns[i][j]);
        printf("\n");
    }
    
    printf("\nDifferences:\n");
    double max_diff = 0;
    int errors = 0;
    for(int i=0; i<n; i++) {
        for(int j=0; j<p_dim; j++) {
            double diff = fabs(C_dns[i][j] - C_mono[i][j]);
            if(diff > max_diff) max_diff = diff;
            if(diff > 1e-6) {
                printf("Error at [%d][%d]: dns=%.6f mono=%.6f diff=%.2e\n", 
                       i, j, C_dns[i][j], C_mono[i][j], diff);
                errors++;
            }
        }
    }
    printf("Max diff: %.2e, Errors: %d\n", max_diff, errors);
    
    return 0;
}
