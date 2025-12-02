#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mulmat.c"

// Định nghĩa các hằng số
#define TOLERANCE 1e-15  // Tighter tolerance to converge to near 0
#define MAX_SWEEPS 500   // More sweeps to allow full convergence
#define NUM_THREADS 8  // Số threads OpenMP (8 = 2³ cho DNS 3D cube)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Macro để truy cập mảng 1 chiều như ma trận 2 chiều
#define AT(arr, r, c, n) (arr)[(r)*(n) + (c)]

// Hàm phụ trợ: Tính bình phương độ lớn x^2
double abs_sq(double x) {
    return x * x;
}

// Hàm tìm cặp (p,q) có phần tử a_pq lớn nhất (norm-reducing strategy)
// Thread-safe version using critical section for index updates
void find_largest_off_diag(int n, double* A, int* p_out, int* q_out) {
    double max_val = 0.0;
    int p = 0, q = 1;
    
    #pragma omp parallel
    {
        double local_max = 0.0;
        int local_p = 0, local_q = 1;
        
        // Each thread searches its portion
        #pragma omp for collapse(2) nowait
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                double val = fabs(AT(A, i, j, n));
                if (val > local_max) {
                    local_max = val;
                    local_p = i;
                    local_q = j;
                }
            }
        }
        
        // Only one thread updates the global maximum at a time
        #pragma omp critical
        {
            if (local_max > max_val) {
                max_val = local_max;
                p = local_p;
                q = local_q;
            }
        }
    }
    
    *p_out = p;
    *q_out = q;
}

// Hàm thực hiện phép quay Jacobi cho ma trận đối xứng thực (Biến thể)
// Áp dụng công thức:
// tau = (App - Aqq) / 2Apq
// t = sgn(tau) / (|tau| + sqrt(1+tau^2))
// c = 1/sqrt(1+t^2), s = t*c
void apply_symmetric_jacobi(int n, double* A, double* V, int p, int q) {
    // Ma trận đối xứng thực
    double a_pp = AT(A, p, p, n);
    double a_qq = AT(A, q, q, n);
    double a_pq = AT(A, p, q, n);

    // Nếu phần tử đã nhỏ, bỏ qua
    if (fabs(a_pq) < TOLERANCE) return;

    // Tính tham số quay tau
    double tau = (a_qq - a_pp) / (2.0 * a_pq);
    double t;
    if (tau >= 0) {
        t = 1.0 / (tau + sqrt(1.0 + tau * tau));
    } else {
        t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
    }
    
    double c = 1.0 / sqrt(1.0 + t * t);
    double s = t * c;

    // Cập nhật phần tử đường chéo
    AT(A, p, p, n) = c*c*a_pp + s*s*a_qq - 2.0*c*s*a_pq;
    AT(A, q, q, n) = s*s*a_pp + c*c*a_qq + 2.0*c*s*a_pq;
    AT(A, p, q, n) = 0.0;
    AT(A, q, p, n) = 0.0;

    // Cập nhật các phần tử ngoài đường chéo
    // Sử dụng OpenMP để song song hóa việc cập nhật hàng/cột
    #pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        if (j != p && j != q) {
            double a_pj = AT(A, p, j, n);
            double a_qj = AT(A, q, j, n);
            
            double new_a_pj = c * a_pj - s * a_qj;
            double new_a_qj = s * a_pj + c * a_qj;
            
            AT(A, p, j, n) = new_a_pj;
            AT(A, q, j, n) = new_a_qj;
            
            // Đối xứng
            AT(A, j, p, n) = new_a_pj;
            AT(A, j, q, n) = new_a_qj;
        }
    }

    // Cập nhật ma trận vector riêng V
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double v_ip = AT(V, i, p, n);
        double v_iq = AT(V, i, q, n);
        
        AT(V, i, p, n) = c * v_ip - s * v_iq;
        AT(V, i, q, n) = s * v_ip + c * v_iq;
    }
}

// ----------------------------------------------------------------------
// Parallel Ordering: Modulus Pivot Selection
// ----------------------------------------------------------------------
typedef struct {
    int p;
    int q;
} IndexPair;

int generate_parallel_pairs(int n, int step, IndexPair* pairs, int max_pairs) {
    int count = 0;
    
    if (n % 2 == 0) {
        for (int i = 0; i < n / 2; i++) {
            int p = i;
            int q = (n - 1 - i + step) % n;
            if (p == q) continue;
            if (p > q) { int tmp = p; p = q; q = tmp; }
            if (count < max_pairs) {
                pairs[count].p = p;
                pairs[count].q = q;
                count++;
            }
        }
    } else {
        int skip = step % n;
        for (int i = 0; i < n; i++) {
            if (i == skip) continue;
            int j = (2 * skip - i + n) % n;
            if (j == skip || j <= i) continue;
            if (count < max_pairs) {
                pairs[count].p = i;
                pairs[count].q = j;
                count++;
            }
        }
    }
    return count;
}

// ----------------------------------------------------------------------
// Hàm chính: Eigen Decomposition using Symmetric Jacobi Variant
// ----------------------------------------------------------------------
void eig(int n, double* A_in, double* w, double* V_out) {
    double* A = (double*)malloc(n * n * sizeof(double));
    double* V = (double*)malloc(n * n * sizeof(double));

    if (!A || !V) {
        if (A) free(A);
        if (V) free(V);
        return;
    }

    // Copy A và khởi tạo V = I
    for (int i = 0; i < n * n; ++i) {
        A[i] = A_in[i];
        int r = i / n;
        int c = i % n;
        V[i] = (r == c) ? 1.0 : 0.0;
    }

    int max_pairs = n / 2 + 2;
    IndexPair* pairs = (IndexPair*)malloc(sizeof(IndexPair) * max_pairs);

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        // Tính Frobenius norm của phần tử ngoài đường chéo: ||A_off||_F = sqrt(sum(|a_ij|^2, i!=j))
        double off_diag_norm_sq = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:off_diag_norm_sq)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    off_diag_norm_sq += abs_sq(AT(A, i, j, n));
                }
            }
        }
        
        double off_diag_norm = sqrt(off_diag_norm_sq);
        
        printf("  [eig] Sweep %3d: ||A_off||_F = %.6e\n", sweep, off_diag_norm);
        
        if (off_diag_norm < TOLERANCE) {
            printf("  [eig] CONVERGED at sweep %d (||A_off||_F = %.6e < %.6e)\n", sweep, off_diag_norm, TOLERANCE);
            break; 
        }

        // Norm-reducing strategy: áp dụng rotations cho tất cả cặp (p,q) với |a_pq| > threshold
        // Để giảm norm xuống 0, ta cần áp dụng nhiều rotations mỗi sweep
        double threshold = off_diag_norm / (n * n);  // Tính threshold động
        
        for (int iter = 0; iter < n; iter++) {
            int p, q;
            find_largest_off_diag(n, A, &p, &q);
            
            // Nếu phần tử lớn nhất nhỏ hơn threshold, dừng lại
            if (fabs(AT(A, p, q, n)) < threshold) break;
            
            apply_symmetric_jacobi(n, A, V, p, q);
        }
    }
    
    free(pairs);

    // Trích xuất kết quả
    // Eigenvalues nằm trên đường chéo của A
    for (int i = 0; i < n; ++i) {
        w[i] = AT(A, i, i, n);
    }

    // V_out lưu dưới dạng row-major (C convention)
    // Để lấy eigenvector thứ j (cột j), dùng: v[i] = V_out[i*n + j]
    // Công thức: eigenvector j = [V_out[0*n+j], V_out[1*n+j], ..., V_out[(n-1)*n+j]]
    for (int i = 0; i < n * n; ++i) {
        V_out[i] = V[i];
    }

    free(A);
    free(V);
}

void calculate_time_inparallel(const char** input_filename, const char *output_filename, int quantity_files) {
    FILE *fout = fopen(output_filename, "w");
    FILE *flog = fopen("./data/output/jacobi_test_log.txt", "w");
    if (fout == NULL) {
        printf("Cannot open output file %s\n", output_filename);
        return;
    }
    if (flog == NULL) {
        printf("Cannot open log file\n");
        flog = stdout;
    }
    
    fprintf(fout, "matrix_size,computation_time_seconds\n");
    fprintf(flog, "=== NORM REDUCING JACOBI EIGENVALUE TEST LOG ===\n");
    fprintf(flog, "Total matrices to process: %d\n\n", quantity_files);
    
    for (int file_idx = 2; file_idx < quantity_files - 18; file_idx++) {
        fprintf(flog, "-------------------------------------------\n");
        fprintf(flog, "TEST %d\n", file_idx);
        fprintf(flog, "-------------------------------------------\n");
        fprintf(flog, "Input file: %s\n", input_filename[file_idx]);
        fflush(flog);
        
        double** A_matrix;
        int rows, cols;
        stream_matrix_reader(input_filename[file_idx], &A_matrix, &rows, &cols);
        fprintf(flog, "Matrix dimensions: %d rows x %d cols\n", rows, cols);
        fflush(flog);
        
        // Compute B = A^T (transpose)
        double** B = (double**)malloc(cols * sizeof(double*));
        for (int i = 0; i < cols; i++) {
            B[i] = (double*)malloc(rows * sizeof(double));
            for (int j = 0; j < rows; j++) {
                B[i][j] = A_matrix[j][i];  // Transpose
            }
        }
        fprintf(flog, "Transposed matrix B: %d rows x %d cols (A^T)\n", cols, rows);
        
        double** C = (double**)malloc(cols * sizeof(double*));
        for (int i = 0; i < cols; i++) {
            C[i] = (double*)calloc(cols, sizeof(double));
        }
        fprintf(flog, "Allocated Gram matrix C: %d x %d\n", cols, cols);
        
        printf("TEST %d: Processing matrix %d: %dx%d -> computing Gram matrix...\n", file_idx, file_idx, rows, cols);
        
        // Matrix multiplication: C = B * A = A^T * A using simple parallel loop
        double time_mul_start = omp_get_wtime();
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < cols; i++) {
            for(int j = 0; j < cols; j++) {
                double sum = 0.0;
                for(int k = 0; k < rows; k++) {
                    sum += B[i][k] * A_matrix[k][j];
                }
                C[i][j] = sum;
            }
        }
        double time_mul_end = omp_get_wtime();
        fprintf(flog, "Gram matrix computed: C = A^T * A (size %dx%d)\n", cols, cols);
        fprintf(flog, "Matrix multiplication time: %.6f seconds\n", time_mul_end - time_mul_start);
        fflush(flog);
        
        int n = cols;
        
        // Convert C to flat array
        double* C_flat = (double*)malloc(n * n * sizeof(double));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C_flat[i * n + j] = C[i][j];
            }
        }
        fprintf(flog, "Converted to flat array\n");
        
        double* w = (double*)malloc(n * sizeof(double));
        double* V = (double*)malloc(n * n * sizeof(double));
        
        fprintf(flog, "Starting eigenvalue computation...\n");
        fflush(flog);
        double start_time = omp_get_wtime();
        eig(n, C_flat, w, V);
        double end_time = omp_get_wtime();
        
        double computation_time = end_time - start_time;
        
        fprintf(flog, "Eigenvalue computation completed!\n");
        fprintf(flog, "Computation time: %.6f seconds\n", computation_time);
        fprintf(flog, "Eigenvalues (first 5):\n");
        for (int i = 0; i < n && i < 5; i++) {
            fprintf(flog, "  λ[%d] = %.10e\n", i, w[i]);
        }
        if (n > 5) fprintf(flog, "  ... (and %d more)\n", n - 5);
        fprintf(flog, "Eigenvector matrix V dimensions: %d x %d\n", n, n);
        fprintf(flog, "TEST %d COMPLETED SUCCESSFULLY\n\n", file_idx);
        fflush(flog);
        
        printf("TEST %d: Completed in %.6f seconds\n", file_idx, computation_time);
        fprintf(fout, "%d,%.6f\n", n, computation_time);
        fflush(fout);
        
        // Cleanup
        for (int i = 0; i < rows; i++) {
            free(A_matrix[i]);
        }
        free(A_matrix);
        
        for (int i = 0; i < cols; i++) {
            free(B[i]);
            free(C[i]);
        }
        free(B);
        free(C);
        free(C_flat);
        free(w);
        free(V);
    }
    fprintf(flog, "=== ALL TESTS COMPLETED ===\n");
    fflush(flog);
    fclose(fout);
    if (flog != stdout) fclose(flog);
    printf("\nLog file written to: ./data/output/jacobi_test_log.txt\n");
}

// ----------------------------------------------------------------------
// Ví dụ sử dụng (Main)
// ----------------------------------------------------------------------
int main() {
    const char *path_input = "./data/input";
    char *input_filename[21];
    const char output_filename[] = "./data/output/norm_reducing_jacobi_output.txt";
    for (int i = 0; i < 21; i++) {
        input_filename[i] = (char*)malloc(150 * sizeof(char));
    }
    for (int i = 0; i < 21; i++) {
        sprintf(input_filename[i], "%s/matrix_%d.txt", path_input, i);
    }
    calculate_time_inparallel((const char**)input_filename, output_filename, 21);
    for (int i = 0; i < 21; i++) {
        free(input_filename[i]);
    }   
    return 0;
}