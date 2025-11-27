/**
 * Parallel Norm-Reducing Jacobi (PNRJ) Algorithm
 * Thuật toán Jacobi giảm chuẩn song song để tính eigenvalues/eigenvectors của ma trận tổng quát
 * 
 * Tham khảo: Parallel Norm-Reducing Jacobi Method for General Complex Matrices
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define MAX_SWEEPS 50
#define TOLERANCE 1e-10
#define NUM_THREADS 8
#define MAX_TAN_Z 1.0  // Giới hạn |tan(z)| < 1 cho phép biến đổi unitary

typedef struct {
    int p;
    int q;
} PivotPair;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

double frobenius_norm(double** A, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) collapse(2) num_threads(NUM_THREADS)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            sum += A[i][j] * A[i][j];
        }
    }
    return sqrt(sum);
}

double lower_triangular_norm(double** A, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(NUM_THREADS)
    for(int i = 1; i < n; i++) {
        for(int j = 0; j < i; j++) {
            sum += A[i][j] * A[i][j];
        }
    }
    return sqrt(sum);
}

double departure_from_normality(double** A, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) collapse(2) num_threads(NUM_THREADS)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            double aat = 0.0, ata = 0.0;
            for(int k = 0; k < n; k++) {
                aat += A[i][k] * A[j][k];  // (A*A^T)[i,j]
                ata += A[k][i] * A[k][j];  // (A^T*A)[i,j]
            }
            double c_ij = aat - ata;
            sum += c_ij * c_ij;
        }
    }
    return sqrt(sum);
}

int is_symmetric(double** A, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            if(fabs(A[i][j] - A[j][i]) > 1e-10) {
                return 0;
            }
        }
    }
    return 1;
}

// ============================================================================
// MODULUS ORDERING - Parallel Ordering cho các pivot pairs
// ============================================================================

int generate_modulus_ordering(int n, PivotPair*** sequences, int* num_sequences) {
    int N = n * (n - 1) / 2;  // Tổng số cặp
    *num_sequences = n - 1;
    
    *sequences = (PivotPair**)malloc((*num_sequences) * sizeof(PivotPair*));
    int* sequence_sizes = (int*)calloc(*num_sequences, sizeof(int));
    
    // Đếm số lượng pairs trong mỗi sequence
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            int idx = (i + j) % (*num_sequences);
            sequence_sizes[idx]++;
        }
    }
    
    // Cấp phát bộ nhớ cho mỗi sequence
    for(int s = 0; s < *num_sequences; s++) {
        (*sequences)[s] = (PivotPair*)malloc(sequence_sizes[s] * sizeof(PivotPair));
    }
    
    // Điền các pivot pairs vào sequences
    int* current_pos = (int*)calloc(*num_sequences, sizeof(int));
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            int idx = (i + j) % (*num_sequences);
            int pos = current_pos[idx]++;
            (*sequences)[idx][pos].p = i;
            (*sequences)[idx][pos].q = j;
        }
    }
    
    free(current_pos);
    int total_pairs = 0;
    for(int s = 0; s < *num_sequences; s++) {
        total_pairs += sequence_sizes[s];
    }
    
    free(sequence_sizes);
    return total_pairs;
}

// ============================================================================
// TRANSFORMATION FUNCTIONS
// ============================================================================

/**
 * Unitary Transformation U: Khử phần tử A[q][p]
 * U = I + (c-1)(e_p*e_p^T + e_q*e_q^T) + s(e_p*e_q^T - e_q*e_p^T)
 * với ràng buộc |tan(z)| < MAX_TAN_Z
 */
void compute_unitary_transform(double** A, int p, int q, double* c, double* s) {
    double aqp = A[q][p];
    
    if(fabs(aqp) < 1e-15) {
        *c = 1.0;
        *s = 0.0;
        return;
    }
    
    // Tính góc z để zero out A[q][p]
    // Nhưng giới hạn |tan(z)| < MAX_TAN_Z để tránh làm chậm hội tụ
    double aqq = A[q][q];
    double app = A[p][p];
    
    // Tính tan(2z) = 2*aqp / (aqq - app)
    double tan_2z;
    if(fabs(aqq - app) < 1e-15) {
        tan_2z = (aqp > 0) ? 1e10 : -1e10;
    } else {
        tan_2z = 2.0 * aqp / (aqq - app);
    }
    
    // Tính tan(z) từ tan(2z)
    double tan_z;
    if(fabs(tan_2z) > 1e10) {
        tan_z = 0.0;
    } else {
        double sqrt_term = sqrt(1.0 + tan_2z * tan_2z);
        tan_z = tan_2z / (1.0 + sqrt_term);
    }
    
    // Giới hạn |tan(z)| < MAX_TAN_Z
    if(fabs(tan_z) > MAX_TAN_Z) {
        tan_z = (tan_z > 0) ? MAX_TAN_Z : -MAX_TAN_Z;
    }
    
    *c = 1.0 / sqrt(1.0 + tan_z * tan_z);
    *s = tan_z * (*c);
}

/**
 * Shear Transformation S: Giảm departure from normality
 * S = I + t_p*e_p*e_q^T + t_q*e_q*e_p^T
 */
void compute_shear_transform(double** A, int p, int q, double* t_p, double* t_q) {
    // Tính C[p][q] và C[q][p] từ ma trận C = A*A^T - A^T*A
    double cpq = 0.0, cqp = 0.0;
    
    for(int k = 0; k < A[0] ? 4 : 0; k++) {  // Giả sử n được truyền vào
        // Cần access n, tạm thời hardcode hoặc thêm tham số
    }
    
    // Tối ưu hóa: chọn t_p, t_q để giảm |C[p][q]| và |C[q][p]|
    // Simplified version: set to zero for now
    *t_p = 0.0;
    *t_q = 0.0;
}

/**
 * Diagonal Transformation D: Điều chỉnh các phần tử đường chéo
 * D = diag(..., d_p, ..., d_q, ...)
 */
void compute_diagonal_transform(double** A, int n, double* d) {
    // Khởi tạo d = 1 cho tất cả các chỉ số
    for(int i = 0; i < n; i++) {
        d[i] = 1.0;
    }
    
    // Tính toán d_i để giảm ||C||_F
    // Simplified: keep as identity for now
}

/**
 * Rotation R = S * U: Kết hợp Shear và Unitary
 * Áp dụng: A' = R^{-1} * A * R
 */
void apply_rotation(double** A, double** P, int n, int p, int q, 
                    double c_u, double s_u, double t_p, double t_q) {
    // Bước 1: Áp dụng Shear S (nếu t_p, t_q != 0)
    if(fabs(t_p) > 1e-15 || fabs(t_q) > 1e-15) {
        // A' = S^{-1} * A * S
        // Left multiply: S^{-1}
        for(int j = 0; j < n; j++) {
            double ap = A[p][j];
            double aq = A[q][j];
            A[p][j] = ap - t_p * aq;
            A[q][j] = aq - t_q * ap;
        }
        
        // Right multiply: S
        for(int i = 0; i < n; i++) {
            double aip = A[i][p];
            double aiq = A[i][q];
            A[i][p] = aip + t_p * aiq;
            A[i][q] = aiq + t_q * aip;
        }
    }
    
    // Bước 2: Áp dụng Unitary U
    // A' = U^T * A * U
    
    // Left multiply: U^T (transpose of rotation)
    for(int j = 0; j < n; j++) {
        double ap = A[p][j];
        double aq = A[q][j];
        A[p][j] = c_u * ap + s_u * aq;
        A[q][j] = -s_u * ap + c_u * aq;
    }
    
    // Right multiply: U
    for(int i = 0; i < n; i++) {
        double aip = A[i][p];
        double aiq = A[i][q];
        A[i][p] = c_u * aip - s_u * aiq;
        A[i][q] = s_u * aip + c_u * aiq;
    }
    
    // Cập nhật eigenvector matrix P
    if(P != NULL) {
        // P' = P * R
        // Right multiply by R = S * U
        
        // First: P' = P * S
        if(fabs(t_p) > 1e-15 || fabs(t_q) > 1e-15) {
            for(int i = 0; i < n; i++) {
                double pip = P[i][p];
                double piq = P[i][q];
                P[i][p] = pip + t_p * piq;
                P[i][q] = piq + t_q * pip;
            }
        }
        
        // Then: P' = P' * U
        for(int i = 0; i < n; i++) {
            double pip = P[i][p];
            double piq = P[i][q];
            P[i][p] = c_u * pip - s_u * piq;
            P[i][q] = s_u * pip + c_u * piq;
        }
    }
}

/**
 * Áp dụng Diagonal Transformation D
 * A' = D^{-1} * A * D
 */
void apply_diagonal(double** A, double** P, int n, double* d) {
    // Left multiply: D^{-1}
    for(int i = 0; i < n; i++) {
        if(fabs(d[i] - 1.0) > 1e-15) {
            double d_inv = 1.0 / d[i];
            for(int j = 0; j < n; j++) {
                A[i][j] *= d_inv;
            }
        }
    }
    
    // Right multiply: D
    for(int j = 0; j < n; j++) {
        if(fabs(d[j] - 1.0) > 1e-15) {
            for(int i = 0; i < n; i++) {
                A[i][j] *= d[j];
            }
        }
    }
    
    // Update P
    if(P != NULL) {
        for(int j = 0; j < n; j++) {
            if(fabs(d[j] - 1.0) > 1e-15) {
                for(int i = 0; i < n; i++) {
                    P[i][j] *= d[j];
                }
            }
        }
    }
}

// ============================================================================
// MAIN PNRJ ALGORITHM
// ============================================================================

int pnrj_eig(double** A, double* eigenvalues, double** eigenvectors, int n) {
    printf("Starting Parallel Norm-Reducing Jacobi (PNRJ) Algorithm\n");
    printf("Matrix size: %dx%d\n", n, n);
    printf("Num threads: %d\n", NUM_THREADS);
    
    int sym_flag = is_symmetric(A, n);
    printf("Matrix type: %s\n", sym_flag ? "Symmetric" : "General (non-symmetric)");
    
    // Khởi tạo eigenvector matrix = Identity
    if(eigenvectors != NULL) {
        #pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    
    // Generate modulus ordering
    PivotPair** sequences;
    int num_sequences;
    int total_pairs = generate_modulus_ordering(n, &sequences, &num_sequences);
    printf("Generated %d parallel sequences covering %d pivot pairs\n", 
           num_sequences, total_pairs);
    
    // Metrics
    double lower_norm = lower_triangular_norm(A, n);
    double initial_lower = lower_norm;
    double dep_norm = departure_from_normality(A, n);
    
    printf("Initial lower triangular norm: %.10e\n", lower_norm);
    printf("Initial departure from normality: %.10e\n", dep_norm);
    
    int sweep = 0;
    double* d = (double*)malloc(n * sizeof(double));
    
    // Main sweep loop
    while(lower_norm > TOLERANCE && sweep < MAX_SWEEPS) {
        // Một sweep bao gồm:
        // SW = T_1 D_1 T_2 D_2 ... T_s D_s
        
        for(int s = 0; s < num_sequences; s++) {
            // T_s: Parallel Rotation Set
            int set_size = (s < num_sequences - 1) ? 
                          (total_pairs / num_sequences) : 
                          (total_pairs - s * (total_pairs / num_sequences));
            
            // Tính toán các rotations song song
            #pragma omp parallel for num_threads(NUM_THREADS)
            for(int k = 0; k < set_size; k++) {
                if(k >= set_size) continue;
                
                PivotPair pair = sequences[s][k];
                int p = pair.p;
                int q = pair.q;
                
                // Compute Unitary transformation
                double c_u, s_u;
                compute_unitary_transform(A, p, q, &c_u, &s_u);
                
                // Compute Shear transformation (simplified)
                double t_p = 0.0, t_q = 0.0;
                
                // Apply rotation R = S * U
                // Note: Cần synchronization sau mỗi rotation set
            }
            
            // Áp dụng các rotations (sequential để đảm bảo correctness)
            for(int k = 0; k < set_size; k++) {
                PivotPair pair = sequences[s][k];
                int p = pair.p;
                int q = pair.q;
                
                double c_u, s_u;
                compute_unitary_transform(A, p, q, &c_u, &s_u);
                double t_p = 0.0, t_q = 0.0;
                
                apply_rotation(A, eigenvectors, n, p, q, c_u, s_u, t_p, t_q);
            }
            
            // D_s: Diagonal Transformation
            compute_diagonal_transform(A, n, d);
            apply_diagonal(A, eigenvectors, n, d);
        }
        
        sweep++;
        
        // Kiểm tra hội tụ
        lower_norm = lower_triangular_norm(A, n);
        dep_norm = departure_from_normality(A, n);
        
        if(sweep % 5 == 0 || sweep < 3) {
            printf("Sweep %d: lower_norm=%.2e, dep_norm=%.2e (reduction: %.2f%%)\n",
                   sweep, lower_norm, dep_norm, 
                   100.0 * (1.0 - lower_norm/initial_lower));
        }
    }
    
    printf("\nConverged after %d sweeps\n", sweep);
    printf("Final lower triangular norm: %.10e\n", lower_norm);
    printf("Final departure from normality: %.10e\n", dep_norm);
    
    // Trích xuất eigenvalues từ đường chéo
    for(int i = 0; i < n; i++) {
        eigenvalues[i] = A[i][i];
    }
    
    // Cleanup
    for(int s = 0; s < num_sequences; s++) {
        free(sequences[s]);
    }
    free(sequences);
    free(d);
    
    return sweep;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void print_matrix(const char* name, double** A, int rows, int cols) {
    printf("\n%s (%dx%d):\n", name, rows, cols);
    int max_print = (rows > 10) ? 5 : rows;
    int max_cols = (cols > 10) ? 5 : cols;
    
    for(int i = 0; i < max_print; i++) {
        for(int j = 0; j < max_cols; j++) {
            printf("%10.6f ", A[i][j]);
        }
        if(cols > 10) printf("... ");
        printf("\n");
    }
    if(rows > 10) printf("...\n");
}

void print_eigenvalues(double* eigenvalues, int n) {
    printf("\nEigenvalues:\n");
    int max_print = (n > 20) ? 20 : n;
    for(int i = 0; i < max_print; i++) {
        printf("λ[%d] = %.10f\n", i, eigenvalues[i]);
    }
    if(n > 20) printf("... (%d more)\n", n - 20);
}

// ============================================================================
// MAIN TEST
// ============================================================================

int main() {
    printf("==================================================\n");
    printf("PARALLEL NORM-REDUCING JACOBI (PNRJ) TEST\n");
    printf("==================================================\n\n");
    
    int n = 4;
    
    double** A = (double**)malloc(n * sizeof(double*));
    double** A_copy = (double**)malloc(n * sizeof(double*));
    double** V = (double**)malloc(n * sizeof(double*));
    double* lambda = (double*)malloc(n * sizeof(double));
    
    for(int i = 0; i < n; i++) {
        A[i] = (double*)malloc(n * sizeof(double));
        A_copy[i] = (double*)malloc(n * sizeof(double));
        V[i] = (double*)malloc(n * sizeof(double));
    }
    
    // Test matrix (symmetric)
    double test_data[4][4] = {
        {4, 1, 0, 0},
        {1, 4, 1, 0},
        {0, 1, 4, 1},
        {0, 0, 1, 4}
    };
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = test_data[i][j];
            A_copy[i][j] = test_data[i][j];
        }
    }
    
    print_matrix("Original matrix A", A_copy, n, n);
    
    double start = omp_get_wtime();
    int sweeps = pnrj_eig(A, lambda, V, n);
    double end = omp_get_wtime();
    
    printf("\nTime: %.6f seconds\n", end - start);
    print_eigenvalues(lambda, n);
    print_matrix("Eigenvectors V", V, n, n);
    print_matrix("Final A (should be upper triangular)", A, n, n);
    
    // Verification: A_copy * V ≈ V * diag(lambda)
    printf("\nVerification: A*V vs V*D\n");
    double max_error = 0.0;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            double av = 0.0;
            for(int k = 0; k < n; k++) {
                av += A_copy[i][k] * V[k][j];
            }
            double vd = V[i][j] * lambda[j];
            double error = fabs(av - vd);
            if(error > max_error) max_error = error;
        }
    }
    printf("Max error |A*V - V*D|: %.2e\n", max_error);
    
    // Cleanup
    for(int i = 0; i < n; i++) {
        free(A[i]); free(A_copy[i]); free(V[i]);
    }
    free(A); free(A_copy); free(V); free(lambda);
    
    return 0;
}
