#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <direct.h>
#include <string.h>
#include <math.h>
#include "stream.c"
#define N 4      // Kích thước ma trận
#define NUM_THREADS 8  // Số thread OpenMP (8 = 2³ cho DNS 3D cube)
#define processor_grid_dim 2 // Kích thước lưới giả lập cho Cannon (2x2=4 threads)


double** transpose_mat(double** A, int rows, int cols) {
    double** T = (double**)malloc(cols * sizeof(double*));
    for (int i = 0; i < cols; i++) {
        T[i] = (double*)malloc(rows * sizeof(double));
        for (int j = 0; j < rows; j++) T[i][j] = A[j][i];
    }
    return T;
}


void matmul_mono(double** A, double** B, double** C, int n, int m, int p){
    // A is n x m, B is m x p, C is n x p
    for(int i=0;i<n;i++){
        for(int j=0;j<p;j++){
            C[i][j] = 0;
            for(int k=0;k<m;k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

// ----------------------------
// 1. Broadcast / Row-Column (1D) Algorithm
// ----------------------------
// Mô tả: Ma trận được phân phối theo hàng/cột trên lưới 1D processors
// - Mỗi processor giữ một hàng của A và một cột của B
// - Broadcast: mỗi processor broadcast hàng của A và cột của B cho nhau
// - Tính toán: mỗi processor tính một phần tử C[i][j] = sum(A[i][k] * B[k][j])
// Complexity: O(n^3/p) với p processors
void matmul_broadcast_1d(double** A, double** B, double** C, int n, int m, int p){
    // A is n x m, B is m x p, C is n x p
    // Broadcast 1D: phân phối processors theo lưới 1D
    // Mỗi processor (pid) được gán một hoặc nhiều phần tử C[i][j]
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int pid = omp_get_thread_num();
        int num_procs = omp_get_num_threads();
        int total_elements = n * p;
        
        // Mỗi processor xử lý một phần các phần tử C[i][j]
        for(int elem = pid; elem < total_elements; elem += num_procs) {
            int i = elem / p;  // hàng của C
            int j = elem % p;  // cột của C
            
            // Processor giữ hàng i của A và cột j của B
            // Broadcast (giả lập): truy cập trực tiếp từ shared memory
            double sum = 0;
            for(int k = 0; k < m; k++) {
                // Tính tích vô hướng: hàng i của A với cột j của B
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// ----------------------------
// 2. Cannon's Algorithm (2D Block Distribution)
// ----------------------------
// Mô tả: Thuật toán Cannon cho nhân ma trận trên lưới 2D processors
// Các bước:
//   1. Initial Alignment (Skewing):
//      - Hàng i của block A: shift trái i positions
//      - Cột j của block B: shift lên j positions
//   2. Loop q iterations:
//      - Mỗi processor (i,j) nhân block local: C[i][j] += A[i][j] * B[i][j]
//      - Shift A blocks: mỗi hàng shift trái 1 position (circular)
//      - Shift B blocks: mỗi cột shift lên 1 position (circular)
// Complexity: O(n^3/p) với p = q^2 processors
// Yêu cầu: Ma trận vuông n x n, n chia hết cho q
void matmul_cannon(double** A, double** B, double** C, int n, int m, int p_dim, int num_threads,int q_param){
    // Tính q dựa trên q_param hoặc NUM_THREADS
    int q = (q_param > 0) ? q_param : (int)sqrt(num_threads);  // q x q processor grid
    while(q*q > num_threads && q > 1) q--;  // Đảm bảo q^2 <= NUM_THREADS
    
    // Kiểm tra điều kiện cơ bản
    if(n != p_dim || q < 2) {
        // Fallback: Nếu không thỏa điều kiện, dùng phương pháp đơn giản tối ưu
        #pragma omp parallel for collapse(2) num_threads(num_threads)
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
    
    // Padding: làm tròn lên để chia hết cho q
    int n_padded = q * ((n + q - 1) / q);  // ceil(n/q) * q
    int m_padded = q * ((m + q - 1) / q);  // ceil(m/q) * q
    int p_dim_padded = q * ((p_dim + q - 1) / q);  // ceil(p_dim/q) * q
    
    // Nếu cần padding, tạo ma trận mới với zero-padding
    double** A_work = A;
    double** B_work = B;
    int need_padding = (n != n_padded || m != m_padded || p_dim != p_dim_padded);
    
    if(need_padding) {
        // Tạo A_padded: n_padded × m_padded
        A_work = (double**)malloc(n_padded * sizeof(double*));
        for(int i = 0; i < n_padded; i++) {
            A_work[i] = (double*)calloc(m_padded, sizeof(double));
            if(i < n) {
                for(int j = 0; j < m; j++) {
                    A_work[i][j] = A[i][j];
                }
            }
        }
        
        // Tạo B_padded: m_padded × p_dim_padded
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
    
    int block_n = n_padded / q;  // Kích thước block cho hàng/cột của kết quả
    int block_m = m_padded / q;  // Kích thước block cho chiều inner

    #pragma omp parallel num_threads(q*q)  // Chỉ dùng q*q threads
    {
        int tid = omp_get_thread_num();
        int bi = tid / q;
        int bj = tid % q;

        int row_start = bi * block_n;
        int col_start = bj * block_n;

        // Local blocks
        double** A_loc = (double**)malloc(block_n*sizeof(double*));
        double** B_loc = (double**)malloc(block_m*sizeof(double*));  // B_loc is block_m x block_n
        double** C_loc = (double**)malloc(block_n*sizeof(double*));

        for(int i=0;i<block_n;i++){
            A_loc[i] = (double*)malloc(block_m*sizeof(double));
            C_loc[i] = (double*)calloc(block_n,sizeof(double));
        }
        for(int i=0;i<block_m;i++){
            B_loc[i] = (double*)malloc(block_n*sizeof(double));
        }

        // Bước 1: Initial Alignment (Skewing)
        // Công thức Cannon: C[i,j] = Σ(k=0..p-1) A[i,(i+j+k)%p] * B[(i+j+k)%p,j]
        // Initial k=0: A_k_block = (bi+bj) % q, B_k_block = (bi+bj) % q
        int A_k_block = (bi + bj) % q;  // Block A ban đầu: A[bi,(bi+bj)%q]
        int B_k_block = (bi + bj) % q;  // Block B ban đầu: B[(bi+bj)%q,bj]
        
        // Bước 2: Loop q iterations - mỗi lần shift và tính toán
        for(int step=0; step<q; step++){
            // Load block A tại vị trí đã skew + shift
            for(int i=0;i<block_n;i++)
                for(int j=0;j<block_m;j++){
                    A_loc[i][j] = A_work[row_start+i][A_k_block*block_m + j];
                }
            
            // Load block B tại vị trí đã skew + shift
            for(int i=0;i<block_m;i++)
                for(int j=0;j<block_n;j++){
                    B_loc[i][j] = B_work[B_k_block*block_m + i][col_start+j];
                }
            
            // Nhân block local: C_loc += A_loc * B_loc
            for(int i=0;i<block_n;i++)
                for(int j=0;j<block_n;j++)
                    for(int k=0;k<block_m;k++)
                        C_loc[i][j] += A_loc[i][k] * B_loc[k][j];

            #pragma omp barrier
            
            // Circular shift cho iteration tiếp theo
            // Shift A trái 1 position (circular)
            A_k_block = (A_k_block - 1 + q) % q;
            // Shift B lên 1 position (circular)
            B_k_block = (B_k_block - 1 + q) % q;
            
            #pragma omp barrier
        }

        // Copy back into global C (chỉ ghi vào phạm vi gốc)
        for(int i=0;i<block_n;i++)
            for(int j=0;j<block_n;j++){
                int global_row = row_start + i;
                int global_col = col_start + j;
                if(global_row < n && global_col < p_dim) {
                    C[global_row][global_col] = C_loc[i][j];
                }
            }

        // Free
        for(int i=0;i<block_n;i++){
            free(A_loc[i]); free(C_loc[i]);
        }
        for(int i=0;i<block_m;i++){
            free(B_loc[i]);
        }
        free(A_loc); free(B_loc); free(C_loc);
    }
    
    // Free padded matrices nếu có
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
}


// ----------------------------
// 3. DNS (Dekel, Nassimi, Sahni) Algorithm - 3D Cube Approach
// ----------------------------
// Mô tả: Thuật toán DNS cho nhân ma trận trên lưới 3D p x p x p processors
// Công thức: C[i,j] = Σ(k=0..p-1) A[i,k] * B[k,j]
// Các bước theo tài liệu:
//   1. Move (Alignment): Di chuyển A[i,k] và B[k,j] đến vị trí ban đầu
//      - A[i,k] từ P[i,k,0] -> P[i,k,k] (broadcast dọc trục k)
//      - B[k,j] từ P[k,j,0] -> P[k,j,k] (broadcast dọc trục k)
//   2. Broadcast:
//      - Broadcast A[i,k] dọc theo trục j: P[i,j,k] nhận A[i,k]
//      - Broadcast B[k,j] dọc theo trục i: P[i,j,k] nhận B[k,j]
//   3. Compute và Reduction:
//      - Mỗi P[i,j,k] tính partial product: A[i,k] * B[k,j]
//      - All-to-one reduction dọc trục k để có C[i,j]
// Complexity: O(n^3/p^3) local computation + O(log p) communication
// Yêu cầu: Ma trận vuông n×n, p^3 processors, n chia hết cho p
void matmul_dns(double** A, double** B, double** C, int n, int m, int p_dim, int num_threads){
    int p = (int)cbrt(num_threads);  // p x p x p cube
    if(p*p*p != num_threads) p = 1;  // fallback nếu không phải lập phương hoàn hảo
    
    // Kiểm tra điều kiện cơ bản
    if(n != p_dim || p < 2) {
        // Fallback: dùng phương pháp tối ưu với static schedule
        #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
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
    
    // Padding: làm tròn lên để chia hết cho p
    int n_padded = p * ((n + p - 1) / p);  // ceil(n/p) * p
    int m_padded = p * ((m + p - 1) / p);  // ceil(m/p) * p
    int p_dim_padded = p * ((p_dim + p - 1) / p);  // ceil(p_dim/p) * p
    
    // Nếu cần padding, tạo ma trận mới với zero-padding
    double** A_work = A;
    double** B_work = B;
    int need_padding = (n != n_padded || m != m_padded || p_dim != p_dim_padded);
    
    if(need_padding) {
        // Tạo A_padded: n_padded × m_padded
        A_work = (double**)malloc(n_padded * sizeof(double*));
        for(int i = 0; i < n_padded; i++) {
            A_work[i] = (double*)calloc(m_padded, sizeof(double));
            if(i < n) {
                for(int j = 0; j < m; j++) {
                    A_work[i][j] = A[i][j];
                }
            }
        }
        
        // Tạo B_padded: m_padded × p_dim_padded
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
    
    int block_size_n = n_padded / p;  // Kích thước block cho chiều n (result dimensions)
    int block_size_m = m_padded / p;  // Kích thước block cho chiều m (inner dimension)
    int p_squared = p * p;
    
    // Shared buffers cho broadcast và computation
    // A_local: block_size_n × block_size_m per thread
    // B_local: block_size_m × block_size_n per thread
    // temp_results: block_size_n × block_size_n per thread (result blocks)
    double* A_local = (double*)malloc(NUM_THREADS * block_size_n * block_size_m * sizeof(double));
    double* B_local = (double*)malloc(NUM_THREADS * block_size_m * block_size_n * sizeof(double));
    double* temp_results = (double*)calloc(NUM_THREADS * block_size_n * block_size_n, sizeof(double));

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        // Map thread ID to 3D cube position (i,j,k)
        int i = tid / p_squared;  // i: 0..p-1 (block row index)
        int j = (tid / p) % p;    // j: 0..p-1 (block col index)  
        int k = tid % p;          // k: 0..p-1 (block inner index)

        int A_buf_offset = tid * block_size_n * block_size_m;
        int B_buf_offset = tid * block_size_m * block_size_n;
        int result_buf_offset = tid * block_size_n * block_size_n;
        
        // Bước 1: Move/Alignment + Bước 2: Broadcast
        // P[i,j,k] cần nhận A[i,k] và B[k,j]
        // Giả lập broadcast: đọc trực tiếp từ global matrix
        int A_row_start = i * block_size_n;
        int A_col_start = k * block_size_m;
        int B_row_start = k * block_size_m;
        int B_col_start = j * block_size_n;
        
        // Load A[i,k] block vào local buffer (block_size_n × block_size_m)
        for(int r = 0; r < block_size_n; r++) {
            for(int c = 0; c < block_size_m; c++) {
                A_local[A_buf_offset + r * block_size_m + c] = A_work[A_row_start + r][A_col_start + c];
            }
        }
        
        // Load B[k,j] block vào local buffer (block_size_m × block_size_n)
        for(int r = 0; r < block_size_m; r++) {
            for(int c = 0; c < block_size_n; c++) {
                B_local[B_buf_offset + r * block_size_n + c] = B_work[B_row_start + r][B_col_start + c];
            }
        }

        #pragma omp barrier
        
        // Bước 3: Local Computation - P[i,j,k] tính partial product A[i,k] * B[k,j]
        // A[i,k]: block_size_n × block_size_m
        // B[k,j]: block_size_m × block_size_n
        // Result: block_size_n × block_size_n
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
        
        // Bước 3: All-to-One Reduction dọc trục k
        // Chỉ thread k=0 của mỗi plane (i,j) thực hiện reduction
        if(k == 0) {
            int C_row_start = i * block_size_n;
            int C_col_start = j * block_size_n;
            int plane_base_tid = i * p_squared + j * p;  // thread ID của (i,j,0)
            
            // Reduction: C[i,j] = Σ(k=0..p-1) A[i,k] * B[k,j]
            for(int r = 0; r < block_size_n; r++) {
                for(int c = 0; c < block_size_n; c++) {
                    double sum = 0;
                    // Cộng tất cả partial products từ P[i,j,0], P[i,j,1], ..., P[i,j,p-1]
                    for(int kk = 0; kk < p; kk++) {
                        int thread_id = plane_base_tid + kk;
                        sum += temp_results[thread_id * block_size_n * block_size_n + r * block_size_n + c];
                    }
                    // Chỉ ghi vào C nếu nằm trong phạm vi gốc (không phải padding)
                    int global_row = C_row_start + r;
                    int global_col = C_col_start + c;
                    if(global_row < n && global_col < p_dim) {
                        C[global_row][global_col] = sum;
                    }
                }
            }
        }
    }
    
    // Free padded matrices nếu có
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
    
    // Free memory
    free(A_local);
    free(B_local);
    free(temp_results);
}

// ----------------------------
// Main
// ----------------------------

void calculate(char** fileins, const char* fileout) {
    double** A;
    int rows, cols;
    FILE* fout = fopen(fileout, "w");
    if (fout == NULL) {
        perror("Error opening output file");
        exit(EXIT_FAILURE);
    }
    
    fprintf(fout, "number_of_computed,matmul_mono,broadcast_time,common_time,dns_time\n");
    for (int f = 0; f < 21; f++) {
        printf("Processing matrix %d...\n", f);
        // Đọc ma trận từ file
        stream_matrix_reader(fileins[f], &A, &rows, &cols);

        // Tạo ma trận B = A^T
        double** B = transpose_mat(A, rows, cols);

        // Khởi tạo ma trận kết quả C (B * A = A^T * A will be cols x cols)
        double** C = (double**)malloc(cols * sizeof(double*));
        for (int i = 0; i < cols; i++) {
            C[i] = (double*)calloc(cols, sizeof(double));
        }

        // 0. Mono (single-threaded baseline)
        double start = omp_get_wtime();
        matmul_mono(B, A, C, cols, rows, cols);
        double end = omp_get_wtime();
        double time_spent_mono = end - start;
        printf("mono done\n");

        // Reset C về 0 trước khi chạy thuật toán tiếp theo
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                C[i][j] = 0;
            }
        }

        // 1. Broadcast algorithm
        start = omp_get_wtime();
        matmul_broadcast_1d(B, A, C, cols, rows, cols);
        end = omp_get_wtime();
        double time_spent_broadcast = end - start;
        printf("broadcast_1d done\n");

        // Reset C về 0
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                C[i][j] = 0;
            }
        }

        // 2. Cannon algorithm
        start = omp_get_wtime();
        // Compute B * A = A^T * A (result is cols x cols)
        // Cho Cannon tự tính q từ NUM_THREADS
        matmul_cannon(B, A, C, cols, rows, cols, NUM_THREADS, 0);  // q_param=0 để tự động tính
        end = omp_get_wtime();
        double time_spent_common = end - start;
        printf("cannon done\n");
        
        // Reset C về 0
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                C[i][j] = 0;
            }
        }

        start = omp_get_wtime();
        // Compute B * A = A^T * A (result is cols x cols)
        matmul_dns(B, A, C, cols, rows, cols, NUM_THREADS);  // Sử dụng NUM_THREADS đã định nghĩa
        end = omp_get_wtime();
        double time_spent_dns = end - start;
        printf("dns done\n");

        // In ra màn hình và ghi file
        printf("{mono: %f, broadcast: %f, cannon: %f, dns: %f}\n", time_spent_mono, time_spent_broadcast, time_spent_common, time_spent_dns);
        fprintf(fout, "%dx%dx%d,%f,%f,%f,%f\n", rows, cols, cols, time_spent_mono, time_spent_broadcast, time_spent_common, time_spent_dns);
        fflush(fout);  // Flush immediately to ensure data is written

        // Giải phóng bộ nhớ
        free_matrix(A, rows);
        free_matrix(B, cols);  // B is A^T, so it's cols x rows
        free_matrix(C, cols);  // C is A^T * A, so it's cols x cols
    }

    fclose(fout);
}

// int main(){
//     const char* path_input = "./data/input/matrix";
//     const char* output_filename = "./data/output/mul_mat_times.csv";
//     char* input_filenames[21];
//     for (int i = 0; i < 21; i++) {

//         input_filenames[i] = (char*)malloc(150 * sizeof(char));
//         sprintf(input_filenames[i], "%s_%d.txt", path_input, i);
//         printf("Processing file: %s\n", input_filenames[i]);
//     }
//     calculate(input_filenames, output_filename);

//     // Giải phóng
//     for(int i=0;i<21;i++) free(input_filenames[i]);
    
//     return 0;
// }
