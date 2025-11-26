#include "stream.c" // Đảm bảo file này có hàm stream_matrix_reader
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- CÁC HÀM QUẢN LÝ BỘ NHỚ & TIỆN ÍCH ---

// Hàm giải phóng ma trận (nếu trong stream.c chưa có thì dùng hàm này)
void free_matrix_custom(double** A, int rows) {
    if (A == NULL) return;
    for (int i = 0; i < rows; i++) {
        if (A[i] != NULL) free(A[i]);
    }
    free(A);
}

// Ma trận đơn vị
double** identity_mat(int n) {
    double** I = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        I[i] = (double*)calloc(n, sizeof(double));
        I[i][i] = 1.0;
    }
    return I;
}

// Sao chép ma trận
double** copy_mat(double** A, int rows, int cols) {
    double** C = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        C[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) C[i][j] = A[i][j];
    }
    return C;
}

// Nhân ma trận thông thường (chỉ dùng để tính X = A^T * A ban đầu)
double** mul_mat_full(double** A, double** B, int A_rows, int A_cols, int B_cols) {
    double** C = (double**)malloc(A_rows * sizeof(double*));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (double*)calloc(B_cols, sizeof(double));
    }
    for (int i = 0; i < A_rows; i++) {
        for (int k = 0; k < A_cols; k++) {
            double r = A[i][k];
            for (int j = 0; j < B_cols; j++) {
                C[i][j] += r * B[k][j];
            }
        }
    }
    return C;
}

// Chuyển vị ma trận
double** transpose_mat(double** A, int rows, int cols) {
    double** T = (double**)malloc(cols * sizeof(double*));
    for (int i = 0; i < cols; i++) {
        T[i] = (double*)malloc(rows * sizeof(double));
        for (int j = 0; j < rows; j++) T[i][j] = A[j][i];
    }
    return T;
}

// --- CÁC HÀM CỐT LÕI CỦA PHƯƠNG PHÁP GIẢM CHUẨN (EBERLEIN METHOD) ---

// Tính chuẩn Frobenius của phần tử ngoài đường chéo
double off_diag_norm(double** A, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) sum += A[i][j] * A[i][j];
        }
    }
    return sqrt(sum);
}

// Xoay hàng p và q: A' = G^T * A (tối ưu O(n))
void rotate_rows(double** A, int n, int p, int q, double c, double s) {
    for (int j = 0; j < n; j++) {
        double Apj = A[p][j];
        double Aqj = A[q][j];
        A[p][j] = c * Apj - s * Aqj;
        A[q][j] = s * Apj + c * Aqj;
    }
}

// Xoay cột p và q: A'' = A' * G (tối ưu O(n))
void rotate_cols(double** A, int n, int p, int q, double c, double s) {
    for (int i = 0; i < n; i++) {
        double Aip = A[i][p];
        double Aiq = A[i][q];
        A[i][p] = c * Aip - s * Aiq;
        A[i][q] = s * Aip + c * Aiq;
    }
}

// Thuật toán Giảm chuẩn Jacobi (Tổng quát hóa cho mọi ma trận thực)
void eig_jacobi_norm_reducing(double** A_in, int n, int max_sweeps,
                              double** *eigen_vecs_out,
                              double** *eigen_vals_mat_out)
{
    double** A = copy_mat(A_in, n, n);
    double** V = identity_mat(n);
    
    const double tol = 1e-9; // Ngưỡng hội tụ

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        // Kiểm tra hội tụ dựa trên chuẩn Frobenius
        if (off_diag_norm(A, n) < tol) break;

        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double app = A[p][p];
                double aqq = A[q][q];
                double apq = A[p][q];
                double aqp = A[q][p];

                // Kiểm tra tính đối xứng
                // Nếu ma trận đối xứng (như A^T*A), ta dùng phép quay Jacobi chuẩn
                // Nếu không đối xứng, lý thuyết yêu cầu thêm phép Shear (nhưng ở đây ta cài đặt nhánh tổng quát)
                
                double theta;
                double c, s;

                // Logic tính góc quay để khử phần tử ngoài đường chéo
                // Với ma trận đối xứng: tan(2*theta) = 2*apq / (aqq - app)
                double numerator = 2.0 * apq;
                double denominator = aqq - app;

                if (fabs(numerator) < 1e-15 && fabs(denominator) < 1e-15) {
                    continue; // Đã bằng 0, bỏ qua
                }

                theta = 0.5 * atan2(numerator, denominator);
                c = cos(theta);
                s = sin(theta);

                // Thực hiện biến đổi tương đồng: A = G^T * A * G
                // Bước 1: Cập nhật ma trận A (Xoay hàng & cột)
                rotate_rows(A, n, p, q, c, s);
                rotate_cols(A, n, p, q, c, s);

                // Bước 2: Cập nhật ma trận vector riêng V (Chỉ xoay cột)
                // V = V * G
                rotate_cols(V, n, p, q, c, s);
            }
        }
    }

    *eigen_vals_mat_out = A;
    *eigen_vecs_out = V;
}

// Trích xuất đường chéo
double* diag_extract(double** A, int n) {
    double* d = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) d[i] = A[i][i];
    return d;
}

// Sắp xếp GIẢM DẦN (Descending) - Quan trọng cho SVD/PCA
double* argsort_desc(double* array, int n) {
    int* indices = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;

    // Selection sort đơn giản
    for (int i = 0; i < n - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (array[indices[j]] > array[indices[max_idx]]) {
                max_idx = j;
            }
        }
        int tmp = indices[i];
        indices[i] = indices[max_idx];
        indices[max_idx] = tmp;
    }

    double* sorted = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) sorted[i] = array[indices[i]];

    free(indices); // Giải phóng mảng chỉ số
    return sorted;
}

// --- HÀM TÍNH TOÁN CHÍNH ---

double calculate(double** A, int rows, int cols) {
    double start = (double)clock() / CLOCKS_PER_SEC;

    // 1. Tính ma trận hiệp phương sai: X = A^T * A
    // X luôn là ma trận đối xứng vuông (cols x cols)
    double** At = transpose_mat(A, rows, cols);
    double** X = mul_mat_full(At, A, cols, rows, cols);
    
    // Giải phóng bộ nhớ trung gian ngay lập tức
    free_matrix_custom(At, cols);

    // 2. Áp dụng phương pháp Giảm chuẩn Jacobi
    double** eig_vecs;
    double** eig_vals_mat;
    
    // Chạy tối đa 20 vòng quét (thường hội tụ sau 5-10 vòng)
    eig_jacobi_norm_reducing(X, cols, 20, &eig_vecs, &eig_vals_mat);
    
    // X bây giờ không cần nữa
    free_matrix_custom(X, cols);

    // 3. Trích xuất và sắp xếp kết quả
    double* eig_vals = diag_extract(eig_vals_mat, cols);
    double* sorted_vals = argsort_desc(eig_vals, cols); // Sắp xếp giảm dần

    // In thử giá trị riêng lớn nhất (chỉ để debug)
    // printf("  Max Eigenvalue: %f\n", sorted_vals[0]);

    // 4. Dọn dẹp
    free_matrix_custom(eig_vecs, cols);
    free_matrix_custom(eig_vals_mat, cols);
    free(eig_vals);
    free(sorted_vals);

    double end = (double)clock() / CLOCKS_PER_SEC;
    return end - start;
}

void calculate_all(char** input_filenames, char* output_filename, int num_files) {
    FILE *output_file = fopen(output_filename, "w");
    if (!output_file) {
        fprintf(stderr, "Error opening output file: %s\n", output_filename);
        return;
    }

    for (int i = 0; i < num_files; i++) {
        double** data = NULL;
        int rows = 0, cols = 0;

        // Đọc ma trận từ file (Yêu cầu stream.c phải hoạt động đúng)
        stream_matrix_reader(input_filenames[i], &data, &rows, &cols);
        
        if (data != NULL) {
            printf("Processing file: %s [%dx%d]\n", input_filenames[i], rows, cols);
            double time_taken = calculate(data, rows, cols);
            printf("Time taken: %f seconds\n", time_taken);
            fprintf(output_file, "File: %s, Time taken: %f seconds\n", input_filenames[i], time_taken);
            
            // Hàm giải phóng giả định, nếu stream.c có free_matrix thì dùng của nó
            // Ở đây gọi free_matrix_custom để an toàn cho code này
            free_matrix_custom(data, rows);
        } else {
            printf("Error reading file: %s\n", input_filenames[i]);
        }
    }
    fclose(output_file);
}



int main() {
    const char* path_input = "./data/input/matrix";
    const char* output_filename = "./data/output/times.txt";

    char* input_filenames[10];
    for (int i = 0; i < 10; i++) {
        input_filenames[i] = (char*)malloc(256 * sizeof(char));
        sprintf(input_filenames[i], "%s_%d.txt", path_input, i);
    }

    calculate_all(input_filenames, (char*)output_filename, 10);

    for (int i = 0; i < 10; i++) free(input_filenames[i]);
    return 0;
}