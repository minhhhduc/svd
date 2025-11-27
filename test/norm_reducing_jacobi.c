#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "mulmat.c"

// Định nghĩa các hằng số dựa trên Section 5 của PDF
#define TOLERANCE 1e-15
#define TAU 1e8
#define MAX_SWEEPS 50

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Macro để truy cập mảng 1 chiều như ma trận 2 chiều
#define AT(arr, r, c, n) (arr)[(r)*(n) + (c)]

// Hàm phụ trợ: Tính bình phương độ lớn |z|^2
double abs_sq(double complex z) {
    return creal(z) * creal(z) + cimag(z) * cimag(z);
}

// Hàm phụ trợ: Tạo số phức từ toạ độ cực (r, theta)
double complex cpolar(double r, double theta) {
    return r * cexp(I * theta);
}

// 1. Shear Transformation (Trang 9, 10)
// Giảm chuẩn của ma trận - CHỈ ÁP DỤNG KHI |c_pq| đủ lớn
void apply_shear(int n, double complex* A, double complex* V, int p, int q) {
    double complex c_pq = 0.0;
    double G_pq = 0.0;

    // Tính C_pq (Commutator) và G_pq
    #pragma omp parallel for reduction(+:c_pq, G_pq)
    for (int j = 0; j < n; ++j) {
        double complex a_pj = AT(A, p, j, n);
        double complex a_qj = AT(A, q, j, n);
        double complex a_jp = AT(A, j, p, n);
        double complex a_jq = AT(A, j, q, n);

        c_pq += a_pj * conj(a_qj) - conj(a_jp) * a_jq;

        if (j != p && j != q) {
            G_pq += abs_sq(a_pj) + abs_sq(a_qj) + abs_sq(a_jp) + abs_sq(a_jq);
        }
    }
    
    // Nếu |c_pq| quá nhỏ, skip shear transformation
    if (cabs(c_pq) < TOLERANCE * 10) {
        return;
    }

    double complex a_qq = AT(A, q, q, n);
    double complex a_pp = AT(A, p, p, n);
    double complex a_qp = AT(A, q, p, n);
    double complex a_pq = AT(A, p, q, n);

    double complex d_pq = a_qq - a_pp;

    // alpha = arg(c_pq) - pi/2
    double alpha = carg(c_pq) - M_PI / 2.0;

    double complex exp_i_alpha = cpolar(1.0, alpha);
    double complex exp_neg_i_alpha = cpolar(1.0, -alpha);

    double complex xi_pq = exp_i_alpha * a_qp + exp_neg_i_alpha * a_pq;

    // Tính tanh y với công thức tối ưu hóa
    double num = -cabs(c_pq);
    double den = 2.0 * (abs_sq(d_pq) + abs_sq(xi_pq)) + G_pq;
    
    if (fabs(den) < TOLERANCE) {
        return;  // Avoid division by zero
    }
    
    double tanh_y = num / den;

    // Giới hạn |tanh y| <= 1/2 để đảm bảo convergence
    if (tanh_y > 0.5) tanh_y = 0.5;
    if (tanh_y < -0.5) tanh_y = -0.5;

    double y = atanh(tanh_y);
    double cosh_y = cosh(y);
    double sinh_y = sinh(y);

    // Ma trận Shear S
    double complex s_pp = cosh_y;
    double complex s_pq = -I * exp_i_alpha * sinh_y;
    double complex s_qp = I * exp_neg_i_alpha * sinh_y;
    double complex s_qq = cosh_y;

    // Ma trận nghịch đảo S^-1 (thay y bằng -y => sinh(-y) = -sinh(y))
    double complex inv_s_pp = cosh_y;
    double complex inv_s_pq = -s_pq; 
    double complex inv_s_qp = -s_qp;
    double complex inv_s_qq = cosh_y;

    // Cập nhật A = S^-1 * A * S
    // Cập nhật hàng p, q (nhân trái với S^-1)
    #pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        double complex a_pj = AT(A, p, j, n);
        double complex a_qj = AT(A, q, j, n);
        AT(A, p, j, n) = inv_s_pp * a_pj + inv_s_pq * a_qj;
        AT(A, q, j, n) = inv_s_qp * a_pj + inv_s_qq * a_qj;
    }
    // Cập nhật cột p, q (nhân phải với S)
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double complex a_ip = AT(A, i, p, n);
        double complex a_iq = AT(A, i, q, n);
        AT(A, i, p, n) = a_ip * s_pp + a_iq * s_qp;
        AT(A, i, q, n) = a_ip * s_pq + a_iq * s_qq;
    }

    // Cập nhật Eigenvectors V = V * S
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double complex v_ip = AT(V, i, p, n);
        double complex v_iq = AT(V, i, q, n);
        AT(V, i, p, n) = v_ip * s_pp + v_iq * s_qp;
        AT(V, i, q, n) = v_ip * s_pq + v_iq * s_qq;
    }
}

// 2. Unitary Transformation (Chuẩn theo PDF Trang 9 - Eq 4, 5)
// Mục đích: Triệt tiêu a_qp nhưng dùng công thức tối ưu cho PNRJ
void apply_unitary(int n, double complex* A, double complex* V, int p, int q) {
    double complex a_qp = AT(A, q, p, n);
    double complex a_pq = AT(A, p, q, n);
    double complex a_pp = AT(A, p, p, n);
    double complex a_qq = AT(A, q, q, n);

    // Nếu phần tử cần khử đã quá nhỏ, bỏ qua
    if (cabs(a_qp) < TOLERANCE) return;

    // Công thức (4) trong PDF: d_pq = a_qq - a_pp
    double complex d_pq = a_qq - a_pp;

    // Tính căn bậc 2 của (d^2 + 4*a_pq*a_qp)
    double complex disc = csqrt(d_pq * d_pq + 4.0 * a_pq * a_qp);

    // Chọn dấu để d_max có độ lớn lớn nhất (Eq 4)
    double complex d_max;
    if (cabs(d_pq + disc) >= cabs(d_pq - disc)) {
        d_max = d_pq + disc;
    } else {
        d_max = d_pq - disc;
    }

    // Xử lý trường hợp đặc biệt: d_max quá nhỏ
    if (cabs(d_max) < TOLERANCE) {
        // Fallback to simple Givens rotation
        double r = sqrt(abs_sq(a_pp) + abs_sq(a_qp));
        if (r < TOLERANCE) return;
        
        double complex c = a_pp / r;
        double complex s = a_qp / r;
        
        double complex u_pp = c;
        double complex u_pq = -conj(s);
        double complex u_qp = s;
        double complex u_qq = conj(c);
        
        double complex uh_pp = conj(c);
        double complex uh_pq = conj(s);
        double complex uh_qp = -s;
        double complex uh_qq = c;
        
        #pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            double complex a_pj = AT(A, p, j, n);
            double complex a_qj = AT(A, q, j, n);
            AT(A, p, j, n) = uh_pp * a_pj + uh_pq * a_qj;
            AT(A, q, j, n) = uh_qp * a_pj + uh_qq * a_qj;
        }
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            double complex a_ip = AT(A, i, p, n);
            double complex a_iq = AT(A, i, q, n);
            AT(A, i, p, n) = a_ip * u_pp + a_iq * u_qp;
            AT(A, i, q, n) = a_ip * u_pq + a_iq * u_qq;
        }
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            double complex v_ip = AT(V, i, p, n);
            double complex v_iq = AT(V, i, q, n);
            AT(V, i, p, n) = v_ip * u_pp + v_iq * u_qp;
            AT(V, i, q, n) = v_ip * u_pq + v_iq * u_qq;
        }
        return;
    }

    // Công thức (5) trong PDF: tan x = (2 * e^i_theta * a_qp) / d_max
    // PDF yêu cầu: "theta is chosen to make the value of tan x real"
    // Đặt z = (2 * a_qp) / d_max. Ta cần e^i_theta * z là số thực dương.
    // => theta = -arg(z). Khi đó tan x = |z|.
    
    double complex z = (2.0 * a_qp) / d_max;
    
    // Góc theta dùng để khử pha ảo
    double theta = -carg(z);
    double tan_x = cabs(z); // Đây là giá trị thực của tan x

    // Giới hạn |tan x| <= 1 (Trang 9, dòng dưới Eq 5)
    // "In practice we need to bound the angle x... We set it to 1"
    if (tan_x > 1.0) {
        tan_x = 1.0;
    }

    // Tính sin x, cos x từ tan x
    double x = atan(tan_x);
    double c = cos(x);          // cos x
    double s_val = sin(x);      // sin x (giá trị thực)

    // Tính s phức: s = e^(-i*theta) * sin x
    // Lưu ý: Trong PDF ma trận U có dạng:
    // U = [ cos x          -e^(i*theta) * sin x ]
    //     [ e^(-i*theta)*sin x      cos x       ]
    // (Xem cấu trúc ma trận ngay đầu mục Unitary Transformation trang 9)
    
    double complex s = cpolar(1.0, -theta) * s_val; 
    
    // Các phần tử của U
    double complex u_pp = c;
    double complex u_pq = -conj(s); // -e^(i*theta) * sin x = -conj(e^-i*theta * sin x)
    double complex u_qp = s;        // e^(-i*theta) * sin x
    double complex u_qq = c;

    // Các phần tử của U* (Conjugate Transpose) - Dùng để nhân bên trái (hàng)
    double complex uh_pp = c;       // conj(real) = real
    double complex uh_pq = conj(s); // conj(s)
    double complex uh_qp = -s;      // conj(-conj(s)) = -s
    double complex uh_qq = c;

    // Cập nhật A = U* A U
    // 1. Nhân trái A = U* * A (Cập nhật hàng p và q)
    #pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        double complex a_pj = AT(A, p, j, n);
        double complex a_qj = AT(A, q, j, n);
        AT(A, p, j, n) = uh_pp * a_pj + uh_pq * a_qj;
        AT(A, q, j, n) = uh_qp * a_pj + uh_qq * a_qj;
    }

    // 2. Nhân phải A = A * U (Cập nhật cột p và q)
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double complex a_ip = AT(A, i, p, n);
        double complex a_iq = AT(A, i, q, n);
        AT(A, i, p, n) = a_ip * u_pp + a_iq * u_qp;
        AT(A, i, q, n) = a_ip * u_pq + a_iq * u_qq;
    }

    // Cập nhật V = V * U (Accumulate eigenvectors)
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double complex v_ip = AT(V, i, p, n);
        double complex v_iq = AT(V, i, q, n);
        AT(V, i, p, n) = v_ip * u_pp + v_iq * u_qp;
        AT(V, i, q, n) = v_ip * u_pq + v_iq * u_qq;
    }
}

// 3. Diagonal Transformation (Trang 10)
void apply_diagonal(int n, double complex* A, double complex* V) {
    double* t = (double*)malloc(n * sizeof(double));
    if (!t) return;

    #pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        double g_j = 0.0;
        double h_j = 0.0;

        for (int k = 0; k < n; ++k) {
            if (k == j) continue;
            g_j += abs_sq(AT(A, k, j, n));
            h_j += abs_sq(AT(A, j, k, n));
        }

        double t_val = 1.0;
        if (g_j > TOLERANCE && h_j > TOLERANCE) {
            t_val = sqrt(sqrt(h_j / g_j));
        }

        if (t_val > TAU) t_val = TAU;
        if (t_val < 1.0 / TAU) t_val = 1.0 / TAU;

        t[j] = t_val;
    }

    // Áp dụng D^-1 A D
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            AT(A, i, j, n) *= (t[j] / t[i]);
        }
    }

    // Cập nhật V = V * D
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            AT(V, i, j, n) *= t[j];
        }
    }

    free(t);
}

// ----------------------------------------------------------------------
// Hàm chính: Schur Decomposition using Norm-Reducing Jacobi
// Input: 
//   n: kích thước ma trận
//   A_in: ma trận đầu vào (phẳng)
// Output:
//   w: mảng chứa các trị riêng (từ đường chéo của ma trận Schur)
//   V_out: Schur vectors (CÓ THỂ KHÔNG PHẢI eigenvectors nếu A non-normal)
// 
// CHÚ Ý: Thuật toán này tính SCHUR DECOMPOSITION: A = V*T*V^H
//   - Ma trận T là dạng tam giác trên (Schur form)
//   - Eigenvalues nằm trên đường chéo của T
//   - Nếu A là ma trận Normal (AA^H = A^H*A), VD: symmetric, hermitian
//     → T là đường chéo, V chứa eigenvectors
//   - Nếu A là ma trận Non-normal (general complex matrix)
//     → T là tam giác trên, V chứa SCHUR VECTORS (không phải eigenvectors)
// ----------------------------------------------------------------------
void eig(int n, double complex* A_in, double complex* w, double complex* V_out) {
    // Cấp phát bộ nhớ cho bản sao A và ma trận V
    double complex* A = (double complex*)malloc(n * n * sizeof(double complex));
    double complex* V = (double complex*)malloc(n * n * sizeof(double complex));

    if (!A || !V) {
        if (A) free(A);
        if (V) free(V);
        return; // Lỗi cấp phát
    }

    // Khởi tạo
    for (int i = 0; i < n * n; ++i) {
        A[i] = A_in[i];
        // V là ma trận đơn vị
        int r = i / n;
        int c = i % n;
        V[i] = (r == c) ? 1.0 : 0.0;
    }

    // Vòng lặp chính
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        // Kiểm tra hội tụ (Norm của phần tam giác dưới ngặt)
        double lower_norm_sq = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:lower_norm_sq)
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                lower_norm_sq += abs_sq(AT(A, i, j, n));
            }
        }
        
        double lower_norm = sqrt(lower_norm_sq);
        
        if (sweep < 5 || sweep % 10 == 0) {
            printf("Sweep %d: lower_norm = %.6e\n", sweep, lower_norm);
        }

        if (lower_norm < TOLERANCE * sqrt(n * n / 2.0)) {
            printf("Converged at sweep %d\n", sweep);
            break; 
        }

        // Quét qua các cặp (p, q): Shear + Unitary cho mỗi cặp
        // Song song hóa sweep qua các cặp không conflict
        #pragma omp parallel for schedule(dynamic)
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                apply_shear(n, A, V, p, q);
                apply_unitary(n, A, V, p, q);
            }
        }

        // Áp dụng Diagonal scaling sau mỗi sweep
        apply_diagonal(n, A, V);
    }
    
    // In ma trận cuối cùng
    printf("\nFinal matrix A (should be upper triangular):\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.4f%+.4fi  ", creal(AT(A, i, j, n)), cimag(AT(A, i, j, n)));
        }
        printf("\n");
    }

    // Trích xuất kết quả
    for (int i = 0; i < n; ++i) {
        w[i] = AT(A, i, i, n);
    }

    for (int i = 0; i < n * n; ++i) {
        V_out[i] = V[i];
    }

    free(A);
    free(V);
}

void calculate_time_inparallel(const char** input_filename, const char *output_filename, int quantity_files) {
    FILE *fout = fopen(output_filename, "w");
    if (fout == NULL) {
        printf("Cannot open output file %s\n", output_filename);
        return;
    }   
    fprintf(fout, "matrix_size,computation_time_seconds\n");
    for (int file_idx = 0; file_idx < quantity_files; file_idx++) {
        double complex** A_matrix;
        int rows, cols;
        stream_matrix_reader(input_filename[file_idx], &A_matrix, &rows, &cols);
        
        // Compute C = A^T * A using mulmat_dns
        double complex** B = transpose_mat(A_matrix, rows, cols);
        double complex** C = (double complex**)malloc(cols * sizeof(double complex*));
        for (int i = 0; i < cols; i++) {
            C[i] = (double complex*)calloc(cols, sizeof(double complex));
        }
        
        printf("Processing matrix %d (%dx%d)...\n", file_idx, rows, cols);
        printf("Computing C = A^T * A using DNS (%dx%d)...\n", cols, cols);
        matmul_dns(B, A_matrix, C, cols, rows, cols, NUM_THREADS);
        printf("Matrix multiplication done.\n");
        
        // Now C is cols x cols square matrix
        int n = cols;
        
        // Convert C to flat array
        double complex* C_flat = (double complex*)malloc(n * n * sizeof(double complex));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C_flat[i * n + j] = C[i][j];
            }
        }
        
        double complex* w = (double complex*)malloc(n * sizeof(double complex));
        double complex* V = (double complex*)malloc(n * n * sizeof(double complex));
        
        printf("Computing eigenvalues for %dx%d matrix...\n", n, n);
        double start_time = omp_get_wtime();
        eig(n, C_flat, w, V);
        double end_time = omp_get_wtime();
        
        double computation_time = end_time - start_time;
        
        printf("Eigenvalue computation time: %.6f seconds\n\n", computation_time);
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