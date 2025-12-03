#ifndef MULMAT_H
#define MULMAT_H

double** transpose_mat(double** A, int rows, int cols);
void matmul_mono(double** A, double** B, double** C, int n, int m, int p);
void matmul_broadcast_1d(double** A, double** B, double** C, int n, int m, int p);
void matmul_cannon(double** A, double** B, double** C, int n, int m, int p_dim, int num_threads, int q_param);
void matmul_dns(double** A, double** B, double** C, int n, int m, int p_dim, int num_threads);

#endif
