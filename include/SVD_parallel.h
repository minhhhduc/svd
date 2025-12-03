#ifndef SVD_PARALLEL_H
#define SVD_PARALLEL_H

/**
 * @brief Computes SVD: A = U * S * V^T (Parallel Version)
 * @param m Rows of A
 * @param n Cols of A
 * @param A Input matrix (flat array m*n)
 * @param U Output matrix (m*n) - Left Singular Vectors
 * @param S Output array (n) - Singular Values
 * @param V Output matrix (n*n) - Right Singular Vectors
 * @param num_threads Number of threads to use
 */
void svd_decomposition_parallel(int m, int n, const double* A, double* U, double* S, double* V, int num_threads);

#endif // SVD_PARALLEL_H
