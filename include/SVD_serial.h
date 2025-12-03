#ifndef SVD_SERIAL_H
#define SVD_SERIAL_H

/**
 * @brief Computes SVD: A = U * S * V^T (Serial Version)
 * @param m Rows of A
 * @param n Cols of A
 * @param A Input matrix (flat array m*n)
 * @param U Output matrix (m*n) - Left Singular Vectors
 * @param S Output array (n) - Singular Values
 * @param V Output matrix (n*n) - Right Singular Vectors
 */
void svd_decomposition(int m, int n, const double* A, double* U, double* S, double* V);

#endif // SVD_SERIAL_H
