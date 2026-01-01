#ifndef NORM_REDUCING_JACOBI_PARALLEL_H
#define NORM_REDUCING_JACOBI_PARALLEL_H

/**
 * @brief Compute eigenvalues and eigenvectors using Parallel Symmetric Jacobi Algorithm
 * 
 * @param n Dimension of the matrix (n x n)
 * @param A_in Input symmetric matrix (flat array n*n)
 * @param w Output eigenvalues (array of size n)
 * @param V_out Output eigenvectors (flat array n*n, stored as columns)
 */
void compute_eigenvalues_parallel(int n, const double* A_in, double* w, double* V_out);

#endif // NORM_REDUCING_JACOBI_PARALLEL_H
