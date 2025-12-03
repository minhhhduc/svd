#ifndef DECOMPOSE_OPERATION_H
#define DECOMPOSE_OPERATION_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs dimensionality reduction using SVD (Serial).
 *        Computes A = U S V^T, then projects A onto the first k principal components: Result = A * V[:, :k].
 * 
 * @param m Rows of input matrix A.
 * @param n Columns of input matrix A.
 * @param k Target dimension (number of components).
 * @param A Input matrix (flattened 1D array of size m*n).
 * @param Result Output matrix (flattened 1D array of size m*k).
 */
void decompose_project_serial(int m, int n, int k, const double* A, double* Result);

/**
 * @brief Performs dimensionality reduction using SVD (Parallel).
 *        Computes A = U S V^T, then projects A onto the first k principal components: Result = A * V[:, :k].
 * 
 * @param m Rows of input matrix A.
 * @param n Columns of input matrix A.
 * @param k Target dimension (number of components).
 * @param A Input matrix (flattened 1D array of size m*n).
 * @param Result Output matrix (flattened 1D array of size m*k).
 * @param num_threads Number of threads to use.
 */
void decompose_project_parallel(int m, int n, int k, const double* A, double* Result, int num_threads);

#ifdef __cplusplus
}
#endif

#endif // DECOMPOSE_OPERATION_H
