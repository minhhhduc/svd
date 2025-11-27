#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOLERANCE 1e-15
#define MAX_ITERATIONS 100

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define AT(arr, r, c, n) (arr)[(r)*(n) + (c)]

// Classical Jacobi method for symmetric real matrices
// Reference: Golub & Van Loan, Matrix Computations
void jacobi_symmetric(int n, double* A, double* eigenvalues, double* eigenvectors) {
    // Copy A to working matrix
    double* a = (double*)malloc(n * n * sizeof(double));
    double* v = (double*)malloc(n * n * sizeof(double));
    
    for (int i = 0; i < n * n; ++i) {
        a[i] = A[i];
    }
    
    // Initialize eigenvectors to identity
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            AT(v, i, j, n) = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Jacobi iterations
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        // Find largest off-diagonal element
        double max_val = 0.0;
        int p = 0, q = 1;
        
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double val = fabs(AT(a, i, j, n));
                if (val > max_val) {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        
        // Check convergence
        if (max_val < TOLERANCE) {
            printf("Converged at iteration %d, max off-diagonal: %.6e\n", iter, max_val);
            break;
        }
        
        if (iter < 5 || iter % 10 == 0) {
            printf("Iteration %d: max off-diagonal = %.6e at (%d,%d)\n", iter, max_val, p, q);
        }
        
        // Calculate rotation angle
        double a_pp = AT(a, p, p, n);
        double a_qq = AT(a, q, q, n);
        double a_pq = AT(a, p, q, n);
        
        double theta;
        if (fabs(a_pp - a_qq) < TOLERANCE) {
            theta = M_PI / 4.0;
        } else {
            theta = 0.5 * atan2(2.0 * a_pq, a_qq - a_pp);
        }
        
        double c = cos(theta);
        double s = sin(theta);
        
        // Apply rotation to A: A' = J^T * A * J
        // Update rows and columns p, q
        for (int i = 0; i < n; ++i) {
            if (i != p && i != q) {
                double a_ip = AT(a, i, p, n);
                double a_iq = AT(a, i, q, n);
                AT(a, i, p, n) = c * a_ip - s * a_iq;
                AT(a, i, q, n) = s * a_ip + c * a_iq;
                AT(a, p, i, n) = AT(a, i, p, n);
                AT(a, q, i, n) = AT(a, i, q, n);
            }
        }
        
        // Update diagonal elements
        double new_a_pp = c*c*a_pp + s*s*a_qq - 2.0*s*c*a_pq;
        double new_a_qq = s*s*a_pp + c*c*a_qq + 2.0*s*c*a_pq;
        AT(a, p, p, n) = new_a_pp;
        AT(a, q, q, n) = new_a_qq;
        AT(a, p, q, n) = 0.0;
        AT(a, q, p, n) = 0.0;
        
        // Update eigenvectors: V = V * J
        for (int i = 0; i < n; ++i) {
            double v_ip = AT(v, i, p, n);
            double v_iq = AT(v, i, q, n);
            AT(v, i, p, n) = c * v_ip - s * v_iq;
            AT(v, i, q, n) = s * v_ip + c * v_iq;
        }
    }
    
    // Extract eigenvalues and eigenvectors
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = AT(a, i, i, n);
    }
    
    for (int i = 0; i < n * n; ++i) {
        eigenvectors[i] = v[i];
    }
    
    free(a);
    free(v);
}

int main() {
    int n = 3;
    
    // Symmetric tridiagonal matrix
    // Expected eigenvalues: 4-sqrt(2), 4, 4+sqrt(2) ≈ 2.586, 4.0, 5.414
    double A[9] = {
        4.0, 1.0, 0.0,
        1.0, 4.0, 1.0,
        0.0, 1.0, 4.0
    };
    
    double A_original[9];
    for (int i = 0; i < 9; ++i) {
        A_original[i] = A[i];
    }
    
    double eigenvalues[3];
    double eigenvectors[9];
    
    printf("Input matrix:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.2f  ", AT(A, i, j, n));
        }
        printf("\n");
    }
    printf("\n");
    
    jacobi_symmetric(n, A, eigenvalues, eigenvectors);
    
    printf("\n=== RESULTS ===\n");
    printf("Eigenvalues:\n");
    for (int i = 0; i < n; ++i) {
        printf("  lambda[%d] = %.10f\n", i, eigenvalues[i]);
    }
    
    printf("\nEigenvectors (columns):\n");
    for (int i = 0; i < n; ++i) {
        printf("  ");
        for (int j = 0; j < n; ++j) {
            printf("%8.4f  ", AT(eigenvectors, i, j, n));
        }
        printf("\n");
    }
    
    // Verification: A * v = lambda * v
    printf("\n=== VERIFICATION ===\n");
    double max_error = 0.0;
    
    for (int j = 0; j < n; ++j) {
        printf("\nEigenvector %d (lambda = %.10f):\n", j, eigenvalues[j]);
        
        double Av[3], lambda_v[3];
        for (int i = 0; i < n; ++i) {
            Av[i] = 0.0;
            for (int k = 0; k < n; ++k) {
                Av[i] += AT(A_original, i, k, n) * AT(eigenvectors, k, j, n);
            }
            lambda_v[i] = eigenvalues[j] * AT(eigenvectors, i, j, n);
        }
        
        double error = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = fabs(Av[i] - lambda_v[i]);
            error += diff * diff;
            printf("  A*v[%d] = %10.6f, lambda*v[%d] = %10.6f, diff = %.2e\n",
                   i, Av[i], i, lambda_v[i], diff);
        }
        error = sqrt(error);
        printf("  ||A*v - lambda*v|| = %.6e\n", error);
        
        if (error > max_error) max_error = error;
    }
    
    printf("\n=== SUMMARY ===\n");
    printf("Maximum error: %.6e\n", max_error);
    printf("Expected eigenvalues: 2.585786, 4.000000, 5.414214\n");
    
    if (max_error < 1e-10) {
        printf("✓ PASS: Results are correct!\n");
    } else if (max_error < 1e-6) {
        printf("⚠ WARNING: Acceptable accuracy\n");
    } else {
        printf("✗ FAIL: Results are incorrect\n");
    }
    
    return 0;
}
