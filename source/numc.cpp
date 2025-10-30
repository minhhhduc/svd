// Concrete implementations for numerical helpers operating on N2Array (double)
#include "../include/numc.h"
#include "../include/n2array.h"
#include <cstring>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Generates an array of evenly spaced double values within a specified range.
 *
 * This function creates a dynamically allocated array of doubles, starting from `start` (inclusive),
 * incrementing by `step`, and stopping before reaching `stop`. The number of elements is determined
 * by the formula: n = (stop - start) / step. If the computed number of elements is less than or equal to zero,
 * the function returns nullptr.
 *
 * @param start The starting value of the sequence (inclusive).
 * @param stop The end value of the sequence (exclusive).
 * @param step The increment between consecutive values.
 * @return Pointer to the dynamically allocated array of doubles, or nullptr if the range is invalid.
 */
double* NumC::arange(double start, double stop, double step) {
    if (step <= 0) throw std::invalid_argument("Step must be positive");
    int n = static_cast<int>((stop - start) / step);
    if (n <= 0) return nullptr;
    
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

/**
 * @brief Generates an array of `num` evenly spaced double values between `start` and `stop`, inclusive.
 * 
 * This function creates a dynamically allocated array of doubles, starting from `start` and ending at `stop`,
 * with a total of `num` elements. If `num` is less than or equal to zero, the function returns nullptr. If `num` is 1,
 * the function returns an array containing only the `start` value.
 * 
 * @param start The starting value of the sequence.
 * @param stop The ending value of the sequence.
 * @param num The number of evenly spaced values to generate.
 * @return Pointer to the dynamically allocated array of doubles, or nullptr if `num` is
 */
double* NumC::linspace(double start, double stop, int num) {
    if (num <= 0) throw std::invalid_argument("num must be positive");
    if (num == 1) {
        double* result = new double[1];
        result[0] = start;
        return result;
    }
    
    double* result = new double[num];
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

/**
 * @brief Creates a 2D N2Array of the specified shape, filled with zeros.
 * 
 * This function allocates a 2D array of doubles with the given number of rows and columns,
 * initializing all elements to 0.0. The resulting array is wrapped in an N2Array object.
 * 
 * @param rows The number of rows in the array.
 * @param cols The number of columns in the array.
 * @return An N2Array object containing the zero-filled array.
 */
N2Array NumC::zeros(int rows, int cols) {
    double** data = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        data[i] = new double[cols];
        for (int j = 0; j < cols; ++j) {
            data[i][j] = 0.0;
        }
    }
    int* shape = new int[2]{rows, cols};
    return N2Array(data, shape);
}

/**
 * @brief Creates a 2D N2Array of the specified shape, filled with ones.
 * 
 * This function allocates a 2D array of doubles with the given number of rows and columns,
 * initializing all elements to 1.0. The resulting array is wrapped in an N2Array object.
 * 
 * @param rows The number of rows in the array.
 * @param cols The number of columns in the array.
 * @return An N2Array object containing the one-filled array.
 */
N2Array NumC::ones(int rows, int cols) {
    double** data = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        data[i] = new double[cols];
        for (int j = 0; j < cols; ++j) {
            data[i][j] = 1.0;
        }
    }
    int* shape = new int[2]{rows, cols};
    return N2Array(data, shape);
}

/**
 * @brief Computes the dot product of two N2Array objects.
 * 
 * This function performs matrix multiplication between two N2Array objects `a` and `b`.
 * The number of columns in `a` must match the number of rows in `b`.
 * The result is a new N2Array object containing the product.
 * 
 * @param a The first N2Array operand.
 * @param b The second N2Array operand.
 * @return An N2Array object containing the result of the dot product.
 * @throws std::invalid_argument if the shapes are incompatible for multiplication.
 */
N2Array NumC::dot(const N2Array& a, const N2Array& b) {
    if (!a.shape || !b.shape) throw std::invalid_argument("Null shape");
    int a_rows = a.shape[0];
    int a_cols = a.shape[1];
    int b_rows = b.shape[0];
    int b_cols = b.shape[1];
    if (a_cols != b_rows) throw std::invalid_argument("Inner dimensions must match for dot");

    int m = a_rows;
    int k = a_cols;
    int n = b_cols;

    double** result = new double*[m];
    for (int i = 0; i < m; ++i) result[i] = new double[n];

    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * a_cols + c]; };
    auto getb = [&](int r, int c) -> double { return b.n2array ? b.n2array[r][c] : b.n1array[r * b_cols + c]; };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sumv = 0.0;
            for (int t = 0; t < k; ++t) {
                sumv += geta(i, t) * getb(t, j);
            }
            result[i][j] = sumv;
        }
    }

    int* shape = new int[2]{m, n};
    return N2Array(result, shape);
}

/**
 * @brief Computes the minimum value in the N2Array.
 * 
 * This function iterates through all elements of the N2Array `a` and finds the minimum value.
 * The result is returned as a new N2Array object with shape (1, 1).
 * 
 * @param a The N2Array to compute the minimum from.
 * @return An N2Array object containing the minimum value.
 * @throws std::invalid_argument if the input array has a null shape.
 */
N2Array NumC::min(const N2Array& a, int axis) {
    if (!a.shape) throw std::invalid_argument("Null shape");
    int rows = a.shape[0];
    int cols = a.shape[1];
    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * cols + c]; };

    if (axis == -1) {
        // Compute min of entire array
        double minv = geta(0,0);
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
            double v = geta(i,j);
            if (v < minv) minv = v;
        }
        double** out = new double*[1]; out[0] = new double[1]; out[0][0] = minv;
        int* shape = new int[2]{1,1};
        return N2Array(out, shape);
    } else if (axis == 0) {
        // Compute min along rows (result: 1 x cols)
        double** result = new double*[1];
        result[0] = new double[cols];
        for (int j = 0; j < cols; ++j) {
            double minv = geta(0, j);
            for (int i = 1; i < rows; ++i) {
                double v = geta(i, j);
                if (v < minv) minv = v;
            }
            result[0][j] = minv;
        }
        int* shape = new int[2]{1, cols};
        return N2Array(result, shape);
    } else if (axis == 1) {
        // Compute min along cols (result: rows x 1)
        double** result = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            result[i] = new double[1];
            double minv = geta(i, 0);
            for (int j = 1; j < cols; ++j) {
                double v = geta(i, j);
                if (v < minv) minv = v;
            }
            result[i][0] = minv;
        }
        int* shape = new int[2]{rows, 1};
        return N2Array(result, shape);
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}

/**
 * @brief Computes the maximum value in the N2Array.
 * 
 * This function iterates through all elements of the N2Array `a` and finds the maximum value.
 * The result is returned as a new N2Array object with shape (1, 1).
 * 
 * @param a The N2Array to compute the maximum from.
 * @return An N2Array object containing the maximum value.
 * @throws std::invalid_argument if the input array has a null shape.
 */
N2Array NumC::max(const N2Array& a, int axis) {
    if (!a.shape) throw std::invalid_argument("Null shape");
    int rows = a.shape[0];
    int cols = a.shape[1];
    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * cols + c]; };

    if (axis == -1) {
        // Compute max of entire array
        double maxv = geta(0,0);
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
            double v = geta(i,j);
            if (v > maxv) maxv = v;
        }
        double** out = new double*[1]; out[0] = new double[1]; out[0][0] = maxv;
        int* shape = new int[2]{1,1};
        return N2Array(out, shape);
    } else if (axis == 0) {
        // Compute max along rows (result: 1 x cols)
        double** result = new double*[1];
        result[0] = new double[cols];
        for (int j = 0; j < cols; ++j) {
            double maxv = geta(0, j);
            for (int i = 1; i < rows; ++i) {
                double v = geta(i, j);
                if (v > maxv) maxv = v;
            }
            result[0][j] = maxv;
        }
        int* shape = new int[2]{1, cols};
        return N2Array(result, shape);
    } else if (axis == 1) {
        // Compute max along cols (result: rows x 1)
        double** result = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            result[i] = new double[1];
            double maxv = geta(i, 0);
            for (int j = 1; j < cols; ++j) {
                double v = geta(i, j);
                if (v > maxv) maxv = v;
            }
            result[i][0] = maxv;
        }
        int* shape = new int[2]{rows, 1};
        return N2Array(result, shape);
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}

/**
 * @brief Computes the sum of elements in the N2Array.
 * 
 * This function sums the elements of the N2Array `a` along the specified axis.
 * If `axis` is -1, it sums all elements. If `axis` is 0, it sums along rows (resulting in a 1 x cols array).
 * If `axis` is 1, it sums along columns (resulting in a rows x 1 array).
 * 
 * @param a The N2Array to compute the sum from.
 * @param axis The axis along which to sum (-1 for all, 0 for rows, 1 for columns).
 * @return An N2Array object containing the sum.
 * @throws std::invalid_argument if the input array has a null shape or if the axis is invalid.
 */
N2Array NumC::sum(const N2Array& a, int axis) {
    if (!a.shape) throw std::invalid_argument("Null shape");
    int rows = a.shape[0];
    int cols = a.shape[1];
    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * cols + c]; };

    if (axis == -1) {
        // Compute sum of entire array
        double s = 0.0;
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) s += geta(i,j);
        double** out = new double*[1]; out[0] = new double[1]; out[0][0] = s;
        int* shape = new int[2]{1,1};
        return N2Array(out, shape);
    } else if (axis == 0) {
        // Compute sum along rows (result: 1 x cols)
        double** result = new double*[1];
        result[0] = new double[cols];
        for (int j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int i = 0; i < rows; ++i) {
                s += geta(i, j);
            }
            result[0][j] = s;
        }
        int* shape = new int[2]{1, cols};
        return N2Array(result, shape);
    } else if (axis == 1) {
        // Compute sum along cols (result: rows x 1)
        double** result = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            result[i] = new double[1];
            double s = 0.0;
            for (int j = 0; j < cols; ++j) {
                s += geta(i, j);
            }
            result[i][0] = s;
        }
        int* shape = new int[2]{rows, 1};
        return N2Array(result, shape);
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}

/**
 * @brief Computes the mean of elements in the N2Array along the specified axis.
 * 
 * This function calculates the mean of the N2Array `a` along the given axis.
 * If `axis` is -1, it computes the mean of all elements. If `axis` is 0, it computes the mean along rows (resulting in a 1 x cols array).
 * If `axis` is 1, it computes the mean along columns (resulting in a rows x 1 array).
 * 
 * @param a The N2Array to compute the mean from.
 * @param axis The axis along which to compute the mean (-1 for all, 0 for rows, 1 for columns).
 * @return An N2Array object containing the mean values.
 * @throws std::invalid_argument if the input array has a null shape or if the axis is invalid.
 */
N2Array NumC::mean(const N2Array& a, int axis) {
    int rows = a.shape[0];
    int cols = a.shape[1];
    
    if (axis == -1) {
        // Compute mean of entire array
        N2Array s = NumC::sum(a, -1);
        double val = (s.n2array ? s.n2array[0][0] : s.n1array[0]) / static_cast<double>(rows * cols);
        double** out = new double*[1]; out[0] = new double[1]; out[0][0] = val;
        int* shape = new int[2]{1,1};
        return N2Array(out, shape);
    } else if (axis == 0) {
        // Compute mean along rows (result: 1 x cols)
        N2Array s = NumC::sum(a, 0);
        double** result = new double*[1];
        result[0] = new double[cols];
        for (int j = 0; j < cols; ++j) {
            result[0][j] = (s.n2array ? s.n2array[0][j] : s.n1array[j]) / static_cast<double>(rows);
        }
        int* shape = new int[2]{1, cols};
        return N2Array(result, shape);
    } else if (axis == 1) {
        // Compute mean along cols (result: rows x 1)
        N2Array s = NumC::sum(a, 1);
        double** result = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            result[i] = new double[1];
            result[i][0] = (s.n2array ? s.n2array[i][0] : s.n1array[i]) / static_cast<double>(cols);
        }
        int* shape = new int[2]{rows, 1};
        return N2Array(result, shape);
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}

/**
 * @brief Computes the standard deviation of elements in the N2Array along the specified axis.
 * 
 * This function calculates the standard deviation of the N2Array `a` along the given axis.
 * If `axis` is -1, it computes the standard deviation of all elements. If `axis` is 0, it computes the standard deviation along rows (resulting in a 1 x cols array).
 * If `axis` is 1, it computes the standard deviation along columns (resulting in a rows x 1 array).
 * 
 * @param a The N2Array to compute the standard deviation from.
 * @param axis The axis along which to compute the standard deviation (-1 for all, 0 for rows, 1 for columns).
 * @return An N2Array object containing the standard deviation values.
 * @throws std::invalid_argument if the input array has a null shape or if the axis is invalid.
 */
N2Array NumC::sd(const N2Array& a, int axis) {
    if (!a.shape) throw std::invalid_argument("Null shape");
    int rows = a.shape[0];
    int cols = a.shape[1];
    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * cols + c]; };

    if (axis == -1) {
        // Compute stdev of entire array
        int N = rows * cols;
        N2Array m = NumC::mean(a, -1);
        double meanv = m.n2array ? m.n2array[0][0] : m.n1array[0];

        double accum = 0.0;
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
            double d = geta(i,j) - meanv;
            accum += d * d;
        }
        double variance = accum / static_cast<double>(N);
        double sigma = std::sqrt(variance);

        double** out = new double*[1]; out[0] = new double[1]; out[0][0] = sigma;
        int* shape = new int[2]{1,1};
        return N2Array(out, shape);
    } else if (axis == 0) {
        // Compute stdev along rows (result: 1 x cols)
        N2Array m = NumC::mean(a, 0);
        double** result = new double*[1];
        result[0] = new double[cols];
        
        for (int j = 0; j < cols; ++j) {
            double meanv = m.n2array ? m.n2array[0][j] : m.n1array[j];
            double accum = 0.0;
            for (int i = 0; i < rows; ++i) {
                double d = geta(i, j) - meanv;
                accum += d * d;
            }
            double variance = accum / static_cast<double>(rows);
            result[0][j] = std::sqrt(variance);
        }
        int* shape = new int[2]{1, cols};
        return N2Array(result, shape);
    } else if (axis == 1) {
        // Compute stdev along cols (result: rows x 1)
        N2Array m = NumC::mean(a, 1);
        double** result = new double*[rows];
        
        for (int i = 0; i < rows; ++i) {
            result[i] = new double[1];
            double meanv = m.n2array ? m.n2array[i][0] : m.n1array[i];
            double accum = 0.0;
            for (int j = 0; j < cols; ++j) {
                double d = geta(i, j) - meanv;
                accum += d * d;
            }
            double variance = accum / static_cast<double>(cols);
            result[i][0] = std::sqrt(variance);
        }
        int* shape = new int[2]{rows, 1};
        return N2Array(result, shape);
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}