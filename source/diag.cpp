#include "../include/n2array.h"
#include "../include/diag.h"
#include <stdexcept>
#include <cstring>

using namespace std;

// Build a size x size diagonal matrix from a 1D buffer of length size
static N2Array build_diag_from_1d(const double* buf, int size) {
    if (size <= 0) throw invalid_argument("size must be positive");
    double** result = new double*[size];
    for (int i = 0; i < size; ++i) {
        result[i] = new double[size];
        // zero initialize row
        std::memset(result[i], 0, sizeof(double) * size);
        result[i][i] = buf[i];
    }
    int* new_shape = new int[2]{size, size};
    return N2Array(result, new_shape);
}

// Extract the diagonal of a square matrix (size x size) into a 1 x size row vector
static N2Array extract_diag_from_square(const double** darray, int size) {
    if (size <= 0) throw invalid_argument("size must be positive");
    double** result = new double*[1];
    result[0] = new double[size];
    for (int i = 0; i < size; ++i) {
        result[0][i] = darray[i][i];
    }
    int* new_shape = new int[2]{1, size};
    return N2Array(result, new_shape);
}

N2Array diag(const N2Array& a) {
    if (!a.shape) throw invalid_argument("Null shape");
    int rows = a.shape[0];
    int cols = a.shape[1];

    // If input looks like 1D (1xN or Nx1), build diagonal matrix
    if (rows == 1 || cols == 1) {
        int size = (rows == 1) ? cols : rows;
        // collect values into a contiguous 1D buffer if needed
        double* buf = nullptr;
        bool allocated = false;
        if (a.n1array) {
            buf = a.n1array;
        } else if (a.n2array) {
            allocated = true;
            buf = new double[size];
            if (rows == 1) {
                for (int j = 0; j < size; ++j) buf[j] = a.n2array[0][j];
            } else { // cols == 1
                for (int i = 0; i < size; ++i) buf[i] = a.n2array[i][0];
            }
        } else {
            throw invalid_argument("Invalid N2Array storage");
        }
        N2Array out = build_diag_from_1d(buf, size);
        if (allocated) delete[] buf;
        return out;
    }

    // If square, extract diagonal to 1 x N row vector
    if (rows == cols) {
        if (!a.n2array) {
            // convert 1D storage to 2D temporary view to reuse function
            double** tmp = new double*[rows];
            for (int i = 0; i < rows; ++i) {
                tmp[i] = new double[cols];
                for (int j = 0; j < cols; ++j) tmp[i][j] = a.n1array[i * cols + j];
            }
            const double** ctmp = const_cast<const double**>(tmp);
            N2Array out = extract_diag_from_square(ctmp, rows);
            for (int i = 0; i < rows; ++i) delete[] tmp[i];
            delete[] tmp;
            return out;
        }
        return extract_diag_from_square(const_cast<const double**>(a.n2array), rows);
    }

    throw invalid_argument("Input must be 1D (1xN or Nx1) or square 2D array");
}

N2Array diag(const double** darray, int* shape) {
    if (!shape) throw invalid_argument("Null shape");
    int rows = shape[0];
    int cols = shape[1];

    if (rows == 1 && cols >= 1) {
        // treat darray[0] as 1D buffer of length cols
        return build_diag_from_1d(darray[0], cols);
    }
    if (cols == 1 && rows >= 1) {
        // build 1D buffer from first column then create diagonal matrix
        double* buf = new double[rows];
        for (int i = 0; i < rows; ++i) buf[i] = darray[i][0];
        N2Array out = build_diag_from_1d(buf, rows);
        delete[] buf;
        return out;
    }
    if (rows == cols && rows > 0) {
        return extract_diag_from_square(darray, rows);
    }
    throw invalid_argument("Input must be 1D (1xN or Nx1) or square 2D array");
}
