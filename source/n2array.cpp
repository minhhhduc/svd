// Implementation of concrete double N2Array
#include "../include/n2array.h"
#include <cstring>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

N2Array::N2Array(double** darray, int* shp) {
    int* own_shape = new int[2];
    own_shape[0] = shp[0];
    own_shape[1] = shp[1];
    this->shape = own_shape;
    this->n2array = darray;
    this->n1array = nullptr;
}

N2Array::N2Array(double* darray, int* shp) {
    int* own_shape = new int[2];
    own_shape[0] = shp[0];
    own_shape[1] = shp[1];
    this->shape = own_shape;
    this->n1array = darray;
    this->n2array = nullptr;
}

N2Array::N2Array(const N2Array& other) {
    // Deep copy shape
    int* own_shape = new int[2];
    own_shape[0] = other.shape[0];
    own_shape[1] = other.shape[1];
    this->shape = own_shape;
    
    int rows = shape[0];
    int cols = shape[1];
    
    if (other.n2array) {
        // Deep copy 2D array
        this->n2array = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            this->n2array[i] = new double[cols];
            for (int j = 0; j < cols; ++j) {
                this->n2array[i][j] = other.n2array[i][j];
            }
        }
        this->n1array = nullptr;
    } else if (other.n1array) {
        // Deep copy 1D array
        this->n1array = new double[rows * cols];
        for (int i = 0; i < rows * cols; ++i) {
            this->n1array[i] = other.n1array[i];
        }
        this->n2array = nullptr;
    } else {
        this->n2array = nullptr;
        this->n1array = nullptr;
    }
}

N2Array::~N2Array() {
    if (n2array) {
        for (int i = 0; i < shape[0]; ++i) {
            delete[] n2array[i];
        }
        delete[] n2array;
        n2array = nullptr;
    }
    if (n1array) {
        delete[] n1array;
        n1array = nullptr;
    }
    if (shape) {
        delete[] const_cast<int*>(shape);
        shape = nullptr;
    }
}

N2Array N2Array::transpose() {
    int rows = shape[0];
    int cols = shape[1];
    double** transposed = new double*[cols];
    for (int i = 0; i < cols; ++i) transposed[i] = new double[rows];

    auto get = [&](int r, int c) -> double {
        if (n2array) return n2array[r][c];
        return n1array[r * cols + c];
    };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = get(i, j);
        }
    }

    int* new_shape = new int[2]{cols, rows};
    return N2Array(transposed, new_shape);
}

N2Array N2Array::operator+(const N2Array& other) {
    // Broadcasting logic: determine output shape and broadcasting rules
    int a_rows = shape[0], a_cols = shape[1];
    int b_rows = other.shape[0], b_cols = other.shape[1];
    
    // Determine output shape (broadcast to larger dimensions)
    int out_rows = std::max(a_rows, b_rows);
    int out_cols = std::max(a_cols, b_cols);
    
    // Check broadcasting compatibility
    if ((a_rows != 1 && a_rows != out_rows) || (a_cols != 1 && a_cols != out_cols) ||
        (b_rows != 1 && b_rows != out_rows) || (b_cols != 1 && b_cols != out_cols)) {
        throw std::invalid_argument("Shapes are not broadcastable for addition");
    }
    
    double** result = new double*[out_rows];
    for (int i = 0; i < out_rows; ++i) result[i] = new double[out_cols];

    auto getA = [&](int r, int c) -> double {
        int ar = (a_rows == 1) ? 0 : r;  // Broadcast along rows if needed
        int ac = (a_cols == 1) ? 0 : c;  // Broadcast along cols if needed
        return n2array ? n2array[ar][ac] : n1array[ar * a_cols + ac];
    };
    auto getB = [&](int r, int c) -> double {
        int br = (b_rows == 1) ? 0 : r;  // Broadcast along rows if needed
        int bc = (b_cols == 1) ? 0 : c;  // Broadcast along cols if needed
        return other.n2array ? other.n2array[br][bc] : other.n1array[br * b_cols + bc];
    };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            result[i][j] = getA(i, j) + getB(i, j);
        }
    }
    int* new_shape = new int[2]{out_rows, out_cols};
    return N2Array(result, new_shape);
}

N2Array N2Array::operator-(const N2Array& other) {
    // Broadcasting logic: determine output shape and broadcasting rules
    int a_rows = shape[0], a_cols = shape[1];
    int b_rows = other.shape[0], b_cols = other.shape[1];
    
    // Determine output shape (broadcast to larger dimensions)
    int out_rows = std::max(a_rows, b_rows);
    int out_cols = std::max(a_cols, b_cols);
    
    // Check broadcasting compatibility
    if ((a_rows != 1 && a_rows != out_rows) || (a_cols != 1 && a_cols != out_cols) ||
        (b_rows != 1 && b_rows != out_rows) || (b_cols != 1 && b_cols != out_cols)) {
        throw std::invalid_argument("Shapes are not broadcastable for subtraction");
    }
    
    double** result = new double*[out_rows];
    for (int i = 0; i < out_rows; ++i) result[i] = new double[out_cols];

    auto getA = [&](int r, int c) -> double {
        int ar = (a_rows == 1) ? 0 : r;  // Broadcast along rows if needed
        int ac = (a_cols == 1) ? 0 : c;  // Broadcast along cols if needed
        return n2array ? n2array[ar][ac] : n1array[ar * a_cols + ac];
    };
    auto getB = [&](int r, int c) -> double {
        int br = (b_rows == 1) ? 0 : r;  // Broadcast along rows if needed
        int bc = (b_cols == 1) ? 0 : c;  // Broadcast along cols if needed
        return other.n2array ? other.n2array[br][bc] : other.n1array[br * b_cols + bc];
    };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            result[i][j] = getA(i, j) - getB(i, j);
        }
    }
    int* new_shape = new int[2]{out_rows, out_cols};
    return N2Array(result, new_shape);
}

N2Array N2Array::operator*(const N2Array& other) {
    // Broadcasting logic: determine output shape and broadcasting rules
    int a_rows = shape[0], a_cols = shape[1];
    int b_rows = other.shape[0], b_cols = other.shape[1];
    
    // Determine output shape (broadcast to larger dimensions)
    int out_rows = std::max(a_rows, b_rows);
    int out_cols = std::max(a_cols, b_cols);
    
    // Check broadcasting compatibility
    if ((a_rows != 1 && a_rows != out_rows) || (a_cols != 1 && a_cols != out_cols) ||
        (b_rows != 1 && b_rows != out_rows) || (b_cols != 1 && b_cols != out_cols)) {
        throw std::invalid_argument("Shapes are not broadcastable for multiplication");
    }
    
    double** result = new double*[out_rows];
    for (int i = 0; i < out_rows; ++i) result[i] = new double[out_cols];

    auto getA = [&](int r, int c) -> double {
        int ar = (a_rows == 1) ? 0 : r;  // Broadcast along rows if needed
        int ac = (a_cols == 1) ? 0 : c;  // Broadcast along cols if needed
        return n2array ? n2array[ar][ac] : n1array[ar * a_cols + ac];
    };
    auto getB = [&](int r, int c) -> double {
        int br = (b_rows == 1) ? 0 : r;  // Broadcast along rows if needed
        int bc = (b_cols == 1) ? 0 : c;  // Broadcast along cols if needed
        return other.n2array ? other.n2array[br][bc] : other.n1array[br * b_cols + bc];
    };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            result[i][j] = getA(i, j) * getB(i, j);
        }
    }
    int* new_shape = new int[2]{out_rows, out_cols};
    return N2Array(result, new_shape);
}

N2Array N2Array::operator/(const N2Array& other) {
    // Broadcasting logic: determine output shape and broadcasting rules
    int a_rows = shape[0], a_cols = shape[1];
    int b_rows = other.shape[0], b_cols = other.shape[1];
    
    // Determine output shape (broadcast to larger dimensions)
    int out_rows = std::max(a_rows, b_rows);
    int out_cols = std::max(a_cols, b_cols);
    
    // Check broadcasting compatibility
    if ((a_rows != 1 && a_rows != out_rows) || (a_cols != 1 && a_cols != out_cols) ||
        (b_rows != 1 && b_rows != out_rows) || (b_cols != 1 && b_cols != out_cols)) {
        throw std::invalid_argument("Shapes are not broadcastable for division");
    }
    
    double** result = new double*[out_rows];
    for (int i = 0; i < out_rows; ++i) result[i] = new double[out_cols];

    auto getA = [&](int r, int c) -> double {
        int ar = (a_rows == 1) ? 0 : r;  // Broadcast along rows if needed
        int ac = (a_cols == 1) ? 0 : c;  // Broadcast along cols if needed
        return n2array ? n2array[ar][ac] : n1array[ar * a_cols + ac];
    };
    auto getB = [&](int r, int c) -> double {
        int br = (b_rows == 1) ? 0 : r;  // Broadcast along rows if needed
        int bc = (b_cols == 1) ? 0 : c;  // Broadcast along cols if needed
        return other.n2array ? other.n2array[br][bc] : other.n1array[br * b_cols + bc];
    };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            double divisor = getB(i, j);
            if (divisor == static_cast<double>(0)) {
                throw std::invalid_argument("Division by zero");
            }
            result[i][j] = getA(i, j) / divisor;
        }
    }
    int* new_shape = new int[2]{out_rows, out_cols};
    return N2Array(result, new_shape);
}

N2Array N2Array::operator+(const double& scalar) {
    int rows = shape[0];
    int cols = shape[1];
    double** result = new double*[rows];
    for (int i = 0; i < rows; ++i) result[i] = new double[cols];
    auto getA = [&](int r, int c) -> double { return n2array ? n2array[r][c] : n1array[r * cols + c]; };
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) result[i][j] = getA(i,j) + scalar;
    int* new_shape = new int[2]{rows, cols};
    return N2Array(result, new_shape);
}

N2Array N2Array::operator-(const double& scalar) {
    int rows = shape[0];
    int cols = shape[1];
    double** result = new double*[rows];
    for (int i = 0; i < rows; ++i) result[i] = new double[cols];
    auto getA = [&](int r, int c) -> double { return n2array ? n2array[r][c] : n1array[r * cols + c]; };
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) result[i][j] = getA(i,j) - scalar;
    int* new_shape = new int[2]{rows, cols};
    return N2Array(result, new_shape);
}

N2Array N2Array::operator*(const double& scalar) {
    int rows = shape[0];
    int cols = shape[1];
    double** result = new double*[rows];
    for (int i = 0; i < rows; ++i) result[i] = new double[cols];
    auto getA = [&](int r, int c) -> double { return n2array ? n2array[r][c] : n1array[r * cols + c]; };
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) result[i][j] = getA(i,j) * scalar;
    int* new_shape = new int[2]{rows, cols};
    return N2Array(result, new_shape);
}

N2Array N2Array::operator/(const double& scalar) {
    if (scalar == static_cast<double>(0)) {
        throw std::invalid_argument("Division by zero");
    }
    int rows = shape[0];
    int cols = shape[1];
    double** result = new double*[rows];
    for (int i = 0; i < rows; ++i) result[i] = new double[cols];
    auto getA = [&](int r, int c) -> double { return n2array ? n2array[r][c] : n1array[r * cols + c]; };
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) result[i][j] = getA(i,j) / scalar;
    int* new_shape = new int[2]{rows, cols};
    return N2Array(result, new_shape);
}

N2Array N2Array::operator=(const N2Array& other) {
    if (shape[0] != other.shape[0] || shape[1] != other.shape[1]) {
        throw std::invalid_argument("Shapes do not match for assignment");
    }
    int rows = shape[0];
    int cols = shape[1];

    if (n2array) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                n2array[i][j] = other.n2array ? other.n2array[i][j] : other.n1array[i * cols + j];
            }
        }
    } else if (n1array) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                n1array[i * cols + j] = other.n2array ? other.n2array[i][j]
                                                      : other.n1array[i * cols + j];
            }
        }
    }
    // Return a copy by value per header signature
    return *this;
}

bool N2Array::operator==(const N2Array& other) {
    if (shape[0] != other.shape[0] || shape[1] != other.shape[1]) {
        return false;
    }
    int rows = shape[0];
    int cols = shape[1];

    auto getA = [&](int r, int c) -> double {
        return n2array ? n2array[r][c] : n1array[r * cols + c];
    };
    auto getB = [&](int r, int c) -> double {
        return other.n2array ? other.n2array[r][c] : other.n1array[r * cols + c];
    };

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (getA(i, j) != getB(i, j)) return false;
        }
    }
    return true;
}

bool N2Array::operator!=(const N2Array& other) {
    return !(*this == other);
}

N2Array N2Array::operator[](int index) {
    if (index < 0 || index >= shape[0]) {
        throw std::out_of_range("Index out of range");
    }
    int cols = shape[1];
    double** row = new double*[1];
    row[0] = new double[cols];

    auto get = [&](int c) -> double {
        return n2array ? n2array[index][c] : n1array[index * cols + c];
    };
    for (int j = 0; j < cols; ++j) row[0][j] = get(j);

    int* new_shape = new int[2]{1, cols};
    return N2Array(row, new_shape);
}

N2Array N2Array::operator[](int* indices) {
    int rowIdx = indices[0];
    int colIdx = indices[1];
    if (rowIdx < 0 || rowIdx >= shape[0] || colIdx < 0 || colIdx >= shape[1]) {
        throw std::out_of_range("Index out of range");
    }
    double** element = new double*[1];
    element[0] = new double[1];
    if (n2array) element[0][0] = n2array[rowIdx][colIdx];
    else element[0][0] = n1array[rowIdx * shape[1] + colIdx];

    int* new_shape = new int[2]{1, 1};
    return N2Array(element, new_shape);
}

double* N2Array::get(int r, int c) {
    if (r < 0 || r >= shape[0] || c < 0 || c >= shape[1]) {
        throw std::out_of_range("Index out of range");
    }
    if (n2array) return &n2array[r][c];
    return &n1array[r * shape[1] + c];
}

const double* N2Array::get(int r, int c) const {
    if (r < 0 || r >= shape[0] || c < 0 || c >= shape[1]) {
        throw std::out_of_range("Index out of range");
    }
    if (n2array) return &n2array[r][c];
    return &n1array[r * shape[1] + c];
}

double** N2Array::toArray() {
    int rows = shape[0];
    int cols = shape[1];
    double** array = new double*[rows];
    for (int i = 0; i < rows; ++i) array[i] = new double[cols];

    auto get = [&](int r, int c) -> double {
        if (n2array) return n2array[r][c];
        return n1array[r * cols + c];
    };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            array[i][j] = get(i, j);
        }
    }
    return array;
}

char* N2Array::toString() {
    int rows = shape[0];
    int cols = shape[1];
    auto get = [&](int r, int c) -> double {
        return n2array ? n2array[r][c] : n1array[r * cols + c];
    };

    std::string result = "[";
    for (int i = 0; i < rows; ++i) {
        result += "[";
        for (int j = 0; j < cols; ++j) {
            result += std::to_string(get(i, j));
            if (j < cols - 1) result += ", ";
        }
        result += "]";
        if (i < rows - 1) result += ",\n ";
    }
    result += "]";

    char* cstr = new char[result.size() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}
