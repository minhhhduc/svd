// Implementation for template class N2Array<T>
// Note: As a template, all definitions live in this .tpp included by the header.

#include <cstring>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
N2Array<T>::N2Array(T** darray, int* shp) {
	// Deep-copy shape (assumes 2D: rows x cols)
	int* own_shape = new int[2];
	own_shape[0] = shp[0];
	own_shape[1] = shp[1];
	this->shape = own_shape;
	// Take ownership of the provided 2D data pointer as-is
	this->n2array = darray;
	this->n1array = nullptr;
}

template <typename T>
N2Array<T>::N2Array(T* darray, int* shp) {
	int* own_shape = new int[2];
	own_shape[0] = shp[0];
	own_shape[1] = shp[1];
	this->shape = own_shape;
	// Take ownership of the provided 1D data pointer as-is
	this->n1array = darray;
	this->n2array = nullptr;
}

template <typename T>
N2Array<T>::~N2Array() {
	if (n2array) {
		// Delete row-by-row then the row pointer array
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
		// shape was deep-copied, so we own it
		delete[] const_cast<int*>(shape);
		shape = nullptr;
	}
}

template <typename T>
N2Array<T> N2Array<T>::transpose() {
	int rows = shape[0];
	int cols = shape[1];
	// Allocate transposed 2D array
	T** transposed = new T*[cols];
	for (int i = 0; i < cols; ++i) transposed[i] = new T[rows];

	auto get = [&](int r, int c) -> T {
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
	return N2Array<T>(transposed, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator+(const N2Array& other) {
	if (shape[0] != other.shape[0] || shape[1] != other.shape[1]) {
		throw std::invalid_argument("Shapes do not match for addition");
	}
	int rows = shape[0];
	int cols = shape[1];
	T** result = new T*[rows];
	for (int i = 0; i < rows; ++i) result[i] = new T[cols];

	auto getA = [&](int r, int c) -> T {
		return n2array ? n2array[r][c] : n1array[r * cols + c];
	};
	auto getB = [&](int r, int c) -> T {
		return other.n2array ? other.n2array[r][c] : other.n1array[r * cols + c];
	};

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result[i][j] = getA(i, j) + getB(i, j);
		}
	}
	int* new_shape = new int[2]{rows, cols};
	return N2Array<T>(result, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator-(const N2Array& other) {
	if (shape[0] != other.shape[0] || shape[1] != other.shape[1]) {
		throw std::invalid_argument("Shapes do not match for subtraction");
	}
	int rows = shape[0];
	int cols = shape[1];
	T** result = new T*[rows];
	for (int i = 0; i < rows; ++i) result[i] = new T[cols];

	auto getA = [&](int r, int c) -> T {
		return n2array ? n2array[r][c] : n1array[r * cols + c];
	};
	auto getB = [&](int r, int c) -> T {
		return other.n2array ? other.n2array[r][c] : other.n1array[r * cols + c];
	};

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result[i][j] = getA(i, j) - getB(i, j);
		}
	}
	int* new_shape = new int[2]{rows, cols};
	return N2Array<T>(result, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator*(const N2Array& other) {
	if (shape[0] != other.shape[0] || shape[1] != other.shape[1]) {
		throw std::invalid_argument("Shapes do not match for multiplication");
	}
	int rows = shape[0];
	int cols = shape[1];
	T** result = new T*[rows];
	for (int i = 0; i < rows; ++i) result[i] = new T[cols];

	auto getA = [&](int r, int c) -> T {
		return n2array ? n2array[r][c] : n1array[r * cols + c];
	};
	auto getB = [&](int r, int c) -> T {
		return other.n2array ? other.n2array[r][c] : other.n1array[r * cols + c];
	};

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result[i][j] = getA(i, j) * getB(i, j);
		}
	}
	int* new_shape = new int[2]{rows, cols};
	return N2Array<T>(result, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator/(const N2Array& other) {
	if (shape[0] != other.shape[0] || shape[1] != other.shape[1]) {
		throw std::invalid_argument("Shapes do not match for division");
	}
	int rows = shape[0];
	int cols = shape[1];
	T** result = new T*[rows];
	for (int i = 0; i < rows; ++i) result[i] = new T[cols];

	auto getA = [&](int r, int c) -> T {
		return n2array ? n2array[r][c] : n1array[r * cols + c];
	};
	auto getB = [&](int r, int c) -> T {
		return other.n2array ? other.n2array[r][c] : other.n1array[r * cols + c];
	};

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (getB(i, j) == static_cast<T>(0)) {
				throw std::invalid_argument("Division by zero");
			}
			result[i][j] = getA(i, j) / getB(i, j);
		}
	}
	int* new_shape = new int[2]{rows, cols};
	return N2Array<T>(result, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator+(const T& scalar) {
	int rows = shape[0];
	int cols = shape[1];
	T** result = new T*[rows];
	for (int i = 0; i < rows; ++i) result[i] = new T[cols];

	auto getA = [&](int r, int c) -> T {
		return n2array ? n2array[r][c] : n1array[r * cols + c];
	};

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result[i][j] = getA(i, j) + scalar;
		}
	}
	int* new_shape = new int[2]{rows, cols};
	return N2Array<T>(result, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator-(const T& scalar) {
	int rows = shape[0];
	int cols = shape[1];
	T** result = new T*[rows];
	for (int i = 0; i < rows; ++i) result[i] = new T[cols];

	auto getA = [&](int r, int c) -> T {
		return n2array ? n2array[r][c] : n1array[r * cols + c];
	};

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result[i][j] = getA(i, j) - scalar;
		}
	}
	int* new_shape = new int[2]{rows, cols};
	return N2Array<T>(result, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator*(const T& scalar) {
	int rows = shape[0];
	int cols = shape[1];
	T** result = new T*[rows];
	for (int i = 0; i < rows; ++i) result[i] = new T[cols];

	auto getA = [&](int r, int c) -> T {
		return n2array ? n2array[r][c] : n1array[r * cols + c];
	};

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result[i][j] = getA(i, j) * scalar;
		}
	}
	int* new_shape = new int[2]{rows, cols};
	return N2Array<T>(result, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator/(const T& scalar) {
	if (scalar == static_cast<T>(0)) {
		throw std::invalid_argument("Division by zero");
	}
	int rows = shape[0];
	int cols = shape[1];
	T** result = new T*[rows];
	for (int i = 0; i < rows; ++i) result[i] = new T[cols];

	auto getA = [&](int r, int c) -> T {
		return n2array ? n2array[r][c] : n1array[r * cols + c];
	};

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result[i][j] = getA(i, j) / scalar;
		}
	}
	int* new_shape = new int[2]{rows, cols};
	return N2Array<T>(result, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator=(const N2Array& other) {
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
				n2array[i][j] = other.n2array ? other.n2array[i][j]
											  : other.n1array[i * cols + j];
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

template <typename T>
bool N2Array<T>::operator==(const N2Array& other) {
	if (shape[0] != other.shape[0] || shape[1] != other.shape[1]) {
		return false;
	}
	int rows = shape[0];
	int cols = shape[1];

	auto getA = [&](int r, int c) -> T {
		return n2array ? n2array[r][c] : n1array[r * cols + c];
	};
	auto getB = [&](int r, int c) -> T {
		return other.n2array ? other.n2array[r][c] : other.n1array[r * cols + c];
	};

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (getA(i, j) != getB(i, j)) return false;
		}
	}
	return true;
}

template <typename T>
bool N2Array<T>::operator!=(const N2Array& other) {
	return !(*this == other);
}

template <typename T>
N2Array<T> N2Array<T>::operator[](int index) {
	if (index < 0 || index >= shape[0]) {
		throw std::out_of_range("Index out of range");
	}
	int cols = shape[1];
	// Deep copy the selected row into a new 2D array of shape 1 x cols
	T** row = new T*[1];
	row[0] = new T[cols];

	auto get = [&](int c) -> T {
		return n2array ? n2array[index][c] : n1array[index * cols + c];
	};
	for (int j = 0; j < cols; ++j) row[0][j] = get(j);

	int* new_shape = new int[2]{1, cols};
	return N2Array<T>(row, new_shape);
}

template <typename T>
N2Array<T> N2Array<T>::operator[](int* indices) {
	// Expect exactly two indices: [row, col]
	int rowIdx = indices[0];
	int colIdx = indices[1];
	if (rowIdx < 0 || rowIdx >= shape[0] || colIdx < 0 || colIdx >= shape[1]) {
		throw std::out_of_range("Index out of range");
	}
	// Return a 1x1 array containing the element
	T** element = new T*[1];
	element[0] = new T[1];
	if (n2array) element[0][0] = n2array[rowIdx][colIdx];
	else element[0][0] = n1array[rowIdx * shape[1] + colIdx];

	int* new_shape = new int[2]{1, 1};
	return N2Array<T>(element, new_shape);
}

template <typename T>
char* N2Array<T>::toString() {
	int rows = shape[0];
	int cols = shape[1];
	auto get = [&](int r, int c) -> T {
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

