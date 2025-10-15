// Minimal demo: include N2Array and print a small matrix
#include "../include/n2array.h"
#include "../include/numc.h"
#include <iostream>
using namespace std;

N2Array testN2Array(double** data, int* shape) {
	// Create a 2x3 matrix using 2D allocation
	cout << "-------------------------------------------" << endl;
	cout << "****N2Array test:****\n";
	N2Array A(data, shape);
	cout << A.toString() << endl;

	// cout << "test operator (scalar):\n";
	// cout << (A*3.0).toString() << endl;
	
	// cout << "First row: " << A[0].toString() << endl;
	// cout << "Shape: " << A[0].shape[0] << " x " << A[0].shape[1] << endl;

	return A;
}

void testNumC(N2Array& A) {
	cout << "------------------------------------------" << endl;
	cout << "****numc test:****\n";
	NumC nc;
	double* arr1d = nc.arange(1.0, 7.0); // 1..6
	int* shape = new int[2]{2, 3};
	N2Array B(arr1d, shape);
	cout << "NumC arange 1..6 as 2x3:\n" << B.toString() << endl;
	
	// Test broadcasting operations
	cout << "\n****Broadcasting tests:****\n";
	cout << "A (2x3):\n" << A.toString() << endl;
	cout << "B (2x3):\n" << B.toString() << endl;
	cout << "A + B (same shape):\n" << (A + B).toString() << endl;
	
	// Test broadcasting with 1x3 array (row vector)
	NumC nc2;
	double* row_data = nc2.arange(10.0, 13.0); // [10, 11, 12]
	int* row_shape = new int[2]{1, 3};
	N2Array C(row_data, row_shape);
	cout << "\nC (1x3 row vector):\n" << C.toString() << endl;
	cout << "A + C (broadcasting 1x3 to 2x3):\n" << (A + C).toString() << endl;
	
	// Test broadcasting with 2x1 array (column vector)
	double** col_data = new double*[2];
	col_data[0] = new double[1]{100.0};
	col_data[1] = new double[1]{200.0};
	int* col_shape = new int[2]{2, 1};
	N2Array D(col_data, col_shape);
	cout << "\nD (2x1 column vector):\n" << D.toString() << endl;
	cout << "A + D (broadcasting 2x1 to 2x3):\n" << (A + D).toString() << endl;
	
	// Test all operations with broadcasting
	cout << "\n****All operations with broadcasting:****\n";
	cout << "A - C:\n" << (A - C).toString() << endl;
	cout << "A * C:\n" << (A * C).toString() << endl;
	cout << "A / C (careful with zeros):\n" << (A / C).toString() << endl;
	
	// Test axis functionality
	cout << "\n****Axis functionality tests:****\n";
	cout << "A (2x3):\n" << A.toString() << endl;
	cout << "Sum all elements (axis=-1): " << sum(A).toString() << endl;
	cout << "Sum by rows (axis=0): " << sum(A, 0).toString() << endl;
	cout << "Sum by columns (axis=1): " << sum(A, 1).toString() << endl;
	cout << "Mean all elements (axis=-1): " << mean(A).toString() << endl;
	cout << "Mean by rows (axis=0): " << mean(A, 0).toString() << endl;
}

void normalize(N2Array& A) {
	N2Array m = mean(A, 0);
	N2Array sd = stdev(A, 0);

	// double sd = stdev(A).n1array[0];
	A = (A - m) / sd;
	cout << "Normalized A:\n" << A.toString() << endl;
}

int main() {
	int* shape = new int[2]{2, 3};
	double** data = new double*[shape[0]];
	for (int i = 0; i < shape[0]; ++i) {
		data[i] = new double[shape[1]];
		for (int j = 0; j < shape[1]; ++j) {
			data[i][j] = i * shape[1] + j + 1; // 1..6
		}
	}
	N2Array A = testN2Array(data, shape);
	normalize(A);
	// testNumC(A);
	return 0;
}
