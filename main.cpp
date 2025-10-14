// Minimal demo: include N2Array and print a small matrix
#include "n2array.h"
#include "numc.h"
#include <iostream>
using namespace std;

N2Array testN2Array(double** data, int* shape) {
	// Create a 2x3 matrix using 2D allocation
	N2Array A(data, shape);
	cout << A.toString() << endl;

	cout << "test operator (scalar):\n";
	cout << (A*3.0).toString() << endl;
	
	cout << "First row: " << A[0].toString() << endl;
	cout << "Shape: " << A[0].shape[0] << " x " << A[0].shape[1] << endl;

	cout << "test transpose:\n";
	cout << "shape: " << A[0].transpose().toString() << endl;

	return A;
}

void testNumC(N2Array& A) {
	NumC nc;
	double* arr1d = nc.arange(1.0, 7.0); // 1..6
	int* shape = new int[2]{2, 3};
	N2Array B(arr1d, shape);
	cout << "NumC arange 1..6 as 2x3:\n" << B.toString() << endl;
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
	testNumC(A);
	return 0;
}