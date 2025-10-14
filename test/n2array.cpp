// Keep this TU minimal for a header-only template. Users should include n2array.h.
#include "../include/n2array.h"
#include <iostream>
using namespace std;

N2Array testN2Array(double** data, int* shape) {
	// Create a 2x3 matrix using 2D allocation
	cout << "-------------------------------------------" << endl;
	cout << "****N2Array test:****\n";
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