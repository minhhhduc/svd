// Minimal demo: include N2Array and print a small matrix
#include "n2array.hpp"
#include <iostream>
using namespace std;

int main() {
	// Create a 2x3 matrix using 2D allocation
	int* shape = new int[2]{2, 3};
	double** data = new double*[shape[0]];
	for (int i = 0; i < shape[0]; ++i) {
		data[i] = new double[shape[1]];
		for (int j = 0; j < shape[1]; ++j) {
			data[i][j] = i * shape[1] + j + 1; // 1..6
		}
	}

	N2Array<double> A(data, shape);
	cout << A.toString() << endl;
	cout << (A*3.0).toString() << endl;
	cout << "First row: " << A[0].toString() << endl;	

	cout << "Shape: " << A[0].shape[0] << " x " << A[0].shape[1] << endl;

	return 0;
}