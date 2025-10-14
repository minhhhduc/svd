// Keep this TU minimal for a header-only template. Users should include n2array.h.
#include "../include/n2array.h"
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

	N2Array A(data, shape);
	cout << A.toString() << endl;
	cout << (A*3.0).toString() << endl;

	return 0;
}