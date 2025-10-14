#include <iostream>
#include "numc.h"
using namespace std;

void testNumC(N2Array& A) {
	cout << "------------------------------------------" << endl;
	cout << "****numc test:****\n";
	NumC nc;
	double* arr1d = nc.arange(1.0, 7.0); // 1..6
	int* shape = new int[2]{2, 3};
	N2Array B(arr1d, shape);
	cout << "NumC arange 1..6 as 2x3:\n" << B.toString() << endl;

	cout << "\n****Axis functionality tests:****\n";
	cout << "Original matrix A:\n" << A.toString() << endl;
	
	// Test sum with different axes
	cout << "\nSum tests:\n";
	cout << "sum(A, axis=-1): " << sum(A).toString() << endl;
	cout << "sum(A, axis=0): " << sum(A, 0).toString() << endl;
	cout << "sum(A, axis=1): " << sum(A, 1).toString() << endl;
	
	// Test mean with different axes  
	cout << "\nMean tests:\n";
	cout << "mean(A, axis=-1): " << mean(A).toString() << endl;
	cout << "mean(A, axis=0): " << mean(A, 0).toString() << endl;
	cout << "mean(A, axis=1): " << mean(A, 1).toString() << endl;
	
	// Test min/max with different axes
	cout << "\nMin/Max tests:\n";
	cout << "min(A, axis=-1): " << min(A).toString() << endl;
	cout << "min(A, axis=0): " << min(A, 0).toString() << endl;
	cout << "min(A, axis=1): " << min(A, 1).toString() << endl;
	
	cout << "max(A, axis=-1): " << max(A).toString() << endl;
	cout << "max(A, axis=0): " << max(A, 0).toString() << endl;
	cout << "max(A, axis=1): " << max(A, 1).toString() << endl;
	
	// Test stdev with different axes
	cout << "\nStandard deviation tests:\n";
	cout << "stdev(A, axis=-1): " << stdev(A, -1).toString() << endl;
	cout << "stdev(A, axis=0): " << stdev(A, 0).toString() << endl;
	cout << "stdev(A, axis=1): " << stdev(A, 1).toString() << endl;
	
	// Test default parameter (should be same as axis=-1)
	cout << "\nDefault parameter test (should equal axis=-1):\n";
	cout << "stdev(A): " << stdev(A).toString() << endl;
}