#ifndef N2ARRAY_H
#define N2ARRAY_H

#include <stdexcept>
#include <algorithm>
#include <cmath>

#pragma once

// Concrete double-only N2Array (non-templated)
class N2Array {
    public:
        double** n2array;
        const int* shape;
        double* n1array;

        N2Array(double** darray, int* shape);
        N2Array(double* darray, int* shape);
        N2Array(const N2Array& other); // Copy constructor
        ~N2Array();

        N2Array transpose();

        N2Array operator+(const N2Array& other);
        N2Array operator-(const N2Array& other);
        N2Array operator*(const N2Array& other);
        N2Array operator/(const N2Array& other);
        N2Array operator+(const double& scalar);
        N2Array operator-(const double& scalar);
        N2Array operator*(const double& scalar);
        N2Array operator/(const double& scalar);
        N2Array operator=(const N2Array& other);

        bool operator==(const N2Array& other);
        bool operator!=(const N2Array& other);

        N2Array operator[](int index);
        N2Array operator[](int* indices);

        double** toArray();
        const int* getShape();
        
        char* toString();
};

#endif
