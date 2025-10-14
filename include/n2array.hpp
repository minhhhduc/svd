#ifndef N2ARRAY_H
#define N2ARRAY_H

#include <stdexcept>
#include <algorithm>
#include <cmath>

#pragma once

template <typename T>
class N2Array {
    public:
        T** n2array;
        const int* shape;
        T* n1array;

        N2Array(T** darray, int* shape);
        N2Array(T* darray, int* shape);
        ~N2Array();

        N2Array transpose();

        N2Array operator+(const N2Array& other);
        N2Array operator-(const N2Array& other);
        N2Array operator*(const N2Array& other);
        N2Array operator/(const N2Array& other);
        N2Array operator+(const T& scalar);
        N2Array operator-(const T& scalar);
        N2Array operator*(const T& scalar);
        N2Array operator/(const T& scalar);
        N2Array operator=(const N2Array& other);

        bool operator==(const N2Array& other);
        bool operator!=(const N2Array& other);

        N2Array operator[](int index);
        N2Array operator[](int* indices);

        char* toString();
};

// Bao gồm phần template implementation
#include "n2array.tpp"

#endif
