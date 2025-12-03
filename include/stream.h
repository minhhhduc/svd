#ifndef STREAM_H
#define STREAM_H

void stream_matrix_reader(const char* filename, double*** data, int* rows, int* cols);
void print_matrix(double** data, int rows, int cols);
void free_matrix(double** data, int rows);

#endif
