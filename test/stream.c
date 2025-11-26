#include <stdio.h>
#include <stdlib.h>

void stream_matrix_reader(const char* filename, double*** data, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Đọc kích thước ma trận
    if (fscanf(file, "%d %d", rows, cols) != 2) {
        fprintf(stderr, "Error: invalid matrix size format.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Cấp phát bộ nhớ cho ma trận
    *data = (double**)calloc(*rows, sizeof(double*));
    if (*data == NULL) {
        fprintf(stderr, "Memory allocation failed for rows.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < *rows; i++) {
        (*data)[i] = (double*)calloc(*cols, sizeof(double));
        if ((*data)[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for row %d.\n", i);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    // Read matrix values from file (rows x cols doubles)
    for (int i = 0; i < *rows; ++i) {
        for (int j = 0; j < *cols; ++j) {
            if (fscanf(file, "%lf", &((*data)[i][j])) != 1) {
                fprintf(stderr, "Error: failed to read matrix element at (%d,%d).\n", i, j);
                // free allocated memory before exiting
                for (int ii = 0; ii <= i; ++ii) free((*data)[ii]);
                free(*data);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);
}

void print_matrix(double** data, int rows, int cols) {
    printf("Matrix (%d x %d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.3lf ", data[i][j]);
        }
        printf("\n");
    }
}

void free_matrix(double** data, int rows) {
    for (int i = 0; i < rows; i++) {
        free(data[i]);
    }
    free(data);
}



// int main() {
//     const char* filename = "./data/matrix.txt";
//     double** data = NULL;
//     int rows = 0, cols = 0;

    // stream_matrix_reader(filename, &data, &rows, &cols);
//     print_matrix(data, rows, cols);
//     free_matrix(data, rows);

//     return 0;
// }
