// #include <stdio.h>
// #include <stdlib.h>
// #include "../include/n2array.h"
// #include "../include/numc.h"


// int main() {
//     printf("N2Array Test\n");
//     // from 1d
//     double arr_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
//     int shape[2] = {2, 3};
//     N2Array* N2Array_z = N2Array_from_1d(arr_data, shape);
//     // from 2d
//     double* arr_2d[2];
//     arr_2d[0] = (double[]){1.0, 2.0, 3.0};
//     arr_2d[1] = (double[]){4.0, 5.0, 6.0};
//     N2Array* N2Array_y = N2Array_from_2d(arr_2d, shape);

//     printf("N2Array from 1D:\n%s\n", N2Array_to_string(N2Array_z));
//     printf("N2Array from 2D:\n%s\n", N2Array_to_string(N2Array_y));

//     //copy
//     N2Array* N2Array_copy_z = N2Array_copy(N2Array_z);
//     printf("N2Array copy of z:\n%s\n", N2Array_to_string(N2Array_copy_z));

//     //free
//     N2Array_free(N2Array_y);
//     N2Array_free(N2Array_copy_z);
    
//     //get
//     double val = N2Array_get(N2Array_z, 1, 2);
//     printf("N2Array_z[1,2] = %f\n", val);
    
//     //set
//     N2Array_set(N2Array_z, 0, 0, 10.0);
//     printf("After setting N2Array_z[0,0] = 10.0:\n%s\n", N2Array_to_string(N2Array_z));

//     //to array
//     double** array_2d = N2Array_to_array(N2Array_z);
//     printf("N2Array_z to 2D array:\n");
//     for (int i = 0; i < shape[0]; i++) {
//         for (int j = 0; j < shape[1]; j++) {
//             printf("%f ", array_2d[i][j]);
//         }
//         printf("\n");
//     }
//     //data
//     double* data_1d = N2Array_data(N2Array_z);
//     printf("N2Array_z to 1D data:\n");
//     for (int i = 0; i < shape[0] * shape[1]; i++) {
//         printf("%f ", data_1d[i]);
//     }
//     printf("\n");

//     //add sub mul div
//     N2Array* N2Array_a = N2Array_copy(N2Array_z);
//     N2Array* N2Array_b = N2Array_copy(N2Array_z);
//     N2Array* N2Array_add_res = N2Array_add(N2Array_a, N2Array_b);
//     N2Array* N2Array_sub_res = N2Array_sub(N2Array_a, N2Array_b);
//     N2Array* N2Array_mul_res = N2Array_mul(N2Array_a, N2Array_b);
//     N2Array* N2Array_div_res = N2Array_div(N2Array_a, N2Array_b);
//     printf("N2Array addition result:\n%s\n", N2Array_to_string(N2Array_add_res));
//     printf("N2Array subtraction result:\n%s\n", N2Array_to_string(N2Array_sub_res));
//     printf("N2Array multiplication result:\n%s\n", N2Array_to_string(N2Array_mul_res));
//     printf("N2Array division result:\n%s\n", N2Array_to_string(N2Array_div_res));
//     N2Array_free(N2Array_a);
//     N2Array_free(N2Array_b);
//     N2Array_free(N2Array_add_res);
//     N2Array_free(N2Array_sub_res);
//     N2Array_free(N2Array_mul_res);
//     N2Array_free(N2Array_div_res);
//     //transpose
//     N2Array* tpose = N2Array_transpose(N2Array_z);
//     printf("N2Array transpose:\n%s\n", N2Array_to_string(tpose));
//     N2Array_free(tpose);

//     //add sub mul div scalar
//     N2Array* N2Array_add_scalar_res = N2Array_add_scalar(N2Array_z, 2.0);
//     N2Array* N2Array_sub_scalar_res = N2Array_sub_scalar(N2Array_z, 2.0);
//     N2Array* mul_scalar_res = N2Array_mul_scalar(N2Array_z, 2.0);
//     N2Array* div_scalar_res = N2Array_div_scalar(N2Array_z, 2.0);
//     printf("N2Array addition with scalar result:\n%s\n", N2Array_to_string(N2Array_add_scalar_res));
//     printf("N2Array subtraction with scalar result:\n%s\n", N2Array_to_string(N2Array_sub_scalar_res));
//     printf("N2Array multiplication with scalar result:\n%s\n", N2Array_to_string(mul_scalar_res));
//     printf("N2Array division with scalar result:\n%s\n  ", N2Array_to_string(div_scalar_res));
//     N2Array_free(N2Array_add_scalar_res);
//     N2Array_free(N2Array_sub_scalar_res);
//     N2Array_free(mul_scalar_res);
//     N2Array_free(div_scalar_res);

//     //equals
//     bool eq = N2Array_equals(N2Array_z, N2Array_z);
//     printf("N2Array_z equals N2Array_z: %s\n", eq ? "true" : "false");

//     //row copy
//     N2Array* row_copy = N2Array_row_copy(N2Array_z, 1);
//     printf("Row copy of row 1:\n%s\n", N2Array_to_string(row_copy));
//     N2Array_free(row_copy);

//     printf("N2Array_z[0,1] = %f\n", N2Array_element(N2Array_z, 0, 1));
//     N2Array_free(N2Array_z);

//     printf("N2Array Test Completed Successfully\n");

//     printf("NumC Test\n");
//     // arange
//     double* arange_result = numc_arange(0.0, 10.0, 0.5);
//     if (arange_result) {
//         printf("numc_arange(0.0, 10.0, 1.0):\n");
//         for (int i = 0; i < 10; i++) {
//             printf("%f ", arange_result[i]);
//         }
//         printf("\n");
//         free(arange_result);
//     } else {
//         printf("numc_arange returned NULL\n");
//     }

//     // linspace
//     double* linspace_result = numc_linspace(0.0, 1.0, 5);
//     if (linspace_result) {
//         printf("numc_linspace(0.0, 1.0, 5):\n");
//         for (int i = 0; i < 5; i++) {
//             printf("%f ", linspace_result[i]);
//         }
//         printf("\n");
//         free(linspace_result);
//     } else {
//         printf("numc_linspace returned NULL\n");
//     }

//     // zeros
//     N2Array* zeros_array = numc_zeros(2, 3);
//     if (zeros_array) {
//         printf("numc_zeros(2, 3):\n%s\n", N2Array_to_string(zeros_array));
//         N2Array_free(zeros_array);
//     } else {
//         printf("numc_zeros returned NULL\n");
//     }

//     // dot product
//     N2Array* a = numc_zeros(2, 3);
//     N2Array* b = numc_zeros(3, 2);
//     if (a && b) {
//         N2Array* dot_result = numc_dot(a, b);
//         if (dot_result) {
//             printf("numc_dot(a, b):\n%s\n", N2Array_to_string(dot_result));
//             N2Array_free(dot_result);
//         } else {
//             printf("numc_dot returned NULL\n");
//         }
//         N2Array_free(a);
//         N2Array_free(b);
//     } else {
//         printf("numc_zeros returned NULL for dot product\n");
//     }

//     // min, max, sum, mean, stdev
//     N2Array* min_result = numc_min(zeros_array, 0);
//     N2Array* max_result = numc_max(zeros_array, 0);
//     N2Array* sum_result = numc_sum(zeros_array, 0);
//     N2Array* mean_result = numc_mean(zeros_array, 0);
//     N2Array* stdev_result = numc_stdev(zeros_array, 0);
//     if (min_result) {
//         printf("numc_min(zeros_array, 0):\n%s\n", N2Array_to_string(min_result));
//         N2Array_free(min_result);
//     } else {
//         printf("numc_min returned NULL\n");
//     }
//     if (max_result) {
//         printf("numc_max(zeros_array, 0):\n%s\n", N2Array_to_string(max_result));
//         N2Array_free(max_result);
//     } else {
//         printf("numc_max returned NULL\n");
//     }
//     if (sum_result) {
//         printf("numc_sum(zeros_array, 0):\n%s\n", N2Array_to_string(sum_result));
//         N2Array_free(sum_result);
//     } else {
//         printf("numc_sum returned NULL\n");
//     }
//     if (mean_result) {
//         printf("numc_mean(zeros_array, 0):\n%s\n", N2Array_to_string(mean_result));
//         N2Array_free(mean_result);
//     } else {
//         printf("numc_mean returned NULL\n");
//     }
//     if (stdev_result) {
//         printf("numc_stdev(zeros_array, 0):\n%s\n", N2Array_to_string(stdev_result));
//         N2Array_free(stdev_result);
//     } else {
//         printf("numc_stdev returned NULL\n");
//     }   

//     // numc
//     return 0;
// }