#include <omp.h>
#include <stdio.h>
#include "mulmat.c"

int main() {

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("Hello from thread %d\n", thread_id);
    }
}