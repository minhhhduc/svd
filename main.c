#include <stdio.h>
#include <stdlib.h>
#include "../include/n2array.h"
#include "../include/numc.h"


int main() {
    N2Array* zero = zeros(2, 3);
    printf("Zero array:\n%s\n", N2Array_to_string(zero));
    N2Array_free(zero);
    return 0;
}