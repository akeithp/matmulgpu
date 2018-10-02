


#include <iostream>
#include <stdlib.h>
#include <stdio.h>

void print_dmatrix(double *matrix, int m, int n ){
        printf("\nMatrix print \n");
        int i,j;
        for(i =0; i<m; i++){
		for( j=0; j<n; j++ ){
			printf("%f ", matrix[ m*i + j ]);
		}
		printf("\n");
        }
	printf("\n");
}





