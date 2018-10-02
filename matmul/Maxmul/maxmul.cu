


#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "../Utils/utils.h"
#include "../Utils/matutils.h"

#define PRINT 1
//#define BSIZE 4

using namespace std;
// pedir el 'n' como parametro de ejecucion del kernel
__global__ void matmul1(int n, double *a, double *b, double *c){
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	//int n = blockDim.x * gridDim.x;
	int k;
	double r = 0.0;
	//c[n*idy + idx] = 0;
	for (k=0; k<n; k++){
		//c[n*idy + idx] +=  a[n*idy + k] *  b[n*k + idx];
		r +=  a[n*idy + k] *  b[n*k + idx];
	}
	c[n*idy + idx] = r;
}

__global__ void matmul_sm(int n, double *a, double *b, double *c){
	// (1) crear memoria compartida
	__shared__ double as[BSIZE*BSIZE];
	__shared__ double bs[BSIZE*BSIZE];
	__shared__ double cs[BSIZE*BSIZE];

	int ltidx = threadIdx.x;
	int ltidy = threadIdx.y;
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	// (2) insertar elementos en cache
	int w=0;
	cs[ltidy*BSIZE + ltidx] = 0.0;
	__syncthreads();
	while(w<n){
		as[ltidy*BSIZE + ltidx] = a[tidy*n + (tidx + w)];
		bs[ltidy*BSIZE + ltidx] = b[(tidy + w)*n + tidx];
		__syncthreads();

		// (3) matmul en cache
		double r = 0.0;
		for (int k=0; k<BSIZE; k++){
			r +=  as[BSIZE*ltidy + k] *  bs[BSIZE*k + ltidx];
		}
		cs[ltidy*BSIZE + ltidx] += r;
		__syncthreads();
		w += BSIZE;
	}
	// (4) escribir en c global
	__syncthreads();
	c[tidy*n + tidx] = cs[ltidy*BSIZE + ltidx];
}


void matmul_cpu(int n, double *a, double *b, double *c){
	//#pragma omp parallel for num_threads(6)
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			double sum=0.0;
			for(int k=0; k<n; ++k){
				sum += a[i*n + k]*b[k*n + j];
			}
			c[i*n + j] = sum;
		}
	}
}

void matmul_cpu_transp_b(int n, double *a, double *b, double *c){
	#pragma omp parallel for num_threads(6)
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			double sum=0.0;
			for(int k=0; k<n; ++k){
				sum += a[i*n + k]*b[j*n + k];
			}
			c[i*n + j] = sum;
		}
	}
}

void transpose(int n, double *m){
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			if(i>j){
				double aux = m[i*n + j];
				m[i*n + j] = m[j*n + i];
				m[j*n + i] = aux;
			}
		}
	}
}

int main( int argc, char**  argv  ){
	int args_needed = 1;
	if (argc < args_needed + 1 ){
		printf(" Arg number error, needed: %d  \n", args_needed);
		return 0;	
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf(" CUDA - Maxmul  \n");

	// Select Device
	// HANDLE_ERROR(  cudaSetDevice(0)  ) ;
	
	// Size
	int n = atoi(argv[1]);

	//Create Data host n x n

	double *a;
	double *b;
	double *c;	

	a = (double *)malloc( sizeof(double) * n * n  );
	b = (double *)malloc( sizeof(double) * n * n  );
	c = (double *)malloc( sizeof(double) * n * n  );

	int i;
	for ( i =0; i<n*n ; i++  ){
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}
	printf("CPU matmul......"); fflush(stdout);
	cudaEventRecord(start);
	print_dmatrix(a,n,n);
	print_dmatrix(b,n,n);
	//transpose(n, b);
	//print_dmatrix(b,n,n);

	printf("computing\n");
	cudaEventRecord(start);
	//matmul_cpu_transp_b(n, a, b, c);
	matmul_cpu(n, a, b, c);
	cudaEventRecord(stop);
	printf("ok\n"); fflush(stdout);
	cudaEventSynchronize(stop);
	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time CPU: %f\n", milliseconds );	
	print_dmatrix(c,n,n);

	// CUDA data
	double *a_dev;
	double *b_dev;
	double *c_dev;


	HANDLE_ERROR(cudaMalloc((void **)&a_dev, sizeof(double)*n*n));
	HANDLE_ERROR(cudaMalloc((void **)&b_dev, sizeof(double)*n*n));
	HANDLE_ERROR(cudaMalloc((void **)&c_dev, sizeof(double)*n*n));

	// Memcpy
	HANDLE_ERROR(cudaMemcpy(a_dev,a,sizeof(double)*n*n,cudaMemcpyHostToDevice)     );
	//transpose(n, b);
	HANDLE_ERROR(cudaMemcpy(b_dev,b,sizeof(double)*n*n,cudaMemcpyHostToDevice)     );	
	
	// Kernel
	dim3 block(BSIZE, BSIZE, 1);
	// asumir que n multiplo de BSIZE	
	dim3 grid(n/BSIZE, n/BSIZE, 1);

	printf("GPU matmul......"); fflush(stdout);
	cudaEventRecord(start);
	//matmul1<<< grid, block >>>(n, a_dev, b_dev, c_dev);
	matmul_sm<<< grid, block >>>(n, a_dev, b_dev, c_dev);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	printf("ok\n"); fflush(stdout);


	// Get data Devices
	HANDLE_ERROR(cudaMemcpy(c, c_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost )     );
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time: %f\n", milliseconds );	
	print_dmatrix(c,n,n);

	//Free
	cudaFree(a_dev);
	cudaFree(b_dev);
	cudaFree(c_dev);
	free(a);
	free(b);
	free(c);
	return 0;
}
