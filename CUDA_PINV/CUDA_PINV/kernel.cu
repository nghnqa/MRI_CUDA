#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <cusolverDn.h>
#include <cublas_v2.h>
/////////////////////
using namespace std;
#define M 3000
#define N 3000
/////////////////////
int readInput(float *A);
/////////////////////
int main()
{		 
	float *h_A, *h_U, *h_S, *h_Vt, *d_A, *d_U, *d_S, *d_Vt, *work;
	int *devInfo, work_size = 0, devInfo_h = 0;

	cusolverDnHandle_t solver_handle; 
	cusolverDnCreate(&solver_handle);
	cusolverDnSgesvd_bufferSize(solver_handle, M, N, &work_size);

	h_A = (float*)malloc(M*N*sizeof(float));
	h_U = (float*)malloc(M*M*sizeof(float));
	h_Vt = (float*)malloc(N*N*sizeof(float));
	h_S = (float*)malloc(N*sizeof(float));

	cudaMalloc((void**)&d_A,M*N*sizeof(float));
	cudaMalloc((void**)&d_U,M*M*sizeof(float));
	cudaMalloc((void**)&d_Vt,N*N*sizeof(float));
	cudaMalloc((void**)&d_S,N*sizeof(float));
	cudaMalloc((void**)&devInfo, sizeof(int));
	cudaMalloc(&work, work_size * sizeof(float));

	readInput(h_A);
	cudaMemcpy(d_A,h_A,M*N*sizeof(float), cudaMemcpyHostToDevice);
	cusolverDnSgesvd(solver_handle, 'A', 'A', M, N, d_A, M, d_S, d_U, M, d_Vt, N, work, work_size, NULL, devInfo);
	cudaDeviceSynchronize();
	cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_U, d_U, M*M*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Vt, d_Vt,N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S, d_S,N*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "devInfo = " << devInfo_h << "\n";
	std::cout << "SVD successful\n\n";

	cudaFree(d_A); cudaFree(d_U); cudaFree(d_Vt); cudaFree(d_S); cudaFree(devInfo); cudaFree(work);
	cusolverDnDestroy(solver_handle);
    
	///////////////////////////////////////////////////

	cublasHandle_t cublas_handler;
	cublasCreate_v2(&cublas_handler);
	cublasDestroy_v2(cublas_handler);
	
	return 0;
}
/////////////////////
int readInput(float *A)
{
	fstream input_matrix;
	input_matrix.open("E:/MRI/test_matrix.txt", ios::in);
	if (input_matrix.is_open())
	{
		for (unsigned int i = 0; i < M*N; i++)
		{
			input_matrix >> A[i];
		}
		input_matrix.close();
		return 0;
	}
	else
	{
		cout << "readInput => ERROR in opening file!" << endl;
		return -1;
	}
}