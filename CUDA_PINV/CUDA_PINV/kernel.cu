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
	float *h_A, *h_U, *h_S, *h_S2, *h_Vt, *d_A, *d_U, *d_S, *d_Vt, *work;
	int *devInfo, work_size = 0, devInfo_h = 0;
	
	cusolverDnHandle_t solver_handle; 
	cusolverDnCreate(&solver_handle);
	cusolverDnSgesvd_bufferSize(solver_handle, M, N, &work_size);

	h_A = (float*)malloc(M*N*sizeof(float));
	h_U = (float*)malloc(M*M*sizeof(float));
	h_Vt = (float*)malloc(N*N*sizeof(float));
	h_S = (float*)malloc(N*sizeof(float)); // this is used for cusolverDn 
	h_S2 = (float*)calloc(M*N,sizeof(float)); // this is full matrixy version of S => we need it to be 0 at first

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

	cudaFree(d_A); cudaFree(d_S); cudaFree(devInfo); cudaFree(work);
	cusolverDnDestroy(solver_handle);
    
	/////////////Repairing S for multiplication///////////////
	float *d_S2;
	cudaMalloc((void**)&d_S2, M*N*sizeof(float));
	for (int i = 0; i < N; i++)
	{
		if (h_S[i] != 0)
		{
			h_S2[i*(N + 1)] = 1.0 / h_S[i];
		}
	}
	cudaMemcpy(d_S2, h_S2, N*M*sizeof(float), cudaMemcpyHostToDevice);
	free(h_S); free(h_S2);
	/////////////cublas_V2 for matrix-matrix multiplication//////////
	cublasHandle_t cublas_handler;
	float *d_KQ1, *d_KQ2, *alpha, *beta;
	alpha = (float*)malloc(sizeof(float)); *alpha = 1.0f;
	beta = (float*)malloc(sizeof(float)); *beta = 1.0f;
	cublasCreate_v2(&cublas_handler);
	cudaMalloc((void**)&d_KQ1, N*M*sizeof(float));					 
	cudaMemset(d_KQ1, 0, N*M*sizeof(float));
	cublasSgemm_v2(cublas_handler, CUBLAS_OP_T, CUBLAS_OP_T, M, N, N, alpha , d_Vt, N, d_S2, M, beta, d_KQ1, N);
	cudaMalloc((void**)&d_KQ2, N*M*sizeof(float));
	cudaMemset(d_KQ1, 0, N*M*sizeof(float));												  
	cublasSgemm_v2(cublas_handler, CUBLAS_OP_T, CUBLAS_OP_T, N, M, M, alpha, d_KQ1, N, d_U, M, beta, d_KQ2, N);
	cudaFree(d_KQ1); cudaFree(d_Vt); cudaFree(d_S2); 
	free(alpha); free(beta);
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