/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <string>
using namespace std;

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

// CUDA Kernel
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

void verifyMalloc(float *a, float *b, float *c) {
	if(a == NULL || b == NULL || c == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
}

void initHostInput(float *a, float *b, size_t n) {
  for (int i = 0; i < n; ++i)
  {
    a[i] = rand()/(float)RAND_MAX;
		b[i] = rand()/(float)RAND_MAX;
	}
}

void checkCUDAError(cudaError_t err, string message) {
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to %s (error code %s)!\n", message.c_str(), cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void verifyResult(float *c, float *a, float *b, int n) {
    for (int i = 0; i < n; ++i)
    {
        if (fabs(a[i] + b[i] - c[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

int main(void)
{
    // Print the vector length to be used, and compute its size
    //int numElements = 50000;
    int numElements = 1000000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A and B, output vector C
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    verifyMalloc(h_A, h_B, h_C);		// Verify that allocations succeeded
		initHostInput(h_A, h_B, numElements);

    // Allocate the device input vector A and B, output vector C
    float *d_A = NULL; 
    float *d_B = NULL;
    float *d_C = NULL;
    checkCUDAError(cudaMalloc((void **)&d_A, size), "allocate device vector A");
    checkCUDAError(cudaMalloc((void **)&d_B, size), "allocate device vector B");
    checkCUDAError(cudaMalloc((void **)&d_C, size), "allocate device vector C");

    // Copy the host inputs to the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    checkCUDAError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "copy vector A HtoD");
    checkCUDAError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "copy vector B HtoD");;

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    checkCUDAError(cudaGetLastError(), "launch vectorAdd kernel");

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    checkCUDAError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "copy vector C DtoH");

    // Verify that the result vector is correct
		verifyResult(h_C, h_A, h_B, numElements);
    printf("Test PASSED\n");

    // Free device global memory
    checkCUDAError(cudaFree(d_A), "free device vector A");
    checkCUDAError(cudaFree(d_B), "free device vector B");
    checkCUDAError(cudaFree(d_C), "free device vector C");

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

