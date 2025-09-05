#include<iostream>
#include<cuda_runtime.h>
#include<stdio.h>

using namespace std;


__global__ void vecadd_kernel(float* a, float *b,  float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
       c[idx] = a[idx] + b[idx]; 
    } 
}


void printArr(float* c, int n) {
    for (int i=0; i<n; i++) {
        printf("%.2f ", c[i]);
    }
    printf("\n");
}

void vecadd(float* a_h, float* b_h, float* c_h, int n) {
    int size = n * sizeof(float);
    float *a_d, *b_d, *c_d;
    cudaError_t cudaStatus = cudaMalloc((void**)&a_d, size);
    if (cudaStatus != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(cudaStatus),__FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaMalloc((void**)&b_d, size);
    if (cudaStatus != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(cudaStatus),__FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaMalloc((void**)&c_d, size);
    if (cudaStatus != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(cudaStatus),__FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);


    int n_threads = 256;
    int n_blocks = ceil(n/256.0);
    vecadd_kernel<<<n_threads, n_blocks>>>(a_d, b_d, c_d, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

}


int main() {
    int n = 5;
    float a[n] = {1., 2., 3., 4., 5.};
    float b[n] = {1., 2., 3., 4., 5.};
    float c[n] = {0.,0.,0.,0.,0.};
    printArr(c, n);
    vecadd(a, b, c, n);
    printArr(c, n);
    return 0;
}