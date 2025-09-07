#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <cmath>

using namespace std;

// Include matmul functions (copy from main implementation)
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matmul(float* A_h, float* B_h, float* C_h, int M, int N, int K) {
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);
    
    float *A_d, *B_d, *C_d;
    
    cudaError_t cudaStatus = cudaMalloc((void**)&A_d, size_A);
    assert(cudaStatus == cudaSuccess);
    
    cudaStatus = cudaMalloc((void**)&B_d, size_B);
    assert(cudaStatus == cudaSuccess);
    
    cudaStatus = cudaMalloc((void**)&C_d, size_C);
    assert(cudaStatus == cudaSuccess);
    
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    matmul_kernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void matmul_cpu(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Test utilities
bool compare_matrices(float* A, float* B, int rows, int cols, float tolerance = 1e-5f) {
    for (int i = 0; i < rows * cols; i++) {
        if (abs(A[i] - B[i]) > tolerance) {
            cout << "Mismatch at index " << i << ": " << A[i] << " vs " << B[i] << endl;
            return false;
        }
    }
    return true;
}

void print_matrix(float* matrix, int rows, int cols, const string& name) {
    cout << name << " (" << rows << "x" << cols << "):" << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        cout << endl;
    }
    cout << endl;
}

// Individual test functions
bool test_2x2_matrices() {
    cout << "Test: 2x2 matrices" << endl;
    
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C_gpu[4], C_cpu[4];
    
    matmul(A, B, C_gpu, 2, 2, 2);
    matmul_cpu(A, B, C_cpu, 2, 2, 2);
    
    bool result = compare_matrices(C_gpu, C_cpu, 2, 2);
    cout << "Result: " << (result ? "PASS" : "FAIL") << endl;
    
    if (!result) {
        print_matrix(A, 2, 2, "A");
        print_matrix(B, 2, 2, "B");
        print_matrix(C_gpu, 2, 2, "C (GPU)");
        print_matrix(C_cpu, 2, 2, "C (CPU)");
    }
    
    return result;
}

bool test_identity_matrix() {
    cout << "Test: Identity matrix multiplication" << endl;
    
    float A[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float B[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float C_gpu[9], C_cpu[9];
    
    matmul(A, B, C_gpu, 3, 3, 3);
    matmul_cpu(A, B, C_cpu, 3, 3, 3);
    
    bool result = compare_matrices(C_gpu, B, 3, 3) && compare_matrices(C_gpu, C_cpu, 3, 3);
    cout << "Result: " << (result ? "PASS" : "FAIL") << endl;
    
    return result;
}

bool test_zero_matrix() {
    cout << "Test: Zero matrix multiplication" << endl;
    
    float A[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float B[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float C_gpu[16], C_cpu[16];
    
    matmul(A, B, C_gpu, 4, 4, 4);
    matmul_cpu(A, B, C_cpu, 4, 4, 4);
    
    bool result = compare_matrices(C_gpu, C_cpu, 4, 4);
    cout << "Result: " << (result ? "PASS" : "FAIL") << endl;
    
    return result;
}

bool test_rectangular_matrices() {
    cout << "Test: Rectangular matrices (2x3) * (3x4)" << endl;
    
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float B[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float C_gpu[8], C_cpu[8];
    
    matmul(A, B, C_gpu, 2, 4, 3);
    matmul_cpu(A, B, C_cpu, 2, 4, 3);
    
    bool result = compare_matrices(C_gpu, C_cpu, 2, 4);
    cout << "Result: " << (result ? "PASS" : "FAIL") << endl;
    
    if (!result) {
        print_matrix(A, 2, 3, "A");
        print_matrix(B, 3, 4, "B");
        print_matrix(C_gpu, 2, 4, "C (GPU)");
        print_matrix(C_cpu, 2, 4, "C (CPU)");
    }
    
    return result;
}

bool test_single_element() {
    cout << "Test: Single element matrices (1x1)" << endl;
    
    float A[] = {2.5f};
    float B[] = {4.0f};
    float C_gpu[1], C_cpu[1];
    
    matmul(A, B, C_gpu, 1, 1, 1);
    matmul_cpu(A, B, C_cpu, 1, 1, 1);
    
    bool result = compare_matrices(C_gpu, C_cpu, 1, 1);
    cout << "Result: " << (result ? "PASS" : "FAIL") << endl;
    
    return result;
}

int main() {
    cout << "Matrix Multiplication Unit Tests" << endl;
    cout << "=================================" << endl;
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run individual tests
    if (test_2x2_matrices()) passed_tests++;
    total_tests++;
    cout << endl;
    
    if (test_identity_matrix()) passed_tests++;
    total_tests++;
    cout << endl;
    
    if (test_zero_matrix()) passed_tests++;
    total_tests++;
    cout << endl;
    
    if (test_rectangular_matrices()) passed_tests++;
    total_tests++;
    cout << endl;
    
    if (test_single_element()) passed_tests++;
    total_tests++;
    cout << endl;
    
    // Summary
    cout << "Test Summary:" << endl;
    cout << "=============" << endl;
    cout << "Total tests: " << total_tests << endl;
    cout << "Passed: " << passed_tests << endl;
    cout << "Failed: " << (total_tests - passed_tests) << endl;
    cout << "Success rate: " << (100.0 * passed_tests / total_tests) << "%" << endl;
    
    return (passed_tests == total_tests) ? 0 : 1;
}
