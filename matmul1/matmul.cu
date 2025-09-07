#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>
#include <random>
#include <chrono>

using namespace std;

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        for (int k_i = 0; k_i < K; k_i++) {
            C[row * N + col] += A[row * K + k_i] * B[k_i * N + col];
        }
    }

}

// Host function to perform matrix multiplication
void matmul(float* A_h, float* B_h, float* C_h, int M, int N, int K) {
    float *A_d, *B_d, *C_d;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    cudaError_t err_A = cudaMalloc((void**)&A_d, size_A);
    if (err_A != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);

    }
    cudaError_t err_b = cudaMalloc((void**)&B_d, size_B);
    if (err_b != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    }

    cudaError_t err_c = cudaMalloc((void**)&C_d, size_C);
    if (err_c != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    }
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

    dim3 n_block(16,16);
    dim3 n_grid(ceil(M/(float)n_block.x), ceil(N/(float)n_block.y));

    matmul_kernel<<<n_grid, n_block>>>(A_d, B_d, C_d, M, N, K);

    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

// CPU reference implementation for verification
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

// Function to initialize matrix with random values
void init_matrix(float* matrix, int rows, int cols, float min_val = -1.0f, float max_val = 1.0f) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(min_val, max_val);

    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

// Function to initialize matrix with specific pattern
void init_matrix_pattern(float* matrix, int rows, int cols, const string& pattern) {
    if (pattern == "identity") {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i * cols + j] = (i == j) ? 1.0f : 0.0f;
            }
           }
    } else if (pattern == "ones") {
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = 1.0f;
        }
    } else if (pattern == "zeros") {
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = 0.0f;
        }
    } else if (pattern == "sequential") {
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = static_cast<float>(i + 1);
        }
    }
}

// Function to print matrix
void print_matrix(float* matrix, int rows, int cols, const string& name = "Matrix") {
    cout << name << " (" << rows << "x" << cols << "):" << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        cout << endl;
    }
    cout << endl;
}

// Function to compare two matrices
bool compare_matrices(float* A, float* B, int rows, int cols, float tolerance = 1e-5f) {
    for (int i = 0; i < rows * cols; i++) {
        if (abs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Function to calculate matrix multiplication performance
void benchmark_matmul(int M, int N, int K, int iterations = 10) {
    cout << "Benchmarking matrix multiplication (" << M << "x" << K << ") * (" << K << "x" << N << ")" << endl;

    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;

    float* A = new float[size_A];
    float* B = new float[size_B];
    float* C_gpu = new float[size_C];
    float* C_cpu = new float[size_C];

    // Initialize matrices
    init_matrix(A, M, K);
    init_matrix(B, K, N);

    // Warm up GPU
    matmul(A, B, C_gpu, M, N, K);
    cudaDeviceSynchronize();

    // Benchmark GPU
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        matmul(A, B, C_gpu, M, N, K);
    }
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    auto gpu_time = chrono::duration_cast<chrono::microseconds>(end - start).count() / iterations;

    // Benchmark CPU
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        matmul_cpu(A, B, C_cpu, M, N, K);
    }
    end = chrono::high_resolution_clock::now();

    auto cpu_time = chrono::duration_cast<chrono::microseconds>(end - start).count() / iterations;

    // Verify correctness
    bool correct = compare_matrices(C_gpu, C_cpu, M, N);

    cout << "GPU Time: " << gpu_time << " μs" << endl;
    cout << "CPU Time: " << cpu_time << " μs" << endl;
    cout << "Speedup: " << (float)cpu_time / gpu_time << "x" << endl;
    cout << "Correctness: " << (correct ? "PASS" : "FAIL") << endl;
    cout << "----------------------------------------" << endl;

    delete[] A;
    delete[] B;
    delete[] C_gpu;
    delete[] C_cpu;
}

int main() {
    cout << "Matrix Multiplication Test Suite" << endl;
    cout << "=================================" << endl;

    // Test 1: Small matrices with known results
    cout << "\nTest 1: Small matrices (2x2)" << endl;
    int M = 2, N = 2, K = 2;
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C_gpu[4], C_cpu[4];

    matmul(A, B, C_gpu, M, N, K);
    matmul_cpu(A, B, C_cpu, M, N, K);

    print_matrix(A, M, K, "A");
    print_matrix(B, K, N, "B");
    print_matrix(C_gpu, M, N, "C (GPU)");
    print_matrix(C_cpu, M, N, "C (CPU)");

    bool test1_pass = compare_matrices(C_gpu, C_cpu, M, N);
    cout << "Test 1 Result: " << (test1_pass ? "PASS" : "FAIL") << endl;

    // Test 2: Identity matrix multiplication
    cout << "\nTest 2: Identity matrix multiplication (3x3)" << endl;
    M = N = K = 3;
    float A_id[9], B_id[9], C_id_gpu[9], C_id_cpu[9];

    init_matrix_pattern(A_id, M, K, "identity");
    init_matrix(B_id, K, N);

    matmul(A_id, B_id, C_id_gpu, M, N, K);
    matmul_cpu(A_id, B_id, C_id_cpu, M, N, K);

    bool test2_pass = compare_matrices(C_id_gpu, B_id, M, N) && compare_matrices(C_id_gpu, C_id_cpu, M, N);
    cout << "Test 2 Result: " << (test2_pass ? "PASS" : "FAIL") << endl;

    // Test 3: Zero matrix multiplication
    cout << "\nTest 3: Zero matrix multiplication (4x4)" << endl;
    M = N = K = 4;
    float A_zero[16], B_zero[16], C_zero_gpu[16], C_zero_cpu[16];

    init_matrix_pattern(A_zero, M, K, "zeros");
    init_matrix(B_zero, K, N);

    matmul(A_zero, B_zero, C_zero_gpu, M, N, K);
    matmul_cpu(A_zero, B_zero, C_zero_cpu, M, N, K);

    bool test3_pass = compare_matrices(C_zero_gpu, C_zero_cpu, M, N);
    cout << "Test 3 Result: " << (test3_pass ? "PASS" : "FAIL") << endl;

    // Test 4: Random matrices
    cout << "\nTest 4: Random matrices (8x8)" << endl;
    M = N = K = 8;
    float* A_rand = new float[M * K];
    float* B_rand = new float[K * N];
    float* C_rand_gpu = new float[M * N];
    float* C_rand_cpu = new float[M * N];

    init_matrix(A_rand, M, K);
    init_matrix(B_rand, K, N);

    matmul(A_rand, B_rand, C_rand_gpu, M, N, K);
    matmul_cpu(A_rand, B_rand, C_rand_cpu, M, N, K);

    bool test4_pass = compare_matrices(C_rand_gpu, C_rand_cpu, M, N);
    cout << "Test 4 Result: " << (test4_pass ? "PASS" : "FAIL") << endl;

    delete[] A_rand;
    delete[] B_rand;
    delete[] C_rand_gpu;
    delete[] C_rand_cpu;

    // Performance benchmarks
    cout << "\nPerformance Benchmarks:" << endl;
    benchmark_matmul(64, 64, 64);
    benchmark_matmul(128, 128, 128);
    benchmark_matmul(256, 256, 256);

    // Overall test result
    bool all_tests_pass = test1_pass && test2_pass && test3_pass && test4_pass;
    cout << "\nOverall Result: " << (all_tests_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << endl;

    return all_tests_pass ? 0 : 1;
}
