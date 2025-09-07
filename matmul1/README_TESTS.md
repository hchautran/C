# Matrix Multiplication Test Suite

This directory contains a comprehensive test suite for CUDA matrix multiplication implementation.

## Files Overview

### Core Implementation
- `matmul.cu` - Main matrix multiplication implementation with built-in tests
- `test_matmul.cu` - Comprehensive test suite with performance benchmarks
- `unit_tests.cu` - Simple unit tests for individual functionality
- `run_tests.sh` - Test runner script

### Build System
- `Makefile` - Build configuration for all components
- `job.slurm` - SLURM job script for cluster execution

## Test Types

### 1. Unit Tests (`unit_tests.cu`)
Simple, focused tests for individual functionality:
- 2x2 matrix multiplication
- Identity matrix multiplication
- Zero matrix multiplication
- Rectangular matrix multiplication (2x3) * (3x4)
- Single element matrices (1x1)

### 2. Comprehensive Tests (`test_matmul.cu`)
Advanced test suite with performance analysis:
- Small matrices (2x2)
- Identity matrix multiplication (3x3)
- Zero matrix multiplication (4x4)
- Random matrices (8x8)
- Large matrices (64x64)
- Rectangular matrices (3x4) * (4x5)
- Edge cases (1x1 matrices)
- Performance benchmarks (32x32 to 256x256)

### 3. Main Program (`matmul.cu`)
Complete implementation with integrated tests and benchmarks.

## Building and Running Tests

### Quick Start
```bash
# Run all tests
make run_tests


# Run only comprehensive tests
make test && ./test_matmul

# Run main program with built-in tests
make all && ./matmul
```

### Individual Commands
```bash
# Build everything
make all_tests

# Build specific components
make all          # Main matmul program
make unit_test    # Unit tests

# Clean up
make clean        # Remove object files
make cleanall     # Remove all generated files
```

## Test Features

### Correctness Testing
- CPU vs GPU result comparison
- Tolerance-based floating-point comparison (1e-5)
- Edge case handling
- Different matrix sizes and shapes

### Performance Testing
- GPU vs CPU speedup measurement
- Multiple matrix sizes (32x32 to 256x256)
- Warm-up runs to ensure accurate timing
- Microsecond precision timing

### Error Handling
- CUDA error checking
- Memory allocation verification
- Kernel launch error detection
- Graceful failure reporting

## Expected Output

### Unit Tests
```
Matrix Multiplication Unit Tests
=================================
Test: 2x2 matrices
Result: PASS

Test: Identity matrix multiplication
Result: PASS

Test: Zero matrix multiplication
Result: PASS

Test: Rectangular matrices (2x3) * (3x4)
Result: PASS

Test: Single element matrices (1x1)
Result: PASS

Test Summary:
=============
Total tests: 5
Passed: 5
Failed: 0
Success rate: 100%
```

### Comprehensive Tests
```
Matrix Multiplication Test Suite
=================================
Running test: Small matrices (2x2)
Running test: Identity matrix multiplication (3x3)
Running test: Zero matrix multiplication (4x4)
Running test: Random matrices (8x8)
Running test: Large matrices (64x64)
Running test: Rectangular matrices (3x4) * (4x5)
Running test: Edge cases (1x1 matrices)

Running performance benchmarks...

Benchmarking 32x32 * 32x32 matrices:
  GPU Time: 45 μs
  CPU Time: 1234 μs
  Speedup: 27.4x

Test Summary:
=============
Small matrices (2x2): PASS (0.123 ms)
Identity matrix multiplication: PASS (0.156 ms)
...
Total: 7/7 tests passed
Success rate: 100%
```

## Requirements

- CUDA Toolkit (tested with CUDA 12.4)
- NVIDIA GPU with compute capability 3.0+
- C++11 compatible compiler
- Linux environment (for SLURM scripts)

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Ensure CUDA toolkit is properly installed
   - Check that nvcc is in your PATH
   - Verify C++11 support

2. **Runtime Errors**
   - Check GPU availability: `nvidia-smi`
   - Verify CUDA driver compatibility
   - Ensure sufficient GPU memory for large matrices

3. **Test Failures**
   - Check floating-point precision tolerance
   - Verify matrix initialization
   - Review CUDA error messages

### Debug Mode
To enable detailed debugging, modify the tolerance in test functions:
```cpp
bool compare_matrices(float* A, float* B, int rows, int cols, float tolerance = 1e-3f)
```

## Performance Notes

- GPU performance depends on matrix size and GPU architecture
- Small matrices may show poor GPU performance due to kernel launch overhead
- Optimal performance typically achieved with matrices 64x64 or larger
- Memory bandwidth is often the limiting factor for large matrices

## Contributing

To add new tests:
1. Add test function to appropriate test file
2. Update test runner if needed
3. Document test purpose and expected behavior
4. Verify test passes on target hardware
