#!/bin/bash

echo "Matrix Multiplication Test Runner"
echo "================================="

# Compile the main matmul program
echo "Compiling matmul..."
make clean
make all

if [ $? -ne 0 ]; then
    echo "Error: Failed to compile matmul"
    exit 1
fi

echo ""
echo "Running main matmul program..."
echo "=============================="
./matmul



echo ""
echo "All tests completed!"
