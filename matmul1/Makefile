OBJ=matmul
TEST_OBJ=test_matmul
UNIT_TEST_OBJ=unit_tests


all: 
	nvcc --std=c++11 -c $(OBJ).cu -o $(OBJ).o
	nvcc --std=c++11 -o $(OBJ) $(OBJ).o

test: $(TEST_OBJ).cu
	nvcc --std=c++11 -o $(TEST_OBJ) $(TEST_OBJ).cu

unit_test: $(UNIT_TEST_OBJ).cu
	nvcc --std=c++11 -o $(UNIT_TEST_OBJ) $(UNIT_TEST_OBJ).cu

all_tests: all test unit_test

clean:
	rm -f $(OBJ).o $(TEST_OBJ) $(UNIT_TEST_OBJ)

cleanall: clean
	rm -f $(OBJ) $(TEST_OBJ) $(UNIT_TEST_OBJ) *.out *.err

run: 
	sbatch job.slurm

run_tests:
	./run_tests.sh

run_unit_tests: unit_test
	./$(UNIT_TEST_OBJ)