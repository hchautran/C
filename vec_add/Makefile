OBJ=vecadd

all:
	nvcc -std=c++11 -c $(OBJ).cu -o $(OBJ).o 
	nvcc -std=c++11 -o $(OBJ) $(OBJ).o 

clean:
	rm -f $(OBJ).o

cleanall: clean
	rm -f $(OBJ) *.out *.err

run:
	sbatch job.slurm