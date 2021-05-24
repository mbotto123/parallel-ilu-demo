# Parallel ILU Demo
This repository includes a test implementation of the Parallel ILU factorization algorithm developed by Chow and Patel (2015). C++ and the Eigen library are used for sparse linear algebra, and OpenMP is used for parallelization. Key parts of the algorithm are timed with OpenMP's wall-clock timer to show the parallel capabilities of the algorithm. Testing was done on the Frontera HPC system at the Texas Advanced Computing Center (TACC), showing good speedup up to 56 threads on an Intel Cascade Lake compute node. A strong scaling analysis is discussed in the report `Parallel_ILU_Demo.pdf` included in this repository.

## Compiling and Running the Code
Compilation is handled with a makefile in the `/src` directory. The code should be run on an HPC system for testing with multiple threads. If you are using an HPC system at TACC, then you can directly follow the instructions in the README included in the `/src` directory. If not, then you will have to change the include path in the makefile to reflect the location of Eigen installation on your system. Also, you may have to change the compiler to `g++` instead of `icpc`. 

Once you have compiled the code successfully, set the desired number of OpenMP threads with `export OMP_NUM_THREADS = ...` and run the executable `pilu.exe`. Timings will be displayed on standard output. 

## References 
Credit goes to Chow and Patel (2015) for the algorithm and the SuiteSparse Matrix collection for the test input matrix.

Chow, E. and Patel, A. (2015). "Fine-Grained Parallel Incomplete LU Factorization." *SIAM J. Sci.
Comput.*, Vol. 37, No. 2, pp. C169-C193.

Davis, T. and Hu, Y. (2011). "The University of Florida Sparse Matrix Collection." *ACM
Transactions on Mathematical Software 38*, 1, Article 1 (December 2011), 25 pages. DOI:
https://doi.org/10.1145/2049662.2049663


