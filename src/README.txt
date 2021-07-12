@@@@@@ Instructions to compile and run PILU code @@@@@@
@@@@@@          Marcos Botto Tornielli           @@@@@@

*There are two directories for this project, 'input' and 'src'
    - 'src' contains the PILU source code, pilu.cpp, and a Makefile
    - 'input' contains the input matrix used by the pilu.cpp code

*Compilation of pilu.cpp has 2 dependencies: the Intel C++ compiler
 and the Eigen library. Before compiling load these modules:

module load intel
module load eigen

*Now navigate to the 'src' directory and run make
    - This should create the executable pilu.x

*To run the code, enter an idev session, set the desired number of
 OpenMP threads with:

export OMP_NUM_THREADS=...

 and execute with ./pilu.x
