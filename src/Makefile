####################################################################################################
# Makefile for an implementation of the Parallel ILU algorithm described in Chow and Patel (2015)
# "Fine-Grained Parallel Incomplete LU Factorization," SIAM J. Sci Comput., Vol. 37, No.2, C169-C193

# C++ and the Eigen library are used for sparse linear algebra, OpenMP is used for parallelization
# Source code is in the same directory as this Makefile

# Developed by Marcos Botto Tornielli, May 2021
####################################################################################################

# C++ Compiler
CXX          = icpc

# Optional compiler flags, for debug and performance
CXXFLAGS     = -g -Wall -O2 -xhost

# Mandatory compiler flags, with optional ones appended
ALL_CXXFLAGS = -I${TACC_EIGEN_INC} -qopenmp $(CXXFLAGS)

pilu : pilu.cpp
	$(CXX) $(ALL_CXXFLAGS) -o pilu.x pilu.cpp

clean : 
	rm -rf pilu.x
