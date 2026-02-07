################################################################################
# Makefile for an implementation of a Parallel ILU algorithm
#
# Reference: Chow and Patel "Fine-Grained Parallel Incomplete LU Factorization"
# SIAM J. Sci Comput 37.2 (2015), pp. 169-193
#
# C++ and the Eigen library are used for sparse linear algebra
# OpenMP is used for parallelization
# Source code is in the same directory as this Makefile
#
# Developed by Marcos Botto Tornielli
# (originally for an independent course project at UT Austin in May 2021)
################################################################################

# C++ Compiler
CXX_LINUX = g++
CXX_TACC  = icpc

COMPILE_ENV = cluster

ifeq ($(COMPILE_ENV), local)
  CXX = $(CXX_LINUX)
endif
ifeq ($(COMPILE_ENV), cluster)
  CXX = $(CXX_TACC)
endif

# Optional compiler flags, for debug and performance
CXXFLAGS_BASE = -g -Wall -O2
ifeq ($(COMPILE_ENV), local)
  CXXFLAGS = $(CXXFLAGS_BASE)
endif
ifeq ($(COMPILE_ENV), cluster)
  CXXFLAGS = $(CXXFLAGS_BASE) -xhost
endif

# Eigen include directory for installation on Ubuntu Linux with apt
EIGEN_INC_LINUX=/usr/include/eigen3

# Mandatory compiler flags, with optional ones appended
ifeq ($(COMPILE_ENV), local)
  ALL_CXXFLAGS = -I$(EIGEN_INC_LINUX) -fopenmp $(CXXFLAGS)
endif
ifeq ($(COMPILE_ENV), cluster)
  ALL_CXXFLAGS = -I${TACC_EIGEN_INC} -qopenmp $(CXXFLAGS)
endif

.PHONY: clean

pilu : pilu.cpp
	$(CXX) $(ALL_CXXFLAGS) -o pilu pilu.cpp

clean : 
	rm -f pilu
