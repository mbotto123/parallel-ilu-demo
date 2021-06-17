# C++ Compiler
CXX          = icpc

# Optional compiler flags, for debug and performance
CXXFLAGS     = -g -Wall -O2 -xhost

# Mandatory compiler flags, with optional ones appended
ALL_CXXFLAGS = -I${TACC_EIGEN_INC} -qopenmp $(CXXFLAGS)

pilu.exe : pilu.cpp
	$(CXX) $(ALL_CXXFLAGS) -o pilu.exe pilu.cpp

clean : 
	rm -rf pilu.exe
