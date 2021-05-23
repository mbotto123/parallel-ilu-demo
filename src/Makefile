CC     = icpc
CFLAGS = -g -Wall -O2 -xhost -qopenmp

LIBS   = -I${TACC_EIGEN_INC} 

pilu.exe : pilu.cpp
	$(CC) $(CFLAGS) $(LIBS) -o pilu.exe pilu.cpp

clean : 
	rm -rf pilu.exe
