# Start from latest LTS Ubuntu at time of writing this Dockerfile
FROM ubuntu:24.04

# Install depdendencies of PILU code
RUN apt update && \
    apt install -y g++ make libeigen3-dev

# Copy source code into the image
COPY src/pilu.cpp src/Makefile /home/parallel-ilu-demo/src/

# Copy input data into the image
COPY input/Pres_Poisson/Pres_Poisson.mtx \
     /home/parallel-ilu-demo/input/Pres_Poisson/

WORKDIR /home/parallel-ilu-demo/src

# Compile with configuration for an Ubuntu Linux environment
RUN make pilu COMPILE_ENV=local

ENV PATH="/home/parallel-ilu-demo/src:$PATH"
