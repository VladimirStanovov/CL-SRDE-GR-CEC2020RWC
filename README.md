# CL-SRDE-GR algorithm for the CEC 2020 Real World Constrained Benchmark Functions in C++

The C++ implementation of the Constrained Linear population size reduction Success Rate based adaptive Differential Evolution with Gradient based Repair. Contains two versions, one with parallel calculation of runs using MPI, and another without. Also contains the results post-processing and comparison code in Jupyter notebook.

# Dependencies: 

Ubuntu 20.04
OpenMPI 4.0.3
GCC 11.4.0
Eigen 3.4.0 (downloaded separately!)

Should work on Windows as well.

# Usage:

To compile CL-SRDE-GR with MPI support run:

mpic++ -std=c++14 -O3 -fexpensive-optimizations CL-SRDE-GR.cpp -o CL-SRDE-GR.out

Then run with 4 threads, for example:

mpirun -np 4 ./CL-SRDE-GR.out

To compile without MPI support:

g++ -std=c++14 -O3 -fexpensive-optimizations CL-SRDE-GR_no_MPI.cpp -o CL-SRDE-GR_no_MPI.out

And run with:

./CL-SRDE-GR_no_MPI.out

# CEC 2020 benchmark code

The original code is written in Matlab by Abhishek Kumar, it can be found at https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation

The C++ implementation used here is available at https://github.com/VladimirStanovov/CEC-2020-RWC-C/tree/main