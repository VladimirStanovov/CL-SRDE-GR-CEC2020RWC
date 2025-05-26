# Dependencies: 

Ubuntu 20.04
OpenMPI 4.0.3
GCC 11.4.0
Eigen 3.4.0

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