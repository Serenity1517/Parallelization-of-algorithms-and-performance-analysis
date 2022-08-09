# Introduction
This is a CUDA-based implementation of the Blelloch Scan algorithm, which is a parallel algorithm for computing scans (prefix sums / maxm. / minm.).

The Design Document contains details about the algorithm and performance analysis.

# Running the code
The input array size can be controlled by editing the ‘ARR_SIZE’ macro defined at the top (line 17).
The operation to be performed (addition / minimum / maximum) can be controlled by editing the ‘OPERATOR’ macro defined at the top (line 16)
To show verbose output, the code can be compiled with an optional command line argument ‘v’ or ‘V’. This will print either the entire or first 100 elements from input and output arrays.

To compile, type : nvcc scan_CUDA.c
To run, type : ./a.out
For verbose output , type : ./a.out v OR ./a.out V
