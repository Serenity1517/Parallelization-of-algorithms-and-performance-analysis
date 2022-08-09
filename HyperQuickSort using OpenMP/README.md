# Introduction
This is an implementation of HyperQuickSort, which is a parallelized quicksort implemented using the hypercube topology.
It is implemented in C using OpenMP.

The details related to the algorithm, implementation and performance analysis can be found in the design document.

# Instructions for running the program

  To compile: gcc -o hyperquicksort_openmp -fopenmp hyperquicksort_openmp.c
    
  To run:  ./hyperquicksort_openmp <no_of_elements(N)> <choice>
  Here <choice> can be 0 or 1:
   0: print the time taken for sorting
   1: print the sorted array as well as time taken for sorting
   
  e.g.: ./hyperquicksort_openmp 32768 0
   
**NOTE: The number of threads spawned will always be a power of 2. This can be changed by altering a MACRO in code.**