# HHRT: Hybrid Hierarchical Runtime library

## Overview

The objective of Hybrid Hierarchical Runtime (HHRT) library is to extend applications' supportable problem scales.
The main targets of HHRT are applications whose problem scales have been limited by the capacity of upper memory layer, such as GPU device memory in GPU clusters.
By using GPUs, applications written with CUDA and MPI have been enjoyed high computing speed and memory bandwidth of GPUs, however, most of them are designed as "in-core", or supported problem sizes are determined by device memory capacity, in spite of the existence of larger memory (storage) layers, including host memory and file systems.

The problem scales of such applications are expanded by executing them with only slight modifications on top of HHRT library.
Basically we assume that the target applications of HHRT have the following characteristics:

* Written on top of CUDA for GPU computation and MPI for inter-process/node communication
* The applications consist of multiple processes working cooperatively
* Data structure that is frequently accessed by each process are put on upper memory layer

For example, many stencil applications on GPU clusters have these characteristics, since they are written in MPI to support multiple nodes, and regions to be simulated are distributed among processes so that each process has smaller local region than device memory capacity.

## Usage

* Edit make.inc for your environment.
* "make" makes lib/libhhrt.a, which is the HHRT library
  Also a sample program 7p2dd (7-point stencil with 2d division) is made.
* Edit application code by adding
  `#include <hhrt.h>`
* Application should be compiled and linked with
  -I[HHRT_ROOT]/lib
  -L[HHRT_ROOT]/lib

## Execution of the sample program

The usage of 7p2dd sample is as follows:

% ./7p2dd [-p PY PZ] [-bt BT] [-nt NT] [NX NY NZ]

like

% ./7p2dd 512 512 512

But this does not exceed the device memory capacity.
In order to do it, process oversubscription is used.

### On single GPU/node envorinment

% mpirun -np 8 ./7p2dd -p 2 4 1024 1024 2048

In this execution, the total problem size (1024x1024x2048xsizeof(float)x2 = 8MiB) exceeds the device memory capacity.
However, the execution speed is very slow for costs for HHRT's implicit memory swapping.

In order to this relieve it, this sample is equipped with "temporal blocking" that improves locality.

% mpirun -np 8 ./7p2dd -p 2 4 -bt 8 1024 1024 2048

Here "-bt 8" specifies temporal block size.

### On GPU cluster

## References

Toshio Endo, Guanghao Jin. Software Technologies Coping with Memory Hierarchy of GPGPU Clusters for Stencil Computations . In Proceedings of IEEE Cluster Computing (CLUSTER2014), pp.132-139, Madrid, September 25, 2014. [DOI:10.1109/CLUSTER.2014.6968747]

Toshio Endo, Yuki Takasaki, Satoshi Matsuoka. Realizing Extremely Large-Scale Stencil Applications on GPU Supercomputers . In Proceedings of The 21st IEEE International Conference on Parallel and Distributed Systems (ICPADS 2015), pp. 625-632, Melbourne, December, 2015. 
[DOI: 10.1109/ICPADS.2015.84]

## Contact

Toshio Endo (endo-at-is.titech.ac.jp)

Twitter: toshioendo
