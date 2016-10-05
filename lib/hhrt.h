/* Included by user programs of HHRT */

#ifndef HHRT_H
#define HHRT_H

#include "hhrt_common.h"

#ifdef __cplusplus
extern "C" {
#endif
#if 0
}
#endif

/* MPI Interface */

/* Tricks in order to users' reduce programming costs */

#define MPI_Init HHMPI_Init
#define MPI_Finalize HHMPI_Finalize

#define MPI_Send HHMPI_Send
#define MPI_Recv HHMPI_Recv
#define MPI_Isend HHMPI_Isend
#define MPI_Irecv HHMPI_Irecv
#define MPI_Wait HHMPI_Wait
#define MPI_Waitall HHMPI_Waitall

#define MPI_Barrier HHMPI_Barrier
#define MPI_Bcast HHMPI_Bcast
#define MPI_Reduce HHMPI_Reduce
#define MPI_Allreduce HHMPI_Allreduce
#define MPI_Gather HHMPI_Gather
#define MPI_Allgather HHMPI_Allgather
#define MPI_Scatter HHMPI_Scatter
#define MPI_Alltoall HHMPI_Alltoall
#define MPI_Comm_split HHMPI_Comm_split

/**************************************************/
/* CUDA interface */

/* Tricks in order to users' reduce programming costs */

#define cudaMalloc HHcudaMalloc
#define cudaFree HHcudaFree
#define cudaSetDevice HHcudaSetDevice

#define cudaMemcpy HHcudaMemcpy
#define cudaMemcpyAsync HHcudaMemcpyAsync
#define cudaMemcpy2D HHcudaMemcpy2D
#define cudaMemcpy2DAsync HHcudaMemcpy2DAsync

#ifdef USE_SWAPHOST
#define malloc HHmalloc
#define calloc HHcalloc
#define free HHfree
#define memalign HHmemalign
#define valloc HHvalloc

#define cudaMallocHost HHcudaMallocHost
#define cudaHostAlloc HHcudaHostAlloc

#endif

#ifdef USE_MMAPSWAP
#warning USE_MMAPSWAP
#endif

#ifdef __cplusplus
}
#endif

#endif /* HHRT_H */
