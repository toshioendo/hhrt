#ifndef HHRT_COMMON_H
#define HHRT_COMMON_H

#include <mpi.h>
#include <pthread.h>
#include <sys/time.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif
#if 0
}
#endif

/* configure */
#define USE_SWAPHOST

//#define USE_MMAPSWAP // testing. use mmaped host buffer, instead of fileheap

#define HHRT_VERSION 20161207

#  define HH(X) HH##X

enum {
  HHMADV_FREED = 0,
  HHMADV_NORMAL,
  HHMADV_CANDISCARD,
  HHMADV_READONLY,
};

enum {
  HHDEV_NORMAL = 0,
  HHDEV_NOTUSED,
};

/************************************************/
/* user programmer writes this function */
int HH_initHeap(size_t heapsize);
int HH_finalizeHeap();
int HH_myid();
int HH_nvps();

/****************************************/
/* memory management */

int HH_madvise(void *p, size_t size, int kind);

int HH_devLock();
int HH_devUnlock();
int HH_devSetMode(int kind);

/****************************************/
/* scheduling */
int HH_yield();

/****************************************/
/* MPI-1 compatible interface */

int HHMPI_Init(int *argcp, char ***argvp);
int HHMPI_Finalize();

int HHMPI_Send( void *buf, int count, MPI_Datatype dt, int dst, 
		int tag, MPI_Comm comm );
int HHMPI_Recv( void *buf, int count, MPI_Datatype dt, int src, 
	       int tag, MPI_Comm comm, MPI_Status *status );
int HHMPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
		int tag, MPI_Comm comm, MPI_Request *reqp);
int HHMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
		int tag, MPI_Comm comm, MPI_Request *reqp);

int HHMPI_Wait(MPI_Request *mreqp, MPI_Status *statp);
int HHMPI_Waitall(int n, MPI_Request mreqs[], MPI_Status stats[]);

int HHMPI_Barrier(MPI_Comm comm);
int HHMPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, 
		 MPI_Comm comm );
int HHMPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
		 MPI_Op op, int root, MPI_Comm comm);
int HHMPI_Allreduce(void *sendbuf, void *recvbuf, int count,
		    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int HHMPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
		 void *recvbuf, int recvcount, MPI_Datatype recvtype, 
		 int root, MPI_Comm comm);
int HHMPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
		    void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int HHMPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
		  void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
		  MPI_Comm comm);
int HHMPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
		   void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int HHMPI_Comm_split(MPI_Comm comm, int color, int key,
		     MPI_Comm *newcomm);


#ifdef USE_CUDA
/**************************************************/
/* CUDA compatible interface */
cudaError_t HHcudaMalloc(void **pp, size_t size);
cudaError_t HHcudaFree(void *p);

cudaError_t HHcudaSetDevice(int device);

cudaError_t HHcudaMemcpy(void * dst,
		       const void * src,
		       size_t count,
		       enum cudaMemcpyKind kind 
			 );
cudaError_t HHcudaMemcpyAsync(void * dst,
			    const void * src,
			    size_t count,
			    enum cudaMemcpyKind kind,
			    cudaStream_t stream 
			      );
cudaError_t HHcudaMemcpy2D(void * dst,
			 size_t dpitch,
			 const void * src,
			 size_t spitch,
			 size_t width,
			 size_t height,
			 enum cudaMemcpyKind kind 
			   );
cudaError_t HHcudaMemcpy2DAsync(void * dst,
			      size_t dpitch,
			      const void * src,
			      size_t spitch,
			      size_t width,
			      size_t height,
			      enum cudaMemcpyKind kind,
			      cudaStream_t stream 
			      );
#endif

/**************/
#ifdef USE_SWAPHOST
/* host memory interface */
void *HHmalloc(size_t size);
void *HHcalloc(size_t nmemb, size_t size);
void HHfree(void *p);
void *HHmemalign(size_t boundary, size_t size);
void *HHvalloc(size_t size);

#ifdef USE_CUDA
cudaError_t HHcudaHostAlloc(void ** pp, size_t size, unsigned int flags);
cudaError_t HHcudaMallocHost(void ** pp, size_t size);
#endif

#endif

/**************/

#ifdef __cplusplus
}
#endif

#endif /* HHRT_COMMON_H */
