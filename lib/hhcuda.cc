#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "hhrt_impl.h"

/* CUDA interface */

// Initialize device structure if it is used for first time
// This may be blocked
int HH_checkDev()
{
  if (HHL->curdevid < 0 || HHL->curdevid >= MAX_LDEVS) {
    fprintf(stderr, 
	    "[HH_checkDev@p%d] ERROR: curdevid %d is invalid\n",
	    HH_MYID, HHL->curdevid);
    exit(1);
  }
  heap *h = HHL2->devheaps[HHL->curdevid];

  if (h != NULL) {
    // device heap is already initialized. Do nothing
#if 0
    fprintf(stderr, 
	    "[HH_checkDev@p%d] devid %d is already initialized.\n",
	    HH_MYID, HHL->curdevid);
#endif
    return 0;
  }

  double st = Wtime_prt(), et;
  fprintf(stderr, 
	  "[HH_checkDev@p%d] [%.2lf] First use of devid %d. initialize...\n",
	  HH_MYID, st, HHL->curdevid);

  // swap out existing heaps
  HH_swapOutIfOver();

  fprintf(stderr, 
	  "[HH_checkDev@p%d] After swapOutIfOver\n",
	  HH_MYID);

  // create heap structure for current GPU
  h = HH_devheapCreate(HH_curdev());
  assert(HHL2->nheaps < MAX_HEAPS-1);
  HHL2->heaps[HHL2->nheaps++] = h;
  HHL2->devheaps[HHL->curdevid] = h;

  // blocked until heaps are accessible
  HH_sleepForMemory();

  et = Wtime_prt();
  fprintf(stderr, 
	  "[HH_checkDev@p%d] [%.2lf] now process restarts. checkDev %.2lfsec\n",
	  HH_MYID, et, et-st);

  HH_printHostMemStat();


  return 0;
}


cudaError_t HHcudaMemcpy(void * dst,
		       const void * src,
		       size_t count,
		       enum cudaMemcpyKind kind 
		       )
{
  HH_checkDev();
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t HHcudaMemcpyAsync(void * dst,
			    const void * src,
			    size_t count,
			    enum cudaMemcpyKind kind,
			    cudaStream_t stream 
			    )
{
  HH_checkDev();
  return cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t HHcudaMemcpy2D(void * dst,
			 size_t dpitch,
			 const void * src,
			 size_t spitch,
			 size_t width,
			 size_t height,
			 enum cudaMemcpyKind kind 
			 )
{
  HH_checkDev();
  return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}


cudaError_t HHcudaMemcpy2DAsync(void * dst,
			      size_t dpitch,
			      const void * src,
			      size_t spitch,
			      size_t width,
			      size_t height,
			      enum cudaMemcpyKind kind,
			      cudaStream_t stream 
			      )
{
  HH_checkDev();
  return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}


/*****************************************************/

int HH_devLock()
{
  dev *d = HH_curdev();
  pthread_mutex_lock(&d->userml);
  return 0;
}

int HH_devUnlock()
{
  dev *d = HH_curdev();
  pthread_mutex_unlock(&d->userml);
  return 0;
}

int HH_devSetMode(int mode)
{
  HH_checkDev();

  assert(HHL->pmode == HHP_RUNNING);
  int prevmode = HHL->devmode;
  HHL->devmode = mode;
  if (prevmode == HHDEV_NOTUSED && mode == HHDEV_NORMAL) {
    /* This may incur swap in */
#ifdef HHLOG_SCHED
    fprintf(stderr, "[HH_devSetMode@p%d] calling sleepForMemory...\n",
	    HH_MYID);
#endif
    HH_sleepForMemory();
  }

  return 0;
}

cudaError_t HHcudaSetDevice(int devid)
{
  HHL->curdevid = devid;

  cudaError_t crc = cudaSetDevice(devid);

  HH_checkDev(); // init heap for new device if not

  return crc;
}

