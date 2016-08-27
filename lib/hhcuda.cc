#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "hhrt_impl.h"

#define HOSTNAMELEN 64

/* CUDA interface */
/* Exceptionally, cudaMalloc/cudaFree are put in hhmem.cc */
/* Also some HHRT specific APIs are here */

membuf *HH_findMembuf(void *p)
{
  return HH_curdevheap()->findMembuf(p);
}

static ssize_t convDevptr2Offs(void *dp, swapper *swapper)
{
  membuf *mbp;
  size_t poffs; // offset inside this memory object
  mbp = HH_findMembuf(dp);
  if (mbp == NULL) {
    fprintf(stderr, "[convDevptr@p%d] ERROR: cannot find pointer %p\n",
	    HH_MYID, dp);
    assert(0);
    exit(1);
  }

  if (mbp->soffs < 0) {
    fprintf(stderr, "[convDevptr@p%d] ERROR: no corresponding host pointer for %p\n",
	    HH_MYID, dp);
    assert(0);
    exit(1);
  }
  poffs = ppsub(dp, HH_curdevheap()->heapptr) - mbp->doffs;
  assert(poffs >= 0 && poffs < mbp->size);

  ssize_t soffs = mbp->soffs + poffs;
#if 0
  fprintf(stderr, "[convDevptr@p%d] convert dp %p -> swap offset %p (offset in obj = %ld)\n",
	  HH_MYID, dp, soffs, poffs);
#endif
  return soffs;
}

static int memcpy_inner(void * dst,
			const void * src0,
			size_t count,
			enum cudaMemcpyKind kind 
			)
{
  void *src = const_cast<void *>(src0);
  assert (HHL->dmode != HHD_ON_DEV);

  if (kind == cudaMemcpyDefault) {
    fprintf(stderr, "[convDstSrc] cudaMemcpyDefault not supported yet!!!\n");
    exit(1);
  }

#if 0
  swapper *srcSwapper = NULL;
  swapper *dstSwapper = NULL;
  ssize_t srcoffs;
  ssize_t dstoffs;
  if (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice) {
    dstSwapper = HHL2->default_devheap_swapper;
    dstoffs = convDevptr2Offs(dst, dstSwapper);
  }

  if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice) {
    srcSwapper = HHL2->default_devheap_swapper;
    srcoffs = convDevptr2Offs(src, srcSwapper);
  }

  size_t cur;
  /* divide into CHUNKSIZE */
  for (cur = 0; cur < count; cur += HH_CHUNKSIZE) {
    size_t lsize = HH_CHUNKSIZE;
    if (cur + lsize > count) lsize = count-cur;

    if (srcSwapper != NULL) {
      srcSwapper->read1(srcoffs, HHL2->copybuf, HHM_HOST, lsize);
    }
    if (dstSwapper != NULL) {
      dstSwapper->write1(dstoffs, HHL2->copybuf, HHM_HOST, lsize);
    }
    srcoffs += lsize;
    dstoffs += lsize;
  }
#else//

  fprintf(stderr, "[HH:memcpy_inner] ERROR: !HHD_ON_DEV mode not supported now \n");
  exit(1);

#endif//
  return 0;
}

static int memcpy2D_inner(void * dst,
			 size_t dpitch,
			 const void * src,
			 size_t spitch,
			 size_t width,
			 size_t height,
			 enum cudaMemcpyKind kind 
			  )
{
#if 1
  size_t i;
  for (i = 0; i < height; i++) {
    memcpy_inner(dst, src, width, kind);

    dst = piadd(dst, dpitch);
    src = piadd(src, spitch);
  }
#else
  void *hdst = dst;
  void *hsrc = const_cast<void *>(src);
  convDstSrc(&hdst, &hsrc, kind);

  size_t i;
  for (i = 0; i < height; i++) {
    memcpy(hdst, hsrc, width);

    hdst = piadd(hdst, dpitch);
    hsrc = piadd(hsrc, spitch);
  }
#endif

  return 0;
}

cudaError_t HHcudaMemcpy(void * dst,
		       const void * src,
		       size_t count,
		       enum cudaMemcpyKind kind 
		       )
{
  if (HHL->dmode == HHD_ON_DEV) {
    return cudaMemcpy(dst, src, count, kind);
  }
  else {
    memcpy_inner(dst, src, count, kind);
    return cudaSuccess;
  }
}

cudaError_t HHcudaMemcpyAsync(void * dst,
			    const void * src,
			    size_t count,
			    enum cudaMemcpyKind kind,
			    cudaStream_t stream 
			    )
{
  if (HHL->dmode == HHD_ON_DEV) {
    return cudaMemcpyAsync(dst, src, count, kind, stream);
  }
  else {
    memcpy_inner(dst, src, count, kind);
    return cudaSuccess;
  }
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
  if (HHL->dmode == HHD_ON_DEV) {
    return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
  }
  else {
    memcpy2D_inner(dst, dpitch, src, spitch, width, height, kind);
    return cudaSuccess;
  }
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
  if (HHL->dmode == HHD_ON_DEV) {
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
  }
  else {
    memcpy2D_inner(dst, dpitch, src, spitch, width, height, kind);
    return cudaSuccess;
  }
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
    HHL->pmode = HHP_RUNNING;
  }

  return 0;
}

cudaError_t HHcudaSetDevice(int device)
{
  /* device is interepreted as virtual device id */

  if (HHL->curdevid >= 0 && HHL->curdevid != device) {
    fprintf(stderr, 
	    "[HHcudaSetDevice@p%d] ERROR: Currently changing device id (other than default id %d) is NOT supported\n",
	    HH_MYID, HHL->curdevid);
    exit(1);
  }

  HHL->curdevid = device;

  return cudaSetDevice(device);
}
