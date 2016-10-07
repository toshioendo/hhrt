#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "hhrt_impl.h"

/* CUDA interface */

dev *HH_curdev()
{
  if (HHL->curdevid < 0) {
    fprintf(stderr, 
	    "[HH_curdev@p%d] ERROR: curdevid is not set\n",
	    HH_MYID);
    exit(1);
  }
  return &HHS->devs[HHL->curdevid];
}

heap *HH_curdevheap()
{
  if (HHL->curdevid < 0) {
    fprintf(stderr, 
	    "[HH_curdevheap@p%d] ERROR: curdevid is not set\n",
	    HH_MYID);
    exit(1);
  }
  heap *h = HHL2->devheaps[HHL->curdevid];
  if (h == NULL) {
    fprintf(stderr, 
	    "[HH_curdevheap@p%d] ERROR: devid %d is not initialized\n",
	    HH_MYID, HHL->curdevid);
    exit(1);
  }
  return h;
}

static int initSharedDevmem(dev *d)
{
  cudaError_t crc;
  int devid = d->devid;
  crc = cudaSetDevice(devid);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] cudaSetDevice for heap failed!\n",
	    HHS->hostname, devid);
    exit(1);
  }
  
  size_t heapsize = d->default_heapsize;
  crc = cudaMalloc(&d->hp_baseptr0, heapsize * HHS->ndh_slots);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] cudaMalloc(%ldMiB) for heap failed!\n",
	    HHS->hostname, devid, heapsize>>20);
    exit(1);
  }

  crc = cudaIpcGetMemHandle(&d->hp_handle, d->hp_baseptr0);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] cudaIpcGetMemhandle for heap failed!\n",
	    HHS->hostname, devid);
    exit(1);
  }
#if 1
  fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] exporting pointer %p\n",
	  HHS->hostname, devid, d->hp_baseptr0);
#endif
  return 0;
}


static int initDev(int i, int lsize, /*int size,*/ hhconf *confp)
{
  cudaError_t crc;
  dev *d = &HHS->devs[i];

  d->devid = i;
  HH_mutex_init(&d->ml);
  HH_mutex_init(&d->userml);
  if (confp->devmem > 0) {
    d->memsize = confp->devmem;
  }
  else {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    d->memsize = prop.totalGlobalMem;
#if 0
    fprintf(stderr, "[HHRT] dev %d: memsize=%ld\n",
	    i, d->memsize);
#endif
  }
  
  /* determine device heap size */
  size_t avail = d->memsize - d->memsize/64L;
#ifdef USE_CUDA_MPS
  avail -= DEVMEM_USED_BY_PROC * 1;
#else
  avail -= DEVMEM_USED_BY_PROC * lsize;
#endif
  d->default_heapsize = avail / HHS->ndh_slots;
  d->default_heapsize = (d->default_heapsize/HEAP_ALIGN)*HEAP_ALIGN;
  
#if 1
  fprintf(stderr, "[HH:initDev@%s:dev%d] memsize=%ld -> default_heapsize=%ld\n",
	  HHS->hostname, i, d->memsize, d->default_heapsize);
#endif

  initSharedDevmem(d);

  int ih;
  /* setup heap slots on device */
  for (ih = 0; ih < HHS->ndh_slots; ih++) {
    d->dhslot_users[ih] = -1;
  }
  
  d->np_in = 0;
  d->np_out = 0;
  return 0;
}

// Init CUDA device structures for GPUs on this node
// Called only by leader (local rank=0) process on the node 
int HH_cudaInitNode(hhconf *confp)
{
  int mydevid;
  cudaError_t crc;
  int ndevs;
  int lsize = HHS->nlprocs;

  ndevs = -1;
  crc = cudaGetDeviceCount(&ndevs);
  if (crc != cudaSuccess || ndevs < 0 || ndevs > MAX_LDEVS) {
    fprintf(stderr, "[HH:initNode@%s] cudaGetDeviceCount ERROR. rc=%d, ndevs=%d\n",
	    HHS->hostname, crc, ndevs);
    exit(1);
  }
  assert(ndevs <= MAX_LDEVS);
  HHS->ndevs = ndevs;

#if 1
  fprintf(stderr, "[HH:initNode@%s] I have %d visible devices\n",
	  HHS->hostname, ndevs);
#endif

  HHS->ndh_slots = confp->dh_slots;
  if (HHS->ndh_slots > lsize) HHS->ndh_slots = lsize;

  crc = cudaGetDevice(&mydevid);
  
  for (int i = 0; i < ndevs; i++) {
    initDev(i, lsize, /*size,*/ confp);
  }
  
  crc = cudaSetDevice(mydevid); // restore

  return 0;
}

int HH_cudaInitProc()
{
  HHL->curdevid = -1;

  // see also devheap::reserveRes()
  HHL->hpid = HHL->lrank % HHS->ndh_slots;

  for (int id = 0; id < MAX_LDEVS; id++) {
    HHL2->devheaps[id] = NULL;
  }

  // We do no substantial work here
  // Instead, process init per GPU is lazily done in HH_cudaCheckDev()
  return 0;
}

// Start using device structure if this process uses it for first time
// This may be blocked
int HH_cudaCheckDev()
{
  if (HHL->curdevid < 0) {
    cudaError_t crc;
    // get device no
    crc = cudaGetDevice(&HHL->curdevid);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:inic_proc@p%d] cudaGetDevice failed. ignored\n", HH_MYID);
      HHL->curdevid = 0;
    }
  }
    if (HHL->curdevid >= MAX_LDEVS) {
    fprintf(stderr, 
	    "[HH_cudaCheckDev@p%d] ERROR: curdevid %d is invalid\n",
	    HH_MYID, HHL->curdevid);
    exit(1);
  }
  heap *h = HHL2->devheaps[HHL->curdevid];

  if (h != NULL) {
    // device heap is already initialized. Do nothing
#if 0
    fprintf(stderr, 
	    "[HH_cudaCheckDev@p%d] devid %d is already initialized.\n",
	    HH_MYID, HHL->curdevid);
#endif
    return 0;
  }

  double st = Wtime_prt(), et;
#ifdef HHLOG_SCHED
  fprintf(stderr, 
	  "[HH_cudaCheckDev@p%d] [%.2lf] First use of devid %d. initialize...\n",
	  HH_MYID, st, HHL->curdevid);
#endif

  // swap out existing heaps
  HH_swapOutIfOver();

#ifdef HHLOG_SCHED
  fprintf(stderr, 
	  "[HH_cudaCheckDev@p%d] After swapOutIfOver\n",
	  HH_MYID);
#endif

  // create heap structure for current GPU
  h = HH_devheapCreate(HH_curdev());
  assert(HHL2->nheaps < MAX_HEAPS-1);
  HHL2->heaps[HHL2->nheaps++] = h;
  HHL2->devheaps[HHL->curdevid] = h;

  // blocked until heaps are accessible
  HH_sleepForMemory();

  et = Wtime_prt();
#ifdef HHLOG_SCHED
  fprintf(stderr, 
	  "[HH_cudaCheckDev@p%d] [%.2lf] now process restarts. checkDev %.2lfsec\n",
	  HH_MYID, et, et-st);
  //HH_printHostMemStat();
#endif

  return 0;
}

/*********************** wrappers of user API ***************/

cudaError_t HHcudaMemcpy(void * dst,
		       const void * src,
		       size_t count,
		       enum cudaMemcpyKind kind 
		       )
{
  HH_cudaCheckDev();
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t HHcudaMemcpyAsync(void * dst,
			    const void * src,
			    size_t count,
			    enum cudaMemcpyKind kind,
			    cudaStream_t stream 
			    )
{
  HH_cudaCheckDev();
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
  HH_cudaCheckDev();
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
  HH_cudaCheckDev();
  return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}

/* Device memory related used API */
/* Wrappers of cudaMalloc/cudaFree */

cudaError_t HHcudaMalloc(void **pp, size_t size)
{
  HH_cudaCheckDev();

  void *p = NULL;

  if (HHL->devmode == HHDEV_NORMAL) {
  }

  p = HH_curdevheap()->alloc(size);
  *pp = p;
  return cudaSuccess;
}

cudaError_t HHcudaFree(void *p)
{
  HH_cudaCheckDev();

  if (p == NULL) return cudaSuccess;

  int rc;
  rc = HH_curdevheap()->free(p);
  if (rc != 0) {
    return cudaErrorInvalidDevicePointer;
  }
  return cudaSuccess;
}

#ifdef USE_SWAPHOST
cudaError_t HHcudaHostAlloc(void ** pp, size_t size, unsigned int flags)
{
  HH_cudaCheckDev();

  void *p;
  if (HH_MYID == 0) {
    fprintf(stderr, "[HHcudaHostAlloc@p%d] WARNING: normal malloc is used now\n",
	    HH_MYID);
  }
  p = HHL2->hostheap->alloc(size);
  *pp = p;
  return cudaSuccess;
}

cudaError_t HHcudaMallocHost(void ** pp, size_t size)
{
  HH_cudaCheckDev();

  return HHcudaHostAlloc(pp, size, cudaHostAllocDefault);
}
#endif

/*****************************************************/
// will be obsolete
int HH_devLock()
{
  dev *d = HH_curdev();
  pthread_mutex_lock(&d->userml);
  return 0;
}

// will be obsolete
int HH_devUnlock()
{
  dev *d = HH_curdev();
  pthread_mutex_unlock(&d->userml);
  return 0;
}

// will be obsolete
int HH_devSetMode(int mode)
{
  HH_cudaCheckDev();

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

  HH_cudaCheckDev(); // init heap for new device if not

  return crc;
}

