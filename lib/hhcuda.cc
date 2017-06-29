#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "hhrt_impl.h"

#ifdef USE_CUDA

/* CUDA interface */

dev *HH_curdev()
{
  if (HHL->cuda.curdevid < 0) {
    fprintf(stderr, 
	    "[HH_curdev@p%d] ERROR: cuda.curdevid is not set\n",
	    HH_MYID);
    exit(1);
  }
  return &HHS->cuda.devs[HHL->cuda.curdevid];
}

devheap *HH_curdevheap()
{
  if (HHL->cuda.curdevid < 0) {
    fprintf(stderr, 
	    "[HH_curdevheap@p%d] ERROR: cuda.curdevid is not set\n",
	    HH_MYID);
    exit(1);
  }
  heap *h = HHL2->devheaps[HHL->cuda.curdevid];
  if (h == NULL) {
    fprintf(stderr, 
	    "[HH_curdevheap@p%d] ERROR: devid %d is not initialized\n",
	    HH_MYID, HHL->cuda.curdevid);
    exit(1);
  }
  return dynamic_cast<devheap *>(h);
}

int HH_resetCudaAll()
{
  for (int id = 0; id < MAX_LDEVS; id++) {
    devheap *dh = dynamic_cast<devheap *>(HHL2->devheaps[id]);
    if (dh != NULL) {
      dh->resetCuda();
    }
  }
  return 0;
}

int HH_printMemHandle(FILE *out, cudaIpcMemHandle_t *handle)
{
  int i;
  unsigned char *p = (unsigned char*)(void*)handle;
  for (i = 0; i < sizeof(cudaIpcMemHandle_t); i++) {
    fprintf(out, "%02x", p[i]);
  }
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
  crc = cudaMalloc(&d->hp_baseptr0, heapsize * HHS->cuda.ndh_slots);
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
#if 0
  fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] exporting pointer %p\n",
	  HHS->hostname, devid, d->hp_baseptr0);
  fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] handle is ", HHS->hostname, devid);
  HH_printMemHandle(stderr, &d->hp_handle);
  fprintf(stderr, "\n");
#endif
  return 0;
}

static int initDev(int devid, int lsize, hhconf *confp)
{
  cudaError_t crc;
  dev *d = &HHS->cuda.devs[devid];

  cudaSetDevice(devid);

  d->devid = devid;
  HH_mutex_init(&d->ml);
  HH_mutex_init(&d->userml);

  // get GPU device memory information
  size_t fsize, tsize;
  crc = cudaMemGetInfo(&fsize, &tsize);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HHRT] dev %d: GetMemInfo failed, rc=%d\n",
	    devid, crc);
  }
#if 0
  fprintf(stderr, "[HHRT] dev %d: GetMemInfo -> free=%ld, total=%ld\n",
	  devid, fsize, tsize);
#endif
  
  if (confp->devmem > 0) {
    d->memsize = confp->devmem;
  }
  else {
    d->memsize = tsize;
  }

  // determine device heap size per proc
  size_t avail = tsize - tsize/64L;
  size_t tax_per_proc = tsize-fsize;

#if 1
  fprintf(stderr, "[HH:initDev:dev%d] Estimated context size per proc -> %ld\n",
	  devid, tax_per_proc);
#endif

#ifdef USE_CUDA_MPS
  avail -= tax_per_proc * 1;
#elif defined USE_DEVRESET
  int nlp = confp->nlphost;
  if (lsize < nlp) nlp = lsize;
  avail -= tax_per_proc * (nlp + 2);
#else
  avail -= tax_per_proc * lsize;
#endif
  d->default_heapsize = avail / HHS->cuda.ndh_slots;
  d->default_heapsize = (d->default_heapsize/HEAP_ALIGN)*HEAP_ALIGN;
  
#if 1
  fprintf(stderr, "[HH:initDev@%s:dev%d] memsize=%ld -> default_heapsize=%ld\n",
	  HHS->hostname, devid, d->memsize, d->default_heapsize);
#endif

  initSharedDevmem(d);

  int ih;
  /* setup heap slots on device */
  for (ih = 0; ih < HHS->cuda.ndh_slots; ih++) {
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
  HHS->cuda.ndevs = ndevs;

#if 1
  fprintf(stderr, "[HH:initNode@%s] I have %d visible devices\n",
	  HHS->hostname, ndevs);
#endif

  HHS->cuda.ndh_slots = confp->dh_slots;
  if (HHS->cuda.ndh_slots > lsize) HHS->cuda.ndh_slots = lsize;

  int orgdevid;
  crc = cudaGetDevice(&orgdevid);
  
  for (int i = 0; i < ndevs; i++) {
    initDev(i, lsize, /*size,*/ confp);
  }
  
  crc = cudaSetDevice(orgdevid); // restore

  return 0;
}

int HH_cudaInitProc()
{
  HHL->cuda.curdevid = -1;

  // see also devheap::reserveRes()
  HHL->cuda.hpid = HHL->lrank % HHS->cuda.ndh_slots;

  for (int id = 0; id < MAX_LDEVS; id++) {
    HHL2->devheaps[id] = NULL;
  }

  // We do no substantial work here
  // Instead, process init per GPU is lazily done in HH_cudaCheckDev()
  return 0;
}

// Start using device structure if this process uses it for first time
// This function creates device heap structure
// This may be blocked
int HH_cudaCheckDev()
{
  if (HHL->cuda.curdevid < 0) {
    cudaError_t crc;
    // get device no
    crc = cudaGetDevice(&HHL->cuda.curdevid);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:inic_proc@p%d] cudaGetDevice failed. ignored\n", HH_MYID);
      HHL->cuda.curdevid = 0;
    }
  }
  if (HHL->cuda.curdevid >= HHS->cuda.ndevs) {
    fprintf(stderr, 
	    "[HH_cudaCheckDev@p%d] ERROR: cuda.curdevid %d is invalid\n",
	    HH_MYID, HHL->cuda.curdevid);
    exit(1);
  }

  heap *h = HHL2->devheaps[HHL->cuda.curdevid];
  if (h != NULL) {
    // device heap is already initialized. Do nothing
#if 0
    fprintf(stderr, 
	    "[HH_cudaCheckDev@p%d] devid %d is already initialized.\n",
	    HH_MYID, HHL->cuda.curdevid);
#endif
    return 0;
  }

  // Do initialization for this device

  double st = Wtime_prt(), et;
#ifdef HHLOG_SCHED
  fprintf(stderr, 
	  "[HH_cudaCheckDev@p%d] [%.2lf] First use of devid %d. initialize...\n",
	  HH_MYID, st, HHL->cuda.curdevid);
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
  HHL2->devheaps[HHL->cuda.curdevid] = h;

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

  void *p = HH_curdevheap()->alloc(size);
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
    if (HHL2->conf.pin_hostbuf == 0) {
      fprintf(stderr, "[HHcudaHostAlloc@p%d] WARNING: normal malloc is used now\n",
	      HH_MYID);
    }
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

cudaError_t HHcudaSetDevice(int devid)
{
  cudaError_t crc;
  assert(devid >= 0 && devid < HHS->cuda.ndevs);
  crc = cudaSetDevice(devid);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HHcudaSetDevice@p%d] cudaSetDevice(%d) FAILED!!!\n", HH_MYID);
    return crc;
  }
#if 1
  fprintf(stderr, "[HHcudaSetDevice@p%d] cudaSetDevice(%d) ok\n",
	  HH_MYID, devid);
#endif

  HHL->cuda.curdevid = devid;
  HH_cudaCheckDev(); // init heap for new device if not

  return crc;
}


#endif // USE_CUDA
