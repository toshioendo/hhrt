#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>
#include "hhrt_impl.h"

/* Host memory management */

/* constants */
#define HOSTHEAP_PTR ((void*)0x700000000000)
#define HOSTHEAP_STEP (1L*1024*1024*1024)

heap *HH_hostheapCreate()
{
  heap *h;

  h = new hostheap();
  /* make memory hierarchy for hostheap */

  heap *filelayer = HHL2->fileheap;
  if (filelayer != NULL) {
    h->addLower(filelayer);
    filelayer->addUpper(h);
  }

  return h;
}

/* statistics about host memory for debug */
int HH_addHostMemStat(int kind, ssize_t incr)
{
  ssize_t s;
  ssize_t olds;
  assert(kind >= 0 && kind < HHST_MAX);
  olds = HHL->hmstat.used[kind];
  HHL->hmstat.used[kind] += incr;
  s = HHL->hmstat.used[kind];

  if (s < 0 || s > (ssize_t)128 << 30) {
    fprintf(stderr, "[HH_addHostMemStat@p%d] host mem usage (kind %s) %ldMB looks STRANGE.\n",
	    HH_MYID, hhst_names[kind], s>>20L);
  }

#if 0
  fprintf(stderr, "[HH_addHostMemStat@p%d] host mem usage (kind %s) %ldMB -> %ldMB.\n",
	  HH_MYID, hhst_names[kind], olds>>20L, s>>20L);
#endif
  return 0;
}

int HH_printHostMemStat()
{
  double t = Wtime_prt();
  int i;
  fprintf(stderr, "[HH_printHostMemStat@p%d] [%.2lf] ",
	  HH_MYID, t);
  for (i = 0; i < HHST_MAX; i++) {
    fprintf(stderr, "%s:%ldMB  ",
	    hhst_names[i], HHL->hmstat.used[i] >>20L);
  }
  fprintf(stderr, "\n");
  return 0;
}


/*****************************************************************/
// hostheap class (child class of heap)
hostheap::hostheap() : heap(0L)
{
  expandable = 1;
  expand_step = HOSTHEAP_STEP;

  heapptr = NULL;
  align = 512L;
  memkind = HHM_HOST;

  strcpy(name, "hostheap");

  swapfd = -1; // anonymous mmap
  mmapflags = MAP_PRIVATE|MAP_ANONYMOUS;

#if 0 && defined USE_CUDA
  cudaError_t crc;
  crc = cudaStreamCreate(&copystream);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:hostheap::init@p%d] ERROR: cudaStreamCreate failed (rc=%d)\n",
	    HH_MYID, crc);
    exit(1);
  }

  copyunit = (size_t)8*1024*1024;
  crc = cudaHostAlloc(&copybuf, copyunit, cudaHostAllocDefault);
  assert(crc == cudaSuccess);
#endif

  return;
}

int hostheap::finalize()
{
  heap::finalize();
  
  HH_lockSched();
  HHL->host_use = 0;
  HH_unlockSched();
  return 0;
}

/****************************************/
void *hostheap::allocCapacity(size_t offset, size_t size)
{
  void *mapp = piadd(heapptr, offset);
  void *resp;

  if (size == 0) {
    return NULL;
  }

  resp = mmap(mapp, size, 
	      PROT_READ|PROT_WRITE, mmapflags,
	      swapfd, offset /* (off_t)0 */);
  if (resp == MAP_FAILED) {
    fprintf(stderr, "[HH:%s::allocCapacity@p%d] ERROR: mmap(0x%lx, 0x%lx) failed\n",
	    name, HH_MYID, mapp, size);
    exit(1);
  }

  if (resp != mapp) {
    fprintf(stderr, "[HH:%s::allocCapacity@p%d] ERROR: unexpected pointer %p -> %p\n",
	    name, HH_MYID, mapp, resp);
    HHstacktrace();
    exit(1);
  }

#ifdef USE_CUDA
  if (HHL2->conf.pin_hostbuf) {
    double st = Wtime();
    cudaError_t crc;
    if (offset != (size_t)0) {
      crc = cudaHostUnregister(heapptr);
      if (crc != cudaSuccess) {
	fprintf(stderr, "[HH:%s::allocCapacity@p%d] ERROR: cudaHostUnregister failed (rc=%d)\n",
		name, HH_MYID, crc);
	HHstacktrace();
	exit(1);
      }
    }
    
    crc = cudaHostRegister(heapptr, offset+size,/*mapp, size, */
			   cudaHostRegisterPortable|cudaHostRegisterMapped);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:%s::allocCapacity@p%d] ERROR: cudaHostRegister failed (rc=%d)\n",
	      name, HH_MYID, crc);
      HHstacktrace();
      exit(1);
    }
    double et = Wtime();
    fprintf(stderr, "[HH:%s::allocCapacity@p%d] [%.2lf-%.2lf] HH_PIN_HOSTBUF pin took %.3lfsec\n",
	    name, Wtime_conv_prt(st), Wtime_conv_prt(et), et-st);
    
  }
  
#endif

  return resp;
}

int hostheap::expandHeapInner(size_t addsize /*reqsize*/)
{
  allocCapacity(heapsize, addsize);
  
  HH_addHostMemStat(HHST_HOSTHEAP, addsize);

  return 0;
}

int hostheap::allocHeapInner()
{
  assert(heapptr == NULL);

  heapptr = HOSTHEAP_PTR;
#if 01
  fprintf(stderr, "[%s::allocHeapInner@p%d] heapptr is set to %p\n",
	  name, HH_MYID, heapptr);
#endif
  if (heapsize == 0) { /* mmap(size=0) fails */
    return 0;
  }

  allocCapacity((size_t)0, heapsize);

  HH_addHostMemStat(HHST_HOSTHEAP, heapsize);

  return 0;
}

int hostheap::releaseHeap()
{
  int rc;

#if 0
  fprintf(stderr, "[%s::releaseHeap@p%d] try to release heap [%p,%p)\n",
	  name, HH_MYID, heapptr, off2ptr(heapsize));
#endif

  if (heapsize > 0L) {
    /* Free memory! */
#ifdef USE_CUDA
    if (HHL2->conf.pin_hostbuf) {
      double st = Wtime();
      cudaError_t crc;
      crc = cudaHostUnregister(heapptr);
      // BUG if heap has been expanded by multiple HostRegister
      if (crc != cudaSuccess) {
	fprintf(stderr, "[HH:%s::releaseHeap@p%d] ERROR: cudaHostUnregister failed (rc=%d)\n",
		name, HH_MYID, crc);
	HHstacktrace();
	exit(1);
      }
      double et = Wtime();
      fprintf(stderr, "[HH:%s::releaseHeap@p%d] [%.2lf-%.2lf] HH_PIN_HOSTBUF unpin took %.3lfsec\n",
	      name, Wtime_conv_prt(st), Wtime_conv_prt(et), et-st);
    }
  
#endif

    rc = munmap(heapptr, heapsize);
    if (rc != 0) {
      fprintf(stderr, "[HH:hostheap::releaseHeap@p%d] munmap(ptr=%p, size=%lx) failed!\n",
	      HH_MYID, heapptr, heapsize);
      exit(1);
    }
  }
  HH_addHostMemStat(HHST_HOSTHEAP, -heapsize);

  return 0;
}

/* allocate heap for swapIn */
int hostheap::restoreHeap()
{
  int rc;
  void *p;
  assert(heapptr != NULL);

  if (heapsize == 0) {
    return 0;
  }

  allocCapacity((size_t)0, heapsize);

  HH_addHostMemStat(HHST_HOSTHEAP, heapsize);
#if 0
  fprintf(stderr, "[HH:%s::restoreHeap@p%d] sucessfully heap [%p,%p) is recoverd\n",
	  name, HH_MYID, heapptr, offs2ptr(heapsize));
#endif

  /* Now we can access heapptr */
  return 0;
}

//////////////////////////////////
// sched_ml should be locked in caller
int HH_countHostUsers()
{
  int i;
  int count = 0;
  for (i = 0; i < HHS->nlprocs; i++) {
    if (HHS->lprocs[i].host_use > 0) count++;
  }
  return count++;
}

// called by heap::checkSwapRes
int hostheap::checkSwapResSelf(int kind, int *pline)
{
  int res = -1;
  int line = -991; // debug


  if (kind == HHSW_OUT) {
    if (HHL2->ncurcomms > 0) {
#if 0
      fprintf(stderr, "[HH:%s::checkSwapRes@p%d] ##### Now %d msgs are flying. Refrain from host swapout\n",
	      name, HH_MYID, HHL2->ncurcomms);
#endif
      res = HHSS_EBUSY;
      line = __LINE__;
    }
#ifdef USE_FIXED_FILESWAP
    if (HHL->lrank < HHL2->conf.nlphost-4) {
      res = HHSS_NONEED; // this process is never swapped out to lower
      line = __LINE__;
    }
#endif
    else {
      res = HHSS_OK;
      line = __LINE__;
    }
    
  }
  else if (kind == HHSW_IN) {
    if (HH_countHostUsers() >= HHL2->conf.nlphost) {
#if 0
      fprintf(stderr, "[HH:%s::checkSwapRes@p%d] hostusers=%d >= nlphost=%d\n",
	      name, HH_MYID, HH_countHostUsers(), HHL2->conf.nlphost);
      usleep(10*1000);
#endif
      res = HHSS_EBUSY;
      line = __LINE__;
    }
    else {
      res = HHSS_OK;
      line = __LINE__;
    }
  }

  if (pline != NULL) {
    *pline = line; // debug info
  }
  return res;
}

// called by heap::checkSwapRes during upper layer's swapping
int hostheap::checkSwapResAsLower(int kind, int *pline)
{
  return HHSS_OK;
}

int hostheap::reserveSwapResSelf(int kind)
{
  if (kind == HHSW_IN) {
    HHL->host_use = 1;
    HH_profBeginAction("F2H");
  }
  else if (kind == HHSW_OUT) {
    HH_profBeginAction("H2F");
  }

  return 0;
}

int hostheap::reserveSwapResAsLower(int kind)
{
  return 0;
}

int hostheap::releaseSwapResSelf(int kind)
{
  if (kind == HHSW_IN) {
    HH_profEndAction("F2H");
  }
  else if (kind == HHSW_OUT) {
    HHL->host_use = 0;
#ifdef HHLOG_SWAP
    fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] [%.2f] I release host capacity\n",
	    name, HH_MYID, Wtime_prt());
#endif
#if defined USE_CUDA && defined USE_DEVRESET
    if (HHL->lrank != 0) {
      HH_unlockSched();
      HH_resetCudaAll();
      HH_lockSched();
    }
#endif
    HH_profEndAction("H2F");
  }
  return 0;
}

int hostheap::releaseSwapResAsLower(int kind)
{
  return 0;
}

// copy from contiguous region to contiguous region
int hostheap::writeSeq(void *tgt, void *buf, int bufkind, size_t size)
{
  void *hp = tgt;
#ifdef USE_CUDA
  assert(bufkind == HHM_DEV || bufkind == HHM_HOST);
#else
  assert(bufkind == HHM_HOST);
#endif

#ifdef USE_CUDA
  if (bufkind == HHM_DEV) {
    cudaError_t crc;
    // obtain tools for D2H copy from upper layer
    cudaStream_t copystream = HH_curdevheap()->copystream;
    size_t copyunit = HH_curdevheap()->copyunit;
    void *copybuf = HH_curdevheap()->copybuf;
    
    if (1 || HHL2->conf.pin_hostbuf) {
      crc = cudaMemcpyAsync(hp, buf, size, cudaMemcpyDeviceToHost, copystream);
      if (crc != cudaSuccess) {
	fprintf(stderr, "[HH:%s::writeSeq@p%d] ERROR: cudaMemcpy(%p,%p,%ldMiB)(D2H)(str=%ld) failed!! rc=%d\n",
		name, HH_MYID, hp,buf,size>>20, copystream, crc);
	exit(1);
      }
      cudaStreamSynchronize(copystream);
    }
    else {
      size_t cur;
      
      for (cur = 0; cur < size; cur += copyunit) {
	size_t copysize = copyunit;
	if (cur+copysize > size) copysize = size-cur;
	crc = cudaMemcpyAsync(copybuf, piadd(buf, cur),copysize,
			      cudaMemcpyDeviceToHost, copystream);
	assert(crc == cudaSuccess);
	cudaStreamSynchronize(copystream);
	memcpy(piadd(hp, cur), copybuf, copysize);
      }
    }
  } 
  else 
#endif
    {
      memcpy(hp, buf, size);
    }

  return 0;
}

int hostheap::readSeq(void *tgt, void *buf, int bufkind, size_t size)
{
  void *hp = tgt;
#ifdef USE_CUDA
  assert(bufkind == HHM_DEV || bufkind == HHM_HOST);
#else
  assert(bufkind == HHM_HOST);
#endif

#ifdef USE_CUDA
  if (bufkind == HHM_DEV) {
    cudaError_t crc;
    // obtain tools for H2D copy from upper layer
    cudaStream_t copystream = HH_curdevheap()->copystream;
    size_t copyunit = HH_curdevheap()->copyunit;
    void *copybuf = HH_curdevheap()->copybuf;

    /* host to device */
    if (1 || HHL2->conf.pin_hostbuf) {
      crc = cudaMemcpyAsync(buf, hp, size, cudaMemcpyHostToDevice, copystream);
      if (crc != cudaSuccess) {
	fprintf(stderr, "[HH:%s::readSeq@p%d] cudaMemcpy(%ldMiB) failed!!\n",
		name, HH_MYID, size>>20);
	exit(1);
      }
      cudaStreamSynchronize(copystream);
    }
    else {
      size_t cur;
      for (cur = 0; cur < size; cur += copyunit) {
	size_t copysize = copyunit;
	if (cur+copysize > size) copysize = size-cur;
	memcpy(copybuf, piadd(hp, cur), copysize);
	crc = cudaMemcpyAsync(piadd(buf, cur), copybuf, copysize,
			      cudaMemcpyHostToDevice, copystream);
	assert(crc == cudaSuccess);
	cudaStreamSynchronize(copystream);
      }
    }
  }
  else 
#endif
    {
      memcpy(buf, hp, size);
    }
  
  return 0;
}



/*****************************************************************/
// hostmmapheap class (child class of hostheap)
// Similar to host heap, but the entire heap is mmaped to a file
// Swapping is done automatically by OS, so this does not have
// a underlying layer explicitly
hostmmapheap::hostmmapheap(fsdir *fsd0) : hostheap()
{
  strcpy(name, "hostmmapheap");

  fsd = fsd0;

  char sfname[256];
  int userid = 99; // check conflict with hhswaper.cc
  
  HH_makeSFileName(fsd, userid, sfname);
  swapfd = HH_openSFile(sfname);

  mmapflags = MAP_SHARED; // data on memory is written to a file
  
  fprintf(stderr, "[HH:%s::init%d] USE_MMAPSWAP: file %s is used\n",
	  name, HH_MYID, sfname);
}

void *hostmmapheap::allocCapacity(size_t offset, size_t size)
{
  int rc;
  void *resp;

  if (size == 0) {
    return NULL;
  }

  assert(swapfd != -1);
  rc = ftruncate(swapfd, offset+size);
  if (rc != 0) {
    fprintf(stderr, "[HH:%s::allocCapacity@p%d] ERROR: ftruncate failed!\n",
	    name, HH_MYID);
    exit(1);
  }

  resp = hostheap::allocCapacity(offset, size);

  madvise(resp, size, MADV_DONTNEED);
  return resp;
}

int hostmmapheap::restoreHeap()
{
  fprintf(stderr, "[HH:%s::restoreHeap@p%d] ERROR: restore is incompatible with USE_MMAPSWAP. This seems a bug of library.\n",
	  name, HH_MYID);
  exit(1);
}


/************************************************/
/* User API */

#ifdef USE_SWAPHOST
void *HHmalloc(size_t size)
{
  void *p;
  p = HHL2->hostheap->alloc(size);
  return p;
}

void *HHcalloc(size_t nmemb, size_t size)
{
  void *p;
  p = HHL2->hostheap->alloc(nmemb*size);
  bzero(p, nmemb*size);
  return p;
}

void HHfree(void *p)
{
  HHL2->hostheap->free(p);
  return;
}

void *HHrealloc(void *p, size_t size)
{
  // a simple version
  size_t orgsize = HHL2->hostheap->getobjsize(p);
  size_t minsize = (orgsize < size)? orgsize: size;
  void *newp = NULL;
  if (size > 0) {
    newp = HHmalloc(size);
  }

  if (minsize > 0) {
    memcpy(newp, p, minsize);
  }

  if (p != NULL && orgsize > 0) {
    HHfree(p);
  }

  return newp;
}

void *HHmemalign(size_t align, size_t size)
{
  // a simple version
  size_t xsize = size+align;
  void *xp = HHmalloc(xsize);
  if (xp == NULL) return NULL;

  void *p = (void*)roundup(xp, align);
  // This is an internal pointer, but HHfree can support it
  return p;
}

int HHposix_memalign(void **memptr, size_t align, size_t size)
{
  void *p = HHmemalign(align, size);
  if (p == NULL) {
    return -1;
  }
  *memptr = p;
  return 0;
}

void *HHaligned_alloc(size_t align, size_t size)
{
  return HHmemalign(align, size);
}

void *HHvalloc(size_t size)
{
  size_t align = sysconf(_SC_PAGESIZE);
  return HHmemalign(align, size);
}

void *HHpvalloc(size_t size)
{
  size_t align = sysconf(_SC_PAGESIZE);
  size_t xsize = roundup(size, align);
  return HHmemalign(align, xsize);
}



#endif
