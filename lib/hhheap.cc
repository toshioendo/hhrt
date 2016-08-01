#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>
#include "hhrt_impl.h"

// heap class

int heap::init(size_t heapsize0)
{
  heapsize = roundup(heapsize0, HEAP_ALIGN); /* heap size */

  if (heapsize > (size_t)0) {
    /* a single large freed area */
    membuf *mbp = new membuf(0L, heapsize, 0L, HHMADV_FREED);
    membufs.push_back(mbp);
  }

  curswapper = NULL;
  strcpy(name, "(HEAP)");

  return 0;
}

int heap::finalize()
{
#if 1
  if (heapptr != NULL) {
    releaseHeap();
  }
#endif
  heapptr = NULL;
  heapsize = 0;
  return 0;
}

void* heap::alloc(size_t size0)
{
  /* round up for alignment */
  size_t size = roundup(size0, align);

 retry:
  void *p = NULL;
  int nused = 0;
  size_t used = 0;
  int nfreed = 0;
  size_t freed = 0;
  int nretries = 0;

  /* search membufs */
  list<membuf *>::iterator it;
  for (it = membufs.begin(); it != membufs.end(); it++) {
    /* scan all buffers in queue */
    membuf *mbp = *it;
    /* stat */
    if (mbp->kind == HHMADV_FREED) {
      freed += mbp->size;
      nfreed++;
    }
    else {
      used += mbp->size;
      nused++;
    }

    if (mbp->kind != HHMADV_FREED || mbp->size < size) {
      continue;
    }
    /* found */
    membuf *umbp = new membuf(mbp->doffs, size, size0, HHMADV_NORMAL);
    membufs.insert(it, umbp);
    p = piadd(heapptr, mbp->doffs);

    /* change the rest free buffer */
    mbp->doffs += size;
    mbp->size -= size;
    mbp->usersize = 0L;
    
    if (mbp->size == 0) {
      membufs.erase(it);
      delete mbp;
    }
    break;
  }

  if (p == NULL) {
    if (expandable) {
      if (nretries > 0) {
	fprintf(stderr, "[%s::alloc@p%d] OUT OF MEMORY after expand? (req %ld bytes)\n",
		name, HH_MYID, size);
	fprintf(stderr, "[%s::alloc@p%d] heap %ld bytes, free %ld bytes (%d chunks), used %ld bytes (%d chunks)\n",
		name, HH_MYID, heapsize, freed, nfreed, used, nused);
	exit(1);
	return NULL;
      }

      expandHeap(size);

      nretries++;
      goto retry;
    }
    else {
      fprintf(stderr, "[%s::alloc@p%d] OUT OF MEMORY (req %ld bytes)\n",
	      name, HH_MYID, size);
      fprintf(stderr, "[%s::alloc@p%d] heap %ld bytes, free %ld bytes (%d chunks), used %ld bytes (%d chunks)\n",
	      name, HH_MYID, heapsize, freed, nfreed, used, nused);
      exit(1);
      return NULL;
    }
  }

  return p;
}

/* aux functions for cudaFree etc. */

list<membuf *>::iterator heap::findMembufIter(ssize_t doffs)
{
  list<membuf *>::iterator it;
  for (it = membufs.begin(); it != membufs.end(); it++) {
    /* scan all buffers in queue */
    membuf *mbp = *it;
    if (doffs >= mbp->doffs && doffs < mbp->doffs + mbp->size) {
      return it;
    }
  }
  return membufs.end();
}

membuf *heap::findMembuf(void *p)
{
  ssize_t doffs = ppsub(p, heapptr);
  if (doffs < 0 || doffs >= heapsize) {
    return NULL;
  }

  list<membuf *>::iterator it;
  it = findMembufIter(doffs);
  if (it == membufs.end()) {
    return NULL;
  }

  return *it;
}

int heap::free(void *p)
{
  if (p == NULL) return 0;

  membuf *mbp = findMembuf(p);
  if (mbp == NULL) {
    fprintf(stderr, "[heap::free@p%d] pointer %p is invalid\n",
	    HH_MYID, p);
    return -1;
  }
  if (mbp->kind == HHMADV_FREED) {
    fprintf(stderr, "[%s::free@p%d] pointer %p is doubly freed\n",
	    name, HH_MYID, p);
    return -1;
  }

  mbp->kind = HHMADV_FREED;
  mbp->usersize = 0L;

  /* TODO: deal with fragmentation */
  return 0;
}

int heap::expandHeap(size_t reqsize)
{
  fprintf(stderr, "heap::expandHeap should not called\n");
  exit(1);
}


/***********************/
int heap::releaseHeap()
{
  fprintf(stderr, "heap::releaseHeap should not called\n");
  exit(1);
}

int heap::swapOut(swapper *curswapper0)
{
  double t0, t1, t2;
  int nmoved = 0, nskipped = 0;
  size_t smoved = 0, sskipped = 0;

  t0 = Wtime();
  assert(curswapper == NULL);

  if (curswapper0 == NULL) {
    /* do not nothing */
    return 0;
  }

  curswapper = curswapper0;
  curswapper->allocBuf();
  curswapper->beginSeqWrite();

#if 1
  fprintf(stderr, "[HH:%s::swapOut@p%d] [%.2lf] start. heap region is [%p,%p)\n",
	  name, HH_MYID, Wtime_conv_prt(t0), heapptr, piadd(heapptr,heapsize));
#endif

  /* for all membufs */
  {
    list<membuf *>::iterator it;
    for (it = membufs.begin(); it != membufs.end(); it++) {
      /* scan all buffers in queue */
      membuf *mbp = *it;
      if (mbp->kind == HHMADV_NORMAL) {
	void *dp = piadd(heapptr, mbp->doffs);
	mbp->soffs = curswapper->allocSeq(mbp->size);

	curswapper->write1(mbp->soffs, dp, memkind, mbp->size);

	nmoved++;
	smoved += mbp->size;
      }
      else {
	mbp->soffs = (ssize_t)-1;
	nskipped++;
	sskipped += mbp->size;
      }
    }
  }


  t1 = Wtime();

#ifdef HHLOG_SWAP
  if (1 || nmoved > 0) {
    double mbps = (smoved > 0)? ((double)(smoved>>20)/(t1-t0)): 0.0;
    fprintf(stderr, "[HH:%s::swapOut@p%d] [%.2f-%.2f] copying %ldMiB (/%ldMiB) took %.1lfms -> %.1lfMiB/s\n",
	    name, HH_MYID, Wtime_conv_prt(t0), Wtime_conv_prt(t1),
	    smoved>>20, (smoved+sskipped)>>20, (t1-t0)*1000.0, mbps);
  }
#endif

  releaseHeap();

  return 0;
}

int heap::allocHeap()
{
  fprintf(stderr, "HH:heap::allocHeap should not called\n");
  exit(1);
}

int heap::restoreHeap()
{
  fprintf(stderr, "HH:heap::restoreHeap should not called\n");
  exit(1);
}


int heap::swapIn(int initing)
{
  double t0, t1;
  int nmoved = 0, nskipped = 0;
  size_t smoved = 0, sskipped = 0;

  if (initing) { // firstswapin
    assert(heapptr == NULL);
    allocHeap();
    return 0;
  }

  if ( curswapper == NULL) {
    return 0;
  }
  
  assert(heapptr != NULL);
  restoreHeap();

  t0 = Wtime();
  /* for all membufs */
  {
    list<membuf *>::iterator it;
    for (it = membufs.begin(); it != membufs.end(); it++) {
      /* scan all buffers in queue */
      membuf *mbp = *it;
      if (mbp->kind == HHMADV_NORMAL) {
	void *dp = piadd(heapptr, mbp->doffs);
	curswapper->read1(mbp->soffs, dp, memkind, mbp->size);

	mbp->soffs = (ssize_t)-1;

	nmoved++;
	smoved += mbp->size;
      }
      else {
	nskipped++;
	sskipped += mbp->size;
      }
    }
  }

  t1 = Wtime();
#ifdef HHLOG_SWAP
  if (1 || smoved > 0) {
    double mbps;
    mbps = (smoved > 0)? (double)(smoved >> 20) / (t1-t0): 0.0;
    fprintf(stderr, "[HH:%s::swapIn@p%d] [%.2f-%.2f] copying %ldMiB (/%ldMiB) took %.1lfms -> %.1lfMiB/s\n",
	    name, HH_MYID, Wtime_conv_prt(t0), Wtime_conv_prt(t1),
	    smoved>>20, (smoved+sskipped)>>20, (t1-t0)*1000.0, mbps);
  }
#endif

  curswapper->releaseBuf();
  curswapper = NULL;

  return 0;
}

int heap::madvise(void *p, size_t size, int kind)
{
  membuf *mbp = findMembuf(p);
  if (mbp == NULL) {
    // unknown pointer
    return -1;
  }
  if (mbp->kind == HHMADV_FREED) {
    fprintf(stderr, "[%s::madvise@p%d] pointer %p is already freed\n",
	    name, HH_MYID, p);
    return -1;
  }

  if (kind == HHMADV_NORMAL) {
    mbp->kind = kind;
  }
  else if (kind == HHMADV_CANDISCARD) {
    ssize_t doffs = ppsub(p, heapptr);
    if (doffs == mbp->doffs && size >= mbp->usersize) {
      /* mark CANDISCARD if the whole region is specified */
      mbp->kind = kind;
    }
    else {
      fprintf(stderr, "[HH:heap::madvise@p%d] [%ld,%ld) specified for size %ld. This madvise is ignored\n",
	      HH_MYID, doffs, size, mbp->usersize);
    }
  }
  return 0;
}

int heap::dump()
{
  fprintf(stderr, "[HH:%s::dump@p%d]\n",
	  name, HH_MYID);
  list<membuf *>::iterator it;
  for (it = membufs.begin(); it != membufs.end(); it++) {
    /* scan all buffers in queue */
    membuf *mbp = *it;
    char kind[16];
    if (mbp->kind == HHMADV_NORMAL) {
      strcpy(kind, "NORMAL");
    }
    else if (mbp->kind == HHMADV_CANDISCARD) {
      strcpy(kind, "CANDISCARD");
    }
    else if (mbp->kind == HHMADV_FREED) {
      strcpy(kind, "FREED");
    }
    else {
      strcpy(kind, "UNKNOWN");
    }

    fprintf(stderr, "  [%p,%p) --> %s\n",
	    piadd(heapptr, mbp->doffs), piadd(heapptr, mbp->doffs+mbp->size),
	    kind);
  }
  return 0;
}

/*****************************************************************/
// devheap class (child class of heap)
int devheap::init(size_t heapsize0)
{
  expandable = 0;
  heap::init(heapsize0);

  heapptr = NULL;
  align = 256L;
  memkind = HHM_DEV;
  
#if 0
  cudaError_t crc;
  crc = cudaStreamCreate(&swapstream);
#endif
  strcpy(name, "devheap");
  hp_baseptr = NULL;

  return 0;
}

int devheap::releaseHeap()
{
  cudaError_t crc;
  dev *d = HH_curdev();

#ifdef USE_CUDA_IPC
  assert(heapptr != NULL);
  // do nothing
#else
  pthread_mutex_lock(&d->ml);
  /* Free device memory! */
  crc = cudaFree(heapptr);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:devheap::releaseHeap@p%d] cudaFree failed!\n",
	    HH_MYID);
    exit(1);
  }
  pthread_mutex_unlock(&d->ml);
#endif // USE_CUDA_IPC

  return 0;
}

void *devheap::allocDevMem(size_t heapsize)
{
  dev *d = HH_curdev();
  cudaError_t crc;
  void *dp;
#ifdef USE_CUDA_IPC
  assert(HHL->hpid >= 0 && HHL->hpid < HHS->nheaps);
  if (hp_baseptr == NULL) {
    if (HHL->lrank == 0) {
      hp_baseptr = d->hp_baseptr0;
    }
    else {
      crc = cudaIpcOpenMemHandle(&hp_baseptr, d->hp_handle, cudaIpcMemLazyEnablePeerAccess);
      if (crc != cudaSuccess) {
	fprintf(stderr, "[HH:%s::allocHeap@p%d] ERROR: cudaIpcOpenMemHandle failed! (rc=%d)\n",
		name, HH_MYID, crc);
	exit(1);
      }
    }
  }

  dp = piadd(hp_baseptr, d->default_heapsize*HHL->hpid);

#else
  int nretries = 0;
  pthread_mutex_lock(&d->ml);
 retry:
  crc = cudaMalloc(&dp, heapsize);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:%s::allocDevMem@p%d] cudaMalloc(0x%lx) failed (rc=%d, dev memsize=%ld)\n",
	    name, HH_MYID, heapsize, crc, d->memsize);
    nretries++;
    if (nretries < 5) {
      sleep(1);
      fprintf(stderr, "[HH:%s::allocDevMem@p%d] retry cudaMalloc...\n",
	      name, HH_MYID);
      goto retry;
    }
    fprintf(stderr, "[%s::allocDevMem@p%d] ERROR: too many retry of cudaMalloc...\n",
	    name, HH_MYID);
    exit(1);
  }
  pthread_mutex_unlock(&d->ml);
#endif
  return dp;
}

int devheap::allocHeap()
{
  dev *d = HH_curdev();
  void *dp;

  assert(heapptr == NULL);

  dp = allocDevMem(heapsize);
  assert(dp != NULL);

  /* first swapin */
  heapptr = dp;
#ifdef HHLOG_SWAP
  fprintf(stderr, "[HH:%s::allocHeap@p%d] Get heap (size=0x%lx) pointer first -> %p\n",
	  name, HH_MYID, heapsize, dp);
#endif

  /* Now we can access HEAPPTR */
  return 0;
}

/* allocate device heap for swapIn */
int devheap::restoreHeap()
{
  dev *d = HH_curdev();
  void *dp;

  assert(heapptr != NULL);

  dp = allocDevMem(heapsize);
  assert(dp != NULL);

  if (heapptr != dp) {
    fprintf(stderr, "[HH:%s::restoreHeap@p%d] pointer restoring failed %p -> %p\n",
	    name, HH_MYID, heapptr, dp);
    exit(1);
  }

  /* Now we can access HEAPPTR */
  return 0;
}

/*****************************************************************/
// hostheap class (child class of heap)
int hostheap::init(size_t heapsize0)
{
  expandable = 1;
  heap::init((size_t)0);

  heapptr = NULL;
  align = 512L;
  memkind = HHM_HOST;

  strcpy(name, "hostheap");

#ifdef USE_MMAPSWAP
  {
    char sfname[256];
    int userid = 90;
    if (HHL->curfsdirid < 0) {
      // fileswap directory not available 
      fprintf(stderr, "[HH:hostheap::init%d] ERROR: swap directory not specified\n",
	      HH_MYID);
      exit(1);
    }

    HH_makeSFileName(userid, sfname);
    sfd = HH_openSFile(sfname);
  }
#endif
  return 0;
}

int hostheap::allocCapacity(size_t offset, size_t size)
{
  void *mapp = piadd(HOSTHEAP_PTR, offset);
  void *resp;

  if (size == 0) {
    return 0;
  }

  resp = mmap(mapp, size, 
	      PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS,
	      -1, (off_t)0);
  if (resp == MAP_FAILED) {
    fprintf(stderr, "[HH:%s::mapHeap@p%d] ERROR: mmap(0x%lx, 0x%lx) failed\n",
	    name, HH_MYID, mapp, size);
    exit(1);
  }

  if (resp != mapp) {
    fprintf(stderr, "[HH:%s::expandHeap@p%d] ERROR: unexpted pointer %p -> %p\n",
	    name, HH_MYID, mapp, resp);
    exit(1);
  }

  return 0;
}

int hostheap::expandHeap(size_t reqsize)
{
  size_t addsize;
  void *p;
  void *mapp;
  if (reqsize > HOSTHEAP_STEP) {
    addsize = roundup(reqsize, HOSTHEAP_STEP);
  }
  else {
    addsize = HOSTHEAP_STEP;
  }

  allocCapacity(heapsize, addsize);
  
  /* expand succeeded */
  /* make a single large free area */
  membuf *mbp = new membuf(heapsize, addsize, 0L, HHMADV_FREED);
  membufs.push_back(mbp);
  
  fprintf(stderr, "[HH:%s::expandHeap@p%d] heap expand succeeded %ldMiB -> %ldMiB\n",
	  name, HH_MYID, heapsize>>20, (heapsize + addsize)>>20);
  
  heapsize += addsize;
  HH_addHostMemStat(HHST_HOSTHEAP, addsize);
  
  return 0;
}

int hostheap::releaseHeap()
{
  int rc;
  cudaError_t crc;

#if 0
  fprintf(stderr, "[%s::releaseHeap@p%d] try to release heap [%p,%p)\n",
	  name, HH_MYID, heapptr, piadd(heapptr,heapsize));
#endif

  /* Free memory! */
  rc = munmap(heapptr, heapsize);
  if (rc != 0) {
    fprintf(stderr, "[HH:hostheap::releaseHeap@p%d] munmap failed!\n",
	    HH_MYID);
    exit(1);
  }
  HH_addHostMemStat(HHST_HOSTHEAP, -heapsize);

  return 0;
}

int hostheap::allocHeap()
{
  void *p;
  assert(heapptr == NULL);

  heapptr = HOSTHEAP_PTR;
  if (heapsize == 0) { /* mmap(size=0) fails */
    return 0;
  }

#if 1 // debugging
  allocCapacity((size_t)0, heapsize);
#else
  /* first allocate */
  p = mmap(HOSTHEAP_PTR, heapsize, 
	   PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS,
	   -1, (off_t)0);
  if (p == MAP_FAILED) {
    fprintf(stderr, "[HH:%s::allocHeap@p%d] mmap(NULL, %ld) failed\n",
	    name, HH_MYID, heapsize);
    exit(1);
  }
  heapptr = p;
#endif

  HH_addHostMemStat(HHST_HOSTHEAP, heapsize);

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
	  name, HH_MYID, heapptr, piadd(heapptr,heapsize));
#endif

  /* Now we can access HEAPPTR */
  return 0;
}

