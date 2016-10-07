#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>
#include "hhrt_impl.h"

/* Host memory management */

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

  heapptr = NULL;
  align = 512L;
  memkind = HHM_HOST;

  strcpy(name, "hostheap");

  swapfd = -1; // anonymous mmap
  mmapflags = MAP_PRIVATE|MAP_ANONYMOUS;

  cudaError_t crc;

  copyunit = 32L*1024*1024;

  int i;
  for (i = 0; i < 2; i++) {
    crc = cudaHostAlloc(&copybufs[i], copyunit, cudaHostAllocDefault);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:hostswapper::init@p%d] cudaHostAlloc(%ldMiB) failed (rc=%d)\n",
	      HH_MYID, copyunit>>20, crc);
      exit(1);
    }
  }

  crc = cudaStreamCreate(&copystream);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:hostswapper::init@p%d] cudaStreamCreate failed (rc=%d)\n",
	    HH_MYID, crc);
    exit(1);
  }

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

void *hostheap::allocCapacity(size_t offset, size_t size)
{
  void *mapp = piadd(HOSTHEAP_PTR, offset);
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

  return resp;
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

#if 1  
  fprintf(stderr, "[HH:%s::expandHeap@p%d] heap expand succeeded %ldMiB -> %ldMiB\n",
	  name, HH_MYID, heapsize>>20, (heapsize + addsize)>>20);
#endif
  heapsize += addsize;
  HH_addHostMemStat(HHST_HOSTHEAP, addsize);
  
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

int hostheap::allocHeap()
{
  void *p;
  assert(heapptr == NULL);

  heapptr = HOSTHEAP_PTR;
#if 01
  fprintf(stderr, "[%s::allocHeap@p%d] heapptr is set to %p\n",
	  name, HH_MYID, heapptr);
#endif
  if (heapsize == 0) { /* mmap(size=0) fails */
    return 0;
  }

  allocCapacity((size_t)0, heapsize);

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
	  name, HH_MYID, heapptr, offs2ptr(heapsize));
#endif

  /* Now we can access HEAPPTR */
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
    res = HHSS_OK;
    line = __LINE__;
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
  }
  else if (kind == HHSW_OUT) {
  }
  //swapping_kind = kind; // remember the kind
  return 0;
}

int hostheap::reserveSwapResAsLower(int kind)
{
  return 0;
}

int hostheap::releaseSwapResSelf(int kind)
{
  if (kind == HHSW_IN) {
  }
  else if (kind == HHSW_OUT) {
    HHL->host_use = 0;
#ifdef HHLOG_SWAP
    fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] [%.2f] I release host capacity\n",
	    name, HH_MYID, Wtime_prt());
#endif
  }
  return 0;
}

int hostheap::releaseSwapResAsLower(int kind)
{
  return 0;
}

// copy from contiguous region to contiguous region
int hostheap::writeSeq(ssize_t offs, void *buf, int bufkind, size_t size)
{
  void *hp = offs2ptr(offs);
  assert(bufkind == HHM_DEV || bufkind == HHM_HOST);

#if 0
  fprintf(stderr, "[HH:hostheap::write_s@p%d] called: hscid=%d, loffs=0x%lx <- [%p,%p) (size=0x%lx)\n",
	  HH_MYID, hscid, loffs, buf, piadd(buf, size), size);
#endif

  if (bufkind == HHM_DEV) {
    cudaError_t crc;
    crc = cudaMemcpyAsync(hp, buf, size, cudaMemcpyDeviceToHost, copystream);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:%s::writeSeq@p%d] cudaMemcpy(%ldMiB) failed!!\n",
	      name, HH_MYID, size>>20);
      exit(1);
    }
    //copyD2H(hp, buf, size, copystream, "HH:hostheap::write_s");
    cudaStreamSynchronize(copystream);
  }
  else {
    memcpy(hp, buf, size);
  }

  return 0;
}

int hostheap::readSeq(ssize_t offs, void *buf, int bufkind, size_t size)
{
  void *hp = offs2ptr(offs);
  assert(bufkind == HHM_DEV || bufkind == HHM_HOST);

  if (bufkind == HHM_DEV) {
    cudaError_t crc;
    crc = cudaMemcpyAsync(buf, hp, size, cudaMemcpyHostToDevice, copystream);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:%s::readSeq@p%d] cudaMemcpy(%ldMiB) failed!!\n",
	      name, HH_MYID, size>>20);
      exit(1);
    }
    cudaStreamSynchronize(copystream);
  }
  else {
    memcpy(buf, hp, size);
  }

  return 0;
}



/*****************************************************************/
// hostmmapheap class (child class of hostheap)
// Similar to host heap, but the entire heap is mmaped to a file
// Swapping is done automatically by OS, so this does not have
// a underlying swapper explicitly
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

int HH_madvise(void *p, size_t size, int kind)
{
  int ih;
  for (ih = 0; ih < HHL2->nheaps; ih++) {
    int rc = HHL2->heaps[ih]->madvise(p, size, kind);
    if (rc == 0) return 0;
  }

  return -1;
}

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

void *HHmemalign(size_t boundary, size_t size)
{
  fprintf(stderr, "[HHmemalign] ERROR: NOT SUPPORTED YET\n");
  exit(1);
}

void *HHvalloc(size_t size)
{
  fprintf(stderr, "[HHvalloc] ERROR: NOT SUPPORTED YET\n");
  exit(1);
}

#endif
