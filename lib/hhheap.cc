#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>
#include "hhrt_impl.h"

heap *HH_devheapCreate(dev *d)
{
  devheap *dh;
  heap *h;
  size_t heapsize = d->default_heapsize;

  dh = new devheap(heapsize, d);
  //dh->device = d;
  h = (heap *)dh;
  /* make memory hierarchy for devheap */

  swapper *s1;
  s1 = (swapper*)(new hostswapper());
  h->setSwapper(s1);

  swapper *s2 = NULL;
  {
    s2 = new fileswapper(0, HH_curfsdir());
    s1->setSwapper(s2);
  }
  return h;
}

#ifdef USE_SWAPHOST
heap *HH_hostheapCreate()
{
  heap *h;

  h = new hostheap();
  /* make memory hierarchy for hostheap */

  swapper *s2 = NULL;
  {
    s2 = new fileswapper(1, HH_curfsdir());
    h->setSwapper(s2);
  }

  return h;
}
#endif

int HH_finalizeHeaps()
{
  assert(HHL->dmode == HHD_ON_DEV);
  int ih;
  for (ih = 0; ih < HHL2->nheaps; ih++) {
    HHL2->heaps[ih]->finalizeRec();
  }

  HH_lockSched();

  HHL->dmode = HHD_NONE;
  HHL->pmode = HHP_RUNNABLE;
  HH_profSetMode("RUNNABLE");

  HH_unlockSched();

  return 0;
}

// heap class

heap::heap(size_t heapsize0) : mempool()
{
  heapsize = roundup(heapsize0, HEAP_ALIGN); /* heap size */

  if (heapsize > (size_t)0) {
    /* a single large freed area */
    membuf *mbp = new membuf(0L, heapsize, 0L, HHMADV_FREED);
    membufs.push_back(mbp);
  }

  curswapper = NULL;
  strcpy(name, "(HEAP)");

  return;
}

int heap::finalize()
{
  if (heapptr != NULL) {
    releaseHeap();
  }

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

  assert(heapptr != NULL);

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
      HHstacktrace();
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

int heap::swapOut()
{
  double t0, t1, t2;
  int nmoved = 0, nskipped = 0;
  size_t smoved = 0, sskipped = 0;

  t0 = Wtime();
  if (swapped == 1) {
    /* do not nothing */
    return 0;
  }

  assert(curswapper != NULL);
  curswapper->allocBuf();
  curswapper->beginSeqWrite();
  curswapper->startContWrite();
  swapped = 1;

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
  curswapper->endContWrite();

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


int heap::swapIn()
{
  double t0, t1;
  int nmoved = 0, nskipped = 0;
  size_t smoved = 0, sskipped = 0;

  if (heapptr == NULL) { // firstswapin
    allocHeap();
#ifdef HHLOG_SWAP
    fprintf(stderr, "[HH:%s::swapIn@p%d] allocHeap() for first swapin\n",
	    name, HH_MYID);
#endif
    return 0;
  }

  if ( curswapper == NULL || swapped == 0) {
    return 0;
  }
  
  assert(heapptr != NULL);
  restoreHeap();
  curswapper->startContRead();

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

  curswapper->endContRead();
  curswapper->releaseBuf();
  swapped = 0;

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
devheap::devheap(size_t heapsize0, dev *device0) : heap(heapsize0)
{
  expandable = 0;

  heapptr = NULL;
  align = 256L;
  memkind = HHM_DEV;

  device = device0;
#if 0
  cudaError_t crc;
  crc = cudaStreamCreate(&swapstream);
#endif
  sprintf(name, "devheap(d%d)", device->devid);
  hp_baseptr = NULL;

  return;
}

int devheap::finalize()
{
  heap::finalize();

  HH_lockSched();
  if (device->dhslot_users[HHL->hpid] == HH_MYID) {
    device->dhslot_users[HHL->hpid] = -1;
  }
  HH_unlockSched();

  return 0;
}

int devheap::releaseHeap()
{
  assert(heapptr != NULL);
  // do nothing

  return 0;
}

void *devheap::allocDevMem(size_t heapsize)
{
  dev *d = device;
  cudaError_t crc;
  void *dp;

  assert(HHL->hpid >= 0 && HHL->hpid < HHS->ndhslots);
  if (hp_baseptr == NULL) {
    if (HHL->lrank == 0) {
      hp_baseptr = d->hp_baseptr0;
    }
    else {
      crc = cudaIpcOpenMemHandle(&hp_baseptr, d->hp_handle, cudaIpcMemLazyEnablePeerAccess);
      if (crc != cudaSuccess) {
	fprintf(stderr, "[HH:%s::allocHeap@p%d] ERROR: cudaIpcOpenMemHandle failed! (%s)\n",
		name, HH_MYID, cudaGetErrorString(crc));
	exit(1);
      }
    }
  }

  dp = piadd(hp_baseptr, d->default_heapsize*HHL->hpid);
  return dp;
}

int devheap::allocHeap()
{
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
  //dev *d = HH_curdev();
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

int devheap::swapOutD2H()
{
  HH_lockSched();
  device->np_out++;
  HH_unlockSched();
  
  swapOut();
  
  // release resource information on device
  HH_lockSched();
  device->np_out--;
  if (device->np_out < 0) {
    fprintf(stderr, "[swapOutD2H@p%d] np_out = %d strange\n",
	    HH_MYID, device->np_out);
  }
#if 1
  assert(HHL->hpid >= 0 && HHL->hpid < HHS->ndhslots);
  assert(device->dhslot_users[HHL->hpid] == HH_MYID);
  device->dhslot_users[HHL->hpid] = -1;
  fprintf(stderr, "[HH_afterDevSwapOut@p%d] [%.2f] I release heap slot %d\n",
    HH_MYID, Wtime_prt(), HHL->hpid);
#endif
  
  HH_unlockSched();
  return 0;
}

int devheap::swapOutH2F()
{
  if (curswapper == NULL || curswapper->curswapper) {
  return 0;
  }

  curswapper->swapOut();
  return 0;
}

int devheap::swapInF2H()
{
  if (curswapper == NULL || curswapper->curswapper) {
    return 0;
  }

  curswapper->swapIn();
  return 0;
}

int devheap::swapInH2D()
{
  HH_lockSched();
  device->np_in++;
  HH_unlockSched();

  swapIn();

  HH_lockSched();
  device->np_in--;
  HH_unlockSched();
  return 0;
}

// check resource availability before actual swapping
int devheap::checkRes(int kind)
{
  if (kind == HHD_SO_D2H) {
    if (device->np_out > 0) return 0;  // someone is doing swapD2H
    return 1;
  }
  else if (kind == HHD_SI_H2D) {
    if (device->np_in > 0) return 0; // someone is doing swapH2D
    if (device->dhslot_users[HHL->hpid] >= 0) return 0; // device heap slot is occupied
    return 1;
  }
  else if (kind == HHD_SO_H2F || kind == HHD_SI_F2H) {
    // check is done by hostheap. is it OK?
    return 1;
  }
  else {
    fprintf(stderr, "[HH:devheap::checkRes@p%d] ERROR: kind %d unknown\n",
	    HH_MYID, kind);
    exit(1);
  }
}

int devheap::reserveRes(int kind)
{
  // Reserve resource information before swapping
  // This must be called after last checkRes(), without releasing schedule lock
  if (kind == HHD_SI_H2D) {
    device->dhslot_users[HHL->hpid] = HH_MYID;
  }
  else if (kind == HHD_SI_F2H) {
    // reserve is done by hostheap. is it OK?
  }
  else {
    fprintf(stderr, "[HH:devheap::reserveRes@p%d] ERROR: kind %d invalid\n",
	    HH_MYID, kind);
    exit(1);
  }
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

  return;
}

int hostheap::finalize()
{
  heap::finalize();
  
  HH_lockSched();
  HHS->nhostusers[HHL->hpid]--;
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
    fprintf(stderr, "[HH:%s::allocCapacity@p%d] ERROR: unexpted pointer %p -> %p\n",
	    name, HH_MYID, mapp, resp);
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
	  name, HH_MYID, heapptr, piadd(heapptr,heapsize));
#endif

  /* Now we can access HEAPPTR */
  return 0;
}

int hostheap::swapOutH2F()
{
  if (curswapper == NULL) {
    return 0;
  }

  swapOut();

  // release resource information
  HH_lockSched();
  HHS->nhostusers[HHL->hpid]--;
  fprintf(stderr, "[HH:%s::swapOutH2F@p%d] [%.2f] I release host capacity\n",
	  name, HH_MYID, Wtime_prt());
  HH_unlockSched();

  return 0;
}

int hostheap::swapInF2H()
{
  if (curswapper == NULL) {
    return 0;
  }

  swapIn();
  return 0;
}

int hostheap::checkRes(int kind)
{
  if (kind == HHD_SO_D2H) {
    return 1;
  }
  else if (kind == HHD_SI_H2D) {
    return 1;
  }
  else if (kind == HHD_SO_H2F) {
    if (curswapper == NULL) return 0;
    
    if (HHL2->conf.nlphost >= HHS->nlprocs) {
      // no need to use fileswapper
      return 0;
    }
    
    fsdir *fsd = ((fileswapper*)curswapper)->fsd;
    if (fsd->np_filein > 0 || fsd->np_fileout > 0) {
      // someone is doing swapF2H or swapH2F
      return 0;
    }
    
    return 1;
  }
  else if (kind == HHD_SI_F2H) {
    fsdir *fsd = ((fileswapper*)curswapper)->fsd;
    if (fsd->np_filein > 0) {
      return 0;
    }
    assert(fsd->np_filein == 0);
    
    int limperslot = (HHL2->conf.nlphost+HHS->ndhslots-1)/HHS->ndhslots;
    if (HHS->nhostusers[HHL->hpid] >= limperslot) {
      return 0;
    }
    
    /* I can start swapF2H */
    return 1;
  }
  else {
    fprintf(stderr, "[HH:devheap@p%d] ERROR: kind %d unknown\n",
	    HH_MYID, kind);
    exit(1);
  }
}

int hostheap::reserveRes(int kind)
{
  // Reserve resource information before swapping
  // This must be called after last checkRes(), without releasing schedule lock
  if (kind == HHD_SI_H2D) {
    // do nothing
  }
  else if (kind == HHD_SI_F2H) {
    HHS->nhostusers[HHL->hpid]++;
  }
  else {
    fprintf(stderr, "[HH:hostheap::reserveRes@p%d] ERROR: kind %d invalid\n",
	    HH_MYID, kind);
    exit(1);
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

