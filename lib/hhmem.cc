#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <malloc.h>
#include <errno.h>
#include "hhrt_impl.h"

/* Memory management (generic) */

int HH_finalizeHeaps()
{
  int ih;
  for (ih = 0; ih < HHL2->nheaps; ih++) {
    HHL2->heaps[ih]->finalizeRec();
  }

  HH_lockSched();

  HHL->pmode = HHP_RUNNABLE;
  HH_profSetMode("RUNNABLE");

  HH_unlockSched();

  return 0;
}

// mempool class 
int mempool::setSwapper(swapper *swapper0) 
{
    if (lower != NULL) {
      fprintf(stderr, "[HH:mempool@p%d] lower set twice, this is ERROR!\n",
	      HH_MYID);
      exit(1);
    }
    lower = swapper0;
    return 0;
}

int mempool::finalizeRec()
{
  if (lower != NULL) {
    /* recursive finalize */
    lower->finalizeRec();
  }

  finalize();
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

  swap_kind = HHD_SWAP_NONE;
  lower = NULL;
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

  assert(lower != NULL);
  lower->allocBuf();
  lower->beginSeqWrite();
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
	mbp->soffs = lower->allocSeq(mbp->size);

	lower->write1(mbp->soffs, dp, memkind, mbp->size);

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

  if ( lower == NULL || swapped == 0) {
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
	lower->read1(mbp->soffs, dp, memkind, mbp->size);

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

  lower->releaseBuf();
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

/* swapper class */
int swapper::beginSeqWrite()
{
  swcur = 0;
  return 0;
}

size_t swapper::allocSeq(size_t size)
{
  size_t cur = swcur;
  swcur += roundup(size, align);
  return cur;
}

