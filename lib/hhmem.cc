#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <malloc.h>
#include <errno.h>
#include "hhrt_impl.h"

/* Memory management and swapping facility (generic base class) */

int HH_finalizeHeaps()
{
  int ih;
  for (ih = 0; ih < HHL2->nheaps; ih++) {
    HHL2->heaps[ih]->finalize();
  }

  HH_lockSched();

  HHL->pmode = HHP_RUNNABLE;
  HH_profSetMode("RUNNABLE");

  HH_unlockSched();

  return 0;
}

/* find a heap structure that contains pointer p */
heap *HH_findHeap(void *p)
{
  int ih;
  for (ih = 0; ih < HHL2->nheaps; ih++) {
    heap *h = HHL2->heaps[ih];
    if (h->doesInclude(p)) {
      return h;
    }
  }
  return NULL;
}

/* Read or write data in buf to tgt. */
/* tgt may be in the heap structure, which is now swapped out */
int HH_accessRec(char rwtype, void *tgt, void *buf, int bufkind, size_t size)
{
  heap *h = HH_findHeap(tgt);
  int rc = 0;
  if (h != NULL) {
    rc = h->accessRec(rwtype, tgt, buf, bufkind, size);
  }
  else {
    // TODO: considering device memory
    if (rwtype == 'R') {
      memcpy(buf, tgt, size);
    }
    else if (rwtype == 'W') {
      memcpy(tgt, buf, size);
    }
    else {
      assert(0);
    }
  }
  return rc;
}

// heap class

heap::heap(size_t heapsize0)
{
  /* init heap tree structure */
  lower = NULL; 
  for (int i = 0; i < MAX_UPPERS; i++) {
    uppers[i] = NULL;
  }

  heapsize = roundup(heapsize0, HEAP_ALIGN); /* heap size */

  if (heapsize > (size_t)0) {
    /* a single large freed area */
    membuf *mbp = new membuf(0L, heapsize, 0L, HHMADV_FREED);
    membufs.push_back(mbp);
  }

#if 1
  swap_stat = HHSW_SWAPPED;
#else
  swapping_kind = HHSW_NONE;
  swapped = 1;
#endif
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

int heap::addLower(heap *h) 
{
  if (lower != NULL) {
    fprintf(stderr, "[HH:%s::addLower@p%d] ERROR: lower set twice!\n",
	    name, HH_MYID);
    exit(1);
  }
  lower = h;
  return 0;
}

int heap::delLower(heap *h) 
{
  if (lower != h) {
    fprintf(stderr, "[HH:%s::delLower@p%d] ERROR: invalid layer (%s) specified!\n",
	    name, HH_MYID, h->name);
    exit(1);
  }
  lower = NULL;
  return 0;
}

int heap::addUpper(heap *h)
{
  int i;
  for (int i = 0; i < MAX_UPPERS; i++) {
    if (uppers[i] == NULL) {
      uppers[i] = h;
      return 0;
    }
  }

  fprintf(stderr, "[HH:%s::addUpper@p%d] ERROR: too many upper layer (>=%d)\n",
	  HH_MYID, name, MAX_UPPERS);
  exit(1);
  return -1;
}

int heap::delUpper(heap *h)
{
  int i;
  for (int i = 0; i < MAX_UPPERS; i++) {
    if (uppers[i] == h) {
      uppers[i] = NULL;
      return 0;
    }
  }

  fprintf(stderr, "[HH:%s::delUpper@p%d] ERROR: invalid layer (%s) specified!\n",
	  HH_MYID, name, h->name);
  exit(1);
  return -1;
}

void* heap::offs2ptr(ssize_t offs)
{
  return piadd(heapptr, offs);
}

ssize_t heap::ptr2offs(void* p)
{
  return ppsub(p, heapptr);
}

int heap::doesInclude(void *p)
{
  if (heapptr == NULL) {
    return 0;
  }

  if (p >= heapptr && p < piadd(heapptr, heapsize)) {
    return 1;
  }
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

  if (heapptr == NULL) {
    fprintf(stderr, "[HH:%s::aloc@p%d] heapptr=%p invalid\n",
	    name, HH_MYID, heapptr);
  }

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
    p = offs2ptr(mbp->doffs);

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
  ssize_t doffs = ptr2offs(p);
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
  ssize_t doffs = ptr2offs(p);
  if (doffs < 0 || doffs >= heapsize) {
    fprintf(stderr, "[HH:heap(%s)::free@p%d] pointer %p is invalid\n",
	    name, HH_MYID, p);
    return -1;
  }

  list<membuf *>::iterator it;
  it = findMembufIter(doffs);
  if (it == membufs.end()) {
    fprintf(stderr, "[HH:heap(%s)::free@p%d] pointer %p is invalid\n",
	    name, HH_MYID, p);
    return -1;
  }

  membuf *mbp = *it;
  assert(mbp != NULL);
  if (mbp->kind == HHMADV_FREED) {
    fprintf(stderr, "[HH:heap(%s)::free@p%d] WARNING: pointer %p is doubly freed\n",
	    name, HH_MYID, p);
    return -1;
  }

  if (mbp->soffs != (ssize_t)-1) {
    // there may be replica in lower layer. this must be invalidated
    mbp->soffs = (ssize_t)-1;
    fprintf(stderr, "[HH:heap(%s)::free@p%d] WARNING: replica may be resident eternally. should be fixed\n",
	    name, HH_MYID);
  }

  mbp->kind = HHMADV_FREED;
  mbp->usersize = 0L;

  /* TODO: deal with fragmentation */
  if (it != membufs.begin()) {
    list<membuf *>::iterator previt = it;
    previt--;
    membuf *prevmbp = *previt;
    if (prevmbp->kind == HHMADV_FREED) {
      // coalesching is possible
#if 0
      fprintf(stderr, "[HH:heap(%s)::free@p%d] try coalesce free area [%ld,%ld) and [%ld,%ld)\n",
	      name, HH_MYID, prevmbp->doffs, prevmbp->doffs+prevmbp->size,
	      mbp->doffs, mbp->doffs+mbp->size);
#endif

      mbp->size += prevmbp->size;
      mbp->doffs = prevmbp->doffs;

      // delete prevmbp
      membufs.erase(previt);
    }
  }

  list<membuf *>::iterator nextit = it;
  nextit++;
  if (nextit != membufs.end()) {
    list<membuf *>::iterator nextit = it;
    nextit++;
    membuf *nextmbp = *nextit;
    if (nextmbp->kind == HHMADV_FREED) {
      // coalescing is possible
#if 0
      fprintf(stderr, "[HH:heap(%s)::free@p%d] try coalesce free area [%ld,%ld) and [%ld,%ld)\n",
	      name, HH_MYID, mbp->doffs, mbp->doffs+mbp->size,
	      nextmbp->doffs, nextmbp->doffs+nextmbp->size);
#endif
      mbp->size += nextmbp->size;
      // delete nextmbp
      membufs.erase(nextit);
    }
  }

  return 0;
}

// mainly for HHrealloc
size_t heap::getobjsize(void *p)
{
  membuf *mbp = findMembuf(p);
  if (mbp == NULL) {
    return (size_t)0;
  }

  return mbp->usersize;
}

int heap::expandHeap(size_t reqsize)
{
  fprintf(stderr, "heap::expandHeap should not called\n");
  exit(1);
}

int heap::releaseHeap()
{
  fprintf(stderr, "heap::releaseHeap should not called\n");
  exit(1);
}


/***********************/
/* swapping facility (at basis class)*/


int heap::checkSwapRes(int kind)
{
  int res = HHSS_OK;
  int line = -999; // debug
  int line2 = -9999;

#if 1
  if (swap_stat == HHSW_IN || swap_stat == HHSW_OUT) {
#else
  if (swapping_kind != HHSW_NONE) {
#endif
    // already swapping is ongoing (this happens in threaded swap)
    res = HHSS_EBUSY;
    line = __LINE__;
    goto out;
  }
  else if (kind == HHSW_OUT) {
    if (lower == NULL) {
      res = HHSS_NONEED;
      line == __LINE__;
      goto out;
    }
#if 1
    else if (swap_stat == HHSW_SWAPPED) {
#else
    else if (swapped) {
#endif
      // swapping-out was already done
      res = HHSS_NONEED;
      line = __LINE__;
      goto out;
    }
    else {
      // check upper heaps
      int ih;
      for (ih = 0; ih < MAX_UPPERS; ih++) {
	heap *h = uppers[ih];
#if 1
	if (h != NULL && h->swap_stat != HHSW_SWAPPED) {
#else
	if (h != NULL && h->swapped == 0) {
#endif
	  break;
	}
      }

      if (ih < MAX_UPPERS) {
	// there is a not-swapped upper host. we should wait
	res = HHSS_EBUSY;
	line = __LINE__;
	goto out;
      }
    }
  }
  else if (kind == HHSW_IN) {
    if (lower == NULL && heapptr != NULL) {
      res = HHSS_NONEED;
      line == __LINE__;
      goto out;
    }
#if 1
    else if (swap_stat == HHSW_NONE) {
#else
    else if (swapped == 0) {
#endif
      // swapping-in was already done
      res = HHSS_NONEED;
      line = __LINE__;
      goto out;
    }
    else {
      // check lower heap
      heap *h = lower;
#if 1
      if (h != NULL && h->swap_stat == HHSW_SWAPPED) {
#else
      if (h != NULL && h->swapped) {
#endif
	// we have to wait lower heap's swapping-in
	res = HHSS_EBUSY;
	line = __LINE__;
	goto out;
      }
    }
  }
  else {
    fprintf(stderr, "[HH:%s::checkSwapRes@p%d] ERROR: kind %d unknown\n",
	    name, HH_MYID, kind);
    exit(1);
  }

  // we still need check
  // class specific check
  res = checkSwapResSelf(kind, &line2);
  if (res != HHSS_OK) {
    //line = __LINE__;
    line = 10000+line2;
    goto out;
  }

  // class specific check of lower layer
  if (lower != NULL) {
    res = lower->checkSwapResAsLower(kind, &line2);
    if (res != HHSS_OK) {
      line = 20000+line2;
    }
    else {
      line = __LINE__;
    }
  }

 out:
#if 1
  // visible from hhview tool
  sprintf(HHL->msg, "[HH:heap(%s)::checkSwapRes@p%d] [%.2lf] for %s result=%s (line=%d)",
    name, HH_MYID, Wtime_prt(), hhsw_names[kind], hhss_names[res], line);
#endif  

#if 0
  if (res == HHSS_OK /*|| rand() % 1024 == 0*/) {
    fprintf(stderr, "[HH:heap(%s)::checkSwapRes@p%d] for %s result=%s (line=%d)\n",
	    name, HH_MYID, hhsw_names[kind], hhss_names[res], line);
  }
#endif
  return res;
}

int heap::reserveSwapRes(int kind)
{
  assert(kind == HHSW_IN || kind == HHSW_OUT);
  reserveSwapResSelf(kind);
  if (lower != NULL) {
    lower->reserveSwapResAsLower(kind);
  }

#if 1
  swap_stat = kind; // remember the kind for following doSwap()
#else
  swapping_kind = kind; // remember the kind for following doSwap()
#endif
  return 0;
}

int heap::doSwap()
{
#if 1
  int kind = swap_stat;
#else
  int kind = swapping_kind;
#endif
  assert(kind == HHSW_IN || kind == HHSW_OUT);

  HH_profBeginAction(hhsw_names[kind]);
  if (kind == HHSW_OUT) {
    swapOut();
  }
  else if (kind == HHSW_IN) {
    swapIn();
  }
  else {
    fprintf(stderr, "[HH:heap(%s)::doSwap@p%d] ERROR: kind %d unknown\n",
	    name, HH_MYID, kind);
    exit(1);
  }
  HH_profEndAction(hhsw_names[kind]);
  return 0;
}

int heap::releaseSwapRes()
{
#if 1
  int kind = swap_stat;
#else
  int kind = swapping_kind;
#endif
  assert(kind == HHSW_IN || kind == HHSW_OUT);

  releaseSwapResSelf(kind);
  if (lower != NULL) {
    lower->releaseSwapResAsLower(kind);
  }

#if 1
  if (kind == HHSW_IN) {
    swap_stat = HHSW_NONE;
  }
  else if (kind == HHSW_OUT) {
    swap_stat = HHSW_SWAPPED;
  }
#else
  swapping_kind = HHSW_NONE; // swap finished
#endif

  return 0;
}


int heap::swapOut()
{
  double t0, t1, t2;
  int nmoved = 0, nskipped = 0;
  size_t smoved = 0, sskipped = 0;

  t0 = Wtime();
  assert(swap_stat == HHSW_OUT);
  assert(lower != NULL);

#if 1
  fprintf(stderr, "[HH:%s::swapOut@p%d] [%.2lf] start. heap region is [%p,%p)\n",
	  name, HH_MYID, Wtime_conv_prt(t0), heapptr, offs2ptr(heapsize));
#endif

  /* for all membufs */
  {
    list<membuf *>::iterator it;
    for (it = membufs.begin(); it != membufs.end(); it++) {
      /* scan all buffers in queue */
      membuf *mbp = *it;
      void *dp = offs2ptr(mbp->doffs);
      if (mbp->kind == HHMADV_NORMAL || mbp->kind == HHMADV_READONLY) {
	if (mbp->soffs == (ssize_t)-1) {
	  void *sp = lower->alloc(mbp->size);
#if 0
	  fprintf(stderr, "[HH:%s::swapOut@p%d] got pointer %p from %s\n",
		  name, HH_MYID, sp, lower->name);
#endif
	  mbp->soffs = lower->ptr2offs(sp);
	  lower->writeSeq(mbp->soffs, dp, memkind, mbp->size);
	  
	  nmoved++;
	  smoved += mbp->size;
	}
	else {
	  // Replica in the lower layer is valid. no need to copy
	  assert(mbp->kind == HHMADV_READONLY);
	  nskipped++;
	  sskipped += mbp->size;
	}
      }
      else {
	assert(mbp->kind == HHMADV_FREED || mbp->kind == HHMADV_CANDISCARD);
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

#if 0
  swapped = 1;
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

  if (swap_stat != HHSW_IN) {
    fprintf(stderr, "[HH:%s::swapIn@p%d] CHECK HHRT: my swap_stat=%s\n",
	    name, HH_MYID, hhsw_names[swap_stat]);
  }
  assert(swap_stat == HHSW_IN);

  if (heapptr == NULL) { // firstswapin
    allocHeap();
#if 0
    swapped = 0;
#endif
#ifdef HHLOG_SWAP
    fprintf(stderr, "[HH:%s::swapIn@p%d] allocHeap() called for first swapin. swapIn finished immediately\n",
	    name, HH_MYID);
#endif
    return 0;
  }

  if ( lower == NULL ) {
#if 0
    swapped = HHSW_NONE;
#endif
    assert(0);
    return 0;
  }
  
#if 1
  if ( swap_stat == HHSW_NONE) {
#else
  if ( swapped == 0) {
#endif
    assert(0);
    return 0;
  }
  
  assert(heapptr != NULL);
#if 0
  swapped = 0;
#endif
  restoreHeap();

  t0 = Wtime();
  /* for all membufs */
  {
    list<membuf *>::iterator it;
    for (it = membufs.begin(); it != membufs.end(); it++) {
      /* scan all buffers in queue */
      membuf *mbp = *it;
      void *dp = offs2ptr(mbp->doffs);
      if (mbp->kind == HHMADV_NORMAL || mbp->kind == HHMADV_READONLY) {
	lower->readSeq(mbp->soffs, dp, memkind, mbp->size);

	if (mbp->kind == HHMADV_NORMAL) {
	  /* Discard replica in lower layer */
	  void *sp = lower->offs2ptr(mbp->soffs);
	  lower->free(sp);
	  mbp->soffs = (ssize_t)-1;
	}

	nmoved++;
	smoved += mbp->size;
      }
      else {
	assert(mbp->kind == HHMADV_FREED || mbp->kind == HHMADV_CANDISCARD);
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

  return 0;
}

/********************/
/* write/read data to heap */
/* This recursive function is diffrent from writeSeq(), readSeq */
/* if this heap is swapped out, swapped region in lower layer is updated. */
/* rwtype: 'W' or 'R' */
int heap::accessRec(char rwtype, void *tgt, void *buf, int bufkind, size_t size)
{
#if 1
  if (swap_stat == HHSW_IN || swap_stat == HHSW_OUT) {
#else
  if (swapping_kind != HHSW_NONE) {
#endif
    fprintf(stderr, "[HH:%s::accessRec@p%d] this heap is now under %s --> EBUSY\n",
	    name, HH_MYID, hhsw_names[swap_stat /*swaping_kind*/]);
    return HHSS_EBUSY;
  }

  if (!doesInclude(tgt)) {
    fprintf(stderr, "[HH:%s::accessRec@p%d] ERROR: this heap does not include ptr %p. Check HHRT implementation\n",
	    name, HH_MYID, tgt);
    return HHSS_ERROR;
  }

#if 1
  if (swap_stat == HHSW_NONE) {
#else
  if (swapped == 0) {
#endif
    /* not swapped */
    int rc;
    if (rwtype == 'W') {
      rc = writeSeq(ptr2offs(tgt), buf, bufkind, size);
    }
    else if (rwtype == 'R') {
      rc = readSeq(ptr2offs(tgt), buf, bufkind, size);
    }
    else {
      fprintf(stderr, "[HH:%s::accessRec@p%d] ERROR: unknown access type %c\n",
	      name, HH_MYID, rwtype);
      rc = -1;
    }

    if (rc != 0) return HHSS_ERROR;
    else return HHSS_OK;
  }

  assert(lower != NULL);
  /* we need to investigate lower layer */
  membuf *mbp = findMembuf(tgt);
  if (mbp == NULL || mbp->kind == HHMADV_FREED) {
    // unknown pointer
    fprintf(stderr, "[HH:%s::accessRec@p%d] ERROR: this heap includes ptr %p, but invalid (maybe freed)\n",
	    name, HH_MYID, tgt);
    return HHSS_ERROR;
  }

  if (mbp->soffs == (ssize_t)-1) {
    fprintf(stderr, "[HH:%s::accessRec@p%d] WARNING: this heap is swapped out, but ptr %p does not have destination\n",
	    name, HH_MYID, tgt);
    return HHSS_ERROR;
  }

  ssize_t inneroffs = ptr2offs(tgt) - mbp->doffs; /* offset inside the object */
  assert(inneroffs >= 0 && inneroffs+size <= mbp->size);
  ssize_t soffs = mbp->soffs + inneroffs;

#if 1
  fprintf(stderr, "[HH:%s::accessRec@p%d] delegate %c access to %s\n",
	  name, HH_MYID, rwtype, lower->name);
#endif
  int rc = lower->accessRec(rwtype, lower->offs2ptr(soffs), buf, bufkind, size);
  return rc;
}

/********************/
int heap::madvise(void *p, size_t size, int kind)
{
  membuf *mbp = findMembuf(p);
  if (mbp == NULL) {
    // unknown pointer
    return -1;
  }
  if (mbp->kind == HHMADV_FREED) {
    fprintf(stderr, "[HH:%s::madvise@p%d] pointer %p is already freed\n",
	    name, HH_MYID, p);
    return -1;
  }

  if (kind == HHMADV_NORMAL) {
    mbp->kind = kind;

    if (mbp->soffs != (ssize_t)-1) {
      // there may be replica in lower layer. this must be invalidated
      mbp->soffs = (ssize_t)-1;
      fprintf(stderr, "[HH:%s::madvise@p%d] WARNING: replica may be resident eternally. should be fixed\n",
	      name, HH_MYID);
    }
  }
  else if (kind == HHMADV_CANDISCARD || kind == HHMADV_READONLY) {
    ssize_t doffs = ptr2offs(p);
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
  fprintf(stderr, "[HH:%s::dump@p%d] heapptr=%p\n",
	  name, HH_MYID, heapptr);
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
    else if (mbp->kind == HHMADV_READONLY) {
      strcpy(kind, "READONLY");
    }
    else if (mbp->kind == HHMADV_FREED) {
      strcpy(kind, "FREED");
    }
    else {
      strcpy(kind, "UNKNOWN");
    }

    fprintf(stderr, "  [%p,%p) --> %s\n",
	    offs2ptr(mbp->doffs), offs2ptr(mbp->doffs+mbp->size),
	    kind);
  }
  return 0;
}

/************************************************/
/* User API */

int HH_madvise(void *p, size_t size, int kind)
{
  heap *h = HH_findHeap(p);
  if (h == NULL) {
    fprintf(stderr, "[HH_madvise@p%d] ptr %p is not managed by HHRT. ignored\n",
	    HH_MYID, p);
    return -1;
  }

  return h->madvise(p, size, kind);
}

