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

  heapsize = roundup(heapsize0, HEAP_ALIGN); /* initial heap size */

  swap_stat = HHSW_SWAPPED; // swapIn() should be called soon for first heap allocation
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
    membuf *umbp = new membuf(mbp->ptr, size, size0, HHMADV_NORMAL);
    membufs.insert(it, umbp);
    p = mbp->ptr;

    /* change the rest free buffer */
    mbp->ptr = piadd(mbp->ptr, size);
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

list<membuf *>::iterator heap::findMembufIter(void *ptr)
{
  list<membuf *>::iterator it;
  for (it = membufs.begin(); it != membufs.end(); it++) {
    /* scan all buffers in queue */
    membuf *mbp = *it;
    if (ptr >= mbp->ptr && ptr < piadd(mbp->ptr, mbp->size)) {
      return it;
    }
  }
  return membufs.end();
}

membuf *heap::findMembuf(void *p)
{
  if (p < heapptr || p >= piadd(heapptr, heapsize)) {
    return NULL;
  }

  list<membuf *>::iterator it;
  it = findMembufIter(p);
  if (it == membufs.end()) {
    return NULL;
  }

  return *it;
}

int heap::free(void *p)
{
  if (p == NULL) return 0;
  if (p < heapptr || p >= piadd(heapptr, heapsize)) {
    fprintf(stderr, "[HH:heap(%s)::free@p%d] pointer %p is invalid (out of heap)\n",
	    name, HH_MYID, p);
    return -1;
  }

  list<membuf *>::iterator it;
  it = findMembufIter(p);
  if (it == membufs.end()) {
    fprintf(stderr, "[HH:heap(%s)::free@p%d] pointer %p is invalid (no such membuf)\n",
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

  if (mbp->sptr != NULL) {
    // there may be replica in lower layer. this must be invalidated
    mbp->sptr = NULL;
    fprintf(stderr, "[HH:heap(%s)::free@p%d] WARNING: replica may be resident eternally. should be fixed\n",
	    name, HH_MYID);
  }

  mbp->kind = HHMADV_FREED;
  mbp->usersize = 0L;

  /* deal with fragmentation */
  if (it != membufs.begin()) {
    list<membuf *>::iterator previt = it;
    previt--;
    membuf *prevmbp = *previt;
    if (prevmbp->kind == HHMADV_FREED) {
      // coalesching is possible

      mbp->size += prevmbp->size;
      mbp->ptr = prevmbp->ptr;

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

  if (swap_stat == HHSW_IN || swap_stat == HHSW_OUT) {
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
    else if (swap_stat == HHSW_SWAPPED) {
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
	if (h != NULL && h->swap_stat != HHSW_SWAPPED) {
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
    else if (swap_stat == HHSW_NONE) {
      // swapping-in was already done
      res = HHSS_NONEED;
      line = __LINE__;
      goto out;
    }
    else {
      // check lower heap
      heap *h = lower;
      if (h != NULL && h->swap_stat == HHSW_SWAPPED) {
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

  swap_stat = kind; // remember the kind for following doSwap()
  return 0;
}

int heap::doSwap()
{
  int kind = swap_stat;
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
  int kind = swap_stat;
  assert(kind == HHSW_IN || kind == HHSW_OUT);

  releaseSwapResSelf(kind);
  if (lower != NULL) {
    lower->releaseSwapResAsLower(kind);
  }

  if (kind == HHSW_IN) {
    swap_stat = HHSW_NONE;
  }
  else if (kind == HHSW_OUT) {
    swap_stat = HHSW_SWAPPED;
  }

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
	  name, HH_MYID, Wtime_conv_prt(t0), heapptr, piadd(heapptr, heapsize));
#endif

  /* for all membufs */
  {
    list<membuf *>::iterator it;
    for (it = membufs.begin(); it != membufs.end(); it++) {
      /* scan all buffers in queue */
      membuf *mbp = *it;
      void *dp = mbp->ptr;
      if (mbp->kind == HHMADV_NORMAL || mbp->kind == HHMADV_READONLY) {
	if (mbp->sptr == NULL) {
	  void *sp = lower->alloc(mbp->size);
#if 0
	  fprintf(stderr, "[HH:%s::swapOut@p%d] got pointer %p from %s\n",
		  name, HH_MYID, sp, lower->name);
#endif
	  mbp->sptr = sp;
	  //lower->writeSeq(lower->ptr2offs(mbp->sptr), dp, memkind, mbp->size); // to be fixed
	  lower->writeSeq(mbp->sptr, dp, memkind, mbp->size);
	  
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
	mbp->sptr = NULL;
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

int heap::allocHeapInner()
{
  fprintf(stderr, "HH:heap::allocHeapInner should not called\n");
  exit(1);
}

int heap::allocHeap()
{
  allocHeapInner();

  /* make a single large free area */
  membuf *mbp = new membuf(heapptr, heapsize, 0L, HHMADV_FREED);
  membufs.push_back(mbp);

  return 0;
}

int heap::expandHeapInner(size_t reqsize)
{
  fprintf(stderr, "HH:heap::expandHeapInner should not called\n");
  exit(1);
}


int heap::expandHeap(size_t reqsize)
{
  assert(expandable);

  size_t addsize;
  if (reqsize > expand_step) {
    addsize = roundup(reqsize, expand_step);
  }
  else {
    addsize = expand_step;
  }

  expandHeapInner(addsize);

  /* expand succeeded */
  /* make a single large free area */
  membuf *mbp = new membuf(piadd(heapptr, heapsize), addsize, 0L, HHMADV_FREED);
  membufs.push_back(mbp);

#if 1  
  fprintf(stderr, "[HH:%s::expandHeap@p%d] heap expand succeeded %ldMiB -> %ldMiB\n",
	  name, HH_MYID, heapsize>>20, (heapsize + addsize)>>20);
#endif

  heapsize += addsize;

  return 0;
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

  if (heapptr == NULL) { // first heap allocation
    // here, lower may be NULL
    allocHeap();
#ifdef HHLOG_SWAP
    fprintf(stderr, "[HH:%s::swapIn@p%d] allocHeap() called for first swapin. swapIn finished immediately\n",
	    name, HH_MYID);
#endif
    return 0;
  }

  assert(lower != NULL);
  assert(heapptr != NULL);
  restoreHeap();

  t0 = Wtime();
  /* for all membufs */
  {
    list<membuf *>::iterator it;
    for (it = membufs.begin(); it != membufs.end(); it++) {
      /* scan all buffers in queue */
      membuf *mbp = *it;
      void *dp = mbp->ptr;
      if (mbp->kind == HHMADV_NORMAL || mbp->kind == HHMADV_READONLY) {
	//lower->readSeq(lower->ptr2offs(mbp->sptr), dp, memkind, mbp->size); // to be fixed
	lower->readSeq(mbp->sptr, dp, memkind, mbp->size);

	if (mbp->kind == HHMADV_NORMAL) {
	  /* Discard replica in lower layer */
	  lower->free(mbp->sptr);
	  mbp->sptr = NULL;
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
  if (swap_stat == HHSW_IN || swap_stat == HHSW_OUT) {
    fprintf(stderr, "[HH:%s::accessRec@p%d] this heap is now under %s --> EBUSY\n",
	    name, HH_MYID, hhsw_names[swap_stat /*swaping_kind*/]);
    return HHSS_EBUSY;
  }

  if (!doesInclude(tgt)) {
    fprintf(stderr, "[HH:%s::accessRec@p%d] ERROR: this heap does not include ptr %p. Check HHRT implementation\n",
	    name, HH_MYID, tgt);
    return HHSS_ERROR;
  }

  if (swap_stat == HHSW_NONE) {
    /* not swapped */
    int rc;
    if (rwtype == 'W') {
      //rc = writeSeq(ptr2offs(tgt), buf, bufkind, size);
      rc = writeSeq(tgt, buf, bufkind, size);
    }
    else if (rwtype == 'R') {
      //rc = readSeq(ptr2offs(tgt), buf, bufkind, size);
      rc = readSeq(tgt, buf, bufkind, size);
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

  if (mbp->sptr == NULL) {
    fprintf(stderr, "[HH:%s::accessRec@p%d] WARNING: this heap is swapped out, but ptr %p does not have destination\n",
	    name, HH_MYID, tgt);
    return HHSS_ERROR;
  }

#if 1
  fprintf(stderr, "[HH:%s::accessRec@p%d] delegate %c access to %s\n",
	  name, HH_MYID, rwtype, lower->name);
#endif
  ssize_t inneroffs = ppsub(tgt, mbp->ptr); /* offset inside the object */
  assert(inneroffs >= 0 && inneroffs+size <= mbp->size);
  int rc = lower->accessRec(rwtype, piadd(mbp->sptr, inneroffs), buf, bufkind, size);

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

    if (mbp->sptr != NULL) {
      // there may be replica in lower layer. this must be invalidated
      mbp->sptr = NULL;
      fprintf(stderr, "[HH:%s::madvise@p%d] WARNING: replica may be resident eternally. should be fixed\n",
	      name, HH_MYID);
    }
  }
  else if (kind == HHMADV_CANDISCARD || kind == HHMADV_READONLY) {
    if (p == mbp->ptr && size >= mbp->usersize) {
      /* mark CANDISCARD if the whole region is specified */
      mbp->kind = kind;
    }
    else {
      fprintf(stderr, "[HH:heap::madvise@p%d] size %ld is specified for size %ld object. This madvise is ignored\n",
	      HH_MYID, size, mbp->usersize);
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
	    mbp->ptr, piadd(mbp->ptr, mbp->size),
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

