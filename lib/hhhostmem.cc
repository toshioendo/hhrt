#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>
#include "hhrt_impl.h"

/* Host memory management */

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

int hostheap::inferSwapMode(int kind0)
{
  int res_kind;
  if (kind0 == HHD_SO_ANY) {
    if (HHL2->conf.nlphost >= HHS->nlprocs) {
      // no need to use fileswapper
      res_kind = HHD_SWAP_NONE;
    }
    else if (curswapper != NULL && swapped == 0) {
      res_kind = HHD_SO_H2F;
    }
    else {
      res_kind = HHD_SWAP_NONE; // nothing requried
    }
  }
  else if (kind0 == HHD_SI_ANY) {
    if (heapptr != NULL && swapped == 0) {
      res_kind = HHD_SWAP_NONE; // nothing requried
    }
    else {
      assert(curswapper != NULL);
      res_kind = HHD_SI_F2H;
    }
  }
  else {
    res_kind = kind0;
  }

#if 0
  if (res_kind != kind0) {
    fprintf(stderr, "[HH:%s::inferSwapMode@p%d] swap_kind inferred %s -> %s\n",
	    name, HH_MYID, hhd_names[kind0], hhd_names[res_kind]);
    usleep(10000);
  }
#endif
  return res_kind;
}

int hostheap::checkSwapRes(int kind0)
{
  int res;
  int line = -200; // debug
  int kind = inferSwapMode(kind0);

  if (swap_kind != HHD_SWAP_NONE) {
    // already swapping is ongoing (this happens in threaded swap)
    res = HHSS_EBUSY;
    line = __LINE__;
  }
  else if (kind == HHD_SWAP_NONE) {
    res = HHSS_NONEED;
    line = __LINE__;
  }
  else if (kind == HHD_SO_D2H) {
    res = HHSS_OK;
    line = __LINE__;
    assert(0);
  }
  else if (kind == HHD_SI_H2D) {
    res = HHSS_OK;
    line = __LINE__;
    assert(0);
  }
  else if (kind == HHD_SO_H2F) {
    if (curswapper == NULL) {
      res = HHSS_OK;
      line = __LINE__;
      assert(0);
    }
    
    fsdir *fsd = ((fileswapper*)curswapper)->fsd;
    if (fsd->np_filein > 0 || fsd->np_fileout > 0) {
      // someone is doing swapF2H or swapH2F
      res = HHSS_EBUSY;
      line = __LINE__;
    }
    else {
      res = HHSS_OK;
      line = __LINE__;
    }
  }
  else if (kind == HHD_SI_F2H) {
    fsdir *fsd = ((fileswapper*)curswapper)->fsd;
    if (fsd->np_filein > 0) {
      res = HHSS_EBUSY;
      line = __LINE__;
    }
    else if (HH_countHostUsers() >= HHL2->conf.nlphost) {
#if 0
      fprintf(stderr, "[HH:%s::checkSwapRes@p%d] hostusers=%d >= nlphost=%d\n",
	      name, HH_MYID, HH_countHostUsers(), HHL2->conf.nlphost);
      usleep(10*1000);
#endif
      res = HHSS_EBUSY;
      line = __LINE__;
    }
    else {
      /* I can start swapF2H */
      res = HHSS_OK;
      line = __LINE__;
    }
  }
  else {
    fprintf(stderr, "[HH:%s::checkSwapRes@p%d] ERROR: kind %d unknown\n",
	    name, HH_MYID, kind);
    exit(1);
  }

#if 0
  if (rand() % 256 == 0) {
    const char *strs[] = {"OK", "EBUSY", "NONEED", "XXX"};
    fprintf(stderr, "[HH:%s::checkSwapRes@p%d] result=%s (line=%d)\n",
	    name, HH_MYID, strs[res], line);
  }
#endif
  return res;
}

int hostheap::reserveSwapRes(int kind0)
{
  int kind = inferSwapMode(kind0);
  swap_kind = kind; // remember the kind
  
  // Reserve resource information before swapping
  // This must be called after last checkSwapRes(), without releasing schedule lock
  if (kind == HHD_SI_H2D) {
    // do nothing
  }
  else if (kind == HHD_SI_F2H) {
    HHL->host_use = 1;

    fsdir *fsd = ((fileswapper*)curswapper)->fsd;
    fsd->np_filein++;
  }
  else if (kind == HHD_SO_H2F) {
    fsdir *fsd = ((fileswapper*)curswapper)->fsd;
    fsd->np_fileout++;
  }
  else {
    // do nothing
  }
  return 0;
}

int hostheap::doSwap()
{
  int kind = swap_kind;
  HH_profBeginAction(hhd_snames[kind]);

  if (kind == HHD_SO_D2H) {
    goto out;
  }
  else if (kind == HHD_SI_H2D) {
    goto out;
  }
  else if (kind == HHD_SO_H2F) {
    if (curswapper == NULL) {
      goto out;
    }
    
    swapOut();
  }
  else if (kind == HHD_SI_F2H) {
    if (curswapper == NULL) {
      goto out;
    }

    swapIn();
  }
  else {
    fprintf(stderr, "[HH:%s::swap@p%d] ERROR: kind %d unknown\n",
	    name, HH_MYID, kind);
    exit(1);
  }
 out:
  HH_profEndAction(hhd_snames[kind]);
  return 0;
}

int hostheap::releaseSwapRes()
{
  int kind = swap_kind;

  // Release resource information after swapping
  if (kind == HHD_SI_H2D) {
    // do nothing
  }
  else if (kind == HHD_SI_F2H) {
    fsdir *fsd = ((fileswapper*)curswapper)->fsd;
    fsd->np_filein--;
    if (fsd->np_filein < 0) {
      fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] np_filein = %d strange\n",
	      name, HH_MYID, fsd->np_filein);
    }
  }
  else if (kind == HHD_SO_H2F) {
    HHL->host_use = 0;
    fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] [%.2f] I release host capacity\n",
	    name, HH_MYID, Wtime_prt());

    fsdir *fsd = ((fileswapper*)curswapper)->fsd;
    fsd->np_fileout--;
    if (fsd->np_fileout < 0) {
      fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] np_fileout = %d strange\n",
	      name, HH_MYID, fsd->np_fileout);
    }
  }
  else {
    // do nothing
  }
  swap_kind = HHD_SWAP_NONE;
  return 0;
}


/******/
hostswapper::hostswapper() : swapper()
{
  /* set up host buffer */
  cudaError_t crc;

  align = 1;
  copyunit = 32L*1024*1024;
  initing = 1;

  sprintf(name, "hostswapper");

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

int hostswapper::finalize()
{
  releaseBuf();
  cudaStreamDestroy(copystream);
  return 0;
}

#if !defined USE_SHARED_HSC && !defined USE_MMAPSWAP
// process local HSC
int HH_hsc_init_node()
{
  return 0;
}

int HH_hsc_init_proc()
{
  return 0;
}

int HH_hsc_fin_node()
{
  return 0;
}

void *HH_hsc_alloc(int id)
{
  void *p = valloc(HSC_SIZE);
  HH_addHostMemStat(HHST_HOSTSWAPPER, HSC_SIZE);
  return p;
}

int HH_hsc_free(void *p)
{
  free(p);
  HH_addHostMemStat(HHST_HOSTSWAPPER, -HSC_SIZE);
  return 0;
}

#elif !defined USE_SHARED_HSC && defined USE_MMAPSWAP
  
// process local HSC, mmaped
int HH_hsc_init_node()
{
  return 0;
}

int HH_hsc_init_proc()
{
  char sfname[256];
  int userid = 98; // check conflict with hhswaper.cc
  if (HHL->curfsdirid < 0) {
    // fileswap directory not available 
    fprintf(stderr, "[HH_hsc_init_proc@p%d] ERROR: swap directory not specified\n",
	    HH_MYID);
    exit(1);
  }
  
  HH_makeSFileName(userid, sfname);
  HHL2->hswfd = HH_openSFile(sfname);
  
  fprintf(stderr, "[HH_hsc_init_proc@p%d] USE_MMAPSWAP: file %s is used\n",
	  HH_MYID, sfname);

  return 0;
}

int HH_hsc_fin_node()
{
  return 0;
}

void *HH_hsc_alloc(int id)
{
  {
    int rc;
    rc = ftruncate(HHL2->hswfd, HSC_SIZE*(id+1));
    if (rc != 0) {
      fprintf(stderr, "[HH_hsc_alloc@p%d] ERROR: ftruncate failed!\n",
	      HH_MYID);
      exit(1);
    }
  }

  void *p = mmap(NULL, HSC_SIZE,
		 PROT_READ|PROT_WRITE, MAP_SHARED,
		 HHL2->hswfd, HSC_SIZE*id);
  if (p == MAP_FAILED) {
    fprintf(stderr, "[HH_hsc_alloc@p%d] ERROR: mmap(0x%lx) failed!\n",
	    HH_MYID, HSC_SIZE);
    exit(1);
  }

  madvise(p, HSC_SIZE, MADV_DONTNEED);

  HH_addHostMemStat(HHST_HOSTSWAPPER, HSC_SIZE);
  return p;
}

int HH_hsc_free(void *p)
{
  int rc;
  rc = munmap(p, HSC_SIZE);
  if (rc != 0) {
    fprintf(stderr, "[HH_hsc_free@p%d] munmap failed!\n",
	    HH_MYID);
    exit(1);
  }
  HH_addHostMemStat(HHST_HOSTSWAPPER, -HSC_SIZE);
  return 0;
}
#endif

/* */
int hostswapper::allocBuf()
{
  cudaError_t crc;

  // hscs will be allocated lazily
  assert(hscs.size() == 0);

  return 0;
}

int hostswapper::releaseBuf()
{
  while (hscs.size() > 0) {
    void *p = hscs.front();
    HH_hsc_free(p);
    hscs.pop_front();
  }

  return 0;
}

/* */

static int copyD2H(void *hp, void *dp, size_t size, 
		   cudaStream_t copystream, const char *aux)
{
  cudaError_t crc;
#if 0
  fprintf(stderr, "[%s@p%d] copyD2H(%p,%p,0x%lx)\n",
	  aux, HH_MYID, hp, dp, size);
#endif
  crc = cudaMemcpyAsync(hp, dp, size, cudaMemcpyDeviceToHost, copystream);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[%s@p%d] ERROR: cudaMemcpyD2H(host %p, dev %p, size 0x%lx) failed!! crc=%d\n", 
	    aux, HH_MYID, hp, dp, size, crc);
    exit(1); 
  } 
  return 0;
}

void *hostswapper::getNthChunk(int n)
{
  if (n >= hscs.size()) {
    fprintf(stderr, "[HH:hostswapper::getNthChunk@p%d] ERROR: chunkid=%d is too big. maybe access to invalid offset is done (%d hscs)\n",
	    HH_MYID, n, hscs.size());
    exit(1);
  }

  list<void *>::iterator it = hscs.begin();
  int i;
  for (i = 0; i < n; i++) it++;
  return (*it);
}

// copy from contiguous region to contiguous region
int hostswapper::write_small(ssize_t offs, void *buf, int bufkind, size_t size)
{
  ssize_t hscid = offs/HSC_SIZE;
  ssize_t loffs = offs%HSC_SIZE;
  if (hscid >= hscs.size()+1) {
    fprintf(stderr, "[HH:hostswapper::write_s@p%d] ERROR: write to invalid offset? offs=%ld, hscs has %d chunks\n",
	    HH_MYID, offs, hscs.size());
    exit(1);
  }
  void *chunk;
  if (hscid == hscs.size()) {
    // new chunk for append write
    assert(loffs == 0);
    chunk = HH_hsc_alloc(hscid);
    hscs.push_back(chunk);
  }

  chunk = getNthChunk(hscid);
  void *hp = piadd(chunk, loffs);
  assert(loffs + size <= HSC_SIZE);

#if 0
  fprintf(stderr, "[HH:hostswapper::write_s@p%d] called: hscid=%d, loffs=0x%lx <- [%p,%p) (size=0x%lx)\n",
	  HH_MYID, hscid, loffs, buf, piadd(buf, size), size);
#endif

  if (bufkind == HHM_DEV) {
    cudaError_t crc;
#if 1
    // hp <- copybuf <- buf
    if (size <= copyunit) {
      copyD2H(copybufs[0], buf, size, copystream, "HH:hostswapper::write_s(s)");
      cudaStreamSynchronize(copystream);
      memcpy(hp, copybufs[0], size);
    }
    else {
      ssize_t cur = 0;
      int bufid = 0;
      // first
      copyD2H(copybufs[0], buf, copyunit, copystream, "HH:hostswapper::write_s(1)");
      cudaStreamSynchronize(copystream);
      for (cur = 0; cur < size; cur += copyunit) {
	// look ahead
	ssize_t nextunit = copyunit;
	if (cur+copyunit+nextunit > size) nextunit = size-(cur+copyunit);
	if (nextunit > 0) {
	  copyD2H(copybufs[1-bufid], piadd(buf, cur+copyunit), nextunit,
		  copystream, "HH:hostswapper::write_s");
	}

	ssize_t unit = copyunit;
	if (cur+unit > size) unit = size-cur;
#if 0
	fprintf(stderr, "[HH:hostswapper::write_s@p%d] memcpy(%p,%p,%ld) cur=%lx, size=%lx\n",
		HH_MYID, piadd(hp, cur), piadd(copybufs[bufid], cur), unit, cur, size);
#endif
	memcpy(piadd(hp, cur), copybufs[bufid], unit);

	crc = cudaStreamSynchronize(copystream);
	if (crc != cudaSuccess) {
	  fprintf(stderr, "[HH:hostswapper::write_s@p%d] cudaSync failed, crc=%ld\n",
		  HH_MYID, crc);
	}
	bufid = 1-bufid;
      }
    }
#else
    copyD2H(hp, buf, size, copystream, "HH:hostswapper::write_s");
    cudaStreamSynchronize(copystream);
#endif
  }
  else {
    memcpy(hp, buf, size);
  }

  return 0;
}


int hostswapper::write1(ssize_t offs, void *buf, int bufkind, size_t size)
{
  size_t cur;
  void *p = buf;

#if 0
  fprintf(stderr, "[HH:hostswapper::write1@p%d] called: [%p,%p) (size=0x%lx)\n",
	  HH_MYID, buf, piadd(buf,size), size);
#endif

  /* divide into copyunit */
  cur = 0;
  while (cur < size) {
    // determine size to write at once 
    size_t lsize = HSC_SIZE;
    if (cur + lsize > size) lsize = size - cur;

    size_t loffs = offs%HSC_SIZE;
    // write_small cannot exceed border of chunks
    if (loffs + lsize > HSC_SIZE) lsize = HSC_SIZE-loffs;

    write_small(offs, p, bufkind, lsize);

    p = piadd(p, lsize);
    cur += lsize;
    offs += lsize;
  }
  
  assert(cur == size);
  return 0;
}

int hostswapper::read_small(ssize_t offs, void *buf, int bufkind, size_t size)
{
  ssize_t hscid = offs/HSC_SIZE;
  ssize_t loffs = offs%HSC_SIZE;

  void *chunk = getNthChunk(hscid);
  void *hp = piadd(chunk, loffs);
  assert(loffs + size <= HSC_SIZE);
  assert(size <= copyunit);

  if (bufkind == HHM_DEV) {
#if 1
    cudaError_t crc;
    memcpy(copybufs[0], hp, size);
    crc = cudaMemcpyAsync(buf, copybufs[0], size, cudaMemcpyHostToDevice, copystream);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:hostswapper::read_s@p%d] cudaMemcpy(%ldMiB) failed!!\n",
	      HH_MYID, size>>20);
      exit(1);
    }
    cudaStreamSynchronize(copystream);
#else
    cudaError_t crc;
    crc = cudaMemcpyAsync(buf, hp, size, cudaMemcpyHostToDevice, copystream);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:hostswapper::read_s@p%d] cudaMemcpy(%ldMiB) failed!!\n",
	      HH_MYID, size>>20);
      exit(1);
    }
    cudaStreamSynchronize(copystream);
#endif
  }
  else {
    memcpy(buf, hp, size);
  }

  return 0;
}

int hostswapper::read1(ssize_t offs, void *buf, int bufkind, size_t size)
{
  size_t cur;
  void *p = buf;
#if 0
  fprintf(stderr, "[HH:hhswapper::read1@p%d] read(offs=%ld, size=%ld)\n",
	  HH_MYID, offs, size);
#endif
  /* divide into copyunit */
  cur = 0;
  while (cur < size) {
    size_t lsize = copyunit; // TODO: change like write1
    if (cur + lsize > size) lsize = size-cur;

    size_t loffs = offs%HSC_SIZE;
    // read_small cannot exceed border of chunks
    if (loffs + lsize > HSC_SIZE) lsize = HSC_SIZE-loffs;

    read_small(offs, p, bufkind, lsize);

    p = piadd(p, lsize);
    cur += lsize;
    offs += lsize;
  }
  assert(cur == size);
  return 0;
}

// Swaps out data FROM myself TO another swapper
int hostswapper::swapOut()
{
  double t0, t1;

  t0 = Wtime();

  if (swapped == 1) {
    /* do not nothing */
    fprintf(stderr, "[HH:hostswapper::swapOut@p%d] SKIP swapOut\n",
	    HH_MYID);
    return 0;
  }

  assert(curswapper != NULL);
  curswapper->allocBuf();
  curswapper->beginSeqWrite();

  /* write the entire swapbuf */
  /* data size is given by swcur (see allocSeq()) */
  size_t allocsize;

  ssize_t hscid;
  ssize_t sum = 0;
  list<void *>::iterator it = hscs.begin();
  for (hscid = 0; it != hscs.end(); hscid++, it++) {
    size_t lsize = HSC_SIZE;
    if (hscid+1 == hscs.size()) {
      lsize = swcur % HSC_SIZE;
    }

    void *chunk = *it;
    curswapper->write1(0, chunk, HHM_HOST, lsize);
    sum += lsize;
  }

#if 1
  fprintf(stderr, "[HH:hostswapper::swapOut@p%d] wrote %ld bytes, swcur=%ld bytes. hscs.size=%d\n",
	  HH_MYID, sum, swcur, hscs.size());
#endif
  
  allocsize = HSC_SIZE * hscs.size();

  t1 = Wtime();

#ifdef HHLOG_SWAP
  if (1) {
    double mbps = (double)(swcur >> 20) / (t1-t0);
    fprintf(stderr, "[HH:hostswapper::swapOut@p%d] [%.2f-%.2f] copying %ldMiB(/%ldMiB) took %.1lfms -> %.1lfMiB/s\n",
	    HH_MYID, Wtime_conv_prt(t0), Wtime_conv_prt(t1),
	    swcur>>20, allocsize>>20, (t1-t0)*1000.0, mbps);
  }
#endif

  releaseBuf();
  swapped = 1;

  return 0;
}

// Swaps in data FROM another swapper (curswapper) TO myself
int hostswapper::swapIn()
{
  double t0, t1;

  if (initing) {
    /* first call */
#if 1
    fprintf(stderr, "[HH:hostswapper::swapIn@p%d] do nothing\n",
	    HH_MYID);
#endif
    initing = 0;
    return 0;
  }

  if ( curswapper == NULL) {
    fprintf(stderr, "[HH:hostswapper::swapIn@p%d] BUG: curswapper==NULL\n",
	    HH_MYID);
    exit(1);
  }

  allocBuf();
  t0 = Wtime();

  /* read the entire swapbuf */
  /* data size is given by swcur (see allocSeq()) set by previous swapOut */

  ssize_t nhsc = (swcur+HSC_SIZE-1)/HSC_SIZE;
  ssize_t sum = 0L; // debug
  ssize_t hscid;
  for (hscid = 0; hscid < nhsc; hscid++) {
    void *chunk;
    chunk = HH_hsc_alloc(hscid);
    hscs.push_back(chunk);

    ssize_t lsize = HSC_SIZE;
    if (hscid+1 == nhsc) { // last chunk
      lsize = swcur % HSC_SIZE;
    }

    curswapper->read1(0, chunk, HHM_HOST, lsize);
    sum += lsize;
  }
  if (sum != swcur) {
    fprintf(stderr, "[HH:hostswapper::swapIn@p%d] read %ld bytes, swcur=%ld bytes. hscs.size=%d, INVALID!\n",
	    HH_MYID, sum, swcur, hscs.size());
    exit(1);
  }
  
  t1 = Wtime();
#ifdef HHLOG_SWAP
  if (1) {
    double mbps = (double)(swcur >> 20) / (t1-t0);
    fprintf(stderr, "[HH:hostswapper::swapIn@p%d] [%.2f-%.2f] copying %ldMiB took %.1lfms -> %.1lfMiB/s\n",
	    HH_MYID, Wtime_conv_prt(t0), Wtime_conv_prt(t1),
	    swcur>>20, (t1-t0)*1000.0, mbps);
  }
#endif

  curswapper->releaseBuf();
  swapped = 0;

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
/* User level API */

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
