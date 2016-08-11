#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <malloc.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include "hhrt_impl.h"


int swapper::beginSeqWrite()
{
  swcur = 0;
  return 0;
}

size_t swapper::allocSeq(size_t size)
{
  size_t cur = swcur;
  //swcur += ((size+align-1)/align)*align;
  swcur += roundup(size, align);
  return cur;
}

/******/
int hostswapper::init(int id)
{
  /* set up host buffer */
  cudaError_t crc;

  swapper::init(id);

  align = 1;
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

  return 0;
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
int hostswapper::swapOut(swapper *curswapper0)
{
  double t0, t1;

  t0 = Wtime();
  assert(curswapper == NULL);

  if (curswapper0 == NULL) {
    /* do not nothing */
    return 0;
  }

  curswapper = curswapper0;
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

  return 0;
}

// Swaps in data FROM another swapper (curswapper) TO myself
int hostswapper::swapIn(int initing)
{
  double t0, t1;

  if (initing) {
    /* first call */
#if 1
    fprintf(stderr, "[HH:hostswapper::swapIn@p%d] do nothing\n",
	    HH_MYID);
#endif
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
  curswapper = NULL;

  return 0;
}


/*** fileswapper ******/

int HH_makeSFileName(int id, char sfname[256])
{
  char *op = sfname;
  fsdir *fsd = HH_curfsdir();

  strcpy(op, fsd->dirname);
  op += strlen(op);

  if (op > sfname && *(op-1) == '/') {
    /* delete last '/' */
    op--;
  }

  /* add filename */
  sprintf(op, "/hhswap-r%d-p%d-%d", HH_MYID, getpid(), id);

#if 1
  fprintf(stderr, "[HH_makeSFileName@p%d] filename is [%s]\n",
	  HH_MYID, sfname);
#endif

  return 0;
}

int HH_openSFile(char sfname[256])
{
  int sfd;
  sfd = open(sfname, O_CREAT | O_RDWR | O_DIRECT, 0700);
  if (sfd == -1) {
    fprintf(stderr, "[HH:fileswapper::openSFile@p%d] ERROR in open(%s)\n",
	    HH_MYID, sfname);
    exit(1);
  }

#ifdef HHLOG_SWAP
  fprintf(stderr, "[HH:fileswapper::openSFile@p%d(L%d)] created a file %s\n",
	  HH_MYID, HHL->lrank, sfname);
#endif

#if 01
  /* This unlink is for automatic cleanup of the file. */
  // from "man 2 unlink"
  // If the name was the last link to a file but any processes still have 
  // the file open the file will remain in existence until the  last  file
  // descriptor  referring  to  it  is closed.

  unlink(sfname);
#else

#ifdef HHLOG_SWAP
  fprintf(stderr, "[HH:fileswapper::openSFile@p%d(L%d)] the file %s will be REMAINED. BE CAREFUL.\n",
	  HH_MYID, HHL->lrank, sfname);
#endif

#endif
  return sfd;
}

// file open is delayed
int fileswapper::openSFileIfNotYet()
{
  if (sfd != -1) return 0;

  /* setup filename from configuration */
  if (HHL->curfsdirid < 0) {
    // fileswap directory not available 
    fprintf(stderr, "[HH_openSFile@p%d] ERROR: swap directory not specified\n",
	    HH_MYID);
    exit(1);
  }

  HH_makeSFileName(userid, sfname);
  sfd = HH_openSFile(sfname);

  return 0;
}

int fileswapper::init(int id)
{
  int rc;
  cudaError_t crc;

  swapper::init(id);
  align = 512;

  userid = id;
  sfd = -1; // open swapfile later

  copyunit = 64L*1024*1024;
  /* prepare copybuf */
  int i;
  for (i = 0; i < 2; i++) {
    copybufs[i] = valloc(copyunit);
    crc = cudaHostRegister(copybufs[i], copyunit, 0 /*cudaHostRegisterPortable*/);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:fileswapper::init@p%d] cudaHostRegister(%ldMiB) failed (rc=%d)\n",
	      HH_MYID, copyunit>>20, crc);
      exit(1);
    }

    crc = cudaStreamCreate(&copystreams[i]);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:fileswapper::init@p%d] cudaStreamCreate failed (rc=%d)\n",
	      HH_MYID, crc);
      exit(1);
    }
  }

  return 0;
}

int fileswapper::finalize()
{
  int i;
  for (i = 0; i < 2; i++) {
    cudaHostUnregister(copybufs[i]);
    free(copybufs[i]);
    cudaStreamDestroy(copystreams[i]);
  }
  if (sfd != -1) {
    close(sfd);
  }

  return 0;
}

int fileswapper::write_small(ssize_t offs, void *buf, int bufkind, size_t size)
{
  void *ptr;
#if 0
  fprintf(stderr, "[HH:fileswapper::write1@p%d] start (align=%ld)\n",
	  HH_MYID, align);
#endif

  if ((offs % align) != 0) {
    fprintf(stderr, "[HH:fileswapper::write1@p%d] offs=0x%lx not supported. to be fixed!\n",
	    HH_MYID, offs);
    return 0;
  }

  if (bufkind == HHM_DEV) {
    cudaError_t crc;
    crc = cudaMemcpyAsync(copybufs[0], buf, size, cudaMemcpyDeviceToHost, copystreams[0]);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:fileswapper::write1@p%d] cudaMemcpy failed\n",
	      HH_MYID);
      exit(1);
    }
    cudaStreamSynchronize(copystreams[0]);
    ptr = copybufs[0];
  }
  else if ((size_t)buf % align != 0) {
#if 1
    fprintf(stderr, "[HH:fileswapper::write1@p%d] buf=0x%lx (size=0x%lx) is not aligned; so copy once more. This is not an error, but slow\n",
	    HH_MYID, buf, size);
#endif
    memcpy(copybufs[0], buf, size);
    ptr = copybufs[0];
  }
  else {
    ptr = buf;
  }

  /* seek the swap file */
  off_t orc = lseek(sfd, offs, SEEK_SET);
  if (orc != offs) {
    fprintf(stderr, "[HH:fileswapper::write1@p%d] ERROR lseek(0x%lx) -> %ld curious!\n",
	    HH_MYID, offs, orc);
    exit(1);
  }

  /* alignment */
  //size = ((size+align-1)/align)*align;
  size = roundup(size, align);

  size_t lrc = write(sfd, ptr, size);
  if (lrc != size) {
    fprintf(stderr, "[HH:fileswapper::write1@p%d] ERROR write(%d,%p,0x%lx) -> %ld curious! (offs=0x%lx, errno=%d)\n",
	    HH_MYID, sfd, ptr, size, lrc, offs, errno);
    exit(1);
  }
#if 0
  fprintf(stderr, "[HH:fileswapper::write1@p%d] OK lseek(0x%lx) and write(%d,%p,0x%lx) (align=%ld)\n",
	  HH_MYID, offs, sfd, ptr, size, align);
#endif
  return 0;
}

int fileswapper::write1(ssize_t offs, void *buf, int bufkind, size_t size)
{
  openSFileIfNotYet();

  size_t cur;
  void *p = buf;
  /* divide into copyunit */
  for (cur = 0; cur < size; cur += copyunit) {
    {
      // Refrain writing if someone is reading

      fsdir *fsd = HH_curfsdir();
      while (fsd->np_filein > 0) {
	usleep(1000);
      }
    }

    size_t lsize = copyunit;
    if (cur + lsize > size) lsize = size-cur;

    write_small(offs, p, bufkind, lsize);
    p = piadd(p, lsize);

    offs += lsize;
  }
  return 0;
}

int fileswapper::read_small(ssize_t offs, void *buf, int bufkind, size_t size)
{
  void *ptr;
#if 0
  fprintf(stderr, "[HH:fileswapper::read_s@p%d] start (align=%ld)\n",
	  HH_MYID, align);
#endif

  if (bufkind == HHM_DEV) {
    ptr = copybufs[0];
  }
  else if ((size_t)buf % align != 0) {
    ptr = copybufs[0];
  }
  else {
    ptr = buf;
  }

  if ((offs % align) != 0) {
    fprintf(stderr, "[HH:fileswapper::read_s@p%d] offs=0x%lx not supported. to be fixed!\n",
	    HH_MYID, offs);
    return 0;
  }

  /* seek the swap file */
  off_t orc = lseek(sfd, offs, SEEK_SET);
  if (orc != offs) {
    fprintf(stderr, "[HH:fileswapper::read_s@p%d] ERROR lseek(0x%lx) -> %ld curious!\n",
	    HH_MYID, offs, orc);
    exit(1);
  }

  /* alignment */
  //size = ((size+align-1)/align)*align;
  size = roundup(size, align);

  size_t lrc = read(sfd, ptr, size);
  if (lrc != size) {
    fprintf(stderr, "[HH:fileswapper::read_s@p%d] ERROR read(%d,%p,0x%lx) -> %ld curious! (errno=%d)\n",
	    HH_MYID, sfd, ptr, size, lrc, errno);
    exit(1);
  }

  if (bufkind == HHM_DEV) {
    cudaError_t crc;
    crc = cudaMemcpyAsync(buf, copybufs[0], size, cudaMemcpyHostToDevice, copystreams[0]);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:fileswapper::read_s@p%d] cudaMemcpy failed\n",
	      HH_MYID);
      exit(1);
    }
    cudaStreamSynchronize(copystreams[0]);
  }
  else if ((size_t)buf % align != 0) {
#if 1
    fprintf(stderr, "[HH:fileswapper::read_s@p%d] buf=0x%lx (size=0x%lx) is not aligned; so copy once more. This is not an error, but slow\n",
	    HH_MYID, buf, size);
#endif
    memcpy(buf, copybufs[0], size);
  }

#if 0
  fprintf(stderr, "[HH:fileswapper::read_s@p%d] OK (align=%ld)\n",
	  HH_MYID, align);
#endif

  return 0;
}

int fileswapper::read1(ssize_t offs, void *buf, int bufkind, size_t size)
{
  openSFileIfNotYet();

  size_t cur;
  void *p = buf;
  /* divide into copyunit */
  for (cur = 0; cur < size; cur += copyunit) {
    size_t lsize = copyunit;
    if (cur + lsize > size) lsize = size-cur;

    read_small(offs, p, bufkind, lsize);
    p = piadd(p, lsize);
    offs += lsize;
  }
  return 0;
}

int fileswapper::swapOut(swapper *curswapper0)
{
  fprintf(stderr, "[HH:fileswapper::swapOut@p%d] ERROR: This should not be called\n",
	  HH_MYID);
  exit(1);
  return -1;
}

int fileswapper::swapIn(int initing)
{
  fprintf(stderr, "[HH:fileswapper::swapIn@p%d] ERROR: This should not be called\n",
	  HH_MYID);
  exit(1);
  return -1;
}


