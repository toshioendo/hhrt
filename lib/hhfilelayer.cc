#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <errno.h>
#include "hhrt_impl.h"

/* File layer management */

/* constants */
#define FILEHEAP_STEP (256L*1024*1024)
#define FILEHEAP_PTR ((void*)512) // the pointer is not accessed


int HH_fileInitNode(hhconf *confp)
{
  int i;
  /* init swap directories */
  for (i = 0; i < confp->n_fileswap_dirs; i++) {
    fsdir *fsd = &HHS->fsdirs[i];
    strcpy(fsd->dirname, confp->fileswap_dirs[i]);
    fsd->np_filein = 0;
    fsd->np_fileout = 0;
  }

  return 0;
}

int HH_fileInitProc()
{
  // default fsdir id
  if (HHL2->conf.n_fileswap_dirs == 0) {
    HHL->curfsdirid = -1;
  }
  else {
    HHL->curfsdirid = HHL->lrank % HHL2->conf.n_fileswap_dirs;
  }
  return 0;
}

fsdir *HH_curfsdir()
{
  if (HHL->curfsdirid < 0) {
    fprintf(stderr, 
	    "[HH_curfsdir@p%d] ERROR: curfsdirid is not set\n",
	    HH_MYID);
    exit(1);
  }
  return &HHS->fsdirs[HHL->curfsdirid];
}

heap *HH_fileheapCreate(fsdir *fsd)
{
  heap *h;
  h = new fileheap(1, fsd);
  return h;
}

/*** fileheap ******/

int HH_makeSFileName(fsdir *fsd, int id, char sfname[256])
{
  char *op = sfname;

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
    fprintf(stderr, "[HH_openSFile@p%d] ERROR in open(%s)\n",
	    HH_MYID, sfname);
    exit(1);
  }

#ifdef HHLOG_SWAP
  fprintf(stderr, "[HH_openSFile@p%d(L%d)] created a file %s\n",
	  HH_MYID, HHL->lrank, sfname);
#endif

#if 1
  /* This unlink is for automatic cleanup of the file. */
  // from "man 2 unlink"
  // If the name was the last link to a file but any processes still have 
  // the file open the file will remain in existence until the  last  file
  // descriptor  referring  to  it  is closed.

  unlink(sfname);
#else

#ifdef HHLOG_SWAP
  fprintf(stderr, "[HH_openSFile@p%d(L%d)] the file %s will be REMAINED. BE CAREFUL.\n",
	  HH_MYID, HHL->lrank, sfname);
#endif

#endif
  return sfd;
}

// file open is delayed
int fileheap::openSFileIfNotYet()
{
  if (sfd != -1) return 0;

  /* setup filename from configuration */
  if (HHL->curfsdirid < 0) {
    // fileswap directory not available 
    fprintf(stderr, "[HH_openSFile@p%d] ERROR: swap directory not specified\n",
	    HH_MYID);
    exit(1);
  }

  HH_makeSFileName(fsd, userid, sfname);
  sfd = HH_openSFile(sfname);

  return 0;
}

fileheap::fileheap(int id, fsdir *fsd0) : heap(0L)
{
  int rc;

  sprintf(name, "fileheap");

  expandable = 1;
#if 1
  swap_stat = HHSW_NONE;
#else
  swapped = 0;
#endif

  heapptr = FILEHEAP_PTR;
  align = 512L;
  memkind = HHM_FILE;

  userid = id;
  sfd = -1; // open swapfile later
  fsd = fsd0; // file swap directory structure

  copyunit = 64L*1024*1024;
  /* prepare copybuf */
  int i;
  for (i = 0; i < 2; i++) {
    copybufs[i] = valloc(copyunit);
#ifdef USE_CUDA
    cudaError_t crc;
    crc = cudaHostRegister(copybufs[i], copyunit, 0 /*cudaHostRegisterPortable*/);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:fileheap::init@p%d] cudaHostRegister(%ldMiB) failed (rc=%d)\n",
	      HH_MYID, copyunit>>20, crc);
      exit(1);
    }

    crc = cudaStreamCreate(&copystreams[i]);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:fileheap::init@p%d] cudaStreamCreate failed (rc=%d)\n",
	      HH_MYID, crc);
      exit(1);
    }
#endif
  }

  return;
}

int fileheap::finalize()
{
  int i;
  for (i = 0; i < 2; i++) {
#ifdef USE_CUDA
    cudaStreamDestroy(copystreams[i]);
    cudaHostUnregister(copybufs[i]);
#endif
    free(copybufs[i]);
  }
  if (sfd != -1) {
    close(sfd);
  }

  heap::finalize();

  return 0;
}

int fileheap::expandHeap(size_t reqsize)
{
  size_t addsize;
  void *p;
  void *mapp;
  if (reqsize > FILEHEAP_STEP) {
    addsize = roundup(reqsize, FILEHEAP_STEP);
  }
  else {
    addsize = FILEHEAP_STEP;
  }

  /* expand succeeded */
  /* make a single large free area */
#if 1
  membuf *mbp = new membuf(piadd(heapptr, heapsize), addsize, 0L, HHMADV_FREED);
#else
  membuf *mbp = new membuf(heapsize, addsize, 0L, HHMADV_FREED);
#endif
  membufs.push_back(mbp);

#if 1  
  fprintf(stderr, "[HH:%s::expandHeap@p%d] heap expand succeeded %ldMiB -> %ldMiB\n",
	  name, HH_MYID, heapsize>>20, (heapsize + addsize)>>20);
#endif
  heapsize += addsize;
  
  return 0;
}


int fileheap::write_small(ssize_t offs, void *buf, int bufkind, size_t size)
{
  void *ptr;
#if 0
  fprintf(stderr, "[HH:fileheap::write_small@p%d] start (offs=%ld, size=%ld, align=%ld)\n",
	  HH_MYID, offs, size, align);
#endif

  if ((offs % align) != 0) {
    fprintf(stderr, "[HH:fileheap::write_small@p%d] offs=0x%lx not supported. to be fixed!\n",
	    HH_MYID, offs);
    return 0;
  }

#ifdef USE_CUDA
  if (bufkind == HHM_DEV) {
    cudaError_t crc;
    crc = cudaMemcpyAsync(copybufs[0], buf, size, cudaMemcpyDeviceToHost, copystreams[0]);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:fileheap::write_small@p%d] cudaMemcpy failed\n",
	      HH_MYID);
      exit(1);
    }
    cudaStreamSynchronize(copystreams[0]);
    ptr = copybufs[0];
  }
  else 
#endif
    if ((size_t)buf % align != 0) {
#if 1
      fprintf(stderr, "[HH:fileheap::write_small@p%d] buf=0x%lx (size=0x%lx) is not aligned; so copy once more. This is not an error, but slow\n",
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
    fprintf(stderr, "[HH:fileheap::write_small@p%d] ERROR lseek(0x%lx) -> %ld curious!\n",
	    HH_MYID, offs, orc);
    exit(1);
  }

  /* alignment */
  //size = ((size+align-1)/align)*align;
  size = roundup(size, align);

  size_t lrc = write(sfd, ptr, size);
  if (lrc != size) {
    fprintf(stderr, "[HH:fileheap::write_small@p%d] ERROR write(%d,%p,0x%lx) -> %ld curious! (offs=0x%lx, errno=%d)\n",
	    HH_MYID, sfd, ptr, size, lrc, offs, errno);
    exit(1);
  }
#if 0
  fprintf(stderr, "[HH:fileheap::write_small@p%d] OK lseek(0x%lx) and write(%d,%p,0x%lx) (align=%ld)\n",
	  HH_MYID, offs, sfd, ptr, size, align);
#endif
  return 0;
}

int fileheap::writeSeq(ssize_t offs, void *buf, int bufkind, size_t size)
{
  openSFileIfNotYet();

#if 0
  fprintf(stderr, "[HH:fileheap::writeSeq@p%d] start (offs=0x%lx, buf=%p, size=%ld, align=%ld)\n",
	  HH_MYID, offs, buf, size, align);
#endif

  size_t cur;
  void *p = buf;
  /* divide into copyunit */
  for (cur = 0; cur < size; cur += copyunit) {
    {
      // Refrain writing if someone is reading
      while (fsd->np_filein > 0) {
#if 0
	fprintf(stderr, "[HH:fileheap::writeSeq@p%d] suspend writing since np_filein=%d\n",
		HH_MYID, fsd->np_filein);
#endif
	usleep(10*1000);
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

int fileheap::read_small(ssize_t offs, void *buf, int bufkind, size_t size)
{
  void *ptr;
#if 0
  fprintf(stderr, "[HH:fileheap::read_s@p%d] start (align=%ld)\n",
	  HH_MYID, align);
#endif

#ifdef USE_CUDA
  if (bufkind == HHM_DEV) {
    ptr = copybufs[0];
  }
  else 
#endif
    if ((size_t)buf % align != 0) {
      ptr = copybufs[0];
    }
    else {
      ptr = buf;
    }
  
  if ((offs % align) != 0) {
    fprintf(stderr, "[HH:fileheap::read_s@p%d] offs=0x%lx not supported. to be fixed!\n",
	    HH_MYID, offs);
    return 0;
  }

  /* seek the swap file */
  off_t orc = lseek(sfd, offs, SEEK_SET);
  if (orc != offs) {
    fprintf(stderr, "[HH:fileheap::read_s@p%d] ERROR lseek(0x%lx) -> %ld curious!\n",
	    HH_MYID, offs, orc);
    exit(1);
  }

  /* alignment */
  size = roundup(size, align);

  size_t lrc = read(sfd, ptr, size);
  if (lrc != size) {
    fprintf(stderr, "[HH:fileheap::read_s@p%d] ERROR read(%d,%p,0x%lx) -> %ld curious! (errno=%d)\n",
	    HH_MYID, sfd, ptr, size, lrc, errno);
    exit(1);
  }

#ifdef USE_CUDA
  if (bufkind == HHM_DEV) {
    cudaError_t crc;
    crc = cudaMemcpyAsync(buf, copybufs[0], size, cudaMemcpyHostToDevice, copystreams[0]);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:fileheap::read_s@p%d] cudaMemcpy failed\n",
	      HH_MYID);
      exit(1);
    }
    cudaStreamSynchronize(copystreams[0]);
  }
  else 
#endif
    if ((size_t)buf % align != 0) {
#if 1
      fprintf(stderr, "[HH:fileheap::read_s@p%d] buf=0x%lx (size=0x%lx) is not aligned; so copy once more. This is not an error, but slow\n",
	      HH_MYID, buf, size);
#endif
      memcpy(buf, copybufs[0], size);
    }
  
#if 0
  fprintf(stderr, "[HH:fileheap::read_s@p%d] OK (align=%ld)\n",
	  HH_MYID, align);
#endif

  return 0;
}

int fileheap::readSeq(ssize_t offs, void *buf, int bufkind, size_t size)
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


/* */
int fileheap::swapOut()
{
  fprintf(stderr, "[HH:fileheap::swapOut@p%d] ERROR: This should not be called\n",
	  HH_MYID);
  exit(1);
  return -1;
}

int fileheap::swapIn()
{
  fprintf(stderr, "[HH:fileheap::swapIn@p%d] ERROR: This should not be called\n",
	  HH_MYID);
  exit(1);
  return -1;
}

int fileheap::checkSwapResAsLower(int kind, int *pline)
{
  int res = -1;
  int line;

  if (kind == HHSW_OUT) {
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
  else if (kind == HHSW_IN) {
    if (fsd->np_filein > 0) {
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
    fprintf(stderr, "[HH:%s::checkSwapResAsLower@p%d] ERROR: kind %d unknown\n",
	    name, HH_MYID, kind);
    exit(1);
  }

  if (pline != NULL) {
    *pline = line; // debug info
  }
  return res;
}

int fileheap::reserveSwapResAsLower(int kind)
{
  if (kind == HHSW_IN) {
    fsd->np_filein++;
  }
  else if (kind == HHSW_OUT) {
    fsd->np_fileout++;
  }
  return 0;
}

int fileheap::releaseSwapResAsLower(int kind)
{
  if (kind == HHSW_IN) {
    fsd->np_filein--;
    assert(fsd->np_filein >= 0);
  }
  else if (kind == HHSW_OUT) {
    fsd->np_fileout--;
    assert(fsd->np_fileout >= 0);
  }
  return 0;
}
