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

/*** fileswapper ******/

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
    fprintf(stderr, "[HH:fileswapper::openSFile@p%d] ERROR in open(%s)\n",
	    HH_MYID, sfname);
    exit(1);
  }

#ifdef HHLOG_SWAP
  fprintf(stderr, "[HH:fileswapper::openSFile@p%d(L%d)] created a file %s\n",
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

  HH_makeSFileName(fsd, userid, sfname);
  sfd = HH_openSFile(sfname);

  return 0;
}

fileswapper::fileswapper(int id, fsdir *fsd0) : swapper()
{
  int rc;
  cudaError_t crc;

  sprintf(name, "fileswapper");

  align = 512;

  userid = id;
  sfd = -1; // open swapfile later
  fsd = fsd0; // file swap directory structure

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

  return;
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
      while (fsd->np_filein > 0) {
#if 0
	fprintf(stderr, "[HH:fileswapper::write1@p%d] suspend writing since np_filein=%d\n",
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


/* */
int fileswapper::swapOut()
{
  fprintf(stderr, "[HH:fileswapper::swapOut@p%d] ERROR: This should not be called\n",
	  HH_MYID);
  exit(1);
  return -1;
}

int fileswapper::swapIn()
{
  fprintf(stderr, "[HH:fileswapper::swapIn@p%d] ERROR: This should not be called\n",
	  HH_MYID);
  exit(1);
  return -1;
}


