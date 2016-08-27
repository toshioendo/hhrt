#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <malloc.h>
#include <errno.h>
#include "hhrt_impl.h"

/* statistics about host memory for debug */
int HH_addHostMemStat(int kind, ssize_t incr)
{
  ssize_t s;
  assert(kind >= 0 && kind < HHST_MAX);
  HHL->hmstat.used[kind] += incr;
  s = HHL->hmstat.used[kind];
  if (s < 0 || s > (ssize_t)128 << 30) {
    fprintf(stderr, "[HH_addHostMemStat@p%d] host mem usage (kind %s) %ldMB looks STRANGE.\n",
	    HH_MYID, hhst_names[kind], s>>20L);
  }
  return 0;
}

/************************************************/

/* Wrappers of cudaMalloc/cudaFree */
cudaError_t HHcudaMalloc(void **pp, size_t size)
{
  void *p = NULL;

  if (HHL->devmode == HHDEV_NORMAL) {
    assert(HHL->dmode == HHD_ON_DEV);
  }

  p = HH_curdevheap()->alloc(size);
  *pp = p;
  return cudaSuccess;
}

cudaError_t HHcudaFree(void *p)
{
  if (p == NULL) return cudaSuccess;

  int rc;
  rc = HH_curdevheap()->free(p);
  if (rc != 0) {
    return cudaErrorInvalidDevicePointer;
  }
  return cudaSuccess;
}

int HH_madvise(void *p, size_t size, int kind)
{
  int rc;
  rc = HH_curdevheap()->madvise(p, size, kind);
  if (rc == 0) return 0;

#ifdef USE_SWAPHOST
  rc = HHL2->hostheap->madvise(p, size, kind);
  if (rc == 0) return 0;
#endif

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

cudaError_t HHcudaHostAlloc(void ** pp, size_t size, unsigned int flags)
{
  void *p;
  if (HH_MYID == 0) {
    fprintf(stderr, "[HHcudaHostAlloc@p%d] WARNING: normal malloc is used now\n",
	    HH_MYID);
  }
  p = HHL2->hostheap->alloc(size);
  *pp = p;
  return cudaSuccess;
}

cudaError_t HHcudaMallocHost(void ** pp, size_t size)
{
  return HHcudaHostAlloc(pp, size, cudaHostAllocDefault);
}
#endif
