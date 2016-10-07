#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "hhrt_impl.h"

/* CUDA device memory management */

heap *HH_curdevheap()
{
  if (HHL->curdevid < 0) {
    fprintf(stderr, 
	    "[HH_curdevheap@p%d] ERROR: curdevid is not set\n",
	    HH_MYID);
    exit(1);
  }
  heap *h = HHL2->devheaps[HHL->curdevid];
  if (h == NULL) {
    fprintf(stderr, 
	    "[HH_curdevheap@p%d] ERROR: devid %d is not initialized\n",
	    HH_MYID, HHL->curdevid);
    exit(1);
  }
  return h;
}

heap *HH_devheapCreate(dev *d)
{
  devheap *dh;
  heap *h;
  size_t heapsize = d->default_heapsize;

  dh = new devheap(heapsize, d);
  //dh->device = d;
  h = (heap *)dh;
  /* make memory hierarchy for devheap */

  assert(HHL2->hostheap != NULL);
  h->addLower(HHL2->hostheap);
  HHL2->hostheap->addUpper(h);

  return h;
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

  assert(HHL->hpid >= 0 && HHL->hpid < HHS->ndh_slots);
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

////////////////////////

// check resource availability before actual swapping
// called by heap::checkSwapRes
int devheap::checkSwapResSelf(int kind, int *pline)
{
  int res = -1;
  int line = -992;

  if (kind == HHSW_OUT) {
    if (device->np_out > 0) {
      res = HHSS_EBUSY;  // someone is doing swap
      line = __LINE__;
    }
    else {
      res = HHSS_OK;
      line = __LINE__;
    }
  }
  else if (kind == HHSW_IN) {
    if (device->np_in > 0) {
      res =  HHSS_EBUSY; // someone is doing swap
      line = __LINE__;
    }
    else if (device->dhslot_users[HHL->hpid] >= 0) {
      res = HHSS_EBUSY; // device heap slot is occupied
      if (device->dhslot_users[HHL->hpid] == HH_MYID) {
	fprintf(stderr, "[HH:%s::checkSwapRes@p%d] I'm devslot's user, STRANGE?\n",
		name, HH_MYID);
	usleep(100*1000);
      }
      line = __LINE__;
    }
    else {
      res = HHSS_OK;
      line = __LINE__;
    }
  }
  else {
    fprintf(stderr, "[HH:%s::checkSwapRes@p%d] ERROR: kind %d unknown\n",
	    name, HH_MYID, kind);
    exit(1);
  }
  
  if (pline != NULL) {
    *pline = line; // debug info
  }
  return res;
}

int devheap::reserveSwapResSelf(int kind)
{
  if (kind == HHSW_IN) {
    if (device->dhslot_users[HHL->hpid] >= 0) {
      fprintf(stderr, "[HH:%s::reserveSwapRes@p%d] devslot[%d]'s user = %d, STRANGE?\n",
	      name, HH_MYID, HHL->hpid, device->dhslot_users[HHL->hpid]);
    }
    assert(device->dhslot_users[HHL->hpid] < 0);
    device->dhslot_users[HHL->hpid] = HH_MYID;
#if 1
    fprintf(stderr, "[HH:%s::reserveSwapRes@p%d] [%.2lf] I reserved devslot[%d]\n",
	    name, HH_MYID, Wtime_prt(), HHL->hpid);
#endif
    device->np_in++;
  }
  else if (kind == HHSW_OUT) {
    device->np_out++;
  }
  else {
    assert(0);
    exit(1);
  }

  swapping_kind = kind; // remember the kind
  return 0;
}

int devheap::releaseSwapResSelf(int kind)
{
  // Release resource information after swapping
  if (kind == HHSW_IN) {
    device->np_in--;
  }
  else if (kind == HHSW_OUT) {
    device->np_out--;
    if (device->np_out < 0) {
      fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] np_out = %d strange\n",
	      name, HH_MYID, device->np_out);
    }

    assert(HHL->hpid >= 0 && HHL->hpid < HHS->ndh_slots);
    assert(device->dhslot_users[HHL->hpid] == HH_MYID);
    device->dhslot_users[HHL->hpid] = -1;
#ifdef HHLOG_SWAP
    fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] [%.2f] I release heap slot %d\n",
	    name, HH_MYID, Wtime_prt(), HHL->hpid);
#endif
  }
  else {
    assert(0);
    exit(1);
  }

  //swapping_kind = HHSW_NONE;
  return 0;
}

/* Device memory related used API */

/* Wrappers of cudaMalloc/cudaFree */
cudaError_t HHcudaMalloc(void **pp, size_t size)
{
  HH_checkDev();

  void *p = NULL;

  if (HHL->devmode == HHDEV_NORMAL) {
  }

  p = HH_curdevheap()->alloc(size);
  *pp = p;
  return cudaSuccess;
}

cudaError_t HHcudaFree(void *p)
{
  HH_checkDev();

  if (p == NULL) return cudaSuccess;

  int rc;
  rc = HH_curdevheap()->free(p);
  if (rc != 0) {
    return cudaErrorInvalidDevicePointer;
  }
  return cudaSuccess;
}

#ifdef USE_SWAPHOST
cudaError_t HHcudaHostAlloc(void ** pp, size_t size, unsigned int flags)
{
  HH_checkDev();

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
  HH_checkDev();

  return HHcudaHostAlloc(pp, size, cudaHostAllocDefault);
}
#endif
