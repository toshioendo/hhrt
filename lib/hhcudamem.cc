#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "hhrt_impl.h"

/* CUDA device memory management */

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

////////////////////////
int devheap::inferSwapMode(int kind0)
{
  int res_kind;
  if (kind0 == HHD_SO_ANY) {
    if (HHS->nlprocs <= HHS->ndh_slots) {
      res_kind = HHD_SWAP_NONE;
    }
    else if (swapped == 0) {
      res_kind = HHD_SO_D2H;
    }
    else if (lower->lower != NULL && lower->swapped == 0) {
      res_kind = HHD_SO_H2F;
    }
    else {
      res_kind = HHD_SWAP_NONE;
    }
  }
  else if (kind0 == HHD_SI_ANY) {
    if (heapptr != NULL && swapped == 0) {
      res_kind = HHD_SWAP_NONE;
    }
    else if (lower->swapped == 0) {
      res_kind = HHD_SI_H2D;
    }
    else {
      assert(lower->lower != NULL);
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
  }
#endif
  return res_kind;
}

// check resource availability before actual swapping
int devheap::checkSwapRes(int kind0)
{
  int res;
  int line = -100; // debug

#if 1
  res = heap::checkSwapRes(kind0);
  if (res == HHSS_OK) {
    // we need further check
    if (kind0 == HHD_SO_ANY) {
      if (device->np_out > 0) {
	res = HHSS_EBUSY;  // someone is doing swapD2H
	line = __LINE__;
      }
      else {
	res = HHSS_OK;
	line = __LINE__;
      }
    }
    else if (kind0 == HHD_SI_ANY) {
      if (device->np_in > 0) {
	res =  HHSS_EBUSY; // someone is doing swapH2D
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
	      name, HH_MYID, kind0);
      exit(1);
    }
  }
#else
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
    if (device->np_out > 0) {
      res = HHSS_EBUSY;  // someone is doing swapD2H
      line = __LINE__;
    }
    else {
      res = HHSS_OK;
      line = __LINE__;
    }
  }
  else if (kind == HHD_SI_H2D) {
    if (device->np_in > 0) {
      res =  HHSS_EBUSY; // someone is doing swapH2D
      line = __LINE__;
    }
    else if (device->dhslot_users[HHL->hpid] >= 0) {
      res = HHSS_EBUSY; // device heap slot is occupied
      if (device->dhslot_users[HHL->hpid] == HH_MYID) {
	fprintf(stderr, "[HH:%s::checkSwapRes@p%d] I'm devslot's user, STRANGE?\n",
		name, HH_MYID);
      }
      line = __LINE__;
    }
    else {
      res = HHSS_OK;
      line = __LINE__;
    }
  }
  else if (kind == HHD_SO_H2F) {
    fsdir *fsd = ((fileheap*)lower->lower)->fsd;
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
    fsdir *fsd = ((fileheap*)lower->lower)->fsd;
    if (fsd->np_filein > 0) {
      // someone is doing swapF2H or swapH2F
      res = HHSS_EBUSY;
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
#endif

#if 0
  if (res == HHSS_OK || rand() % 256 == 0) {
    const char *strs[] = {"OK", "EBUSY", "NONEED", "XXX"};
    fprintf(stderr, "[HH:%s::checkSwapRes@p%d] result=%s (line=%d)\n",
	    name, HH_MYID, strs[res], line);
  }
#endif
  return res;
}

int devheap::reserveSwapRes(int kind0)
{
#if 1
  if (kind0 == HHD_SI_ANY) {
    device->dhslot_users[HHL->hpid] = HH_MYID;
    device->np_in++;
  }
  else if (kind0 == HHD_SO_ANY) {
    device->np_out++;
  }
  else {
    fprintf(stderr, "[HH:%s::reserveSR@p%d] ERROR: kind %d unknown\n",
	    name, HH_MYID, kind0);
    exit(1);
  }

  swap_kind = kind0; // remember the kind
#else
  int kind = inferSwapMode(kind0);
  swap_kind = kind; // remember the kind

  // Reserve resource information before swapping
  // This is called after last checkSwapRes(), without releasing schedule lock
  if (kind == HHD_SI_H2D) {
    device->dhslot_users[HHL->hpid] = HH_MYID;
    device->np_in++;
  }
  else if (kind == HHD_SO_D2H) {
    device->np_out++;
  }
  else if (kind == HHD_SI_F2H) {
    // reserve is done by hostheap. is it OK?
  }
  else {
    // do nothing
  }
#endif
  return 0;
}

int devheap::doSwap()
{
#if 1
  heap::doSwap();
#else
  int kind = swap_kind;
  HH_profBeginAction(hhd_snames[kind]);

  if (kind == HHD_SO_D2H) {
    swapOut();
  }
  else if (kind == HHD_SI_H2D) {
    swapIn();
  }
  else if (kind == HHD_SO_H2F) {
    if (lower == NULL || lower->lower == NULL) {
      fprintf(stderr, "[HH:%s::swap(H2F)@p%d] SKIP\n",
	      name, HH_MYID);
      goto out;
    }
    
    lower->swapOut();
  }
  else if (kind == HHD_SI_F2H) {
    if (lower == NULL || lower->lower == NULL) {
      fprintf(stderr, "[HH:%s::swap(F2H)@p%d] SKIP\n",
	      name, HH_MYID);
      goto out;
    }
    
    lower->swapIn();
  }
  else {
    fprintf(stderr, "[HH:devheap::swap@p%d] ERROR: kind %d unknown\n",
	    HH_MYID, kind);
    exit(1);
  }
 out:
  HH_profEndAction(hhd_snames[kind]);
#endif
  return 0;
}

int devheap::releaseSwapRes()
{
  int kind = swap_kind;

  // Release resource information after swapping
  if (kind == HHD_SI_ANY) {
    device->np_in--;
  }
  else if (kind == HHD_SO_ANY) {
    device->np_out--;
    if (device->np_out < 0) {
      fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] np_out = %d strange\n",
	      name, HH_MYID, device->np_out);
    }

    assert(HHL->hpid >= 0 && HHL->hpid < HHS->ndh_slots);
    assert(device->dhslot_users[HHL->hpid] == HH_MYID);
    device->dhslot_users[HHL->hpid] = -1;
    fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] [%.2f] I release heap slot %d\n",
	    name, HH_MYID, Wtime_prt(), HHL->hpid);
  }
  else {
    fprintf(stderr, "[HH:%s::releaseSR@p%d] ERROR: kind %d unknown\n",
	    name, HH_MYID, kind);
    exit(1);
  }

  swap_kind = HHD_SWAP_NONE;
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
