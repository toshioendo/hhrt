#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "hhrt_impl.h"

/* CUDA device memory management */

#ifdef USE_CUDA

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
  // We do not do substantial initialization work yet
  expandable = 0;

  heapptr = NULL;
  align = 256L;
  memkind = HHM_DEV;

  copyunit = 0L;
  copybuf = NULL;

  device = device0;
  sprintf(name, "devheap(d%d)", device->devid);
  hp_baseptr = NULL;

  cudaflag = (HHL->lrank == 0)? 1: 0;

  return;
}

int devheap::finalize()
{
  heap::finalize();

#if 0 && defined USE_DEVRESET
  if (HHL->lrank != 0 && cudaflag == 1) {
    resetCuda();
    sleep(1);
  }
#endif

  HH_lockSched();
  if (device->dhslot_users[HHL->cuda.hpid] == HH_MYID) {
    device->dhslot_users[HHL->cuda.hpid] = -1;
  }
  HH_unlockSched();

  return 0;
}

/*************************************/
int HH_printMemHandle(FILE *out, cudaIpcMemHandle_t *handle);

int devheap::initCopyBufs()
{
  cudaError_t crc;
  copyunit = (size_t)8*1024*1024;
  crc = cudaHostAlloc(&copybuf, copyunit, cudaHostAllocDefault);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH_cudaInitBufs@p%d] ERROR: cudaHostAlloc failed (rc=%d)\n",
	    HH_MYID, crc);
    exit(1);
  }

  crc = cudaStreamCreate(&copystream);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH_cudaInitBufs@p%d] ERROR: cudaStreamCreate failed (rc=%d)\n",
	    HH_MYID, crc);
    exit(1);
  }

  return 0;
}

int devheap::resetCuda()
{
#if 01
  fprintf(stderr, "[HH:%s::resetCuda@p%d] [%.2f] Here I am going to deep sleep (cudaDeviceReset)\n",
	  name, HH_MYID, Wtime_prt());
#endif
  cudaError_t crc;
  crc = cudaDeviceReset();
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:%s::resetCuda@p%d] [%.2f] ERROR: cudaDeviceReset failed! (rc=%d)\n",
	    name, HH_MYID, Wtime_prt(), crc);
  }
#if 1
  fprintf(stderr, "[HH:%s::resetCuda@p%d] [%.2f] cudaDeviceReset finished\n",
	  name, HH_MYID, Wtime_prt());
#endif
  cudaflag = 0;
  return 0;
}

void *devheap::allocCapacity1st(size_t heapsize)
{
  dev *d = device;
  void *org_baseptr = hp_baseptr;
  if (HHL->lrank == 0) {
    hp_baseptr = d->hp_baseptr0;
  }
  else {
    cudaError_t crc;
    crc = cudaIpcOpenMemHandle(&hp_baseptr, d->hp_handle, cudaIpcMemLazyEnablePeerAccess);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:%s::allocCapacity@p%d] ERROR: cudaIpcOpenMemHandle failed! (%s)\n",
	      name, HH_MYID, cudaGetErrorString(crc));
      fprintf(stderr, "[HH:%s::allocCapacity@p%d]  handle is ",
	      name, HH_MYID);
      HH_printMemHandle(stderr, &d->hp_handle);
      fprintf(stderr, "\n");

      system("nvidia-smi");
      fflush(0);

      int devid = -1;
      cudaGetDevice(&devid);
      fprintf(stderr, "[HH:%s::allocCapacity@p%d]  current using dev %d\n",
	      name, HH_MYID, devid);
      exit(1);
    }
  }
  // here hp_baseptr is valid

  if (org_baseptr != NULL) {
    if (hp_baseptr != org_baseptr) {
      fprintf(stderr, "[HH:%s::allocCapacity@p%d] ERROR: new baseptr %p != org %p !!! Reconsider HHRT implementation...\n",
	      name, HH_MYID, hp_baseptr, org_baseptr);
      exit(1);
    }
    fprintf(stderr, "[HH:%s::allocCapacity@p%d] OK: baseptr %p recovered!!\n",
	    name, HH_MYID, hp_baseptr);
  }
  else {
    fprintf(stderr, "[HH:%s::allocCapacity@p%d] I got baseptr %p for the first time!!\n",
	    name, HH_MYID, hp_baseptr);
  }

  initCopyBufs();
  cudaflag = 1; // this process is connected to CUDA
  return 0;
}

void *devheap::allocCapacity(size_t dummy, size_t heapsize)
{
  dev *d = device;
  void *dp;

  assert(HHL->cuda.hpid >= 0 && HHL->cuda.hpid < HHS->cuda.ndh_slots);

  if (hp_baseptr == NULL || cudaflag == 0) {
    allocCapacity1st(heapsize);
  }

  dp = piadd(hp_baseptr, d->default_heapsize*HHL->cuda.hpid);
  return dp;
}

int devheap::allocHeapInner()
{
  void *dp;

  assert(heapptr == NULL);

  dp = allocCapacity(0L, heapsize);
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

int devheap::releaseHeap()
{
  assert(heapptr != NULL);
  // do nothing

  return 0;
}

/* allocate device heap for swapIn */
int devheap::restoreHeap()
{
  void *dp;

  assert(heapptr != NULL);

  dp = allocCapacity(0L, heapsize);
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
    else if (device->dhslot_users[HHL->cuda.hpid] >= 0) {
      res = HHSS_EBUSY; // device heap slot is occupied
      if (device->dhslot_users[HHL->cuda.hpid] == HH_MYID) {
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
    if (device->dhslot_users[HHL->cuda.hpid] >= 0) {
      fprintf(stderr, "[HH:%s::reserveSwapRes@p%d] devslot[%d]'s user = %d, STRANGE?\n",
	      name, HH_MYID, HHL->cuda.hpid, device->dhslot_users[HHL->cuda.hpid]);
    }
    assert(device->dhslot_users[HHL->cuda.hpid] < 0);
    device->dhslot_users[HHL->cuda.hpid] = HH_MYID;
#if 1
    fprintf(stderr, "[HH:%s::reserveSwapRes@p%d] [%.2lf] I reserved devslot[%d]\n",
	    name, HH_MYID, Wtime_prt(), HHL->cuda.hpid);
#endif
    device->np_in++;
    HH_profBeginAction("H2D");
  }
  else if (kind == HHSW_OUT) {
    device->np_out++;
    HH_profBeginAction("D2H");
  }
  else {
    assert(0);
    exit(1);
  }

  return 0;
}

int devheap::doSwap()
{
  int olddevid;
  cudaGetDevice(&olddevid);
  cudaSetDevice(device->devid);
  heap::doSwap(); // parent class
  cudaSetDevice(olddevid); // restore devid
  return 0;
}

int devheap::releaseSwapResSelf(int kind)
{
  // Release resource information after swapping
  if (kind == HHSW_IN) {
    device->np_in--;
    HH_profEndAction("H2D");
  }
  else if (kind == HHSW_OUT) {
    device->np_out--;
    if (device->np_out < 0) {
      fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] np_out = %d strange\n",
	      name, HH_MYID, device->np_out);
    }

    assert(HHL->cuda.hpid >= 0 && HHL->cuda.hpid < HHS->cuda.ndh_slots);
    assert(device->dhslot_users[HHL->cuda.hpid] == HH_MYID);
    device->dhslot_users[HHL->cuda.hpid] = -1;
#ifdef HHLOG_SWAP
    fprintf(stderr, "[HH:%s::releaseSwapRes@p%d] [%.2f] I release devslot[%d]\n",
	    name, HH_MYID, Wtime_prt(), HHL->cuda.hpid);
#endif
    HH_profEndAction("D2H");
  }
  else {
    assert(0);
    exit(1);
  }

  return 0;
}


#endif // USE_CUDA
