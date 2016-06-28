#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <malloc.h>
#include <errno.h>
#include "hhrt_impl.h"

int HH_initHeap(size_t heapsize)
{
  /* OBSOLETE */
  fprintf(stderr, "[HH_initHeap@p%d] OBSOLETE: The call is ignored\n",
	  HH_MYID);
  return 0;
}

/* This may cause waiting */
int HH_initHeap_inner()
{
  cudaError_t crc;
  dev *d = HH_curdev();
  size_t heapsize = d->default_heapsize;

  HHL2->devheap = new devheap();
  HHL2->devheap->init(heapsize);

  int usefile = 1; //(HHL->lrank >= HHL2->conf.nlphost);

  HHL2->devheap_hostswapper = (swapper*)(new hostswapper());
  HHL2->devheap_hostswapper->init(0);

  HHL2->devheap_fileswapper = NULL;
  if (usefile) {
    HHL2->devheap_fileswapper = (swapper*)(new fileswapper());
    HHL2->devheap_fileswapper->init(0);
  }

#ifdef USE_SWAPHOST
  HHL2->hostheap = new hostheap();
  HHL2->hostheap->init(0L);

  HHL2->hostheap_fileswapper = NULL;
  if (usefile) {
    HHL2->hostheap_fileswapper = new fileswapper();
    HHL2->hostheap_fileswapper->init(1);
  }
#endif

  HHL->pmode = HHP_RUNNABLE;
  HHL->dmode = HHD_ON_FILE;
  HH_sleepForMemory(1);
  HHL->pmode = HHP_RUNNING;

  return 0;
}

int HH_finalizeHeap()
{
  assert(HHL->dmode == HHD_ON_DEV);
#ifdef USE_SWAPHOST
  if (HHL2->hostheap_fileswapper) {
    HHL2->hostheap_fileswapper->finalize();
  }
  HHL2->hostheap->finalize();
#endif

  if (HHL2->devheap_fileswapper) {
    HHL2->devheap_fileswapper->finalize();
  }
  if (HHL2->devheap_hostswapper) {
    HHL2->devheap_hostswapper->finalize();
  }
  HHL2->devheap->finalize();

  lock_log(&HHS->sched_ml);

  dev *d = HH_curdev();
  assert(d->hp_user[HHL->hpid] == HH_MYID);
  d->hp_user[HHL->hpid] = -1;

  HHS->nhostusers[HHL->hpid]--;

  HHL->dmode = HHD_ON_FILE;
  HHL->pmode = HHP_RUNNABLE;

  pthread_mutex_unlock(&HHS->sched_ml);

  return 0;
}

/*****************************************/

int HH_afterDevSwapOut()
{
  dev *d = HH_curdev();
  assert(HHL->hpid >= 0 && HHL->hpid < HHS->nheaps);

  d->hp_user[HHL->hpid] = -1;
  fprintf(stderr, "[HH_afterDevSwapOut@p%d] [%.2f] I release heap slot %d\n",
	  HH_MYID, Wtime_prt(), HHL->hpid);

  return 0;
}

// D2H
// This function assumes sched_ml is locked
int HH_swapOutD2H()
{
  dev *d;

  assert (HHL->dmode == HHD_ON_DEV);
  HHL->dmode = HHD_SO_D2H;
#if 0
  fprintf(stderr, "[swapOut@p%d] START\n",
	  HH_MYID);
#endif

  d = HH_curdev();
  d->np_out++;
#ifndef DEBUG_SEQ_SWAP
  pthread_mutex_unlock(&HHS->sched_ml);
#endif

  /* D -> H */
  HHL2->devheap->swapOut(HHL2->devheap_hostswapper);

#ifndef DEBUG_SEQ_SWAP
  lock_log(&HHS->sched_ml);
#endif
  HH_afterDevSwapOut();
  d->np_out--;
  if (d->np_out < 0) {
    fprintf(stderr, "[swapOut@p%d] np_out = %d strange\n",
	    HH_MYID, d->np_out);
  }

  HHL->dmode = HHD_ON_HOST;

  return 0;
}

#ifdef USE_FILESWAP_THREAD

static void *swapOutH2Fthread(void *arg)
{
#ifdef USE_SWAPHOST
  /* Host Heap */
  /* H -> F */
  if (HHL2->hostheap_fileswapper) {
    HHL2->hostheap->swapOut(HHL2->hostheap_fileswapper);
  }
#endif

  if (HHL2->devheap_fileswapper) {
    /* H -> F */
    HHL2->devheap_hostswapper->swapOut(HHL2->devheap_fileswapper);
  }

  return NULL;
}

// This function assumes sched_ml is locked
int HH_startSwapOutH2F()
{
  assert(HHL->dmode == HHD_ON_HOST);
  HHL->dmode = HHD_SO_H2F;

  fsdir *fsd = HH_curfsdir();
  fsd->np_fileout++;

  pthread_create(&HHL2->fileswap_tid, NULL, swapOutH2Fthread, NULL);

  return 0;
}

// This function assumes sched_ml is locked
int HH_tryfinSwapOutH2F()
{
  int rc;
  void *retval;
  assert(HHL->dmode == HHD_SO_H2F);

  rc= pthread_tryjoin_np(HHL2->fileswap_tid, &retval);
  if (rc == EBUSY) {
    // thread has not yet terminated
    return 0;
  }

  // thread has terminated

  fsdir *fsd = HH_curfsdir();
  fsd->np_fileout--;

  HHL->dmode = HHD_ON_FILE;
  HHS->nhostusers[HHL->hpid]--;
  fprintf(stderr, "[HH_swapOutH2F@p%d] [%.2f] I release host capacity\n",
	  HH_MYID, Wtime_prt());

  return 1;
}

// This function assumes sched_ml is locked
int HH_swapOutH2F()
{
  HH_startSwapOutH2F();
  while (1) {
    int rc = HH_tryfinSwapOutH2F();
    if (rc > 0) break;
    pthread_mutex_unlock(&HHS->sched_ml);
    usleep(100);
    lock_log(&HHS->sched_ml);
  }
  return 0;
}

#else // !USE_FILESWAP_THREAD

// H2F
// This function assumes sched_ml is locked
int HH_swapOutH2F()
{
  assert(HHL->dmode == HHD_ON_HOST);
  HHL->dmode = HHD_SO_H2F;

  fsdir *fsd = HH_curfsdir();
  fsd->np_fileout++;

  pthread_mutex_unlock(&HHS->sched_ml);

  if (HHL2->devheap_fileswapper) {
    /* H -> F */
    HHL2->devheap_hostswapper->swapOut(HHL2->devheap_fileswapper);
  }

#ifdef USE_SWAPHOST
  /* Host Heap */
  /* H -> F */
  if (HHL2->hostheap_fileswapper) {
    HHL2->hostheap->swapOut(HHL2->hostheap_fileswapper);
  }
#endif

  lock_log(&HHS->sched_ml);

  fsdir *fsd = HH_curfsdir();
  fsd->np_fileout--;

  HHL->dmode = HHD_ON_FILE;

  HHS->nhostusers[HHL->hpid]--;
#if 0
  fprintf(stderr, "[HH_swapOutH2F@p%d] [%.2f] I release host capacity\n",
	  HH_MYID, Wtime_prt());
#endif

  return 0;
}

#endif // !USE_FILESWAP_THREAD

// H2D
// This function assumes sched_ml is locked
int HH_swapInH2D(int initing)
{
  dev *d;
  assert(HHL->dmode == HHD_ON_HOST);
  HHL->dmode = HHD_SI_H2D;
  d = HH_curdev();
  d->np_in++;

#ifndef DEBUG_SEQ_SWAP
  pthread_mutex_unlock(&HHS->sched_ml);
#endif
  /* H -> D */
  HHL2->devheap->swapIn(initing);

#ifndef DEBUG_SEQ_SWAP
  lock_log(&HHS->sched_ml);
#endif
  HHL->dmode = HHD_ON_DEV;
  d->np_in--;

  return 0;
}

#ifdef USE_FILESWAP_THREAD
static void *swapInF2Hthread(void *arg)
{
  int initing = (int)((long)arg);
#ifdef USE_SWAPHOST
  /* Host Heap */
  HHL2->hostheap->swapIn(initing);
#endif

  /* Device Heap */
  if (HHL2->devheap_fileswapper) {
    /* F -> H */
    HHL2->devheap_hostswapper->swapIn(initing);
  }

  return NULL;
}

// This function assumes sched_ml is locked
int HH_startSwapInF2H(int initing)
{
  assert(HHL->dmode == HHD_ON_FILE);

  fsdir *fsd = HH_curfsdir();
  fsd->np_filein++;

  HHS->nhostusers[HHL->hpid]++;
  pthread_create(&HHL2->fileswap_tid, NULL, swapInF2Hthread, (void*)((long)initing));
  HHL->dmode = HHD_SI_F2H;

  return 0;
}

// This function assumes sched_ml is locked
int HH_tryfinSwapInF2H()
{
  int rc;
  void *retval;
  assert(HHL->dmode == HHD_SI_F2H);

  rc= pthread_tryjoin_np(HHL2->fileswap_tid, &retval);
  if (rc == EBUSY) {
    // thread has not yet terminated
    return 0;
  }

  // thread has terminated
  fsdir *fsd = HH_curfsdir();
  fsd->np_filein--;

  HHL->dmode = HHD_ON_HOST;

  return 1;
}

#else // !USE_FILESWAP_THREAD
// F2H
// This function assumes sched_ml is locked
int HH_swapInF2H(int initing)
{
  assert(HHL->dmode == HHD_ON_FILE);

  fsdir *fsd = HH_curfsdir();
  fsd->np_filein++;

  HHS->nhostusers[HHL->hpid]++;
  HHL->dmode = HHD_SI_F2H;
  pthread_mutex_unlock(&HHS->sched_ml);

#ifdef USE_SWAPHOST
  /* Host Heap */
  HHL2->hostheap->swapIn(initing);
#endif

  /* Device Heap */
  if (HHL2->devheap_fileswapper) {
    /* F -> H */
    HHL2->devheap_hostswapper->swapIn(initing);
  }

  lock_log(&HHS->sched_ml);

  fsdir *fsd = HH_curfsdir();
  fsd->np_filein--;

  HHL->dmode = HHD_ON_HOST;

  return 0;
}

#endif // !USE_FILESWAP_THREAD

/************************************************/
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

  p = HHL2->devheap->alloc(size);
  *pp = p;
  return cudaSuccess;
}

cudaError_t HHcudaFree(void *p)
{
  if (p == NULL) return cudaSuccess;

  int rc;
  rc = HHL2->devheap->free(p);
  if (rc != 0) {
    return cudaErrorInvalidDevicePointer;
  }
  return cudaSuccess;
}

int HH_madvise(void *p, size_t size, int kind)
{
  int rc;
  rc = HHL2->devheap->madvise(p, size, kind);
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
    fprintf(stderr, "[HHcudaHostAlloc] WARNING: normal malloc is used now\n");
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