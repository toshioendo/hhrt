#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "hhrt_impl.h"

int HH_lockSched()
{
  double st = Wtime(), et;
  pthread_mutex_lock(&HHS->sched_ml);
  et = Wtime();
  if (et-st > 0.1) {
    fprintf(stderr, "[HH_lockSched@p%d] [%.2lf-%.2lf] %s:%d LOCK TOOK LONG\n",
	    HH_MYID, Wtime_conv_prt(st), Wtime_conv_prt(et), __FILE__, __LINE__);
  }
  return 0;
}

int HH_unlockSched()
{
  pthread_mutex_unlock(&HHS->sched_ml);
  return 0;
}

//------------------------------- D2H
int HH_checkD2H()
{
  dev *d = HH_curdev();
  if (d->np_out > 0) return 0;
  return 1;
}

static int afterSwapOutD2H()
{
  dev *d = HH_curdev();
  assert(HHL->hpid >= 0 && HHL->hpid < HHS->ndhslots);

  d->dhslot_users[HHL->hpid] = -1;
  fprintf(stderr, "[HH_afterDevSwapOut@p%d] [%.2f] I release heap slot %d\n",
	  HH_MYID, Wtime_prt(), HHL->hpid);

#if 0
  d->np_out--;
  if (d->np_out < 0) {
    fprintf(stderr, "[swapOut@p%d] np_out = %d strange\n",
	    HH_MYID, d->np_out);
  }
#endif

  HHL->dmode = HHD_ON_HOST;

  return 0;
}

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

  HH_unlockSched();

  /* D -> H */
  HHL2->devheap->swapOutD2H();
#ifdef USE_SWAPHOST
  HHL2->hostheap->swapOutD2H(); // do nothing
#endif

  HH_lockSched();
  afterSwapOutD2H();

  return 0;
}

//------------------------------- H2D
// check whether resource is available for swapIn
int HH_checkH2D()
{
  dev *d = HH_curdev();
  if (d->np_in > 0 || d->dhslot_users[HHL->hpid] >= 0) {
    return 0;
  }
  return 1;
}

// reserve resource for swapIn. called soon after HH_checkH2D
// (before scheduling lock is released)
int HH_reserveResH2D()
{
  dev *d = HH_curdev();
  d->dhslot_users[HHL->hpid] = HH_MYID;
  return 0;
}

// This function assumes sched_ml is locked
int HH_swapInH2D()
{
  assert(HHL->dmode == HHD_ON_HOST);
  HHL->dmode = HHD_SI_H2D;
  HH_unlockSched();

  /* H -> D */
  HHL2->devheap->swapInH2D();
#ifdef USE_SWAPHOST
  HHL2->hostheap->swapInH2D();
#endif

  HH_lockSched();
  HHL->dmode = HHD_ON_DEV;

  return 0;
}



//----------------------------- H2F
int HH_checkH2F()
{
  if (HHL2->conf.nlphost >= HHS->nlprocs) {
    // no need to use fileswapper
    return 0;
  }

  fsdir *fsd = HH_curfsdir();
  if (fsd->np_filein > 0 || fsd->np_fileout > 0) {
    return 0;
  }

  return 1;
}

// This function assumes sched_ml is locked
static int beforeSwapOutH2F()
{
  assert(HHL->dmode == HHD_ON_HOST);
  HHL->dmode = HHD_SO_H2F;

  fsdir *fsd = HH_curfsdir();
  fsd->np_fileout++;

  return 0;
}

static int mainSwapOutH2F()
{
#ifdef USE_SWAPHOST
  /* Host Heap */
  /* H -> F */
  HHL2->hostheap->swapOutH2F();
#endif

  HHL2->devheap->swapOutH2F();
  return 0;
}

static int afterSwapOutH2F()
{
  fsdir *fsd = HH_curfsdir();
  fsd->np_fileout--;

  HHS->nhostusers[HHL->hpid]--;
  fprintf(stderr, "[HH_swapOutH2F@p%d] [%.2f] I release host capacity\n",
	  HH_MYID, Wtime_prt());

  HHL->dmode = HHD_ON_FILE;
  return 0;
}

#ifdef USE_FILESWAP_THREAD

static void *swapOutH2Fthread(void *arg)
{
  mainSwapOutH2F();
  return NULL;
}

// This function assumes sched_ml is locked
int HH_tryfinSwapOutH2F()
{
  int rc;
  void *retval;
  assert(HHL->dmode == HHD_SO_H2F);

  rc= pthread_tryjoin_np(HHL2->fileswap_tid, &retval);
  if (rc == EBUSY) return 0;

  // thread has terminated
  afterSwapOutH2F();

  return 1;
}

int HH_startSwapOutH2F()
{
  beforeSwapOutH2F();
  pthread_create(&HHL2->fileswap_tid, NULL, swapOutH2Fthread, NULL);
  return 0;
}

// This function assumes sched_ml is locked
int HH_swapOutH2F()
{
  beforeSwapOutH2F();
  pthread_create(&HHL2->fileswap_tid, NULL, swapOutH2Fthread, NULL);

  while (1) {
    int rc = HH_tryfinSwapOutH2F();
    if (rc > 0) break;
    HH_unlockSched();
    usleep(100);
    HH_lockSched();
  }

  return 0;
}

#else // !USE_FILESWAP_THREAD

// H2F
// This function assumes sched_ml is locked
int HH_swapOutH2F()
{
  beforeSwapOutH2F();
  HH_unlockSched();
  mainSwapOutH2F();
  HH_lockSched();
  afterSwapOutH2F();
  return 0;
}

#endif // !USE_FILESWAP_THREAD

//----------------------------- F2H
int HH_checkF2H()
{

  fsdir *fsd = HH_curfsdir();
  if (fsd->np_filein > 0) {
    return 0;
  }
  assert(fsd->np_filein == 0);

  int limperslot = (HHL2->conf.nlphost+HHS->ndhslots-1)/HHS->ndhslots;
  if (HHS->nhostusers[HHL->hpid] >= limperslot) {
    return 0;
  }

  /* I can start F2H */
  return 1;
}

static int beforeSwapInF2H()
{
  assert(HHL->dmode == HHD_ON_FILE || HHL->dmode == HHD_NONE);

  fsdir *fsd = HH_curfsdir();
  fsd->np_filein++;

  HHS->nhostusers[HHL->hpid]++;
  HHL->dmode = HHD_SI_F2H;
  return 0;
}

static int mainSwapInF2H()
{
#ifdef USE_SWAPHOST
  /* Host Heap */
  HHL2->hostheap->swapInF2H();
#endif
  /* Device Heap */
  HHL2->devheap->swapInF2H();
  return 0;
}

static int afterSwapInF2H()
{
  fsdir *fsd = HH_curfsdir();
  fsd->np_filein--;
  HHL->dmode = HHD_ON_HOST;
  return 0;
}

#ifdef USE_FILESWAP_THREAD
static void *swapInF2Hthread(void *arg)
{
  mainSwapInF2H();
  return NULL;
}

// This function assumes sched_ml is locked
int HH_startSwapInF2H()
{
  beforeSwapInF2H();
  pthread_create(&HHL2->fileswap_tid, NULL, swapInF2Hthread, NULL);
  return 0;
}

// This function assumes sched_ml is locked
int HH_tryfinSwapInF2H()
{
  int rc;
  void *retval;
  assert(HHL->dmode == HHD_SI_F2H);

  rc= pthread_tryjoin_np(HHL2->fileswap_tid, &retval);
  if (rc == EBUSY) return 0;

  // thread has terminated
  afterSwapInF2H();
  return 1;
}

#else // !USE_FILESWAP_THREAD
// F2H
// This function assumes sched_ml is locked
int HH_swapInF2H()
{
  beforeSwapInF2H();
  HH_unlockSched();
  mainSwapInF2H();
  HH_lockSched();

  fsdir *fsd = HH_curfsdir();
  fsd->np_filein--;

  HHL->dmode = HHD_ON_HOST;

  return 0;
}

#endif // !USE_FILESWAP_THREAD

   
/************************************************************/
int HH_swapInIfOk()
{
  if (HHL->dmode == HHD_ON_DEV) {
    // do nothing 
    return 0;
  }
  else if (HHL->dmode == HHD_ON_HOST) {
    if (!HH_checkH2D()) {
      return 0;
    }

    HH_lockSched();
    if (!HH_checkH2D()) {
      HH_unlockSched();
      return 0;
    }

#if 0
    fprintf(stderr, "[HH_swapInIfOk@p%d] [%.2lf] Now I start H2D (heap slot %d)\n",
	    HH_MYID, Wtime_prt(), HHL->hpid);
#endif

    // now I can proceed!
    HH_reserveResH2D();

    HH_swapInH2D();

    assert(HHL->dmode == HHD_ON_DEV);
    HH_unlockSched();
    return 1;
  }
  else if (HHL->dmode == HHD_ON_FILE || HHL->dmode == HHD_NONE) {
    if (!HH_checkF2H()) {
      return 0;
    }

    HH_lockSched();
    if (!HH_checkF2H()) {
      HH_unlockSched();
      return 0;
    }

#if 0
    fprintf(stderr, "[HH_swapInIfOk@p%d] [%.2lf] Now I start F2H\n",
	    HH_MYID, Wtime_prt());
#endif

#ifdef USE_FILESWAP_THREAD
    HH_startSwapInF2H();
    assert(HHL->dmode == HHD_SI_F2H);
#else
    HH_swapInF2H();
    assert(HHL->dmode == HHD_ON_HOST);
#endif

    HH_unlockSched();
    return 1;
  }
  else {
    fprintf(stderr, "[HH_swapInIfOk@p%d] [%.2lf] dmode %s strange\n",
	    HH_MYID, Wtime_prt(), hhd_names[HHL->dmode]);
    assert(0);
  }
  
  return 0;
}

/* swapOut may be called in this function */
int HH_swapOutIfBetter()
{
  if (HHL->dmode == HHD_ON_FILE || HHL->dmode == HHD_NONE) {
    // do nothing
    return 0;
  }
  else if (HHL->dmode == HHD_ON_HOST) {
#ifdef USE_MMAPSWAP
    return 0; // do nothing
#else // !USE_MMAPSWAP
    if (!HH_checkH2F()) {
      return 0;
    }

    HH_lockSched();
    if (!HH_checkH2F()) {
      HH_unlockSched();
      return 0;
    }

#if 0
    fprintf(stderr, "[HH_swapOutIfBetter@p%d] [%.2lf] Now H2F starts\n",
	    HH_MYID, Wtime_prt());
#endif

#ifdef USE_FILESWAP_THREAD
    HH_startSwapOutH2F();
    assert(HHL->dmode == HHD_SO_H2F);
#else
    HH_swapOutH2F();
    assert(HHL->dmode == HHD_ON_FILE);
#endif
    HH_unlockSched();
    return 1;
#endif // !USE_MMAPSWAP
  }
  else if (HHL->dmode == HHD_ON_DEV) {
    if (!HH_checkD2H()) {
      return 0;
    }

    HH_lockSched();
    if (!HH_checkD2H()) {
      HH_unlockSched();
      return 0;
    }

    HH_swapOutD2H();
    assert(HHL->dmode == HHD_ON_HOST);
    HH_unlockSched();
    return 1;
  }
  else {
    assert(0);
  }
  return 0;
}

int HH_swapOutIfOver()
{
  HH_lockSched();
  fprintf(stderr, "[HH_swapOutIfOver@p%d] %s before SwapOut\n",
	  HH_MYID, hhd_names[HHL->dmode]);
  if (HHL->dmode == HHD_ON_DEV &&
      HHS->nlprocs > HHS->ndhslots) {
    HH_swapOutD2H();
  }

  if (HHL->dmode == HHD_ON_HOST &&
      HHS->nlprocs > HHL2->conf.nlphost) {
    HH_swapOutH2F();
  }

  fprintf(stderr, "[HH_swapOutIfOver@p%d] %s after SwapOut\n",
	  HH_MYID, hhd_names[HHL->dmode]);

  HH_unlockSched();

  return 0;
}

/* This function must be called by blocking functions periodically. */
int HH_progressSched()
{
  int rc;
#ifdef USE_FILESWAP_THREAD
  if (HHL->dmode == HHD_SO_H2F) {
    HH_lockSched();
    rc = HH_tryfinSwapOutH2F();
    HH_unlockSched();
    return rc;
  }
  else if (HHL->dmode == HHD_SI_F2H) {
    HH_lockSched();
    rc = HH_tryfinSwapInF2H();
    HH_unlockSched();
    return rc;
  }
#endif

  if (HHL->pmode == HHP_BLOCKED) {
    rc = HH_swapOutIfBetter();
  }
  else if (HHL->pmode == HHP_RUNNABLE) {
    rc = HH_swapInIfOk();
  }

  return rc; /* progress */
}

/* swapIn may be called in this function */
/* This function may be blocked */
int HH_sleepForMemory()
{
  if (HHL->dmode == HHD_ON_DEV ||
      HHL->devmode == HHDEV_NOTUSED) {
    /* we do nothing, immediate return */
#if 1 && defined HHLOG_SCHED
    fprintf(stderr, "[sleepForMemory@p%d] wake up, but data are in SWAPPED_OUT mode!!!\n",
	    HH_MYID);
#endif
    return 0;
  }

  HHL->pmode = HHP_RUNNABLE;

#ifdef HHLOG_SCHED
  fprintf(stderr, "[sleepForMemory@p%d] sleep for heap capacity\n",
	  HH_MYID);
#endif

  do {
    HH_progressSched();
    if (HHL->dmode == HHD_ON_DEV) {
      break;
    }

    usleep(1000);
  } while (1);

#if 1 && defined HHLOG_SCHED
  fprintf(stderr, "[sleepForMemory@p%d] wake up!\n",
	  HH_MYID);
#endif
  return 0;
}



/***/
int HH_enterAPI(const char *str)
{
  if (HHL->in_api == 0) {
#ifdef HHLOG_API
    strcpy(HHL2->api_str, str);
    fprintf(stderr, "[HH_enterAPI@p%d] API [%s] start\n",
	    HH_MYID, HHL2->api_str);
#endif
    assert(HHL->pmode == HHP_RUNNING);
    HHL->pmode = HHP_BLOCKED;
#ifdef HHLOG_API
    fprintf(stderr, "[HH_enterAPI@p%d] API [%s] end\n",
	    HH_MYID, HHL2->api_str);
#endif
  }
  HHL->in_api++;
  return 0;
}

int HH_exitAPI()
{
  assert(HHL->in_api >= 0);
  HHL->in_api--;
  if (HHL->in_api == 0) {
    assert(HHL->pmode == HHP_BLOCKED);
    HH_sleepForMemory();
    /* now I'm awake */
    HHL->pmode = HHP_RUNNING;
#ifdef HHLOG_API
    fprintf(stderr, "[HH_exitAPI@p%d] API [%s] end\n",
	    HH_MYID, HHL2->api_str);
#endif
  }
  return 0;
}

int HH_enterGComm(const char *str)
{

  if (HHL->in_api == 0) {
#ifdef HHLOG_API
    strcpy(HHL2->api_str, str);
    fprintf(stderr, "[HH_enterGComm@p%d] GComm [%s] start\n",
	    HH_MYID, HHL2->api_str);
#endif
    assert(HHL->pmode == HHP_RUNNING);

    /* When device is oversubscribed, I sleep eagerly */
    HH_swapOutIfOver();

    HHL->pmode = HHP_BLOCKED;
  }
  HHL->in_api++;


  return 0;
}

int HH_exitGComm()
{
  assert(HHL->in_api >= 0);
  HHL->in_api--;
  if (HHL->in_api == 0) {
    assert(HHL->pmode == HHP_BLOCKED);
    HH_sleepForMemory();
    /* now I'm awake */
    HHL->pmode = HHP_RUNNING;
#ifdef HHLOG_API
    fprintf(stderr, "[HH_exitGComm@p%d] API [%s] end\n",
	    HH_MYID, HHL2->api_str);
#endif
  }
  return 0;
}


/******************************/

int HH_yield()
{
  assert(HHL->pmode == HHP_RUNNING);
#if 1 || defined HHLOG_API
  fprintf(stderr, "[HH_yield@p%d] start\n", HH_MYID);
#endif

  /* I may be swapped out if appropriate */
  HH_swapOutIfBetter();

  HHL->pmode = HHP_RUNNABLE;
  HH_sleepForMemory();

  HHL->pmode = HHP_RUNNING;

  return 0;
}
