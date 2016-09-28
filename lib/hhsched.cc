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

// sched_ml should be locked in caller
int HH_countProcsInMode(int mode)
{
  int n = 0;
  for (int ip = 0; ip < HHS->nlprocs; ip++) {
    if (HHS->lprocs[ip].pmode == mode) {
      n++;
    }
  }
  return n;
}

// check resouce availability before swapping
int HH_checkRes(int kind)
{
  for (int ih = 0; ih < HHL2->nheaps; ih++) {
    if (!HHL2->heaps[ih]->checkRes(kind)) return 0;
  }
  return 1;
}

// This function assumes sched_ml is locked
// reserve resource for swapIn. called soon after HH_checkRes
static int beforeSwap(int kind)
{
  for (int ih = 0; ih < HHL2->nheaps; ih++) {
    HHL2->heaps[ih]->reserveRes();
  }

  HHL->dmode = kind;
  return 0;
}

static int mainSwap(int kind)
{
  HH_profBeginAction(hhd_snames[kind]);
  for (int ih = 0; ih < HHL2->nheaps; ih++) {
    HHL2->heaps[ih]->swap();
  }
  HH_profEndAction(hhd_snames[kind]);
  return 0;
}

// This function assumes sched_ml is locked
static int afterSwap(int kind)
{
  for (int ih = 0; ih < HHL2->nheaps; ih++) {
    HHL2->heaps[ih]->releaseRes();
  }

  // state transition
  if (kind == HHD_SO_D2H) {
    HHL->dmode = HHD_ON_HOST;
  }
  else if (kind == HHD_SI_H2D) {
    HHL->dmode = HHD_ON_DEV;
  }
  else if (kind == HHD_SO_H2F) {
    HHL->dmode = HHD_ON_FILE;
  }
  else if (kind == HHD_SI_F2H) {
    HHL->dmode = HHD_ON_HOST;
  }
  else {
    fprintf(stderr, "[HH:afterSwap@p%d] ERROR: kind %d unknown\n",
	    HH_MYID, kind);
    exit(1);
  }
  return 0;
}

static void *swap_thread_func(void *arg)
{
  int kind = (int)(long)arg;
  mainSwap(kind);
  return NULL;
}

// This function assumes sched_ml is locked
int HH_startSwap(int kind)
{
  beforeSwap(kind);
  HHL2->swap_kind = kind;
  pthread_create(&HHL2->swap_tid, NULL, swap_thread_func, (void*)(long)kind);
  return 0;
}

// This function assumes sched_ml is locked
int HH_tryfinSwap()
{
  int kind = HHL2->swap_kind;
  int rc;
  void *retval;

  rc= pthread_tryjoin_np(HHL2->swap_tid, &retval);
  if (rc == EBUSY) return 0;

  // thread has terminated
  afterSwap(kind);
  HHL2->swap_kind = -1;
  return 1;
}

// This function assumes sched_ml is locked
int HH_swap(int kind)
{
  beforeSwap(kind);
  HH_unlockSched();

  mainSwap(kind);

  HH_lockSched();
  afterSwap(kind);
  return 0;
}
   
/************************************************************/
int HH_swapInIfOk()
{
  int kind = -1;

  if (HHL->dmode == HHD_ON_HOST) {
    kind = HHD_SI_H2D;
  }
  else if (HHL->dmode == HHD_ON_FILE || HHL->dmode == HHD_NONE) {
    kind = HHD_SI_F2H;
  }
  else {
    // do nothing 
    return 0;
  }

  if (!HH_checkRes(kind)) return 0;

  HH_lockSched();
  if (!HH_checkRes(kind)) {
    HH_unlockSched();
    return 0;
  }

#ifdef USE_SWAP_THREAD
  HH_startSwap(kind);
  assert(HHL->dmode == kind);
#else
  HH_swap(kind);
#endif

  HH_unlockSched();
  return 1;
}

/* swapOut may be called in this function */
int HH_swapOutIfBetter()
{
  int kind = -1;
  if (HHL->dmode == HHD_ON_HOST) {
    kind = HHD_SO_H2F;
  }
  else if (HHL->dmode == HHD_ON_DEV) {
    kind = HHD_SO_D2H;
  }
  else {
    // do nothing
    return 0;
  }
  
  if (!HH_checkRes(kind)) return 0;

  HH_lockSched();
  if (!HH_checkRes(kind)) {
    HH_unlockSched();
    return 0;
  }

#if 1
  HH_startSwap(kind);
  assert(HHL->dmode == kind);
#else
  HH_swap(kind);
#endif
  HH_unlockSched();

  return 1;
}

// This function assumes sched_ml is locked
int HH_swapWithCheck(int kind)
{
  double st = Wtime(), et;
  while (1) {
    if (HH_checkRes(kind)) {
      // can proceed
      break;
    }
    HH_unlockSched();
    et = Wtime();
#if 0
    if (et-st > 1.0) {
      fprintf(stderr, "[HH_swapWithCheck@p%d] [%.2lf-%.2lf] Waiting long for %s...\n",
	      HH_MYID, Wtime_conv_prt(st), Wtime_conv_prt(et), hhd_names[kind]);
    }
#endif
    usleep(10*1000);
    HH_lockSched();
  }

  et = Wtime();
  if (et-st > 1.0) {
    fprintf(stderr, "[HH_swapWithCheck@p%d] [%.2lf-%.2lf] Waited long for %s\n",
	    HH_MYID, Wtime_conv_prt(st), Wtime_conv_prt(et), hhd_names[kind]);
  }
  
  HH_swap(kind);

  return 0;
}

int HH_swapOutIfOver()
{
  HH_lockSched();
  fprintf(stderr, "[HH_swapOutIfOver@p%d] %s before SwapOut\n",
	  HH_MYID, hhd_names[HHL->dmode]);

  if (HHL->dmode == HHD_ON_DEV &&
      HHS->nlprocs > HHS->ndh_slots) {
    HH_swapWithCheck(HHD_SO_D2H);
  }
  else {
    fprintf(stderr, "[HH_swapOutIfOver@p%d] D2H skipped (OK?)\n",
	    HH_MYID);
  }

  if (HHL->dmode == HHD_ON_HOST &&
      HHS->nlprocs > HHL2->conf.nlphost) {
    HH_swapWithCheck(HHD_SO_H2F);
  }
  else {
    fprintf(stderr, "[HH_swapOutIfOver@p%d] H2F skipped (OK?)\n",
	    HH_MYID);
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
#ifdef USE_SWAP_THREAD
  if (HHL->dmode == HHD_SO_H2F || HHL->dmode == HHD_SI_F2H ||
      HHL->dmode == HHD_SO_D2H || HHL->dmode == HHD_SI_H2D) {
    HH_lockSched();
    rc = HH_tryfinSwap();
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
  HHL->pmode = HHP_RUNNABLE;
  HH_profSetMode("RUNNABLE");
  
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
  
  // check #running procs
  do {
    HH_lockSched();
    int nrp = HH_countProcsInMode(HHP_RUNNING);
    if (nrp+1 <= HHL2->conf.maxrp) { // ok
      HHL->pmode = HHP_RUNNING;
      HH_unlockSched();
      HH_profSetMode("RUNNING");
      break;
    }
    HH_unlockSched();
    usleep(10*1000);
  } while (1);


#if 1 && defined HHLOG_SCHED
  fprintf(stderr, "[HH_sleepForMemory@p%d] wake up!\n",
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
    HH_profSetMode("BLOCKED");
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
    assert(HHL->pmode == HHP_RUNNING);
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
    fprintf(stderr, "[HH_enterGComm@p%d] [%.2lf] GComm [%s] start\n",
	    HH_MYID, Wtime_prt(), HHL2->api_str);
#endif
    assert(HHL->pmode == HHP_RUNNING);

    HHL->pmode = HHP_BLOCKED;
    HH_profSetMode("BLOCKED");

    /* When device is oversubscribed, I sleep eagerly */
    HH_swapOutIfOver();

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
    assert(HHL->pmode == HHP_RUNNING);
#ifdef HHLOG_API
    fprintf(stderr, "[HH_exitGComm@p%d] [%.2lf] API [%s] end\n",
	    HH_MYID, Wtime_prt(), HHL2->api_str);
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

  HH_sleepForMemory();

  assert(HHL->pmode == HHP_RUNNING);
  return 0;
}
