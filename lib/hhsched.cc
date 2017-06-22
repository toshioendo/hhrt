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

int HH_prioInc(int rank, int inc)
{
  /* is rank on the same node? */
  for (int ip = 0; ip < HHS->nlprocs; ip++) {
    proc *pp = &HHS->lprocs[ip];
    if (pp->rank == rank) {
      pp->prio_score += inc;
      return 0;
    }
  }
  return 0;
}

int HH_prioGetMax(int *argmaxp)
{
  /* get the highest prio_score of local&runnable processes */
  int ans = -99999;
  int argmax = -1;
  for (int ip = 0; ip < HHS->nlprocs; ip++) {
    proc *pp = &HHS->lprocs[ip];
    if (pp->pmode == HHP_RUNNABLE && pp->prio_score > ans) {
      ans = pp->prio_score;
      argmax = pp->rank;
    }
  }

  if (argmaxp != NULL) *argmaxp = argmax;
  return ans;
}

// This function assumes sched_ml is locked
// This must be called after h->checkSwapRes() returns OK
int HH_swapHeap(heap *h, int kind)
{
  h->reserveSwapRes(kind);
  HH_unlockSched();

  h->doSwap();

  HH_lockSched();
  h->releaseSwapRes();
  return 0;
}

#ifdef USE_SWAP_THREAD
static void *swapheap_thread_func(void *arg)
{
  heap *h = (heap *)arg;
  h->doSwap();
  return NULL;
}

// Thread version of HH_swap()
// This function assumes sched_ml is locked
// This must be called after h->checkRes() returns OK
int HH_startSwapHeap(heap *h, int kind)
{
  assert(HHL2->swapping_heap == NULL);
  h->reserveSwapRes(kind);
  pthread_create(&HHL2->swap_tid, NULL, swapheap_thread_func, (void*)h);
  HHL2->swapping_heap = h;
  return 0;
}

// This function assumes sched_ml is locked
int HH_tryfinSwap()
{
  int rc;
  void *retval;

  if (HHL2->swapping_heap == NULL) return 0;

  rc= pthread_tryjoin_np(HHL2->swap_tid, &retval);
  if (rc == EBUSY) return 0;

  // thread has terminated
  heap *h = (heap *)HHL2->swapping_heap;
  assert(h != NULL);
  h->releaseSwapRes();
  HHL2->swapping_heap = NULL;
  return 1;
}
#endif

/************************************************************/
int HH_swapIfOk(int kind)
{
  for (int ih = 0; ih < HHL2->nheaps; ih++) {
    heap *h = HHL2->heaps[ih];
    if (h->checkSwapRes(kind) == HHSS_OK) {
      HH_lockSched();
      if (h->checkSwapRes(kind) == HHSS_OK) {
	// can proceed swap
#ifdef USE_SWAP_THREAD
	HH_startSwapHeap(h, kind);
#else
	HH_swapHeap(h, kind);
#endif
	HH_unlockSched(); // debug 16/12/07
	return 1;
      }

      HH_unlockSched();
    }
  }

  return 0; // nothing done
}

// This function assumes sched_ml is locked
int HH_isSwapCompleted(int kind) // kind should be HHSW_IN or HHSW_OUT
{
#ifdef USE_SWAP_THREAD
  if (HHL2->swapping_heap != NULL) {
    return 0;
  }
#endif

  for (int ih = 0; ih < HHL2->nheaps; ih++) {
    heap *h = HHL2->heaps[ih];
    if (h->checkSwapRes(kind) != HHSS_NONEED) {
      return 0;
    }
  }
  return 1;
}


int HH_swapOutIfOver()
{
#ifdef HHLOG_SCHED
  fprintf(stderr, "[HH_swapOutIfOver@p%d] started\n",
	  HH_MYID);
#endif

  while (1) {
    HH_lockSched();
    if (HH_isSwapCompleted(HHSW_OUT)) {
      HH_unlockSched();
      break;
    }
    HH_unlockSched();

    HH_swapIfOk(HHSW_OUT);

#ifdef USE_SWAP_THREAD
    HH_lockSched();
    if (HHL2->swapping_heap != NULL) {
      while (1) {
	if (HH_tryfinSwap()) break;

	HH_unlockSched();
	usleep(1000);
	HH_lockSched();
      }
    }
    HH_unlockSched();
#endif

    usleep(1000);
  }

#ifdef HHLOG_SCHED
  fprintf(stderr, "[HH_swapOutIfOver@p%d] finished\n",
	  HH_MYID);
#endif

  return 0;
}

/* This function must be called by blocking functions periodically. */
int HH_progressSched()
{
  int rc;

#ifdef USE_SWAP_THREAD
  HH_lockSched();
  if (HHL2->swapping_heap != NULL) {
    rc = HH_tryfinSwap();
    HH_unlockSched();
    return rc;
  }
  HH_unlockSched();
#endif


  if (HHL->pmode == HHP_BLOCKED) {
    rc = HH_swapIfOk(HHSW_OUT);
  }
  else if (HHL->pmode == HHP_RUNNABLE) {
    rc = HH_swapIfOk(HHSW_IN);
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
  fprintf(stderr, "[HH_sleepForMemory@p%d] start sleeping (pid=%d)\n",
	  HH_MYID, getpid());
#endif

#if 0
  // check priority
  // BUG: This still causes  deadlock
  do {
    HH_lockSched();
    int hrank;
    int hp = HH_prioGetMax(&hrank);
    if (HHL->prio_score >= hp) {
      HH_unlockSched();
      fprintf(stderr, "[HH_sleepForMemory@p%d] [%.2lf] My prio_score %d, proceed\n",
	      HH_MYID, Wtime_prt(), HHL->prio_score);
      break;
    }
    fprintf(stderr, "[HH_sleepForMemory@p%d] [%.2lf] My prio_score %d < %d (p%d), thus I backoff now\n",
	    HH_MYID, Wtime_prt(), HHL->prio_score, hp, hrank);
    HH_unlockSched();
    usleep(50*1000);
  } while (1);
#endif
  
  do {
    HH_progressSched();

    HH_lockSched();
    if (HH_isSwapCompleted(HHSW_IN)) {
      HH_unlockSched();
      break;
    }
    HH_unlockSched();
    
    usleep(10*1000);
  } while (1);
  
  // check #running procs
  do {
    HH_lockSched();
    int nrp = HH_countProcsInMode(HHP_RUNNING);
    if (nrp+1 <= HHL2->conf.maxrp) { // ok
      HHL->pmode = HHP_RUNNING;

      fprintf(stderr, "[HH_sleepForMemory@p%d] I'm going to wake up; prio_score=%d\n",
	      HH_MYID, HHL->prio_score);
      HHL->prio_score = 0;

      HH_unlockSched();
      HH_profSetMode("RUNNING");
      break;
    }
    HH_unlockSched();
    usleep(10*1000);
  } while (1);


#if 1 && defined HHLOG_SCHED
  fprintf(stderr, "[HH_sleepForMemory@p%d] wake up! (pid=%d)\n",
	  HH_MYID, getpid());
#endif
  
  return 0;
}



int HH_enterBlocking()
{
#if 0
  assert(HHL->in_api == 0); ///
  if (HHL->in_api == 0) {
#endif
#ifdef HHLOG_API
    fprintf(stderr, "[HH_enterBlocking@p%d] [%.2lf] start\n",
	    HH_MYID, Wtime_prt());
#endif
    assert(HHL->pmode == HHP_RUNNING);
    HHL->pmode = HHP_BLOCKED;
    HH_profSetMode("BLOCKED");
#if 0
  }
  HHL->in_api++;
#endif
  return 0;
}

int HH_exitBlocking()
{
#if 0
  assert(HHL->in_api >= 0);
  HHL->in_api--;
  if (HHL->in_api == 0) {
#endif
    assert(HHL->pmode == HHP_BLOCKED);
    HH_sleepForMemory();
    /* now I'm awake */
    assert(HHL->pmode == HHP_RUNNING);
#ifdef HHLOG_API
    fprintf(stderr, "[HH_exitBlocking@p%d] [%.2lf] end\n",
	    HH_MYID, Wtime_prt());
#endif
#if 0
  }
  assert(HHL->in_api == 0); ///
#endif
  return 0;
}

int HH_enterGComm(const char *str)
{
#if 0
  assert(HHL->in_api == 0); ///

  if (HHL->in_api == 0) {
#endif
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

#if 0
  }
  HHL->in_api++;
#endif

  return 0;
}

int HH_exitGComm()
{
#if 0
  assert(HHL->in_api >= 0);
  HHL->in_api--;
  if (HHL->in_api == 0) {
#endif
    assert(HHL->pmode == HHP_BLOCKED);
    HH_sleepForMemory();
    /* now I'm awake */
    assert(HHL->pmode == HHP_RUNNING);
#ifdef HHLOG_API
    fprintf(stderr, "[HH_exitGComm@p%d] [%.2lf] API [%s] end\n",
	    HH_MYID, Wtime_prt(), HHL2->api_str);
#endif
#if 0
  }

  assert(HHL->in_api == 0); ///
#endif

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
  HH_swapIfOk(HHSW_OUT);

  HH_sleepForMemory();

  assert(HHL->pmode == HHP_RUNNING);
  return 0;
}
