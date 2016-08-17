#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "hhrt_impl.h"


/***/

int HH_checkF2H()
{

  fsdir *fsd = HH_curfsdir();
  if (fsd->np_filein > 0) {
    return 0;
  }
  assert(fsd->np_filein == 0);

  int limperslot = (HHL2->conf.nlphost+HHS->nheaps-1)/HHS->nheaps;
  if (HHS->nhostusers[HHL->hpid] >= limperslot) {
    return 0;
  }

  /* I can start F2H */
  return 1;
}
    
int HH_swapInIfOk()
{
  dev *d = HH_curdev();

  if (HHL->dmode == HHD_ON_DEV) {
    // do nothing 
    return 0;
  }
  else if (HHL->dmode == HHD_ON_HOST) {
    if (d->np_in > 0 || d->hp_user[HHL->hpid] >= 0) {
      return 0;
    }

    lock_log(&HHS->sched_ml);
    if (d->np_in > 0 || d->hp_user[HHL->hpid] >= 0) {
      pthread_mutex_unlock(&HHS->sched_ml);
      return 0;
    }

#if 0
    fprintf(stderr, "[HH_swapInIfOk@p%d] [%.2lf] Now I start H2D (heap slot %d)\n",
	    HH_MYID, Wtime_prt(), HHL->hpid);
#endif

    // now I can proceed!
    d->hp_user[HHL->hpid] = HH_MYID;
    d->hp_nextuser[HHL->hpid] = -1;

    HH_swapInH2D();

    assert(HHL->dmode == HHD_ON_DEV);
    pthread_mutex_unlock(&HHS->sched_ml);
    return 1;
  }
  else if (HHL->dmode == HHD_ON_FILE || HHL->dmode == HHD_NONE) {
    if (!HH_checkF2H()) {
      return 0;
    }

    lock_log(&HHS->sched_ml);
    if (!HH_checkF2H()) {
      pthread_mutex_unlock(&HHS->sched_ml);
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

    pthread_mutex_unlock(&HHS->sched_ml);
    return 1;
  }
  else {
    fprintf(stderr, "[HH_swapInIfOk@p%d] [%.2lf] dmode %s strange\n",
	    HH_MYID, Wtime_prt(), hhd_names[HHL->dmode]);
    assert(0);
  }
  
  return 0;
}

/* */

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

/* swapOut may be called in this function */
int HH_swapOutIfBetter()
{
  dev *d = HH_curdev();
  if (HHL->dmode == HHD_ON_FILE || HHL->dmode == HHD_NONE) {
    // do nothing
    return 0;
  }
  else if (HHL->dmode == HHD_ON_HOST) {
#ifdef USE_MMAPSWAP
    return 0; // do nothing
#else
    if (!HH_checkH2F()) {
      return 0;
    }

    lock_log(&HHS->sched_ml);
    if (!HH_checkH2F()) {
      pthread_mutex_unlock(&HHS->sched_ml);
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
    pthread_mutex_unlock(&HHS->sched_ml);
    return 1;
#endif // !USE_MMAPSWAP
  }
  else if (HHL->dmode == HHD_ON_DEV) {
    if (d->np_out > 0) {
      return 0;
    }

    lock_log(&HHS->sched_ml);
    if (d->np_out > 0) {
      pthread_mutex_unlock(&HHS->sched_ml);
      return 0;
    }

    HH_swapOutD2H();
    assert(HHL->dmode == HHD_ON_HOST);
    pthread_mutex_unlock(&HHS->sched_ml);
    return 1;
  }
  else {
    assert(0);
  }
  return 0;
}

int HH_swapOutIfOver()
{
  lock_log(&HHS->sched_ml);
  fprintf(stderr, "[HH_swapOutIfOver@p%d] %s before SwapOut\n",
	  HH_MYID, hhd_names[HHL->dmode]);
  if (HHL->dmode == HHD_ON_DEV &&
      HHS->nlprocs > HHS->nheaps) {
    HH_swapOutD2H();
  }

  if (HHL->dmode == HHD_ON_HOST &&
      HHS->nlprocs > HHL2->conf.nlphost) {
    HH_swapOutH2F();
  }

  fprintf(stderr, "[HH_swapOutIfOver@p%d] %s after SwapOut\n",
	  HH_MYID, hhd_names[HHL->dmode]);

  pthread_mutex_unlock(&HHS->sched_ml);

  return 0;
}

/* called by blocking functions periodically. */
int HH_progressSched()
{
  int rc;
#ifdef USE_FILESWAP_THREAD
  if (HHL->dmode == HHD_SO_H2F) {
    lock_log(&HHS->sched_ml);
    rc = HH_tryfinSwapOutH2F();
    pthread_mutex_unlock(&HHS->sched_ml);
    return rc;
  }
  else if (HHL->dmode == HHD_SI_F2H) {
    lock_log(&HHS->sched_ml);
    rc = HH_tryfinSwapInF2H();
    pthread_mutex_unlock(&HHS->sched_ml);
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
    dev *d = HH_curdev();

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
