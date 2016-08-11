#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>

#include <assert.h>
#include "hhrt_impl.h"

// shared host memory buffers
// currently, used for hostswapper chunks
#ifdef USE_SHARED_HSC

//#define USE_PIN_HSC

int HH_hsc_init_node()
{
  HHS->nshsc = 0;
  HH_mutex_init(&HHS->shsc_ml);
  return 0;
}

int HH_hsc_init_proc()
{
  int i;
  for (i = 0; i < MAX_SHSC; i++) {
    HHL2->shsc_ptrs[i] = NULL;
  }
  return 0;
}

int HH_hsc_fin_node()
{
  int i;
  for (i = 0; i < HHS->nshsc; i++) {
    int shmid;
    shmid = shmget(SHSC_KEY(i), HSC_SIZE, S_IRWXU);
    shmctl(shmid, IPC_RMID, NULL);
  }
  return 0;
}

static void *hsc_get_ptr(int i, int isnew)
{
  assert(HHL2->shsc_ptrs[i] == NULL);

  void *p;
  int flag = (isnew)? IPC_CREAT|IPC_EXCL|S_IRWXU: S_IRWXU;

  int shmid;
  // this is the first slot for me
  shmid = shmget(SHSC_KEY(i), HSC_SIZE, flag);
  if (shmid < 0) {
    fprintf(stderr, "[HH:hsc_get_ptr@p%d] shmget for SHSC hsc-slot %d (isnew=%d) failed!\n",
	    HH_MYID, i);
    exit(1);
  }

#if 0
  void *objp = piadd(SHSC_PTR0, HSC_SIZE*i);
  p = shmat(shmid, objp, 0);
  if (p != objp) {
    fprintf(stderr, "[HH:hsc_get_ptr@p%d] shmat for hsc-slot %d -> rc=%p != %p\n",
	    HH_MYID, i, p, objp);
  }
  else {
    fprintf(stderr, "[HH:hsc_get_ptr@p%d] shmat for hsc-slot %d -> rc=%p == %p\n",
	    HH_MYID, i, p, objp);
  }
#else  
  p = shmat(shmid, NULL, 0);
#endif
  if (p == (void*)-1) {
    fprintf(stderr, "[HH:hsc_get_ptr@p%d] shmat for SHSC hsc-slot %d failed!\n",
	    HH_MYID, i);
    exit(1);
  }

  bzero(p, HSC_SIZE);

#ifdef USE_PIN_HSC
  cudaError_t crc;
  crc = cudaHostRegister(p, HSC_SIZE, cudaHostRegisterPortable);
  if (crc == cudaErrorHostMemoryAlreadyRegistered) {
    fprintf(stderr, "[HH:hsc_get_ptr@p%d] cudaHostRegister ALREADY hsc-slot %d (isnew=%d)\n",
	    HH_MYID, i, isnew);
  }
  else if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:hsc_get_ptr@p%d] cudaHostRegister FAILED for hsc-slot %d (isnew=%d) failed! crc=%d\n",
	    HH_MYID, i, isnew, crc);
    exit(1);
  }
  else {
    fprintf(stderr, "[HH:hsc_get_ptr@p%d] cudaHostRegister OK for hsc-slot %d (isnew=%d)\n",
	    HH_MYID, i, isnew);
  }
#endif  

  return p;
}

// HH_sched_ml should be locked by caller
void *HH_hsc_alloc(int id /* not used */)
{
  int i;
  int me = HH_MYID;
  double st, et;

  lock_log(&HHS->shsc_ml);
  st = Wtime();
 retry:  
  for (i = 0; i < HHS->nshsc; i++) {
    if (HHS->shsc_users[i] < 0) {
      // empty slot found
      void *p;
      HHS->shsc_users[i] = me;

      if (HHL2->shsc_ptrs[i] == NULL) {
	p = hsc_get_ptr(i, 0);
	HHL2->shsc_ptrs[i] = p;
      }
      else {
	p = HHL2->shsc_ptrs[i];
      }

#if 1
      et = Wtime();
      fprintf(stderr, "[HH_hsc_alloc@p%d] ok (%.1lfms) -> hsc-slot %d, p=%p\n",
	      HH_MYID, (et-st)*1000.0, i, p);
#endif
      pthread_mutex_unlock(&HHS->shsc_ml);

      return p;
    }
  }

  if (HHS->nshsc >= MAX_SHSC) {
    fprintf(stderr, "[HH_hsc_alloc@p%d] ERROR: Too many shscs.\n",
	    HH_MYID);
    exit(1);
  }

  // not found, so I create a new region
  {
    i = HHS->nshsc;
    HHL2->shsc_ptrs[i] = hsc_get_ptr(i, 1);

#if 1
    fprintf(stderr, "[HH_hsc_alloc@p%d] hsc array expanded. now %d chunks\n",
	    HH_MYID, i);
#endif

    HHS->shsc_users[i] = -1;
    HHS->nshsc++;

  }
  goto retry;

  return 0;
}

// HH_sched_ml should be locked by caller
int HH_hsc_free(void *p)
{
  int rc;
  int i;

  lock_log(&HHS->shsc_ml);

  rc = shmdt(p);
  if (rc != 0) {
    fprintf(stderr, "[HH_hsc_free@p%d] shmdt for SHSC (p=%p) failed!\n",
	    HH_MYID, p);
    exit(1);
  }

  for (i = 0; i < HHS->nshsc; i++) {
    if (HHL2->shsc_ptrs[i] == p) {
      // found;
      break;
    }
  }
  if (i >= HHS->nshsc) {
    fprintf(stderr, "[HH_hsc_free@p%d] ERROR: p=%p not found?\n",
	    HH_MYID, p);
    exit(1);
  }

#if 1
  fprintf(stderr, "[HH_hsc_free@p%d] ok p=%p -> hsc-slot %d\n",
	  HH_MYID, p, i);
#endif

  HHS->shsc_users[i] = -1;

  pthread_mutex_unlock(&HHS->shsc_ml);
  return 0;
}


#endif // USE_SHARED_HSC

