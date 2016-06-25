// HHRT status tool for debug

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <malloc.h>
#include "hhrt_impl.h"

shdata *HHS;

int main(int argc, char *argv[])
{
  FILE *out = stdout;
  /* for debug print */
  char hostname[HOSTNAMELEN];
  memset(hostname, 0, HOSTNAMELEN);
  gethostname(hostname, HOSTNAMELEN-1);

  ipsm_setloglevel(IPSM_LOG_WARN);
  //ipsm_setloglevel(IPSM_LOG_INFO);
  putenv((char*)"IPSM_SHMSIZE=100000"); /* in hex */
  
  //HHS = (struct shdata*)ipsm_join(HH_IPSM_KEY);
  HHS = (struct shdata*)ipsm_tryjoin(HH_IPSM_KEY, 100*1000, 5);
  if (HHS == NULL) {
    int rc;
    rc = ipsm_getlasterror();
    if (rc == IPSM_ENOTREADY) {
      fprintf(stderr, "HHVIEW@%s ipsm_join failed. no HHRT app running?\n", 
	      hostname);
      exit(1);
    }
    fprintf(stderr, "HHVIEW@%s ipsm_join failed!! lasterror=0x%x. abort..\n", 
	    hostname, rc);
    exit(1);
  }

  /****************/
  int i, j;

  fprintf(out, "HHVIEW@%s [%.2lf]\n", hostname, Wtime_prt());
  fprintf(out, "nps=%d, nlps=%d, ndevs=%d, nheaps=%d\n",
	  HHS->nprocs, HHS->nlprocs, HHS->ndevs, HHS->nheaps);
  fprintf(out, "nhostusers: ");
  for (j = 0; j < HHS->nheaps; j++) {
    fprintf(out, "%d, ", HHS->nhostusers[j]);
  }
  fprintf(out, "\n");

  double st = Wtime(), et;
  pthread_mutex_lock(&HHS->sched_ml);
  et = Wtime();
  fprintf(out, "lock sched_ml took %.1lfms\n",
	  (et-st)*1000);

  ssize_t hmsums[HHST_MAX];
  for (j = 0; j < HHST_MAX; j++) hmsums[j] = 0L;

  fprintf(out, "----------------------\n");
  for (i = 0; i < HHS->nlprocs; i++) {
    struct proc *HHL = &HHS->lprocs[i];
    fprintf(out, "P%03d/L%03d/pid%05d:  %-10s %-10s hpid=%d in_api=%d\n",
	    HHL->rank, HHL->lrank, HHL->pid,
	    hhp_names[HHL->pmode], hhd_names[HHL->dmode], HHL->hpid, HHL->in_api);
    fprintf(out, "    host mem stat(MiB): ");
    for (j = 0; j < HHST_MAX; j++) {
      fprintf(out, "%s=%ld  ", hhst_names[j], HHL->hmstat.used[j]>>20L);
      hmsums[j] += HHL->hmstat.used[j];
    }
    fprintf(out, "\n");
  }	  

  pthread_mutex_unlock(&HHS->sched_ml);

  fprintf(out, "----------------------\n");
  /* sum */
  fprintf(out, "    total host mem stat(MiB): ");
  for (j = 0; j < HHST_MAX; j++) {
    fprintf(out, "%s=%ld  ", hhst_names[j], hmsums[j]>>20L);
  }
  fprintf(out, "\n");

  fprintf(out, "----------------------\n");

  for (i = 0; i < HHS->ndevs; i++) {
    struct dev *d = &HHS->devs[i];
    fprintf(out, "DEVICE %d: memsize=%ldMiB, heapsize=%ldMiB, np_in=%d, np_out=%d \n",
	    d->devid, d->memsize>>20, d->default_heapsize>>20, d->np_in, d->np_out);
    fprintf(out, "    heap_user: ");
    for (j = 0; j < HHS->nheaps; j++) {
      fprintf(out, "%d, ", d->hp_user[j]);
    }
    fprintf(out, "\n");
  }
  fprintf(out, "\n");

  exit(0);
}
