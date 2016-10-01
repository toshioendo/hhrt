#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include "hhrt_impl.h"

#define PROFFN "%s/hhprof-p%d.log"

int HH_profInit()
{
  if (!HHL2->conf.use_prof) {
    return 0; // do nothing
  }

  char fn[CONFSTRLEN+32];
  sprintf(fn, PROFFN, HHL2->conf.prof_dir, HH_MYID);
  HHL2->prof.fp = fopen(fn, "w");
  if (HHL2->prof.fp == NULL) {
    fprintf(stderr, "[HH_profInit@p%d] fopen %s failed\n",
	    HH_MYID, fn);
    exit(1);
  }

  strcpy(HHL2->prof.mode, "");
  HHL2->prof.modest = Wtime();
  strcpy(HHL2->prof.act, "");
  HHL2->prof.actst = Wtime();

  return 0;
}

int HH_profSetMode(const char *str)
{
  if (!HHL2->conf.use_prof) {
    return 0; // do nothing
  }

  double et = Wtime();
  if (strlen(HHL2->prof.mode) > 0) {
    // output info of previous mode 
    double st = HHL2->prof.modest;

    if (et-st > 0.002) {
      fprintf(HHL2->prof.fp,
	      "%d MODE %s %.3lf %.3lf\n",
	      HH_MYID, HHL2->prof.mode, Wtime_conv_prt(st), Wtime_conv_prt(et));
      fflush(HHL2->prof.fp);

#if 1
      fprintf(stderr,
	      "[HH_profSetMode@p%d] [%.2lf] Change Mode %s to %s\n",
	      HH_MYID, Wtime_conv_prt(et), HHL2->prof.mode, str);
#endif
    }
  }

  strcpy(HHL2->prof.mode, str);
  HHL2->prof.modest = et;
  return 0;
}

int HH_profBeginAction(const char *str)
{
  if (!HHL2->conf.use_prof) {
    return 0; // do nothing
  }

  double t = Wtime();
  strcpy(HHL2->prof.act, str);
  HHL2->prof.actst = t;
#if 0
  fprintf(stderr, "[HH_profBeginAction@p%d] begin %s\n",
	  HH_MYID, str);
#endif
  return 0;
}

int HH_profEndAction(const char *str)
{
  if (!HHL2->conf.use_prof) {
    return 0; // do nothing
  }

  double st = HHL2->prof.actst;
  double et = Wtime();
  if (strcmp(str, HHL2->prof.act) != 0) {
    fprintf(stderr, "[HH_profEndAction@p%d] %s is specified now, but current action has been %s\n",
	    HH_MYID, str, HHL2->prof.act);
  }
  assert(strcmp(str, HHL2->prof.act) == 0);

#if 0
  fprintf(stderr, "[HH_profEndAction@p%d] end %s\n",
	  HH_MYID, str);
#endif
  if (et-st > 0.002) {
    fprintf(HHL2->prof.fp,
	    "%d ACT %s %.3lf %.3lf\n",
	    HH_MYID, str, Wtime_conv_prt(st), Wtime_conv_prt(et));
    fflush(HHL2->prof.fp);
  }

  return 0;
}

