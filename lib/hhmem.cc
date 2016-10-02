#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <malloc.h>
#include <errno.h>
#include "hhrt_impl.h"

/* statistics about host memory for debug */
int HH_addHostMemStat(int kind, ssize_t incr)
{
  ssize_t s;
  ssize_t olds;
  assert(kind >= 0 && kind < HHST_MAX);
  olds = HHL->hmstat.used[kind];
  HHL->hmstat.used[kind] += incr;
  s = HHL->hmstat.used[kind];

  if (s < 0 || s > (ssize_t)128 << 30) {
    fprintf(stderr, "[HH_addHostMemStat@p%d] host mem usage (kind %s) %ldMB looks STRANGE.\n",
	    HH_MYID, hhst_names[kind], s>>20L);
  }

#if 0
  fprintf(stderr, "[HH_addHostMemStat@p%d] host mem usage (kind %s) %ldMB -> %ldMB.\n",
	  HH_MYID, hhst_names[kind], olds>>20L, s>>20L);
#endif
  return 0;
}

int HH_printHostMemStat()
{
  double t = Wtime_prt();
  int i;
  fprintf(stderr, "[HH_printHostMemStat@p%d] [%.2lf] ",
	  HH_MYID, t);
  for (i = 0; i < HHST_MAX; i++) {
    fprintf(stderr, "%s:%ldMB  ",
	    hhst_names[i], HHL->hmstat.used[i] >>20L);
  }
  fprintf(stderr, "\n");
  return 0;
}

