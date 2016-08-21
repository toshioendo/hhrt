#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <malloc.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include "hhrt_impl.h"


int mempool::setSwapper(swapper *swapper0) 
{
    if (curswapper != NULL) {
      fprintf(stderr, "[HH:mempool@p%d] curswapper set twice, this is ERROR!\n",
	      HH_MYID);
      exit(1);
    }
    curswapper = swapper0;
    return 0;
}

int mempool::finalizeRec()
{
  if (curswapper != NULL) {
    /* recursive finalize */
    curswapper->finalizeRec();
  }

  finalize();
  return 0;
}
